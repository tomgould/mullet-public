"""opencode tool — runs a coding task via opencode using free/Kimi models.

Dispatches via `opencode run <task> --model X --format json --dir Y` (a
one-shot CLI invocation, not the ACP/JSON-RPC session protocol): opencode's
ACP `session/new` "model" parameter is silently ignored in practice — it
always dispatches to whatever ~/.config/opencode/opencode.json's top-level
"model" field says, regardless of what the ACP client requests. `opencode
run --model` is the one invocation opencode actually honours per-call
(confirmed empirically: the same task explicitly targeting different
providers via `--model` correctly produces provider-specific errors/results,
where ACP mode always produced the config-default provider's behaviour).

`--format json` emits line-delimited JSON events on stdout:
  step_start   → a reasoning/tool-call step begins
  text         → part.text is a response text chunk (accumulate)
  tool_use     → part.tool/part.state describe a file-edit/bash/etc call
  step_finish  → part.tokens/part.cost for that step (accumulate across steps)
  error        → part/error.data.message is opencode's real, immediate,
                 correctly-diagnosed failure — surfaced without waiting out
                 SILENT_TIMEOUT_S
Completion is signalled by the process exiting (stdout EOF); exit code 0
is success, non-zero is failure.

Zero Anthropic tokens: uses opencode/big-pickle (free) by default,
or moonshot/kimi-k2.7-code when ~/.config/opencode/opencode.json configures Moonshot.
"""
import json
import logging
import os
import select
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from mullet.core.providers.model_info import get_context_length
from mullet.core.providers.model_registry import CONTEXT_SAFETY_MARGIN
from mullet.core.providers.tokenize import estimate_tokens

logger = logging.getLogger(__name__)

# Free model, no API key required
_DEFAULT_MODEL = "opencode/big-pickle"


def _resolve_configured_model() -> str:
    """Read the effective default model from ~/.config/opencode/opencode.json, if set."""
    try:
        cfg_path = os.path.expanduser("~/.config/opencode/opencode.json")
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        m = cfg.get("model")
        if m and isinstance(m, str):
            return m
    except Exception:
        pass
    return _DEFAULT_MODEL


def _resolve_kimi_key() -> str | None:
    """Return KIMI_API_KEY from environment, falling back to .mcp.json."""
    key = os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
    if key:
        return key
    for candidate in [
        os.path.join(os.getcwd(), ".mcp.json"),
        os.path.expanduser("~/.mcp.json"),
    ]:
        try:
            data = json.loads(open(candidate).read())
            for _srv in data.get("mcpServers", {}).values():
                k = _srv.get("env", {}).get("KIMI_API_KEY") or _srv.get("env", {}).get("MOONSHOT_API_KEY")
                if k:
                    return k
        except Exception:
            pass
    return None


# Provider API keys that should be forwarded to opencode so cloud models
# (openrouter, anthropic direct, openai, etc.) work without the MCP server
# needing them in its own process environment.
_FORWARDED_API_KEYS = (
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "XAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "QWEN_API_KEY",
)


def _read_dotenv(path: str) -> dict[str, str]:
    """Parse a .env file into a dict (no shell expansion, no exports)."""
    out: dict[str, str] = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k:
                    out[k] = v
    except Exception:
        pass
    return out


def _resolve_forwarded_keys() -> dict[str, str]:
    """Resolve API keys for cloud providers, checking env then known .env files.

    Searches (in order): process env, ~/.mullet/.env, $PWD/.env.
    Returns only keys that resolved to a value.
    """
    found: dict[str, str] = {}
    dotenv_candidates = [
        os.path.expanduser("~/.mullet/.env"),
        os.path.join(os.getcwd(), ".env"),
    ]
    dotenv_data: dict[str, str] = {}
    for path in dotenv_candidates:
        if os.path.isfile(path):
            for k, v in _read_dotenv(path).items():
                dotenv_data.setdefault(k, v)

    for name in _FORWARDED_API_KEYS:
        val = os.environ.get(name) or dotenv_data.get(name)
        if val:
            found[name] = val
    return found


def _mullet_run_fallback(task: str, cwd: str | None) -> dict:
    """Fallback when opencode is not installed: delegate to Kimi via mullet_run."""
    logger.warning("opencode not installed — falling back to mullet_run/Kimi (no filesystem access)")
    try:
        from mullet.server.mcp_handlers import handle_mullet_run
        result = handle_mullet_run({
            "task": task,
            "tier": "STANDARD",
            "target_dir": cwd or os.getcwd(),
        })
        return {
            "completed": True,
            "response": result.get("response") or result.get("error") or str(result),
            "duration": result.get("duration", 0),
            "fallback": "mullet_run/kimi",
            "warning": "opencode not installed — install with: npm i -g opencode-ai@latest",
        }
    except Exception as exc:
        return {
            "error": f"opencode not installed and mullet_run fallback failed: {exc}",
            "install_hint": "npm i -g opencode-ai@latest",
        }


def _git_status_snapshot(cwd: str) -> set[str] | None:
    """Return the set of changed-file paths per `git status --short`, or None if unavailable.

    Used to diff before/after a delegate run so out-of-scope edits (a stray
    config file touched alongside the intended task) are surfaced in the
    result instead of silently going unnoticed.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=cwd, capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        return {line[3:].strip() for line in result.stdout.splitlines() if line.strip()}
    except Exception:
        return None


def _classify_error(message: str) -> str:
    """Classify an opencode failure message into a coarse reason bucket.

    One of: "context_overflow", "auth", "transient", "unknown". Lets callers
    (escalation logic, the calling agent) react differently instead of
    blind-retrying every failure the same way — e.g. a context_overflow
    should jump straight to the largest-context model, not the next small
    step in the normal chain; an auth failure won't be fixed by retrying at
    all.
    """
    msg = message.lower()
    if any(s in msg for s in ("n_ctx", "n_keep", "context length", "context window", "exceeds", "too many tokens")):
        return "context_overflow"
    if any(s in msg for s in ("unauthorized", "401", "403", "forbidden", "invalid api key", "authentication", "api key not valid")):
        return "auth"
    if any(s in msg for s in ("no output for", "overloaded", "timed out", "timeout", "rate limit", "429", "503", "connection reset", "exited unexpectedly")):
        return "transient"
    return "unknown"


# Escalation targets for a context_overflow failure specifically, NOT the
# same as _escalate_model's normal chain below: that chain isn't
# monotonically increasing in context size (Kimi's 262K rung is SMALLER
# than OpenRouter DeepSeek V4 Flash's 1M), so blindly following it after a
# context failure could land on an even smaller context and fail again for
# the identical reason. Picks the largest by actual get_context_length(),
# so it stays correct as model_info.py's context figures change.
_CONTEXT_OVERFLOW_ESCALATION_CANDIDATES = (
    "openrouter/qwen/qwen3-coder",
    "openrouter/deepseek/deepseek-v4-flash",
    "moonshot/kimi-k2.7-code",
)


_KNOWN_BAD_MODELS_PATH = Path.home() / ".mullet" / "opencode_known_bad_models.json"
_KNOWN_BAD_TTL_S = 900  # 15 min: long enough to skip repeat failures within a
                          # working session, short enough to self-heal soon
                          # after the underlying issue is fixed (e.g. LM
                          # Studio reloaded with a bigger context) without
                          # needing a manual cache-clear.
_known_bad_lock = threading.Lock()


def _load_known_bad_models() -> dict:
    """Return {model_id: {"reason", "message", "ts"}} for still-fresh entries.

    Same persisted-JSON-with-TTL shape as ConstraintRegistry's learned
    constraints (constraints.py) — a different cache for a different
    purpose (skip a structurally-broken model entirely, not reshape its
    request params), same pattern.
    """
    try:
        if not _KNOWN_BAD_MODELS_PATH.exists():
            return {}
        data = json.loads(_KNOWN_BAD_MODELS_PATH.read_text())
        cutoff = time.time() - _KNOWN_BAD_TTL_S
        return {k: v for k, v in data.items() if v.get("ts", 0) >= cutoff}
    except Exception:
        return {}


def _mark_model_bad(model: str, reason: str, message: str) -> None:
    """Persist that `model` just failed with `reason`, atomically.

    Only called for context_overflow — a structural, deterministic failure
    (same task shape will fail the same way again), unlike transient/auth
    failures which may resolve on their own or need human intervention that
    skip-caching wouldn't help with anyway.
    """
    try:
        with _known_bad_lock:
            data = _load_known_bad_models()
            data[model] = {"reason": reason, "message": message[:200], "ts": time.time()}
            _KNOWN_BAD_MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=_KNOWN_BAD_MODELS_PATH.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f)
                os.replace(tmp, _KNOWN_BAD_MODELS_PATH)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
    except Exception:
        pass


def _known_bad_reason(model: str) -> str | None:
    """Return the cached failure reason for `model` if still fresh, else None."""
    return _load_known_bad_models().get(model, {}).get("reason")


def _escalate_for_context_overflow(current_model: str) -> str | None:
    """Pick the largest-context candidate that isn't the current model."""
    candidates = [m for m in _CONTEXT_OVERFLOW_ESCALATION_CANDIDATES if m != current_model]
    if not candidates:
        return None
    return max(candidates, key=get_context_length)


def _escalate_model(model: str) -> str | None:
    """Return the next model in the fallback chain, or None if already at the end.

    local/lmstudio (small/unknown context) -> OpenRouter DeepSeek V4 Flash (1M ctx)
    openrouter/*                           -> Kimi K2.7 Code (262K ctx)
    kimi/moonshot (already top of chain)   -> None, nothing bigger to try
    anything else (incl. opencode's own free default, which has no context
    guarantee) -> same first hop as local, since it's equally unverified.
    """
    if "kimi" in model.lower() or model.startswith("moonshot/"):
        return None
    if model.startswith(("local/", "lmstudio/")) or "local" in model:
        return "openrouter/deepseek/deepseek-v4-flash"
    if "openrouter" in model:
        return "moonshot/kimi-k2.7-code"
    return "openrouter/deepseek/deepseek-v4-flash"


def _provider_from_model(model: str) -> str:
    """Map a model ID string to a provider name for routing history."""
    if not model or model.startswith("opencode/") or "local" in model or "lmstudio" in model or "ollama" in model:
        return "local"
    if model.startswith("moonshot/") or "kimi" in model.lower():
        return "kimi"
    if model.startswith("openrouter/"):
        return "openrouter"
    if model.startswith("anthropic/") or "claude" in model.lower():
        return "anthropic"
    return "openrouter"  # safe default for unknown cloud providers


def _update_hud_model(agent_id: str | None, model: str) -> None:
    """Best-effort HUD correction when opencode_run silently swaps models.

    Soft-fails — a HUD write issue must never affect the actual delegation.
    """
    if not agent_id:
        return
    try:
        from mullet.hud.agent_state import update_agent_model
        update_agent_model(agent_id, _provider_from_model(model), model)
    except Exception:
        pass


def opencode_run(
    task: str,
    cwd: str | None = None,
    model: str | None = None,
    agent_id: str | None = None,
) -> dict:
    """Run a coding task via opencode ACP with native filesystem access.

    Args:
        task: Natural language coding task.
        cwd: Project directory. Defaults to current working directory.
        model: Model override. Defaults to opencode global config model.
        agent_id: HUD agent id (from ``add_agent``). When given, the HUD
            entry's provider/model are corrected in place on escalation, so
            a long-running retry never shows a stale (attempt-1) model.

    Returns:
        dict with ``completed``, ``response``, ``duration``, ``model``,
        ``usage`` on success, or ``error`` key on failure.
    """
    if not shutil.which("opencode"):
        return _mullet_run_fallback(task, cwd)

    env = os.environ.copy()
    kimi_key = _resolve_kimi_key()
    if kimi_key:
        env.setdefault("KIMI_API_KEY", kimi_key)
        env.setdefault("MOONSHOT_API_KEY", kimi_key)
    for name, val in _resolve_forwarded_keys().items():
        env.setdefault(name, val)

    cwd_resolved = os.path.abspath(cwd or os.getcwd())
    resolved_model = model or _resolve_configured_model()
    task_tokens = estimate_tokens(task)
    before_status = _git_status_snapshot(cwd_resolved)

    t0 = time.monotonic()
    last_err = None
    last_err_reason = None

    from shutil import which as _which
    _prio_prefix = []
    if _which("nice"):
        _prio_prefix += ["nice", "-n", "19"]
    if _which("ionice"):
        _prio_prefix += ["ionice", "-c", "3"]

    for attempt in range(1, 3):
        # Known-bad check: a model that recently failed with context_overflow
        # is structurally broken for this environment (e.g. LM Studio loaded
        # with a smaller context than model_info.py assumes) — the static
        # pre-flight guard below can't catch this since it only knows the
        # ASSUMED context, not what's actually loaded. Skip straight to
        # escalation instead of repeating a call known to fail.
        known_bad_reason = _known_bad_reason(resolved_model)
        if known_bad_reason == "context_overflow":
            last_err = RuntimeError(
                f"{resolved_model} recently failed with context_overflow — skipping known-bad model"
            )
            last_err_reason = "context_overflow"
            logger.warning("opencode_run: skipping known-bad model on attempt %d: %s", attempt, last_err)
            next_model = _escalate_for_context_overflow(resolved_model)
            if attempt == 1 and next_model is not None:
                resolved_model = next_model
                _update_hud_model(agent_id, resolved_model)
                continue
            break

        # Pre-flight D1/D4-style guard (see context_gate.py / v1.8.1 delegation
        # safety): reject before spawning a subprocess when the task can't
        # possibly fit the target model's context window, instead of burning
        # a full SILENT_TIMEOUT_S waiting on a call that was never going to
        # succeed. Soft-fails (skips the check) if context lookup itself errors.
        try:
            ctx_len = get_context_length(resolved_model)
            overflow = ctx_len > 0 and task_tokens * CONTEXT_SAFETY_MARGIN > ctx_len
        except Exception as exc:
            logger.debug("opencode_run: context lookup for %s failed (%s); skipping pre-flight check", resolved_model, exc)
            overflow = False

        if overflow:
            last_err = RuntimeError(
                f"task (~{task_tokens} tokens) exceeds {resolved_model}'s context "
                f"window ({ctx_len} tokens) — not dispatching"
            )
            last_err_reason = "context_overflow"
            logger.warning("opencode_run: pre-flight context check failed on attempt %d: %s", attempt, last_err)
            next_model = _escalate_for_context_overflow(resolved_model)
            if attempt == 1 and next_model is not None:
                resolved_model = next_model
                _update_hud_model(agent_id, resolved_model)
                continue
            break

        try:
            proc = subprocess.Popen(
                _prio_prefix + [
                    "opencode", "run", task,
                    "--model", resolved_model,
                    "--format", "json",
                    "--dir", cwd_resolved,
                    "--auto",  # non-interactive: no TTY to approve permission prompts
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # line-buffered: each readline() returns on \n, not on 4KB block
                env=env,
            )
        except Exception as exc:
            return {"error": f"failed to spawn opencode: {exc}"}

        # Maximum silent stretch before assuming opencode has wedged.
        # 90s was originally chosen but proved too aggressive for substantial
        # tasks: a complex prompt against a deep-reasoning model (gpt-5, opus,
        # DeepSeek R1) plus a few sequential tool calls can exceed 90s of
        # silence on stdout between events. 240s is a more forgiving default;
        # override per-call via OPENCODE_SILENT_TIMEOUT_S.
        SILENT_TIMEOUT_S = float(os.environ.get("OPENCODE_SILENT_TIMEOUT_S", "240"))

        def _read_run_events() -> dict:
            """Read `opencode run --format json` line-delimited events until
            the process exits. A task can involve multiple tool_use +
            step_finish cycles (read a file, edit it, then respond) before
            the final text — token/cost usage is summed across all of them,
            not just the last one.

            Completion is the process exiting: stdout EOF, then check the
            real exit code (0 = success). An `error` event is opencode's own
            immediate, correctly-diagnosed failure — raised right away so
            the caller's retry/escalation logic sees it without waiting out
            SILENT_TIMEOUT_S.
            """
            text_chunks: list[str] = []
            tokens_total = tokens_in = tokens_out = 0
            cost_total = 0.0
            stderr_lines: list[str] = []
            session_id = None
            last_activity = time.monotonic()

            while True:
                poll_result = proc.poll()

                ready, _, _ = select.select([proc.stdout, proc.stderr], [], [], 0.5)

                if ready:
                    last_activity = time.monotonic()

                stdout_eof = False
                for stream in ready:
                    if stream is proc.stderr:
                        line = proc.stderr.readline()
                        if line:
                            stripped = line.rstrip("\n")
                            stderr_lines.append(stripped)
                            logger.warning("opencode stderr: %s", stripped)
                    elif stream is proc.stdout:
                        line = proc.stdout.readline()
                        if not line:
                            stdout_eof = True
                            break
                        try:
                            m = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        session_id = m.get("sessionID") or session_id
                        ev_type = m.get("type", "")
                        part = m.get("part") or {}

                        if ev_type == "text":
                            chunk = part.get("text", "")
                            if chunk:
                                text_chunks.append(chunk)
                                logger.debug("opencode chunk: %s", chunk[:80])

                        elif ev_type == "tool_use":
                            state = part.get("state") or {}
                            logger.info(
                                "opencode tool: %s [%s] %s",
                                part.get("tool", ""), state.get("status", ""), part.get("title", ""),
                            )

                        elif ev_type == "step_finish":
                            tok = part.get("tokens") or {}
                            tokens_total += tok.get("total", 0) or 0
                            tokens_in += tok.get("input", 0) or 0
                            tokens_out += tok.get("output", 0) or 0
                            cost_total += part.get("cost", 0) or 0

                        elif ev_type == "error":
                            err = m.get("error") or {}
                            msg = (err.get("data") or {}).get("message") or err.get("message") or str(err)
                            if proc.poll() is None:
                                proc.kill()
                            raise RuntimeError(f"opencode run error: {msg}")

                if stdout_eof:
                    break

                if poll_result is not None and not ready:
                    break

                if time.monotonic() - last_activity > SILENT_TIMEOUT_S and proc.poll() is None:
                    proc.kill()
                    stderr_msg = "\n".join(stderr_lines) if stderr_lines else "(no stderr)"
                    raise RuntimeError(
                        f"opencode: no output for {int(SILENT_TIMEOUT_S)}s — API may be overloaded; "
                        f"retrying. Override with OPENCODE_SILENT_TIMEOUT_S env var. "
                        f"stderr: {stderr_msg}"
                    )

            try:
                exit_code = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                exit_code = proc.wait(timeout=5)
            if exit_code != 0:
                stderr_msg = "\n".join(stderr_lines) if stderr_lines else "(no stderr)"
                raise RuntimeError(f"opencode exited with code {exit_code}. stderr: {stderr_msg}")

            return {
                "text": "".join(text_chunks),
                "session_id": session_id,
                "usage": {
                    "total_tokens": tokens_total,
                    "input_tokens": tokens_in,
                    "output_tokens": tokens_out,
                    "cached_tokens": 0,
                    "cost_usd": cost_total,
                    "currency": "USD",
                },
            }

        try:
            result = _read_run_events()
            duration = round(time.monotonic() - t0, 1)

            after_status = _git_status_snapshot(cwd_resolved)
            files_changed = (
                sorted(after_status - before_status)
                if before_status is not None and after_status is not None
                else None
            )

            _success_result = {
                "completed": True,
                "response": result["text"] or "(no text output)",
                "duration": duration,
                "model": resolved_model,
                "session_id": result["session_id"],
                "usage": result["usage"],
                "files_changed": files_changed,
            }
            try:
                from mullet.core.routing_history import make_outcome, record as _rh_record
                _usage = result["usage"]
                _rh_record(make_outcome(
                    task=task,
                    provider=_provider_from_model(resolved_model),
                    model=resolved_model,
                    tier="STANDARD",
                    success=True,
                    tokens_in=_usage.get("input_tokens") or 0,
                    tokens_out=_usage.get("output_tokens") or 0,
                    wall_time_ms=int(duration * 1000),
                    tool_family="opencode",
                    tags=["opencode"],
                ))
            except Exception:
                pass
            return _success_result

        except (RuntimeError, TimeoutError) as exc:
            last_err = exc
            last_err_reason = _classify_error(str(exc))
            logger.warning(
                "opencode attempt %d failed (%s) [reason=%s], retrying...",
                attempt, exc, last_err_reason,
            )
            if last_err_reason == "context_overflow":
                _mark_model_bad(resolved_model, "context_overflow", str(exc))
            if attempt == 1:
                next_model = (
                    _escalate_for_context_overflow(resolved_model)
                    if last_err_reason == "context_overflow"
                    else _escalate_model(resolved_model)
                )
                if next_model is not None:
                    resolved_model = next_model
                    _update_hud_model(agent_id, resolved_model)
            else:
                break
        except Exception as exc:
            try:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=5)
            except Exception:
                pass
            return {"error": f"opencode run error: {exc}"}

    _fail_duration = round(time.monotonic() - t0, 1)
    try:
        from mullet.core.routing_history import make_outcome, record as _rh_record
        _rh_record(make_outcome(
            task=task,
            provider=_provider_from_model(resolved_model),
            model=resolved_model,
            tier="STANDARD",
            success=False,
            tokens_in=0,
            tokens_out=0,
            wall_time_ms=int(_fail_duration * 1000),
            error_class=type(last_err).__name__ if last_err else "unknown",
            error_message=str(last_err) if last_err else None,
            tool_family="opencode",
            tags=["opencode", "failed", f"reason:{last_err_reason or 'unknown'}"],
        ))
    except Exception:
        pass

    after_status = _git_status_snapshot(cwd_resolved)
    files_changed = (
        sorted(after_status - before_status)
        if before_status is not None and after_status is not None
        else None
    )
    return {
        "error": f"opencode failed after 2 attempts: {last_err}",
        "error_reason": last_err_reason or "unknown",
        "files_changed": files_changed,
    }
