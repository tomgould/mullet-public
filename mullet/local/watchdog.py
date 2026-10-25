"""Watchdog for the local LLM server (LM Studio on localhost:1234 by default).

``probe()`` asks the server which models it has, without sending a completion
or loading anything. ``check()`` compares that with the previous check (kept
in a small JSON file) and decides whether the server needs a restart:

1. unreachable, or a probe slower than 10 s, on two checks in a row:
   "not answering";
2. a model "loading" for more than 5 minutes across checks:
   "stuck loading <id>";
3. a loaded model whose one-token generation times out on two checks in a
   row: "generation hangs on <id>". A model that isn't loaded is never sent
   a generation, so a check never loads a model.

Every outside call (HTTP, clock) is a parameter so tests can use fakes.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from mullet.jobs.notify import notify

SLOW_PROBE_S = 10.0
STUCK_LOADING_S = 5 * 60.0
GENERATION_TIMEOUT_S = 30.0
RESTART_COOLDOWN_S = 5 * 60.0

Fetch = Callable[..., tuple[int, Any]]
GenerationFn = Callable[..., bool]


@dataclass
class Probe:
    reachable: bool
    latency_s: float | None = None
    models: list[tuple[str, str]] = field(default_factory=list)
    error: str | None = None


@dataclass
class Verdict:
    action: str  # "ok" | "restart" | "busy"
    reason: str
    probe: Probe


@dataclass
class RestartResult:
    ok: bool
    steps: list[list[str]]
    error: str | None = None


def _default_base_url() -> str:
    from mullet.core import env

    return env.local_llm_url()


def _root(base_url: str) -> str:
    url = base_url.rstrip("/")
    return url[: -len("/v1")] if url.endswith("/v1") else url


def _urllib_fetch(url: str, *, timeout: float) -> tuple[int, Any]:
    """GET ``url`` and return (status, parsed JSON). Raises on any failure."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        status = resp.status
        body = resp.read()
    return status, json.loads(body.decode("utf-8") or "null")


def _model_list(data: Any, *, native: bool) -> list[tuple[str, str]] | None:
    """The (id, state) pairs in a model list response, or None if it isn't one."""
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        return None
    models: list[tuple[str, str]] = []
    for entry in data["data"]:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        state = str(entry.get("state") or "unknown") if native else "unknown"
        models.append((str(entry["id"]), state))
    return models


def probe(
    base_url: str | None = None,
    *,
    timeout: float = 5.0,
    fetch: Fetch = _urllib_fetch,
    clock: Callable[[], float] = time.monotonic,
) -> Probe:
    """Ask the server which models it has. Never raises, never loads a model.

    Tries LM Studio's native ``{root}/api/v0/models`` first (it reports each
    model's state), then the OpenAI-style ``{base_url}/models``.
    """
    try:
        base = (base_url or _default_base_url()).rstrip("/")
        attempts = [
            (f"{_root(base)}/api/v0/models", True),
            (f"{base}/models", False),
        ]
    except Exception as exc:  # noqa: BLE001
        return Probe(reachable=False, error=f"{type(exc).__name__}: {exc}")

    error: str | None = None
    for url, native in attempts:
        started = clock()
        try:
            status, data = fetch(url, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            error = f"{url}: {type(exc).__name__}: {exc}"
            continue
        latency = max(0.0, clock() - started)
        if status != 200:
            error = f"{url}: HTTP {status}"
            continue
        models = _model_list(data, native=native)
        if models is None:
            error = f"{url}: not a model list"
            continue
        return Probe(reachable=True, latency_s=latency, models=models)
    return Probe(reachable=False, error=error)


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return True
        return "timed out" in str(reason).lower()
    return "timed out" in str(exc).lower()


def generate_once(base_url: str, model_id: str, *, timeout: float = GENERATION_TIMEOUT_S) -> bool:
    """Send one tiny completion to a loaded model.

    Returns False only when the request timed out; any other failure counts as
    an answer, since only a hang is a reason to restart.
    """
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            resp.read()
    except Exception as exc:  # noqa: BLE001
        return not _is_timeout(exc)
    return True


def default_log_path() -> Path:
    from mullet.core import paths

    return paths.state_dir() / "local-watchdog.log"


def _port(base_url: str | None) -> str:
    if base_url is None:
        from mullet.core import env

        base_url = env.local_llm_url()
    parsed = urllib.parse.urlparse(base_url)
    return str(parsed.port or 1234)


def _log_restart(path: Path, entry: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
    except OSError:
        pass


def last_restart(log_path: Path | str | None = None) -> dict[str, Any] | None:
    """Return the last entry from the restart log, or None if it cannot be read."""
    path = Path(log_path) if log_path is not None else default_log_path()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except ValueError:
            continue
    return None


def _on_cooldown(now: float, log_path: Path | str | None) -> bool:
    last = last_restart(log_path)
    if last is None:
        return False
    when = last.get("time")
    if not isinstance(when, (int, float)):
        return False
    return now - when < RESTART_COOLDOWN_S


def restart(
    *,
    hard: bool = False,
    run: Callable[..., object] = subprocess.run,
    unit: str = "lmstudio.service",
    base_url: str | None = None,
    probe_fn: Callable[[str], Probe] = probe,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    wait_s: float = 90.0,
    reason: str = "manual",
    log_path: Path | str | None = None,
    notify_fn: Callable[..., Any] = notify,
) -> RestartResult:
    """Restart the local LLM server and wait for it to answer.

    Uses ``systemctl --user`` when the unit exists, otherwise the ``lms`` CLI.
    Every outside call is injectable so tests can use fakes.
    """
    from mullet.core.providers.lmstudio_manager import _find_lms_cli

    steps: list[list[str]] = []
    first_error: str | None = None
    ok = False

    try:
        lms = _find_lms_cli()
        base = (base_url or _default_base_url()).rstrip("/")

        def run_step(argv: list[str], timeout: float) -> None:
            nonlocal first_error
            try:
                run(argv, timeout=timeout, check=False, capture_output=True)
            except Exception as exc:  # noqa: BLE001
                if first_error is None:
                    first_error = f"{type(exc).__name__}: {exc}"
            steps.append(argv)

        if hard:
            run_step([lms, "daemon", "down"], 30)
            run_step([lms, "daemon", "up"], 30)

        has_unit = False
        try:
            result = run(["systemctl", "--user", "cat", unit], timeout=10, check=False, capture_output=True)
            has_unit = getattr(result, "returncode", -1) == 0
        except Exception:  # noqa: BLE001
            has_unit = False

        if has_unit:
            run_step(["systemctl", "--user", "restart", unit], 60)
        else:
            run_step([lms, "server", "stop"], 30)
            run_step([lms, "server", "start", "--port", _port(base_url)], 30)

        deadline = clock() + wait_s
        while True:
            try:
                if probe_fn(base).reachable:
                    ok = True
                    break
            except Exception:  # noqa: BLE001
                pass
            if clock() >= deadline:
                break
            sleep(min(2.0, deadline - clock()))

        if not ok and first_error is None:
            first_error = "probe did not become reachable"
    except Exception as exc:  # noqa: BLE001
        if first_error is None:
            first_error = f"{type(exc).__name__}: {exc}"

    final_log_path = Path(log_path) if log_path is not None else default_log_path()
    entry = {
        "time": time.time(),
        "reason": reason,
        "hard": hard,
        "ok": ok,
        "steps": steps,
        "error": first_error,
    }
    _log_restart(final_log_path, entry)

    try:
        body = "The local LLM is answering again." if ok else "The local LLM did not come back."
        notify_fn(f"Local LLM restarted: {reason}", body)
    except Exception:  # noqa: BLE001
        pass

    return RestartResult(ok=ok, steps=steps, error=first_error)


def default_state_path() -> Path:
    from mullet.core import paths

    return paths.state_dir() / "local-watchdog.json"


def _load_state(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2, sort_keys=True)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError:
        pass


def _gpu_busy() -> str | None:
    """The purpose of the GPU lease's holder, or None when nobody holds it."""
    try:
        from mullet.jobs import machine

        holder = machine.current_holder()
        return holder.purpose if holder else None
    except Exception:  # an unreadable lease must never stop a check
        return None


def check(
    state_path: Path | str | None = None,
    *,
    base_url: str | None = None,
    now: float | None = None,
    probe_fn: Callable[[str], Probe] = probe,
    generation_fn: GenerationFn = generate_once,
    log_path: Path | str | None = None,
    gpu_busy_fn: Callable[[], str | None] | None = None,
) -> Verdict:
    """Probe the server and decide whether it needs a restart.

    The previous check's facts are read from and written back to
    ``state_path``. A missing or corrupt state file means no history.
    While another Mullet process holds the GPU lease (a local job attempt,
    a reindex), no generation is sent: it would queue behind that work,
    time out and restart the server under it. ``gpu_busy_fn`` returns the
    holder's purpose, or None when the GPU is free.
    """
    path = Path(state_path) if state_path is not None else default_state_path()
    base = (base_url or _default_base_url()).rstrip("/")
    now = time.time() if now is None else float(now)

    prev = _load_state(path)
    prev_loading = prev.get("loading_since")
    if not isinstance(prev_loading, dict):
        prev_loading = {}
    prev_timeouts = prev.get("generation_timeouts")
    if not isinstance(prev_timeouts, list):
        prev_timeouts = []

    result = probe_fn(base)
    bad_probe = not result.reachable or (
        result.latency_s is not None and result.latency_s > SLOW_PROBE_S
    )
    state: dict[str, Any] = {
        "checked_at": now,
        "bad_probe": bad_probe,
        "loading_since": {},
        "generation_timeouts": [],
    }

    def done(action: str, reason: str) -> Verdict:
        _save_state(path, state)
        return Verdict(action=action, reason=reason, probe=result)

    if not result.reachable:
        # Nothing else can be judged; keep the loading history for next time.
        state["loading_since"] = prev_loading
        if prev.get("bad_probe") is True:
            if _on_cooldown(now, log_path):
                return done("busy", "not answering")
            return done("restart", "not answering")
        return done("ok", f"no answer once: {result.error or 'unreachable'}")

    restart_reason: str | None = None
    if bad_probe and prev.get("bad_probe") is True:
        restart_reason = "not answering"

    for model_id, model_state in result.models:
        if model_state != "loading":
            continue
        since = prev_loading.get(model_id)
        if not isinstance(since, (int, float)):
            since = now
        state["loading_since"][model_id] = since
        if restart_reason is None and now - since > STUCK_LOADING_S:
            restart_reason = f"stuck loading {model_id}"

    busy = (gpu_busy_fn or _gpu_busy)() if restart_reason is None else None
    if busy:
        state["generation_timeouts"] = []
        _save_state(path, state)
        return Verdict(action="ok", reason=f"answering; generation not checked while the GPU is in use ({busy})",
                       probe=result)

    if restart_reason is None:
        for model_id, model_state in result.models:
            if model_state != "loaded":
                continue
            answered = generation_fn(base, model_id, timeout=GENERATION_TIMEOUT_S)
            if answered:
                continue
            state["generation_timeouts"].append(model_id)
            if model_id in prev_timeouts:
                restart_reason = f"generation hangs on {model_id}"
                break

    if restart_reason is not None:
        if _on_cooldown(now, log_path):
            return done("busy", restart_reason)
        return done("restart", restart_reason)
    if bad_probe:
        return done("ok", f"slow answer once: {result.latency_s:.1f} s")
    if state["generation_timeouts"]:
        return done("ok", "generation timed out once on " + ", ".join(state["generation_timeouts"]))
    return done("ok", "answering")
