"""MCP tool handlers for Mullet orchestration.

These extend Trove's existing MCP server with Mullet-specific tools.
When registered, Claude Code can use Mullet as a backend for:
- Running agents with intelligent routing
- Delegating subtasks
- Checking provider status and budget
- Searching project knowledge via Mullet's router
"""

import json
import logging
import os
from pathlib import Path
import sys
from typing import Tuple, List

# Legacy single-slot writes moved into OpenAICompatibleProvider._hud_agent_track
# (see mullet/core/providers/openai_compat.py). The provider hook adds the agent
# on create() entry and removes on exit, with a per-call id so concurrent
# calls coexist — writing here as well would duplicate every row.
# These shims keep the existing call sites inert so behaviour is preserved on
# pre-hook paths that haven't been removed yet.
def _write_agent_state_fn(*_args, **_kwargs) -> None:  # noqa: D401 — no-op shim
    """No-op shim — provider hook handles HUD state (see openai_compat.py)."""
    return None


def _clear_agent_state() -> None:  # noqa: D401 — no-op shim
    """No-op shim — provider hook removes its own entry on exit."""
    return None


_write_agent_state = _write_agent_state_fn

logger = logging.getLogger(__name__)


_ANTHROPIC_ALIASES = {"haiku", "sonnet", "opus-4-8", "opus-4-7", "opus-4-6"}


def _anthropic_only_needs_native_handoff(allowed: set[str] | None) -> bool:
    """True when the caller restricted routing to Anthropic-only aliases and
    no ANTHROPIC_API_KEY is configured.

    Mullet only reaches Anthropic through AnthropicProvider, which needs a
    real ANTHROPIC_API_KEY, so resolution would just fail here. But if this
    MCP call is coming from Claude Code itself (or any other caller that can
    act on the hint), the honest answer isn't "provider unavailable" -- it's
    "you're already an Anthropic client, just do this yourself instead of
    asking Mullet to do it on your behalf."
    """
    if not allowed or not allowed.issubset(_ANTHROPIC_ALIASES):
        return False
    return not os.environ.get("ANTHROPIC_API_KEY")


_NATIVE_HANDOFF_MESSAGE = (
    "Anthropic-only routing requested but ANTHROPIC_API_KEY is not set. "
    "Set ANTHROPIC_API_KEY (pay-per-token) if you want Mullet to route to "
    "Anthropic models on its own. If you're calling this from Claude Code, "
    "just do the work yourself in-context instead of delegating it here."
)


_security_log = logging.getLogger("mullet.security.audit")


def _sanitize_paths(files: List[str], target_dir: str) -> Tuple[List[str], str, str | None]:
    """Sanitize file paths to prevent directory traversal."""
    safe_target_str = str(target_dir) if target_dir is not None else ""
    try:
        safe_target = Path(target_dir).resolve()
        safe_target_str = str(safe_target)
        safe_files = []
        for file_path in files:
            resolved = (safe_target / file_path).resolve()
            if not str(resolved).startswith(str(safe_target)):
                _security_log.warning(
                    "security.path_traversal blocked file=%r target_dir=%s", file_path, safe_target
                )
                return [], safe_target_str, f"Path traversal detected: {file_path}"
            safe_files.append(str(resolved.relative_to(safe_target)))
        return safe_files, safe_target_str, None
    except Exception as e:
        logger.warning(
            "_sanitize_paths: path resolution failed for target_dir=%r files=%r: %s",
            target_dir, files, e,
        )
        return [], safe_target_str, f"Path sanitization error: {e}"


def _rlog(msg: str) -> None:
    """Always-visible routing log — writes to stderr regardless of log level."""
    print(f"\033[2m[mullet] {msg}\033[0m", file=sys.stderr, flush=True)


def _delegation_off_reply() -> str | None:
    """The refusal message when MULLET_DELEGATION=off, else None.

    Checked first by every handler that sends work to a model provider
    (mullet_run, mullet_delegate, mullet_swarm, mullet_opencode), before any
    routing, so nothing reaches a provider while delegation is turned off.
    Trawl, status, budget and the other read-only tools don't check it.
    """
    from mullet.core.env import DELEGATION_OFF_MESSAGE, delegation_enabled
    if delegation_enabled():
        return None
    return DELEGATION_OFF_MESSAGE


def _active_preset():
    """Return the active Preset, or None if unavailable (fail-open)."""
    try:
        from mullet.core.presets import PresetManager
        return PresetManager().active()
    except Exception:
        return None


# Tool definitions in MCP format
MCP_TOOLS = [
    {
        "name": "mullet_run",
        "description": "For code editing tasks (reading/modifying files), use mullet_opencode instead: it runs the edit as a gated Mullet job "
                       "and applies the result to the working tree only after the gates pass. "
                       "Use mullet_run only for: analysis, planning, research, review, or text generation tasks where no files need to be written. "
                       "Routes to the optimal provider (local GPU or cloud) based on complexity.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The task to execute."},
                "client": {"type": "string", "description": "Trove client for project context (optional)."},
                "tier": {"type": "string", "enum": ["LOW", "STANDARD", "ADVANCED", "CRITICAL", "PREMIUM"],
                         "description": "Force routing tier (optional). "
                                        "PREMIUM routes to Opus when budget pressure is low (<0.30), "
                                        "falling back to Sonnet → Haiku → Kimi as pressure rises."},
                "max_iterations": {"type": "integer", "description": "Max iterations. Default: 90."},
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File paths to include as context for the delegate (ignored — use mullet_opencode for file edits).",
                },
                "apply": {
                    "type": "boolean",
                    "description": "Deprecated. Always returns an error. Use mullet_opencode for file edits.",
                },
                "target_dir": {
                    "type": "string",
                    "description": "Directory to resolve file paths against and apply changes to. "
                                   "Defaults to current working directory.",
                },
                "providers": {
                    "type": "string",
                    "description": "Comma-separated allowlist of providers for this task. "
                                   "Restricts routing to only these providers. "
                                   "Example: 'kimi,haiku,sonnet' excludes local and gemini. "
                                   "Valid: local, kimi, gemini, openrouter, haiku, sonnet, opus-4-8, opus-4-7, opus-4-6",
                },
                "force_single": {
                    "type": "boolean",
                    "description": "If true, disable auto-swarm routing and always use a single "
                                   "provider call even when the heuristic would fan out to a swarm. "
                                   "Default: false (auto-swarm fires when signals are detected).",
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "mullet_delegate",
        "description": "Delegate a subtask to a Mullet child agent. The child runs "
                       "independently with its own context and tools, routed by complexity.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "What the child agent should accomplish."},
                "context": {"type": "string", "description": "Additional context for the child."},
                "client": {"type": "string", "description": "Trove client for project context."},
                "providers": {
                    "type": "string",
                    "description": "Comma-separated allowlist of providers for this task. "
                                   "Restricts routing to only these providers. "
                                   "Example: 'kimi,haiku,sonnet' excludes local and gemini. "
                                   "Valid: local, kimi, gemini, openrouter, haiku, sonnet, opus-4-8, opus-4-7, opus-4-6",
                },
            },
            "required": ["goal"],
        },
    },
    {
        "name": "mullet_status",
        "description": "Check Mullet's current status — active tasks, provider health, budget.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "mullet_providers",
        "description": "List available LLM providers and their health status.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "mullet_search",
        "description": "Search project knowledge for relevant context. Uses Trove's hybrid search engine.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "client": {"type": "string", "description": "Trove client name (project)."},
                "top_k": {"type": "integer", "description": "Max results. Default: 10."},
            },
            "required": ["query", "client"],
        },
    },
    {
        "name": "mullet_remember",
        "description": "Store a memory in a project's knowledge base for future sessions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Short key for this memory."},
                "content": {"type": "string", "description": "Memory content to store."},
                "client": {"type": "string", "description": "Trove client name."},
            },
            "required": ["key", "content", "client"],
        },
    },
    {
        "name": "mullet_analyze",
        "description": "Analyze a task's complexity and show what provider/tier the router would select. Useful for understanding routing decisions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The task to analyze."},
                "agent_type": {"type": "string", "description": "Agent type (planner, executor, etc.)."},
            },
            "required": ["task"],
        },
    },
    {
        "name": "mullet_budget",
        "description": "Query current token spend and budget limits.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "mullet_opencode",
        "description": "PREFERRED for code editing tasks. Runs the task as a single gated Mullet job on the delegation ladder "
                       "(the cheapest worker that can pass the project's gates does the work) and applies the result to the "
                       "working tree only after the gates pass. Changes are left unstaged and are never committed or pushed. "
                       "Returns JSON with status (applied, passed_not_applied, failed, blocked, unverified or timeout), job id, "
                       "report path, rung, model, cost, gate summary and files changed.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Coding task to perform (natural language)."},
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Files or paths the task may change (the job's scope). Defaults to the whole repo.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Project directory. Defaults to the git repo containing the files, else the server's working directory.",
                },
                "timeout": {
                    "type": "integer",
                    "default": 20,
                    "minimum": 1,
                    "maximum": 60,
                    "description": "Minutes to wait for the job before giving up (default 20, max 60).",
                },
                "providers": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["kimi", "openrouter", "free", "local", "claude"]},
                    "description": "Limit the ladder to these rungs. Omit to use the whole ladder.",
                },
                "model": {"type": "string", "description": "Deprecated and ignored: the delegation ladder picks the worker."},
            },
            "required": ["task"],
        },
    },
    {
        "name": "mullet_swarm",
        "description": "Fan-out a goal across many Kimi K2 calls in parallel and synthesize the results. Use for tasks that can be split across multiple files, items, or independent sub-questions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "High-level objective for the swarm"},
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File paths to include as context (glob patterns expanded by caller)",
                },
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Arbitrary string items to process (alternative to files)",
                },
                "concurrency": {"type": "integer", "description": "Max parallel Kimi calls (default 20)"},
                "output_shape": {"type": "string", "description": "Desired output format (default: 'markdown report')"},
                "dry_run": {"type": "boolean", "description": "Decompose only — return task plan without dispatching"},
                "apply": {
                    "type": "boolean",
                    "description": "Deprecated. Always returns an error. Use mullet_opencode for file edits.",
                },
                "target_dir": {
                    "type": "string",
                    "description": "Directory to resolve file paths against. Defaults to cwd.",
                },
            },
            "required": ["goal"],
        },
    },
]


def _resolve_provider(
    task: str = "",
    force_tier: str | None = None,
    allowed: set[str] | None = None,
    context=None,
    agent_type: str | None = None,
    speed_mode: str = "balanced",
):
    """Find the best provider for an MCP task; enforces allowlist as post-pick assertion."""

    if allowed is not None and len(allowed) == 0:
        raise ValueError("allowed_providers is empty set — refusing to route")

    def _assert(prov):
        _resolved_name = getattr(prov, "_mcp_resolved_alias", None) or getattr(prov, "name", None)
        if prov is not None and allowed is not None and _resolved_name not in allowed:
            logger.error(
                "BUG: _provider_for_tier returned %s not in allowlist %s",
                _resolved_name, allowed,
            )
            raise RuntimeError(
                f"provider {_resolved_name!r} not in allowlist {allowed!r}"
            )
        return prov

    if force_tier:
        return _assert(_provider_for_tier(force_tier, allowed=allowed, speed_mode=speed_mode))

    if task:
        from mullet.core.router.complexity import extract_signals, score_complexity, score_to_tier, is_simple_message, classify_task_type, TASK_TYPE_PROVIDERS
        if is_simple_message(task):
            return _assert(_provider_for_tier("LOW", allowed=allowed, speed_mode=speed_mode))
        task_type = classify_task_type(task)
        type_tier = TASK_TYPE_PROVIDERS.get(task_type, "STANDARD")
        signals = extract_signals(task)
        # Pre-flight: ask cheap LLM for output estimate + complexity refinement
        from mullet.core.router.preflight import analyze_prompt
        _preflight = analyze_prompt(
            task,
            files=list(context.get("files", [])) if context else [],
            agent_type=agent_type,
            speed_mode=speed_mode,
        )
        score = score_complexity(signals)
        tier = score_to_tier(score)
        tier_order = {"LOW": 0, "STANDARD": 1, "ADVANCED": 2, "CRITICAL": 3, "PREMIUM": 4}
        # Soften the type-table floor: very low scores don't need STANDARD.
        if is_simple_message(task) or score < 15:
            final_tier = tier.value
        else:
            final_tier = max(tier.value, type_tier, key=lambda t: tier_order.get(t, 1))
        # Override with preflight complexity tier when available
        if _preflight and _preflight.complexity_tier:
            final_tier = _preflight.complexity_tier.value
        logger.debug("Task type=%s (tier=%s), complexity=%s → final=%s", task_type, type_tier, tier.value, final_tier)
        return _assert(_provider_for_tier(final_tier, allowed=allowed, preflight=_preflight, speed_mode=speed_mode))

    return _assert(_provider_for_tier("LOW", allowed=allowed, speed_mode=speed_mode))


def _provider_for_tier(tier: str, allowed: set[str] | None = None, preflight=None, speed_mode: str = "balanced"):
    """Return the best provider for a given tier; allowlist filtered in one loop."""
    import os

    def _ok(name: str) -> bool:
        return allowed is None or name in allowed

    kimi_key = os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
    gemini_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    def _mk_kimi():
        if not kimi_key:
            return None
        try:
            from mullet.core.providers.kimi import KimiProvider
            from mullet.core.providers.model_registry import get_registry
            kimi_model = get_registry().get_best_model(
                "kimi", tier, allowed_providers=allowed, preflight=preflight, speed_mode=speed_mode
            )
            return KimiProvider(api_key=kimi_key, model=kimi_model)
        except Exception as e:
            logger.warning("Kimi unavailable: %s", e)
            return None

    def _mk_gemini():
        if not gemini_key:
            return None
        try:
            from mullet.core.providers.gemini import GeminiProvider
            return GeminiProvider(api_key=gemini_key)
        except Exception as e:
            logger.warning("Gemini unavailable: %s", e)
            return None

    def _mk_openrouter():
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_key:
            return None
        try:
            from mullet.core.providers.openrouter import OpenRouterProvider
            from mullet.core.providers.model_registry import get_registry
            openrouter_model = get_registry().get_best_model(
                "openrouter", tier, allowed_providers=allowed, preflight=preflight, speed_mode=speed_mode
            )
            prov = OpenRouterProvider(api_key=openrouter_key, model=openrouter_model)
            # Lets the provider pick a same-tier replacement if OpenRouter
            # says this model is unavailable.
            prov.routing_tier = tier
            return prov
        except Exception as e:
            logger.warning("OpenRouter unavailable: %s", e)
            return None

    def _mk_anthropic(model_id: str):
        def _inner():
            if not anthropic_key:
                # Anthropic needs ANTHROPIC_API_KEY. The caller-side
                # _anthropic_only_needs_native_handoff() check catches the
                # MCP-request case and hands the task back to Claude Code
                # instead; this factory just reports "not configured".
                return None
            try:
                from mullet.core.providers.anthropic import AnthropicProvider
                return AnthropicProvider(api_key=anthropic_key, model=model_id)
            except Exception as e:
                logger.warning("Anthropic (%s) unavailable: %s", model_id, e)
                return None
        return _inner

    def _mk_local():
        from mullet.core.providers.local_detection import detect_local_provider
        provider = detect_local_provider()
        if not provider:
            return None
        from mullet.core.router.resource_guard import get_resource_guard
        guard = get_resource_guard()
        ok, reason = guard.can_dispatch_local()
        if not ok:
            logger.warning("Skipping local LLM: %s — escalating to cloud", reason)
            return None
        return provider

    candidates: list[tuple[str, object]] = []
    if tier == "PREMIUM":
        candidates.append(("opus-4-8", _mk_anthropic("claude-opus-4-8")))
        candidates.append(("opus-4-7", _mk_anthropic("claude-opus-4-7")))
        candidates.append(("opus-4-6", _mk_anthropic("claude-opus-4-6")))
        candidates.append(("sonnet",   _mk_anthropic("claude-sonnet-4-6")))
        candidates.append(("haiku",    _mk_anthropic("claude-haiku-4-5-20251001")))
        candidates.append(("kimi", _mk_kimi))
        candidates.append(("gemini", _mk_gemini))
        candidates.append(("openrouter", _mk_openrouter))
    elif tier in ("STANDARD", "ADVANCED", "CRITICAL"):
        candidates.append(("kimi", _mk_kimi))
        candidates.append(("gemini", _mk_gemini))
        candidates.append(("openrouter", _mk_openrouter))
    candidates.append(("haiku",    _mk_anthropic("claude-haiku-4-5-20251001")))
    candidates.append(("sonnet",   _mk_anthropic("claude-sonnet-4-6")))
    candidates.append(("opus-4-8", _mk_anthropic("claude-opus-4-8")))
    candidates.append(("opus-4-7", _mk_anthropic("claude-opus-4-7")))
    candidates.append(("opus-4-6", _mk_anthropic("claude-opus-4-6")))
    candidates.append(("local", _mk_local))
    candidates.append(("kimi", _mk_kimi))
    candidates.append(("gemini", _mk_gemini))
    candidates.append(("openrouter", _mk_openrouter))

    for name, factory in candidates:
        if not _ok(name):
            continue
        try:
            prov = factory()
        except Exception as e:
            logger.warning("Provider factory %s raised: %s", name, e)
            continue
        if prov is not None:
            prov._mcp_resolved_alias = name
            return prov
    return None


def _build_fallback_chain(
    allowed: set[str] | None = None,
    requested_tier: str | None = None,
    prompt_tokens: int = 0,
    filtered_out: dict[str, str] | None = None,
) -> list:
    """Build fallback chain; raises if allowlist + constraints eliminate every candidate.

    If *filtered_out* is provided, providers rejected by the ConstraintRegistry are
    recorded as ``{alias: reason}`` entries for observability.
    """
    import os
    from mullet.core.providers.constraints import get_constraint_registry

    from mullet.core.providers.errors import ModelContextExceeded, ObedienceError

    if allowed is not None and len(allowed) == 0:
        raise ObedienceError(
            message="allowed_providers is empty set — refusing to build fallback chain",
            allowed_providers=[],
            requested_tier=requested_tier,
        )

    constraint_registry = get_constraint_registry()
    needs = {
        "context_tokens": prompt_tokens,
        "requires_tools": True,
        "requires_streaming": False,
    }

    fallbacks: list = []
    had_candidates = False
    # Record filtered-out providers so callers can surface them on failure.
    # If the caller didn't pass a dict we still build a local one for the
    # error message below.
    local_filtered: dict[str, str] = filtered_out if filtered_out is not None else {}

    def _ok(name: str) -> bool:
        return allowed is None or name in allowed

    kimi_key = os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
    if kimi_key and _ok("kimi"):
        had_candidates = True
        try:
            from mullet.core.providers.kimi import KimiProvider
            from mullet.core.providers.model_registry import get_registry
            kimi_model = get_registry().get_best_model(
                "kimi", requested_tier or "STANDARD",
                allowed_providers=allowed, context_tokens=prompt_tokens,
            )
            # Reject deprecated moonshot-v1 models for ADVANCED/CRITICAL tiers —
            # they are low-context legacy endpoints not suitable for complex tasks.
            _high_tiers = {"ADVANCED", "CRITICAL", "MAXIMUM"}
            if kimi_model.startswith("moonshot-v1") and (requested_tier or "STANDARD") in _high_tiers:
                reason = f"{kimi_model} is a deprecated low-context model; excluded from {requested_tier} tier"
                local_filtered["kimi"] = reason
                logger.info("constraint-filter: skipping kimi model=%s (reason=%s)", kimi_model, reason)
            else:
                constraints = constraint_registry.get(kimi_model)
                if constraints.can_serve(needs):
                    fallbacks.append(KimiProvider(api_key=kimi_key, model=kimi_model))
                    fallbacks[-1]._mcp_resolved_alias = "kimi"
                else:
                    reason = (
                        f"context_tokens={prompt_tokens} exceeds {kimi_model} "
                        f"max_context={constraints.max_context}"
                        if prompt_tokens > constraints.max_context
                        else f"{kimi_model} cannot serve requested capabilities"
                    )
                    local_filtered["kimi"] = reason
                    logger.info(
                        "constraint-filter: skipping kimi model=%s (reason=%s)",
                        kimi_model, reason,
                    )
        except ModelContextExceeded as _mce:
            # get_best_model refused every Kimi candidate for this prompt.
            # Record the refusal so ObedienceError carries a meaningful
            # filtered_providers entry rather than an empty dict.
            local_filtered["kimi"] = (
                f"context_tokens={prompt_tokens} exceeds kimi candidates "
                f"(tried={_mce.tried_candidates}, "
                f"max_candidate_context={_mce.max_candidate_context})"
            )
            logger.warning("_build_fallback_chain: Kimi refused all candidates: %s", _mce)
        except (ImportError, Exception) as e:
            logger.warning("_build_fallback_chain: Kimi fallback unavailable: %s", e)

    gemini_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if gemini_key and _ok("gemini"):
        had_candidates = True
        try:
            from mullet.core.providers.gemini import GeminiProvider
            constraints = constraint_registry.get("gemini-2.5-flash")
            if constraints.can_serve(needs):
                fallbacks.append(GeminiProvider(api_key=gemini_key))
                fallbacks[-1]._mcp_resolved_alias = "gemini"
            else:
                reason = (
                    f"context_tokens={prompt_tokens} exceeds gemini-2.5-flash "
                    f"max_context={constraints.max_context}"
                )
                local_filtered["gemini"] = reason
                logger.info("constraint-filter: skipping gemini (reason=%s)", reason)
        except ImportError as _exc:
            logger.warning("mcp_handlers: Gemini fallback provider import failed: %s", _exc)

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    model_map = [
        ("haiku",    "claude-haiku-4-5-20251001"),
        ("sonnet",   "claude-sonnet-4-6"),
        ("opus-4-8", "claude-opus-4-8"),
        ("opus",     "claude-opus-4-8"),
        ("opus-4-7", "claude-opus-4-7"),
    ]
    # Without ANTHROPIC_API_KEY, Anthropic simply isn't a fallback candidate.
    if anthropic_key:
        for alias, model_id in model_map:
            if _ok(alias):
                had_candidates = True
                try:
                    constraints = constraint_registry.get(model_id)
                    if constraints.can_serve(needs):
                        from mullet.core.providers.anthropic import AnthropicProvider
                        fallbacks.append(AnthropicProvider(api_key=anthropic_key, model=model_id))
                        fallbacks[-1]._mcp_resolved_alias = alias
                    else:
                        reason = (
                            f"context_tokens={prompt_tokens} exceeds {model_id} "
                            f"max_context={constraints.max_context}"
                        )
                        local_filtered[alias] = reason
                        logger.info(
                            "constraint-filter: skipping %s model=%s (reason=%s)",
                            alias, model_id, reason,
                        )
                except (ImportError, Exception) as _exc:
                    logger.warning("mcp_handlers: Anthropic fallback provider import failed: %s", _exc)

    if allowed is not None and not fallbacks and had_candidates:
        logger.error(
            "allowlist=%s intersected with ConstraintRegistry eliminated all candidates "
            "(prompt_tokens=%d, requested_tier=%s) filtered=%s",
            allowed, prompt_tokens, requested_tier, local_filtered,
        )
        # ObedienceError subclasses ValueError so every existing
        # `except ValueError:` catcher (CLI, mcp_handlers, router) still
        # catches it; new callers can `except ObedienceError` for the
        # structured fields below.
        raise ObedienceError(
            message=(
                f"All providers in allowlist {sorted(allowed)} filtered out by constraint registry "
                f"(prompt_tokens={prompt_tokens}, requested_tier={requested_tier})"
            ),
            allowed_providers=sorted(allowed),
            filtered_providers=dict(local_filtered),
            requested_tier=requested_tier,
        )

    return fallbacks


def handle_mullet_run(arguments: dict) -> dict:
    """Execute a task through Mullet's engine.

    D7 (Cycle 17): always returns a ``dict``. The MCP protocol wrapper in
    ``mullet.server.mcp.handle_request`` JSON-serialises the dict before
    sending it over stdio. In-process callers (Python tests,
    other MCP handlers) can rely on ``isinstance(result, dict)`` without
    a ``json.loads()`` round-trip.
    """
    _off = _delegation_off_reply()
    if _off:
        return {"completed": False, "error": _off, "delegation": "off"}

    import time as _rh_time
    from mullet.core.routing_history import make_outcome, record as rh_record

    _rh_task = arguments.get("task", "")
    _rh_tier = arguments.get("tier") or "STANDARD"
    _rh_t0 = _rh_time.monotonic()

    # --- Routing priors re-rank (MULLET_ROUTING_PRIORS: on|off|auto, default auto) ---
    # Auto: activates once routing_history has >= 20 total outcomes, so priors only
    # influence dispatch after enough data has accumulated to be meaningful.
    # Set MULLET_SEMANTIC_PRIORS=off to disable semantic path and use only exact-sig.
    _rh_rerank_tag: str | None = None
    _rh_priors_mode = os.environ.get("MULLET_ROUTING_PRIORS", "auto").lower() or "auto"
    _rh_priors_enabled = False
    if _rh_priors_mode == "on":
        _rh_priors_enabled = True
    elif _rh_priors_mode == "auto":
        try:
            from mullet.core.routing_history import iter_recent as _iter_h
            _rh_priors_enabled = sum(1 for _ in _iter_h(limit=25)) >= 20
        except Exception:
            _rh_priors_enabled = False
    if _rh_priors_enabled:
        try:
            _providers_raw = arguments.get("providers", "")
            _provider_list = [p.strip() for p in _providers_raw.split(",") if p.strip()] if _providers_raw else []
            if _provider_list:
                _picked = _provider_list[0]
                _candidates = _provider_list[1:]
                _new_pick: str = _picked
                _reason: str | None = None
                # Try semantic path first (unless explicitly disabled)
                if os.environ.get("MULLET_SEMANTIC_PRIORS", "on").lower() != "off":
                    from mullet.core.routing_priors import pick_better_provider_semantic
                    _new_pick, _reason = pick_better_provider_semantic(
                        _rh_task, _rh_tier, _picked, _candidates
                    )
                # Fall back to exact-signature path if semantic returned no change
                if _new_pick == _picked:
                    from mullet.core.routing_priors import pick_better_provider
                    _new_pick, _reason = pick_better_provider(_rh_task, _rh_tier, _picked, _candidates)
                if _new_pick != _picked and _reason:
                    logger.info(
                        "routing priors: re-rank %s → %s for sig=%s reason=%s",
                        _picked, _new_pick,
                        __import__("mullet.core.routing_history", fromlist=["make_signature"]).make_signature(_rh_task, "run", _rh_tier),
                        _reason,
                    )
                    # Rebuild providers string with new_pick first
                    reranked = [_new_pick] + [p for p in _provider_list if p != _new_pick]
                    arguments = dict(arguments)
                    arguments["providers"] = ",".join(reranked)
                    _rh_rerank_tag = f"rerank:{_picked}→{_new_pick}"
        except Exception as _pr_exc:
            logger.warning("routing_priors: re-rank failed (non-fatal): %s", _pr_exc)

    result = _handle_mullet_run_inner(arguments)

    try:
        _rh_wall_ms = int((_rh_time.monotonic() - _rh_t0) * 1000)
        _rh_provider = result.get("actual_provider", "unknown")
        _rh_model = result.get("actual_model", "unknown")
        _rh_success = bool(
            result
            and not result.get("error")
            and not result.get("failure")
        )
        # When actual_provider is missing, use the last tried provider if available
        if _rh_provider == "unknown":
            _tried = result.get("tried_providers") or []
            if _tried:
                _rh_provider = _tried[-1]
        _rh_tokens_in = result.get("prompt_tokens", 0) or result.get("input_tokens", 0) or 0
        _rh_tokens_out = result.get("completion_tokens", 0) or result.get("output_tokens", 0) or 0
        # Fall back to total tokens when no per-direction split is available
        # (the agent runner returns a "tokens" total)
        if not _rh_tokens_in and not _rh_tokens_out:
            _rh_tokens_in = result.get("tokens", 0) or 0
        # Error classification for failed calls was removed with the harness layer;
        # routing history still records the failure, just without a classified reason.
        _rh_error_class: str | None = None
        # Determine tags by agent/swarm signals in result
        if result.get("auto_swarm"):
            _rh_tags = ["mcp", "swarm"]
        elif "filtered_providers" in result or "tried_providers" in result:
            _rh_tags = ["mcp", "agent"]
        else:
            _rh_tags = ["mcp", "agent"]
        if _rh_rerank_tag:
            _rh_tags.append(_rh_rerank_tag)
        _rh_outcome = make_outcome(
            task=_rh_task,
            provider=_rh_provider,
            model=_rh_model,
            tier=_rh_tier,
            success=_rh_success,
            tokens_in=_rh_tokens_in,
            tokens_out=_rh_tokens_out,
            wall_time_ms=_rh_wall_ms,
            error_class=_rh_error_class,
            tool_family="run",
            tags=_rh_tags,
        )
        rh_record(_rh_outcome)
    except Exception as _rh_exc:
        logger.debug("routing_history: record failed (non-fatal): %s", _rh_exc)

    return result


def _handle_mullet_run_inner(arguments: dict) -> dict:
    """Inner implementation for handle_mullet_run — do not call directly."""
    from mullet.cli.main import build_tool_registry
    from mullet.core.agent.runner import AgentRunner

    task = arguments.get("task", "")
    if not task:
        return {"error": "task is required"}
    client = arguments.get("client")
    mode = arguments.get("mode", "standard")
    if mode != "standard":
        return {
            "error": f"mode '{mode}' was removed. For multi-step work, use `mullet job`.",
        }
    tier = arguments.get("tier")
    max_iter = arguments.get("max_iterations", 90)
    files = arguments.get("files") or []
    apply_blocks = arguments.get("apply", False)
    target_dir = arguments.get("target_dir") or os.getcwd()
    providers_raw = arguments.get("providers", "")
    force_single: bool = bool(arguments.get("force_single", False))
    allowed_providers: set[str] | None = (
        {p.strip() for p in providers_raw.split(",") if p.strip()}
        if providers_raw else None
    )

    # Apply active preset when caller did not supply providers
    speed_mode = "balanced"
    if allowed_providers is None:
        preset = _active_preset()
        if preset is not None:
            if preset.native_only:
                return {"error": f"Preset '{preset.name}' is native-only — Claude Code handles this directly; not delegating. Do the work in-context."}
            allowed_providers = preset.mcp_allowlist()
            speed_mode = preset.speed_mode

    if _anthropic_only_needs_native_handoff(allowed_providers):
        return {"error": _NATIVE_HANDOFF_MESSAGE}

    # Sanitize paths
    safe_files, safe_target, error = _sanitize_paths(files, target_dir)
    if error:
        return {"error": error}
    files = safe_files
    target_dir = safe_target

    # Auto-swarm routing: fan out to a swarm when the heuristic fires,
    # unless the caller explicitly opts out with force_single=True.
    if not force_single:
        try:
            from mullet.core.router.auto_swarm import _should_swarm
            should, reason = _should_swarm(task, files, prompt_len=len(task), tier=tier)
            if should:
                logger.info("auto-swarm: routing to swarm (%s)", reason)
                import asyncio
                from mullet.core.orchestration.swarm import run_swarm
                from mullet.core.orchestration.swarm_types import SwarmContext
                import time as _time

                _start = _time.time()
                ctx = SwarmContext(
                    files=files,
                    output_shape="markdown report",
                    concurrency=10,
                )
                swarm_result = asyncio.run(run_swarm(task, ctx, allowed_providers=frozenset(allowed_providers) if allowed_providers else None))
                if len(swarm_result.tasks) == 0:
                    logger.warning("auto-swarm produced zero tasks, falling back to single-provider dispatch")
                else:
                    duration = _time.time() - _start
                    swarm_providers = sorted({r.provider for r in swarm_result.results if r.provider})
                    actual_provider = "+".join(swarm_providers) if swarm_providers else "unknown"
                    return {
                        "completed": not swarm_result.partial_failure,
                        "response": swarm_result.synthesis,
                        "tasks": len(swarm_result.tasks),
                        "partial_failure": swarm_result.partial_failure,
                        "duration": round(duration, 2),
                        "tokens": swarm_result.total_tokens.total_tokens,
                        "auto_swarm": True,
                        "swarm_reason": reason,
                        "actual_provider": actual_provider,
                    }
        except Exception as e:
            logger.warning("auto-swarm failed, falling back to single call: %s", e)

    # apply=true is not supported — use mullet_opencode for file edits
    if apply_blocks:
        return {"error": "apply=true is not supported — use mullet_opencode for file edits."}

    # Write placeholder row so HUD shows activity during provider resolution
    # (local LLM model-load can block up to 30s before real provider is ready).
    _write_agent_state_fn(task, "resolving", "…")

    try:
        provider = _resolve_provider(task=task, force_tier=tier, allowed=allowed_providers, speed_mode=speed_mode)
    except (ValueError, RuntimeError) as _re:
        _clear_agent_state()
        return {"error": f"provider resolution failed: {_re}"}
    if not provider:
        _clear_agent_state()
        return {"error": "No LLM provider available"}

    # Write active agent info for HUD display
    _write_agent_state_fn(task, getattr(provider, "name", "?"), getattr(provider, "model", "?"))

    import time as _time
    _start = _time.time()

    result = None  # Initialize result to None

    registry = build_tool_registry()
    from mullet.core.agent.prompts import build_base_prompt
    model_name = getattr(provider, "model", "") if provider else ""
    system_prompt = build_base_prompt(model=model_name)

    # Load knowledge if client specified
    if client:
        try:
            from mullet.core.knowledge.manager import build_knowledge_environment

            _km, _context, system_prompt = build_knowledge_environment(
                client, task, registry, system_prompt,
            )
        except Exception as e:
            logger.warning("Knowledge load failed: %s", e)

    # All toolsets enabled for full capability (matches CLI)
    toolsets = ["terminal", "file", "web", "code", "browser", "delegation", "session", "git", "testing", "vision", "docker"]
    if client:
        toolsets.append("trove")

    # D1: estimator must never break dispatch — fall back to char-based count
    # if tiktoken itself raises.
    try:
        from mullet.core.providers.tokenize import estimate_tokens as _estimate_tokens
        _agent_prompt_tokens = _estimate_tokens(task)
    except Exception as _est_err:
        logger.warning(
            "estimate_tokens failed in agent mode (%s); using char fallback",
            _est_err,
        )
        _agent_prompt_tokens = max(1, len(task or "") // 6)
    agent_filtered_providers: dict[str, str] = {}
    try:
        fb_chain = _build_fallback_chain(
            allowed=allowed_providers,
            requested_tier=tier,
            prompt_tokens=_agent_prompt_tokens,
            filtered_out=agent_filtered_providers,
        )
    except ValueError as _ve:
        _clear_agent_state()
        _err: dict = {"error": f"allowlist/constraint conflict: {_ve}"}
        _extra = getattr(_ve, "filtered_providers", None)
        if isinstance(_extra, dict) and _extra:
            _err["filtered_providers"] = _extra
        elif agent_filtered_providers:
            _err["filtered_providers"] = agent_filtered_providers
        _allowed_attr = getattr(_ve, "allowed_providers", None)
        if _allowed_attr:
            _err["allowed_providers"] = list(_allowed_attr)
        _tier_attr = getattr(_ve, "requested_tier", None)
        if _tier_attr:
            _err["requested_tier"] = _tier_attr
        return _err

    # Defense-in-depth: if the resolved provider is not in the allowlist (can
    # happen when _resolve_provider is patched or allowlist enforcement races),
    # demote it and promote the first fallback to primary.
    if allowed_providers and fb_chain:
        prov_name = getattr(provider, "_mcp_resolved_alias", None) or getattr(provider, "name", None)
        if prov_name not in allowed_providers:
            logger.warning(
                "handle_mullet_run: primary provider %r not in allowlist %s — "
                "switching to fallback %r",
                prov_name, sorted(allowed_providers), getattr(fb_chain[0], "name", "?"),
            )
            provider = fb_chain[0]
            fb_chain = fb_chain[1:]

    agent = AgentRunner(
        provider=provider,
        tool_registry=registry,
        toolsets=toolsets,
        system_prompt=system_prompt,
        max_iterations=max_iter,
        fallback_providers=fb_chain,
    )

    try:
        result = agent.run(task)
        duration = _time.time() - _start
        out_agent: dict = {
            "completed": result.completed,
            "response": result.final_response,
            "tool_calls": result.tool_calls_made,
            "duration": round(duration, 2),
            "tokens": result.tokens_used.total_tokens,
            "provider": getattr(agent.provider, "name", "unknown"),
            "actual_provider": getattr(result, "provider_used", getattr(agent.provider, "name", "unknown")),
            "actual_model": getattr(agent.provider, "model", "unknown"),
        }
        # Surface fallback history if the AgentRunner internally tried more
        # than one provider during this run — keeps the legacy shape when
        # the first provider succeeded.
        tried = sorted(getattr(agent, "_tried_providers", set()) or set())
        if len(tried) > 1:
            out_agent["tried_providers"] = tried
        return out_agent
    except Exception as e:
        return {
            "error": f"Agent execution failed: {e}",
            "actual_provider": getattr(agent.provider, "name", "unknown"),
            "tried_providers": sorted(getattr(agent, "_tried_providers", set()) or set()),
        }
    finally:
        _clear_agent_state()
        try:
            from mullet.core.router.resource_guard import get_resource_guard
            get_resource_guard().flush_resources()
        except Exception as _re:
            logger.debug("Resource flush failed (non-fatal): %s", _re)


def handle_mullet_delegate(arguments: dict) -> str:
    """Delegate a subtask to a child agent."""
    _off = _delegation_off_reply()
    if _off:
        return json.dumps({"error": _off, "delegation": "off"})

    from mullet.cli.main import build_tool_registry
    from mullet.core.agent.runner import AgentRunner
    from mullet.core.agent.delegation import delegate_task

    goal = arguments.get("goal", "")
    if not goal:
        return json.dumps({"error": "goal is required"})
    context = arguments.get("context", "")
    providers_raw = arguments.get("providers", "")
    allowed_providers: set[str] | None = (
        {p.strip() for p in providers_raw.split(",") if p.strip()}
        if providers_raw else None
    )

    # Apply active preset when caller did not supply providers
    speed_mode = "balanced"
    if allowed_providers is None:
        preset = _active_preset()
        if preset is not None:
            if preset.native_only:
                return json.dumps({"error": f"Preset '{preset.name}' is native-only — Claude Code handles this directly; not delegating. Do the work in-context."})
            allowed_providers = preset.mcp_allowlist()
            speed_mode = preset.speed_mode

    if _anthropic_only_needs_native_handoff(allowed_providers):
        return json.dumps({"error": _NATIVE_HANDOFF_MESSAGE})

    # Placeholder row while resolving (local LLM model-load can take 30s).
    _write_agent_state_fn(goal, "resolving", "…")

    try:
        provider = _resolve_provider(task=goal, allowed=allowed_providers, speed_mode=speed_mode)
    except (ValueError, RuntimeError) as _re:
        _clear_agent_state()
        return json.dumps({"error": f"provider resolution failed: {_re}"})
    if not provider:
        _clear_agent_state()
        return json.dumps({"error": "No LLM provider available"})

    _write_agent_state_fn(goal, getattr(provider, "name", "?"), getattr(provider, "model", "?"))

    registry = build_tool_registry()
    parent = AgentRunner(
        provider=provider,
        tool_registry=registry,
        toolsets=["terminal", "file", "web", "code", "browser", "session", "git", "testing", "vision", "docker"],
        max_iterations=50,
        allowed_providers=frozenset(allowed_providers) if allowed_providers else None,
    )

    try:
        result = delegate_task(goal=goal, context=context, parent=parent)
        return json.dumps({
            "completed": result.completed,
            "summary": result.summary,
            "error": result.error,
        })
    finally:
        _clear_agent_state()
        try:
            from mullet.core.router.resource_guard import get_resource_guard
            get_resource_guard().flush_resources()
        except Exception as _re:
            logger.debug("Resource flush failed (non-fatal): %s", _re)


def handle_mullet_status(arguments: dict) -> str:
    """Return current Mullet status with resource info."""
    import time
    from mullet.core.providers.local_detection import detect_local_provider
    from mullet.core.router.resource_guard import get_resource_guard
    from mullet.hud.agent_state import _read_current
    local = detect_local_provider()
    guard = get_resource_guard()
    ok, reason = guard.can_dispatch_local() if local else (False, "no local provider")

    # Budget pressure
    budget_info = None
    try:
        from mullet.hud.guard import check_budget
        br = check_budget()
        budget_info = {"status": br.status.value, "message": br.message}
    except Exception:
        pass

    # Running agents with elapsed time and cost
    now = time.time()
    agents = []
    for a in _read_current():
        elapsed = now - float(a.get("started_at", now))
        agents.append({
            "id": a.get("id"),
            "description": a.get("description", ""),
            "provider": a.get("provider", ""),
            "model": a.get("model", ""),
            "status": a.get("status", "running"),
            "elapsed_seconds": round(elapsed),
            "cost_usd": a.get("cost_usd"),
        })

    return json.dumps({
        "status": "ok",
        "local_provider": local.name if local else None,
        "local_can_dispatch": ok,
        "local_reason": reason,
        "resources": guard.get_system_snapshot(),
        "version": __import__("mullet").__version__,
        "budget": budget_info,
        "running_agents": agents,
        "agent_count": len(agents),
    })


def handle_mullet_providers(arguments: dict) -> str:
    """List available providers with live health checks and resource status."""
    import os
    from mullet.core.providers.local_detection import detect_local_provider
    from mullet.core.router.resource_guard import get_resource_guard
    from mullet.core.providers.model_registry import get_registry

    guard = get_resource_guard()
    registry = get_registry()
    providers = []

    # Local — live health check + resource gate
    local = detect_local_provider()
    if local:
        ok, reason = guard.can_dispatch_local()
        providers.append({
            "name": "local", "model": local.model,
            "healthy": local.health_check(), "can_dispatch": ok,
            "reason": reason, "cost": "free",
        })

    if os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY"):
        kimi_model = registry.get_best_model("kimi", "STANDARD")
        providers.append({
            "name": "kimi",
            "model": kimi_model,
            "actual_model_standard": kimi_model,
            "actual_model_critical": registry.get_best_model("kimi", "CRITICAL"),
            "healthy": True,
            "cost": "cheap",
        })

    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        gemini_model = registry.get_best_model("gemini", "STANDARD")
        providers.append({
            "name": "gemini",
            "model": gemini_model,
            "actual_model_standard": gemini_model,
            "actual_model_critical": registry.get_best_model("gemini", "CRITICAL"),
            "healthy": True,
            "cost": "budget",
        })

    if os.environ.get("ANTHROPIC_API_KEY"):
        anthropic_model = registry.get_best_model("anthropic", "STANDARD")
        providers.append({
            "name": "anthropic",
            "model": anthropic_model,
            "actual_model_standard": anthropic_model,
            "actual_model_critical": registry.get_best_model("anthropic", "CRITICAL"),
            "healthy": True,
            "cost": "premium",
        })

    if os.environ.get("OPENROUTER_API_KEY"):
        providers.append({"name": "openrouter", "model": "various", "healthy": True, "cost": "variable"})

    return json.dumps({
        "providers": providers,
        "resources": guard.get_system_snapshot(),
    })


def handle_mullet_budget(arguments: dict) -> str:
    """Return budget status from actual BudgetTracker."""
    try:
        from mullet.core.router.budget import BudgetTracker
        tracker = BudgetTracker()
        return json.dumps(tracker.status())
    except Exception as e:
        logger.warning("handle_mullet_budget: BudgetTracker init/status failed: %s", e)
        return json.dumps({
            "session_tokens": 0,
            "session_cost": 0.0,
            "daily_limit": 10.0,
            "note": "Budget tracker unavailable — resets on restart.",
        })


def handle_mullet_search(arguments: dict) -> str:
    """Search project knowledge via Trove."""
    query = arguments.get("query", "")
    client = arguments.get("client", "")
    top_k = arguments.get("top_k", 10)

    if not query or not client:
        return json.dumps({"error": "query and client required"})

    try:
        from mullet.core.knowledge.manager import knowledge_session
        with knowledge_session(client) as km:
            results = km.search(query, top_k=top_k)

        formatted = []
        for r in results:
            formatted.append({
                "content": r.get("content", r.get("text", ""))[:500],
                "source": r.get("source", r.get("metadata", {}).get("source", "")),
                "score": r.get("score", 0),
            })
        return json.dumps({"results": formatted, "count": len(formatted)})
    except Exception as e:
        logger.warning("MCP search failed: %s", e)
        return json.dumps({"error": "Search failed"})


def handle_mullet_remember(arguments: dict) -> str:
    """Store a memory in a project's knowledge base."""
    key = arguments.get("key", "")
    content = arguments.get("content", "")
    client = arguments.get("client", "")

    if not key or not content or not client:
        return json.dumps({"error": "key, content, and client required"})

    try:
        from mullet.core.knowledge.manager import knowledge_session
        with knowledge_session(client) as km:
            km.remember(key, content, namespace="mcp")
        return json.dumps({"stored": True, "key": key, "client": client})
    except Exception as e:
        logger.warning("MCP remember failed: %s", e)
        return json.dumps({"error": "Memory store failed"})


def handle_mullet_analyze(arguments: dict) -> str:
    """Analyze task complexity and show routing decision."""
    task = arguments.get("task", "")
    agent_type = arguments.get("agent_type")

    if not task:
        return json.dumps({"error": "task required"})

    from mullet.core.router.complexity import extract_signals, score_complexity, score_to_tier, is_simple_message
    from mullet.core.router.decision import AGENT_TIER_OVERRIDES

    # Check simple message fast-path
    is_simple = is_simple_message(task)

    # Full analysis
    signals = extract_signals(task, agent_type)
    score = score_complexity(signals)
    tier = score_to_tier(score)

    # Check agent override
    override = None
    if agent_type and agent_type in AGENT_TIER_OVERRIDES:
        override = AGENT_TIER_OVERRIDES[agent_type]
        if override:
            override = override.value

    result = {
        "task": task[:100],
        "is_simple": is_simple,
        "complexity_score": round(score, 1),
        "tier": tier.value,
        "agent_override": override,
        "signals": {
            "word_count": signals.lexical.word_count,
            "file_paths": signals.lexical.file_path_count,
            "architecture_keywords": signals.lexical.has_architecture_keywords,
            "debugging_keywords": signals.lexical.has_debugging_keywords,
            "simple_keywords": signals.lexical.has_simple_keywords,
            "risk_keywords": signals.lexical.has_risk_keywords,
            "question_depth": signals.lexical.question_depth,
            "subtasks": signals.structural.estimated_subtasks,
            "cross_file": signals.structural.cross_file_dependencies,
            "domain": signals.structural.domain_specificity,
            "impact": signals.structural.impact_scope,
        },
    }

    # Add budget headroom advice
    try:
        from mullet.hud.budget_monitor import check_headroom
        result["headroom"] = check_headroom()
    except Exception as _exc:
        logger.warning("mcp_handlers: Failed to check budget headroom: %s", _exc)

    return json.dumps(result)


def handle_mullet_swarm(arguments: dict) -> str:
    """MCP handler for mullet_swarm tool."""
    _off = _delegation_off_reply()
    if _off:
        return json.dumps({"error": _off, "delegation": "off"})

    import asyncio
    import uuid
    from mullet.core.orchestration.swarm import run_swarm
    from mullet.core.orchestration.swarm_types import SwarmContext
    from mullet.hud.agent_state import add_agent, remove_agent, update_agent_tokens
    from mullet.core.agent.cost import estimate_cost

    goal = arguments.get("goal", "")
    if not goal:
        return json.dumps({"error": "goal is required"})

    files = arguments.get("files") or []
    items = arguments.get("items") or []
    concurrency = int(arguments.get("concurrency", 20))
    output_shape = arguments.get("output_shape", "markdown report")
    dry_run = bool(arguments.get("dry_run", False))
    # The swarm returns text only and never writes files, so apply=true could
    # only ever report a success that didn't happen.
    if arguments.get("apply", False):
        return json.dumps({"error": "apply=true is not supported: the swarm never writes files. Use mullet_opencode for file edits."})
    target_dir = arguments.get("target_dir") or os.getcwd()
    providers_raw = arguments.get("providers", "")
    allowed_providers: set[str] | None = (
        {p.strip() for p in providers_raw.split(",") if p.strip()}
        if providers_raw else None
    )

    safe_files, safe_target, path_err = _sanitize_paths(files, target_dir)
    if path_err:
        return json.dumps({"error": path_err})
    files, target_dir = safe_files, safe_target

    ctx = SwarmContext(
        files=files,
        items=items,
        concurrency=concurrency,
        output_shape=output_shape,
        apply=False,
        target_dir=target_dir,
    )

    swarm_id = f"swarm-{uuid.uuid4().hex[:8]}"
    add_agent(swarm_id, goal[:120], "swarm", "kimi")

    def on_progress(result, completed, total):
        """Per-subtask swarm callback — accumulate tokens + estimated cost into the agent record."""
        cost_delta = estimate_cost("kimi", "k2", result.tokens)
        update_agent_tokens(
            swarm_id,
            result.tokens.input_tokens,
            result.tokens.output_tokens,
            cost_delta,
        )

    try:
        result = asyncio.run(run_swarm(goal, ctx, dry_run=dry_run, allowed_providers=frozenset(allowed_providers) if allowed_providers else None, on_progress=on_progress))
    except Exception as exc:
        logger.error("Swarm failed: %s", exc, exc_info=True)
        return json.dumps({"error": str(exc)})
    finally:
        remove_agent(swarm_id)

    succeeded = sum(1 for r in result.results if r.success)
    response: dict = {
        "goal": result.goal,
        "synthesis": result.synthesis,
        "task_count": len(result.tasks),
        "success_count": succeeded,
        "partial_failure": result.partial_failure,
        "synthesis_fallback": result.synthesis_fallback,
        "wall_clock_seconds": round(result.wall_clock_seconds, 2),
        "total_tokens": {
            "input": result.total_tokens.input_tokens,
            "output": result.total_tokens.output_tokens,
        },
        "tasks": [
            {"id": t.id, "instruction": t.instruction}
            for t in result.tasks
        ],
    }
    return json.dumps(response)


_OPENCODE_ARGS = {"task", "files", "paths", "cwd", "repo_root", "timeout", "providers"}

_OPENCODE_IGNORED = {
    "model": "the delegation ladder picks the worker",
    "tier": "the delegation ladder picks the worker",
    "speed_mode": "the delegation ladder picks the worker",
    "provider": "the delegation ladder picks the worker; use 'providers' to limit its rungs",
    "apply": "the result is applied only after the job's gates pass, and never committed",
}

_OPENCODE_DEFAULT_TIMEOUT_MIN = 20
_OPENCODE_MAX_TIMEOUT_MIN = 60


def _opencode_reply(status: str, reason: str, notes: list[str]) -> dict:
    """A result in the same shape as a delegation's, for when no job ran."""
    return {
        "status": status,
        "reason": reason,
        "job_id": None,
        "report_path": None,
        "rung": None,
        "model": None,
        "cost": 0.0,
        "gates": [],
        "files_changed": [],
        "notes": notes,
    }


def _string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    return [str(p) for p in value if str(p).strip()]


def handle_mullet_opencode(arguments: dict) -> dict:
    """Run a coding task as a single gated Mullet job on the delegation ladder.

    A thin wrapper over mullet.jobs.delegate.delegate: the job runs in the
    foreground and its diff reaches the caller's working tree (unstaged, never
    committed) only when the node passed its gates. Arguments from the old
    direct-opencode tool are accepted and ignored, each with a note.
    """
    _off = _delegation_off_reply()
    if _off:
        return {"completed": False, "status": "blocked", "reason": _off, "error": _off, "delegation": "off"}

    notes: list[str] = []
    for key in arguments:
        if key in _OPENCODE_ARGS:
            continue
        why = _OPENCODE_IGNORED.get(key, "not an argument of mullet_opencode")
        notes.append(f"ignored '{key}': {why}")

    preset = _active_preset()
    if preset is not None and preset.native_only:
        message = (
            f"Preset '{preset.name}' handles edits natively in Claude Code, so "
            "delegation is disabled for this preset. Make the edit directly in Claude Code."
        )
        reply = _opencode_reply("blocked", message, notes)
        reply["error"] = message
        return reply

    task = str(arguments.get("task") or "").strip()
    if not task:
        return _opencode_reply("failed", "no task given", notes)

    paths = _string_list(arguments.get("files")) + _string_list(arguments.get("paths"))
    cwd = arguments.get("cwd") or arguments.get("repo_root") or None

    timeout = arguments.get("timeout")
    try:
        timeout_min = _OPENCODE_DEFAULT_TIMEOUT_MIN if timeout is None else int(timeout)
    except (TypeError, ValueError):
        notes.append(f"ignored 'timeout': {timeout!r} is not a whole number of minutes")
        timeout_min = _OPENCODE_DEFAULT_TIMEOUT_MIN
    clamped = max(1, min(_OPENCODE_MAX_TIMEOUT_MIN, timeout_min))
    if clamped != timeout_min:
        notes.append(f"timeout {timeout_min} minutes clamped to {clamped}")

    providers = [p.lower() for p in _string_list(arguments.get("providers"))] or None

    try:
        from mullet.jobs import delegate as delegate_mod
        from mullet.jobs.ledger import Ledger

        with Ledger() as ledger:
            result = delegate_mod.delegate(
                ledger, task=task, paths=paths or None, cwd=cwd,
                timeout_min=clamped, providers=providers,
            )
    except Exception as exc:  # job creation errors (bad paths, git) become a result, not a crash
        logger.warning("mullet_opencode: delegation failed: %s", exc, exc_info=True)
        return _opencode_reply("failed", str(exc) or type(exc).__name__, notes)

    result = dict(result)
    result["notes"] = notes
    return result


# Handler dispatch
MCP_HANDLERS = {
    "mullet_run": handle_mullet_run,
    "mullet_delegate": handle_mullet_delegate,
    "mullet_status": handle_mullet_status,
    "mullet_providers": handle_mullet_providers,
    "mullet_budget": handle_mullet_budget,
    "mullet_search": handle_mullet_search,
    "mullet_remember": handle_mullet_remember,
    "mullet_analyze": handle_mullet_analyze,
    "mullet_swarm": handle_mullet_swarm,
    "mullet_opencode": handle_mullet_opencode,
}
