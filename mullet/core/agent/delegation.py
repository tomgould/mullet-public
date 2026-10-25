"""Agent delegation — spawn isolated child agents for subtasks.

From Hermes delegate_tool.py: children get their own conversation,
tools, and iteration budget. Only the final summary returns to the
parent context. Children can use different providers for cost savings.
"""

import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from mullet.core.agent.profiles import AGENT_PROFILES

from mullet.core.agent.types import DelegationResult
# Providers that run as API endpoints with no local tool-dispatch
# infrastructure. Delegates routed to these MUST be told they have
# no access to MCP tools, filesystem, subagents, or web.
_API_ONLY_PROVIDERS = frozenset({"kimi", "gemini", "haiku", "openrouter"})


# Toolsets that require local execution (filesystem, shell, git, docker,
# browser, MCP delegation). API-only delegates (Kimi, Gemini, Haiku via API,
# OpenRouter) cannot actually execute these — advertising them leads to
# hallucinated tool calls. Stripped from the child runner's toolset list
# when the delegate provider is API-only.
_API_ONLY_STRIPPED_TOOLSETS = frozenset({
    "delegation",
    "file",
    "terminal",
    "browser",
    "docker",
    "testing",
    "git",
})

if TYPE_CHECKING:
    from mullet.core.agent.runner import AgentRunner
    from mullet.core.providers.base import Provider

logger = logging.getLogger(__name__)

MAX_DEPTH = 2
MAX_CONCURRENT = 3


def _log_spend(goal: str, provider_name: str, tokens_used: int, success: bool) -> None:
    """Log delegation token spend to SQLite (best-effort, never crashes).

    Creates delegation_spend.db in the state directory with schema:
        CREATE TABLE IF NOT EXISTS spend (
            id INTEGER PRIMARY KEY,
            ts TEXT,
            goal TEXT,
            provider TEXT,
            tokens INTEGER,
            success INTEGER
        )
    """
    try:
        from mullet.core.paths import state_dir
        db_path = state_dir() / "delegation_spend.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(db_path), timeout=5.0)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS spend (
                id INTEGER PRIMARY KEY,
                ts TEXT,
                goal TEXT,
                provider TEXT,
                tokens INTEGER,
                success INTEGER
            )
            """
        )

        now_iso = datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"
        goal_truncated = goal[:200]
        success_int = 1 if success else 0

        cursor.execute(
            """
            INSERT INTO spend (ts, goal, provider, tokens, success)
            VALUES (?, ?, ?, ?, ?)
            """,
            (now_iso, goal_truncated, provider_name, tokens_used, success_int),
        )

        conn.commit()
        conn.close()
    except Exception as e:
        logger.debug("Failed to log spend (best-effort): %s", e)


def delegate_task(
    goal: str,
    context: str = "",
    toolsets: list[str] | None = None,
    provider_override: "Provider | None" = None,
    max_iterations: int = 50,
    parent: "AgentRunner | None" = None,
    _depth: int = 0,
    complexity_hint: str | None = None,
    allowed_providers: frozenset[str] | None = None,
) -> DelegationResult:
    """Spawn an isolated child agent to handle a subtask.

    The child gets its own conversation, tools, and budget.
    Only the final summary returns to the parent context.

    Args:
        complexity_hint: Optional override for complexity tier
            ("LOW", "STANDARD", "ADVANCED", "CRITICAL"). When None, the
            tier is inferred from the goal text via recommend_tier().
        allowed_providers: Constraint envelope for the child agent.
            None (default) -> inherit from parent.allowed_providers.
            Empty frozenset() -> explicit override to unrestricted.
            Non-empty frozenset -> explicit restriction, replaces inherited.
    """
    if parent is None:
        return DelegationResult(goal=goal, completed=False, summary="", error="No parent agent provided")

    if _depth >= MAX_DEPTH:
        return DelegationResult(
            goal=goal, completed=False, summary="",
            error=f"Max delegation depth ({MAX_DEPTH}) reached — no grandchildren allowed",
        )

    from mullet.core.agent.runner import AgentRunner

    # Log recommended complexity tier (informational — does not change provider automatically)
    try:
        from mullet.core.router.complexity import recommend_tier
        effective_hint = complexity_hint or recommend_tier(goal).value
        logger.debug("delegate_task: goal='%s...' complexity=%s", goal[:60], effective_hint)
    except Exception as _exc:
        logger.warning("delegation: Failed to recommend tier: %s", _exc)

    # Inheritance: None -> inherit parent's allowlist; empty frozenset -> explicit
    # unrestricted; non-empty -> explicit restriction.
    if allowed_providers is None:
        effective_allowlist = getattr(parent, "allowed_providers", None)
    else:
        effective_allowlist = allowed_providers

    provider = provider_override or parent.provider
    tool_registry = parent.tool_registry

    # Check for agent profile in goal
    profile = None
    for profile_name, agent_profile in AGENT_PROFILES.items():
        if profile_name in goal.lower():
            profile = agent_profile
            break
    child_toolsets = toolsets or (profile.default_toolsets if profile else parent.toolsets)

    # Strip local-execution toolsets when the delegate is an API-only provider.
    # Kimi/Gemini/Haiku-via-API/OpenRouter cannot dispatch filesystem, shell,
    # git, docker, browser, or subagent-delegation tools, so advertising them
    # produces hallucinated tool calls. Anthropic (claude_cli / first-party
    # SDK) is left unmodified because its tool-use interface can dispatch
    # MCP tools server-side.
    provider_name = getattr(provider, "name", "") or ""
    if provider_name.lower() in _API_ONLY_PROVIDERS and child_toolsets:
        stripped = [t for t in child_toolsets if t not in _API_ONLY_STRIPPED_TOOLSETS]
        if stripped != list(child_toolsets):
            logger.debug(
                "delegate_task: stripping local-execution toolsets for API-only "
                "provider %s: %s -> %s",
                provider_name, list(child_toolsets), stripped,
            )
        child_toolsets = stripped

    system_prompt = _build_child_system_prompt(goal, context)

    child = AgentRunner(
        provider=provider,
        tool_registry=tool_registry,
        toolsets=child_toolsets,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
        allowed_providers=effective_allowlist,
    )
    child._delegation_depth = _depth + 1  # Track depth for child's delegate tool
    with parent._interrupt_lock:
        parent._active_children.append(child)

    try:
        result = child.run(goal)
        _log_spend(goal, provider.name, 0, result.completed)
        return DelegationResult(
            goal=goal,
            completed=result.completed,
            summary=result.final_response,
            agent_result=result,
        )
    except Exception as e:
        logger.exception("Delegation failed for: %s", goal)
        _log_spend(goal, provider.name, 0, False)
        return DelegationResult(
            goal=goal,
            completed=False,
            summary="",
            error=str(e),
        )
    finally:
        with parent._interrupt_lock:
            if child in parent._active_children:
                parent._active_children.remove(child)


def delegate_with_fallback(
    goal: str,
    context: str = "",
    toolsets: list[str] | None = None,
    provider_chain: list["Provider"] | None = None,
    max_iterations: int = 50,
    parent: "AgentRunner | None" = None,
    _depth: int = 0,
    allowed_providers: frozenset[str] | None = None,
) -> DelegationResult:
    """Delegate with provider fallback chain.

    Tries providers in order until one succeeds. Logs each attempt
    and failure. Falls back to delegate_task() if provider_chain is empty.

    allowed_providers follows the same semantics as delegate_task().
    """
    if not provider_chain:
        return delegate_task(goal, context, toolsets, None, max_iterations, parent, _depth, allowed_providers=allowed_providers)

    errors_by_provider = {}
    for provider in provider_chain:
        provider_name = getattr(provider, "name", "unknown")
        logger.info("Trying provider %s for: %s", provider_name, goal[:60])
        try:
            result = delegate_task(goal, context, toolsets, provider, max_iterations, parent, _depth, allowed_providers=allowed_providers)
            if result.completed:
                return result
            if result.error:
                errors_by_provider[provider_name] = result.error
                logger.warning("Provider %s failed: %s", provider_name, result.error)
        except Exception as e:
            error_str = str(e)
            errors_by_provider[provider_name] = error_str
            logger.warning("Provider %s failed: %s", provider_name, error_str)

    error_summary = "; ".join(f"{p}: {e}" for p, e in errors_by_provider.items())
    return DelegationResult(
        goal=goal,
        completed=False,
        summary="",
        error=f"All providers exhausted: {error_summary}",
    )


def delegate_batch(
    tasks: list[dict],
    parent: "AgentRunner | None" = None,
    provider_override: "Provider | None" = None,
    allowed_providers: frozenset[str] | None = None,
) -> list[DelegationResult]:
    """Spawn multiple child agents in parallel.

    Each task dict has: goal (required), context, toolsets, max_iterations.
    Up to MAX_CONCURRENT run simultaneously.
    """
    if not tasks:
        return []

    if parent is None:
        return [
            DelegationResult(goal=task["goal"], completed=False, summary="", error="No parent agent provided for batch delegation")
            for task in tasks
        ]

    results: list[DelegationResult] = [
        DelegationResult(goal=task["goal"], completed=False, summary="") for task in tasks
    ]

    # Limit concurrency to 1 if ANY task routes to a local provider
    is_local = any(
        getattr(t.get("_provider_override") or provider_override or (parent.provider if parent else None), "name", "") == "local"
        for t in tasks
    )
    max_workers = 1 if is_local else min(MAX_CONCURRENT, len(tasks))
    if is_local:
        logger.info("Local provider: limiting batch concurrency to 1")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for i, task in enumerate(tasks):
            parent_depth = getattr(parent, "_delegation_depth", 0) if parent else 0
            future = executor.submit(
                delegate_task,
                goal=task["goal"],
                context=task.get("context", ""),
                toolsets=task.get("toolsets"),
                provider_override=task.get("_provider_override") or provider_override,
                _depth=parent_depth,
                max_iterations=task.get("max_iterations", 50),
                parent=parent,
                allowed_providers=task.get("allowed_providers", allowed_providers),
            )
            future_to_idx[future] = i

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = DelegationResult(
                    goal=tasks[idx]["goal"],
                    completed=False,
                    summary="",
                    error=str(e),
                )

    return results


def _build_child_system_prompt(goal: str, context: str) -> str:
    """Build a focused system prompt for a child agent."""
    # Check for agent profile in goal
    profile_system_prompt = ""
    for profile_name, agent_profile in AGENT_PROFILES.items():
        if profile_name in goal.lower():
            profile_system_prompt = agent_profile.system_prompt + "\n\n"
            break
    parts = [
        "You are a focused sub-agent working on a specific task.",
        "Complete the task thoroughly, then provide a concise summary of what you did.",
        "",
        f"YOUR TASK: {goal}",
    ]
    if context:
        parts.extend(["", f"CONTEXT: {context}"])
    return profile_system_prompt + "\n".join(parts)
