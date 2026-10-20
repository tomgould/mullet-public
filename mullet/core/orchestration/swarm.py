"""Top-level swarm orchestrator — wires Decomposer → Dispatcher → Synthesizer."""

from __future__ import annotations

import datetime
import json
import logging
import os
import time
from pathlib import Path

from mullet.core.agent.types import TokenUsage
from mullet.core.orchestration.swarm_dispatcher import SwarmDispatcher
from mullet.core.orchestration.swarm_types import SwarmContext, SwarmResult, SwarmTask
from mullet.core.orchestration.synthesizer import Synthesizer

logger = logging.getLogger(__name__)


def _estimate_cost(tasks: list) -> tuple[int, float]:
    """Estimate total input tokens and cost for a list of tasks.

    Uses the canonical ``mullet.core.providers.tokenize.estimate_tokens``
    helper (tiktoken when available; conservative 6-char fallback otherwise),
    plus per-task overhead for the system prompt (~200 tokens).
    Returns (estimated_tokens, estimated_cost_usd).
    """
    from mullet.core.providers.tokenize import estimate_tokens

    try:
        from mullet.core.providers.model_registry import get_registry
        from mullet.core.providers.model_info import get_model_costs
        model_id = get_registry().get_best_model("kimi", "STANDARD")
        input_cost_per_1k, _ = get_model_costs(model_id)
    except Exception:
        input_cost_per_1k = 0.60 / 1000  # Kimi K2 fallback: $0.60/1K

    total_tokens = 0
    for task in tasks:
        # estimate_tokens uses tiktoken (cl100k_base) when installed —
        # accurate for routing/cost estimation. +200 covers the per-task
        # system prompt overhead added at dispatch time.
        task_tokens = estimate_tokens(task.instruction) + 200
        total_tokens += task_tokens

    cost = total_tokens * input_cost_per_1k / 1000
    return total_tokens, cost


_RETRYABLE_ERRORS = {"timeout", "rate_limit"}


def _dedupe_tasks(tasks: list) -> list:
    """Remove tasks with identical instruction text (case + whitespace normalized).

    Decomposer LLMs occasionally return the same instruction multiple times —
    e.g. prompting "summarise X" without decomposition cues produces N copies
    of the same task. Dispatching duplicates wastes tokens and confuses the
    synthesizer. First occurrence wins (preserves original task id + order).
    """
    seen: set[str] = set()
    deduped = []
    for t in tasks:
        key = " ".join(t.instruction.lower().split())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(t)
    if len(deduped) != len(tasks):
        logger.info(
            "swarm: deduped %d → %d tasks (decomposer returned duplicates)",
            len(tasks), len(deduped),
        )
    return deduped


async def _retry_failed_tasks(
    tasks: list,
    results: list,
    concurrency: int,
    on_progress=None,
    allowed_providers: frozenset[str] | None = None,
) -> list:
    """Retry transient failures at CRITICAL tier. Returns merged result list."""
    from mullet.core.orchestration.swarm_types import SwarmTask

    result_by_id = {r.task_id: r for r in results}
    task_by_id = {t.id: t for t in tasks}

    to_retry = [
        r for r in results
        if not r.success and r.error in _RETRYABLE_ERRORS
    ]
    if not to_retry:
        return results

    logger.info("swarm: retrying %d failed tasks at CRITICAL tier", len(to_retry))
    retry_tasks = []
    for r in to_retry:
        original = task_by_id[r.task_id]
        # Escalate to CRITICAL tier for the retry attempt
        escalated = SwarmTask(
            id=original.id,
            instruction=original.instruction,
            context_refs=original.context_refs,
            tier="CRITICAL",
            apply=original.apply,
        )
        retry_tasks.append(escalated)

    retry_dispatcher = SwarmDispatcher(
        concurrency=min(concurrency, len(retry_tasks)),
        allowed_providers=allowed_providers,
    )
    retry_results = await retry_dispatcher.dispatch(retry_tasks, on_progress=on_progress)

    for rr in retry_results:
        result_by_id[rr.task_id] = rr

    return [result_by_id[t.id] for t in tasks]


def _sum_tokens(results: list) -> TokenUsage:
    total = TokenUsage()
    for r in results:
        total.input_tokens += r.tokens.input_tokens
        total.output_tokens += r.tokens.output_tokens
    return total


def _select_synthesizer_provider(allowed_providers: frozenset[str] | None):
    """Pick an async synthesizer provider that respects the allowlist.

    Preference order when allowed_providers is set: kimi > haiku > sonnet.
    Returns None if the allowlist excludes all three (or no API key is
    present for the chosen provider). When allowed_providers is None,
    preserves legacy behaviour: always returns a KimiAsyncProvider
    (api_key may be empty — this matches pre-G19 behaviour).
    """
    from mullet.core.providers.kimi_async import KimiAsyncProvider
    from mullet.core.providers.model_registry import get_registry

    def _kimi_async(require_key: bool):
        api_key = os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
        if require_key and not api_key:
            return None
        from mullet.core.providers.kimi import KimiProvider
        model = get_registry().get_best_model("kimi", "ADVANCED")
        try:
            return KimiAsyncProvider(provider=KimiProvider(api_key=api_key or None, model=model))
        except Exception as exc:
            logger.debug("Kimi synthesizer provider unavailable: %s", exc)
            return None

    def _anthropic_async(model: str):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        from mullet.core.providers.anthropic import AnthropicProvider
        # KimiAsyncProvider is duck-typed: it calls ._sync.create(messages=..., max_tokens=...)
        # AnthropicProvider exposes the same .create signature returning .content/.usage.
        return KimiAsyncProvider(provider=AnthropicProvider(api_key=api_key, model=model))

    if allowed_providers is None:
        # Legacy behaviour — always return Kimi wrapper (key may be empty).
        return _kimi_async(require_key=False)

    for name in ("kimi", "haiku", "sonnet"):
        if name not in allowed_providers:
            continue
        if name == "kimi":
            p = _kimi_async(require_key=True)
        elif name == "haiku":
            p = _anthropic_async("claude-haiku-4-5")
        else:  # sonnet
            p = _anthropic_async("claude-sonnet-4-6")
        if p is not None:
            return p

    return None


def _select_decomposer(allowed_providers: frozenset[str] | None):
    """Pick a decomposer that respects the allowlist.

    Preference order when allowed_providers is set: kimi > haiku > sonnet.
    If none of those are in the allowlist, or no API key is present for the
    chosen provider, falls back to StaticDecomposer (no LLM required). When
    allowed_providers is None, preserves existing behaviour (Kimi if key
    present, else Static).
    """
    from mullet.core.orchestration.static_decomposer import StaticDecomposer

    def _kimi_decomposer():
        kimi_key = os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
        if not kimi_key:
            return None
        from mullet.core.orchestration.llm_decomposer import LLMDecomposer
        from mullet.core.providers.kimi import KimiProvider
        return LLMDecomposer(KimiProvider(api_key=kimi_key))

    def _anthropic_decomposer(model: str):
        anth_key = os.environ.get("ANTHROPIC_API_KEY")
        if not anth_key:
            return None
        from mullet.core.orchestration.llm_decomposer import LLMDecomposer
        from mullet.core.providers.anthropic import AnthropicProvider
        return LLMDecomposer(AnthropicProvider(api_key=anth_key, model=model))

    if allowed_providers is None:
        return _kimi_decomposer() or StaticDecomposer()

    for name in ("kimi", "haiku", "sonnet"):
        if name not in allowed_providers:
            continue
        if name == "kimi":
            d = _kimi_decomposer()
        elif name == "haiku":
            d = _anthropic_decomposer("claude-haiku-4-5")
        else:  # sonnet
            d = _anthropic_decomposer("claude-sonnet-4-6")
        if d is not None:
            return d

    return StaticDecomposer()


async def run_swarm(
    goal: str,
    context: SwarmContext,
    *,
    decomposer=None,
    concurrency: int | None = None,
    dry_run: bool = False,
    report_path: Path | None = None,
    on_progress=None,
    allowed_providers: frozenset[str] | None = None,
) -> SwarmResult:
    """Orchestrate a full swarm run: decompose → dispatch → synthesize.

    Args:
        goal: The high-level objective.
        context: Files/items/settings for this run.
        decomposer: A Decomposer instance (StaticDecomposer or LLMDecomposer).
                    If None, LLMDecomposer is used if Kimi key present, else Static.
        concurrency: Override context.concurrency if provided.
        dry_run: Decompose only — print tasks and return without dispatching.
        report_path: If given, write JSON report to this path.
        allowed_providers: Optional frozenset of provider names that restricts
                    the swarm path. Raises ValueError downstream (via
                    SwarmDispatcher) if the allowlist excludes 'kimi', since
                    the swarm path currently only supports Kimi.
    """
    wall_start = time.monotonic()

    effective_concurrency = concurrency if concurrency is not None else context.concurrency

    # Per-call hook handles every HUD add/remove; no outer cleanup needed.
    return await _run_swarm_inner(
        goal=goal,
        context=context,
        decomposer=decomposer,
        effective_concurrency=effective_concurrency,
        dry_run=dry_run,
        report_path=report_path,
        on_progress=on_progress,
        allowed_providers=allowed_providers,
        wall_start=wall_start,
    )


async def _run_swarm_inner(
    *,
    goal: str,
    context: SwarmContext,
    decomposer,
    effective_concurrency: int,
    dry_run: bool,
    report_path: Path | None,
    on_progress,
    allowed_providers: frozenset[str] | None,
    wall_start: float,
) -> SwarmResult:
    # Choose decomposer. When the caller supplied explicit files or items the
    # split is already known — StaticDecomposer maps 1:1 to SwarmTasks with no
    # LLM round-trip (fixes "No JSON array found in response" failures when an
    # LLM decomposer is asked to enumerate something that was already listed).
    if decomposer is None:
        if context.files or context.items:
            from mullet.core.orchestration.static_decomposer import StaticDecomposer
            decomposer = StaticDecomposer()
        else:
            decomposer = _select_decomposer(allowed_providers)

    tasks: list[SwarmTask] = decomposer.decompose(goal, context)
    tasks = _dedupe_tasks(tasks)

    # Diversity gate: if the decomposer produced zero tasks, bail out.
    # A single task is still dispatched (avoids the "one agent" degenerate
    # case where callers think swarm failed).
    if len(tasks) == 0:
        logger.warning(
            "swarm: decomposer produced %d unique task(s) — not enough diversity for a swarm. "
            "Caller should fall back to single-provider dispatch.",
            len(tasks),
        )

    # Per-task HUD visibility is handled by OpenAICompatibleProvider._hud_agent_track,
    # which adds/removes rows exactly around each in-flight provider.create() call.
    # Writing here would double every row during dispatch.

    if dry_run:
        logger.info("dry_run: %d tasks planned (not dispatched)", len(tasks))
        estimated_tokens, estimated_cost = _estimate_cost(tasks)
        return SwarmResult(
            goal=goal,
            tasks=tasks,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            estimated_input_tokens=estimated_tokens,
            estimated_cost_usd=estimated_cost,
        )

    if not tasks:
        logger.warning("Swarm: no tasks produced by decomposer — returning empty result")
        return SwarmResult(
            goal=goal,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    # Dispatch + synthesize — per-task HUD entries cleaned up by the provider hook.
    dispatcher = SwarmDispatcher(
        concurrency=effective_concurrency,
        allowed_providers=allowed_providers,
    )
    try:
        results = await dispatcher.dispatch(tasks, on_progress=on_progress)

        # Cross-task retry: escalate transient failures to CRITICAL tier
        results = await _retry_failed_tasks(
            tasks,
            results,
            effective_concurrency,
            on_progress,
            allowed_providers=allowed_providers,
        )

        # Aggregate tokens
        total_tokens = _sum_tokens(results)
        partial_failure = any(not r.success for r in results)

        # Synthesize — pick an allowlist-compliant provider (kimi > haiku > sonnet)
        syn_provider = _select_synthesizer_provider(allowed_providers)
        if syn_provider is None:
            logger.info(
                "swarm: no synthesizer provider available under allowed_providers=%s — "
                "falling back to raw concatenation",
                allowed_providers,
            )
            from mullet.core.orchestration.synthesizer import _format_raw_results
            synthesis_text = _format_raw_results(results)
            synthesis_fallback = True
        else:
            synthesizer = Synthesizer(provider=syn_provider)
            try:
                synthesis_text, syn_tokens = await synthesizer.synthesize(
                    goal=goal,
                    results=results,
                    output_shape=context.output_shape,
                )
                synthesis_fallback = False
                total_tokens.input_tokens += syn_tokens.input_tokens
                total_tokens.output_tokens += syn_tokens.output_tokens
            except Exception as exc:
                logger.warning("Synthesis failed (%s) — concatenating raw results", exc)
                from mullet.core.orchestration.synthesizer import _format_raw_results
                synthesis_text = _format_raw_results(results)
                synthesis_fallback = True

        wall_seconds = time.monotonic() - wall_start
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        swarm_result = SwarmResult(
            goal=goal,
            tasks=tasks,
            results=results,
            synthesis=synthesis_text,
            partial_failure=partial_failure,
            synthesis_fallback=synthesis_fallback,
            total_tokens=total_tokens,
            wall_clock_seconds=wall_seconds,
            timestamp=timestamp,
        )

        if report_path:
            _write_report(swarm_result, report_path)

        return swarm_result
    except Exception:
        # Per-task hook cleans its own rows; nothing else to undo here.
        raise


def _write_report(result: SwarmResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "goal": result.goal,
        "timestamp": result.timestamp,
        "wall_clock_seconds": result.wall_clock_seconds,
        "partial_failure": result.partial_failure,
        "synthesis_fallback": result.synthesis_fallback,
        "total_tokens": {
            "input": result.total_tokens.input_tokens,
            "output": result.total_tokens.output_tokens,
        },
        "task_count": len(result.tasks),
        "success_count": sum(1 for r in result.results if r.success),
        "synthesis": result.synthesis,
        "tasks": [
            {
                "id": t.id,
                "instruction": t.instruction,
                "context_refs": t.context_refs,
            }
            for t in result.tasks
        ],
        "results": [
            {
                "task_id": r.task_id,
                "success": r.success,
                "response": r.response,
                "error": r.error,
                "elapsed_seconds": r.elapsed_seconds,
            }
            for r in result.results
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Swarm report written to %s", path)
