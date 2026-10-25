"""SwarmDispatcher — async fan-out over multiple providers with semaphore concurrency control.

Design decisions (from swarm-review.md):
- asyncio.gather(..., return_exceptions=True): one task raise never cancels the swarm
- Semaphore acquired fresh per retry attempt, released before sleep: prevents convoy
  effect where retrying tasks starve the queue
- RateLimitError gets its own return path after final attempt (no fall-through to None)

Phase C2: multi-provider swarm — any provider whose sync .create(messages=, max_tokens=)
returns an object with .content (str) and .usage is swarm-capable via the KimiAsyncProvider
duck-typed wrapper.  SWARM_CAPABLE_PROVIDERS lists the verified set.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from mullet.core.agent.types import TokenUsage
from mullet.core.orchestration.swarm_types import SwarmTask, TaskResult

if TYPE_CHECKING:
    from mullet.core.providers.kimi_async import KimiAsyncProvider

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 20
DEFAULT_TASK_TIMEOUT = 120.0

# Providers that satisfy the duck-typed contract:
#   sync_provider.create(messages=<list>, max_tokens=<int>) -> obj with .content (str) and .usage
# All of these extend mullet.core.providers.base.Provider and return LLMResponse.
# "haiku", "sonnet", "opus" are Anthropic sub-aliases — same AnthropicProvider, different model.
SWARM_CAPABLE_PROVIDERS: frozenset[str] = frozenset({
    "kimi",
    "gemini",
    "openrouter",
    "anthropic",
    "haiku",
    "sonnet",
    "opus",
    "local",
})

# Anthropic model aliases (provider_name → model string)
_ANTHROPIC_ALIASES: dict[str, str] = {
    "haiku":      "claude-haiku-4-5",
    "sonnet":     "claude-sonnet-4-6",
    "opus":       "claude-opus-4-8",
    "anthropic":  "claude-sonnet-4-6",  # generic alias → sonnet
}

# Tier-aware default models for providers that don't use the full registry lookup.
# Used when we can't (or don't need to) call get_best_model for non-kimi providers.
_PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "gemini":     "gemini-2.5-flash",
    "openrouter": "deepseek/deepseek-v3",
    "local":      "default",
}


def _live_pressure() -> float:
    """Live budget pressure (0.0–1.0); fail-open to 0.0 (treat as full headroom).

    Phase F: lets the swarm's per-task provider pick exploit prepaid headroom the same
    way the main router does, instead of assuming pressure=0.0 unconditionally.
    """
    try:
        from mullet.hud.guard import compute_pressure
        from mullet.hud.usage import fetch_usage
        limits = fetch_usage()
        return compute_pressure(limits) if limits else 0.0
    except Exception:
        return 0.0


class RateLimitError(Exception):
    """Raised by KimiAsyncProvider when a 429 is received."""


class SwarmDispatcher:
    """Fan out SwarmTasks to async-wrapped providers with bounded concurrency.

    Guarantees:
    - dispatch() never raises; all failures are TaskResult(success=False)
    - Concurrency cap enforced via asyncio.Semaphore
    - Semaphore released before retry sleep (no convoy effect)
    - Ctrl-C (CancelledError) propagates cleanly; partial results returned by caller

    Phase C2: provider is selected per-task via _pick_task_provider, which uses
    the allowed_providers allowlist.  Kimi remains the default when allowed and capable.
    """

    def __init__(
        self,
        concurrency: int = DEFAULT_CONCURRENCY,
        task_timeout: float = DEFAULT_TASK_TIMEOUT,
        max_retries: int = 1,
        allowed_providers: frozenset[str] | None = None,
    ):
        self._sem = asyncio.Semaphore(concurrency)
        self._task_timeout = task_timeout
        self._max_retries = max_retries
        self._allowed_providers = allowed_providers

        # Phase C2: replace kimi-only guard with a "at least one swarm-capable provider" check.
        # When allowed_providers is set, require at least one provider in it to be swarm-capable.
        if allowed_providers is not None:
            capable = allowed_providers & SWARM_CAPABLE_PROVIDERS
            if not capable:
                raise ValueError(
                    f"SwarmDispatcher: no swarm-capable provider in allowed_providers={sorted(allowed_providers)}. "
                    f"Swarm-capable providers are: {sorted(SWARM_CAPABLE_PROVIDERS)}"
                )

    def _pick_task_provider(self, tier: str) -> str:
        """Select the provider_name to use for a given tier from the allowed set.

        Phase C2: uses pick_provider_for_tier (Phase B) when a Router is available.
        Falls back to a simple static preference order:
            kimi > gemini > openrouter > haiku > sonnet > anthropic > local
        Kimi is first to maintain backward compat: existing kimi-only swarms
        produce identical behaviour.

        Returns the provider_name string (always a member of SWARM_CAPABLE_PROVIDERS).
        """
        allowed = self._allowed_providers
        if allowed is None:
            # Legacy: no allowlist → always kimi
            return "kimi"

        # Intersect allowed with swarm-capable set
        capable_allowed = allowed & SWARM_CAPABLE_PROVIDERS
        if not capable_allowed:
            # Should not happen — __init__ guards against this — but be defensive
            return "kimi"

        # Phase B/F: pressure-aware routing — compute live budget pressure (fail-open to 0.0).
        pressure = _live_pressure()
        try:
            from mullet.core.router.resolver import build_default_router
            router, _ = build_default_router()  # returns (Router, ProviderResolver)
            chosen = router.pick_provider_for_tier(tier, capable_allowed, pressure)
            if chosen is not None and chosen in capable_allowed:
                return chosen
        except Exception as exc:
            logger.debug("swarm: pick_provider_for_tier failed (%s); using static fallback", exc)

        # Static fallback preference order within the capable_allowed set.
        # Phase E: gemini demoted to last-resort (not cheap vs kimi/openrouter).
        # local promoted above gemini — free, always reachable.
        _PREFERENCE = ["kimi", "openrouter", "haiku", "sonnet", "opus", "anthropic", "local", "gemini"]
        for name in _PREFERENCE:
            if name in capable_allowed:
                return name

        # Unreachable: _PREFERENCE covers all of SWARM_CAPABLE_PROVIDERS.
        return next(iter(capable_allowed))  # pragma: no cover

    def _build_sync_provider(self, provider_name: str, model_id: str):
        """Construct the correct sync provider for provider_name with model_id."""
        import os

        if provider_name == "kimi":
            from mullet.core.providers.kimi import KimiProvider
            api_key = os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY") or ""
            return KimiProvider(api_key=api_key or None, model=model_id)

        if provider_name == "gemini":
            from mullet.core.providers.gemini import GeminiProvider
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
            return GeminiProvider(api_key=api_key, model=model_id)

        if provider_name == "openrouter":
            from mullet.core.providers.openrouter import OpenRouterProvider
            api_key = os.environ.get("OPENROUTER_API_KEY") or ""
            return OpenRouterProvider(api_key=api_key, model=model_id)

        if provider_name in ("anthropic", "haiku", "sonnet", "opus"):
            from mullet.core.providers.anthropic import AnthropicProvider
            api_key = os.environ.get("ANTHROPIC_API_KEY") or ""
            return AnthropicProvider(api_key=api_key, model=model_id)

        if provider_name == "local":
            from mullet.core.env import local_llm_url
            from mullet.core.providers.openai_compat import OpenAICompatibleProvider
            return OpenAICompatibleProvider(base_url=local_llm_url(), model=model_id)

        raise ValueError(f"swarm: unknown provider_name {provider_name!r}")

    def _make_provider(
        self,
        tier: str,
        prompt_len: int,
        provider_name: str = "kimi",
        prompt: str | None = None,
    ) -> "KimiAsyncProvider":
        """Create a KimiAsyncProvider wrapping the correct sync provider for provider_name.

        For kimi: uses the registry-based model selection with token-estimate/escalation
        logic (D1 — preserving full backward compat).
        For all other providers: selects a sensible default model via the registry when
        available, falling back to _PROVIDER_DEFAULT_MODELS.

        Delegation safety D1 (kimi path): use ``estimate_tokens`` on the actual prompt
        (when provided) and thread ``context_tokens=`` into every ``get_best_model`` call.
        Falls back to ``prompt_len // 6`` if the estimator itself raises.
        """
        _FALLBACK_KIMI = "kimi-k2.7-code"  # k2.6 → k2.7-code (thinking ON, same price)
        _TIER_ORDER = ["LOW", "STANDARD", "ADVANCED", "CRITICAL"]

        from mullet.core.providers.kimi_async import KimiAsyncProvider

        # --- Kimi path: full registry-based model selection (D1, backward compat) ---
        if provider_name == "kimi":
            try:
                import os
                from mullet.core.providers.kimi import KimiProvider
                from mullet.core.providers.model_registry import get_registry
                from mullet.core.providers.model_info import get_model_info

                api_key = os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY") or ""

                # D1: real token estimate via tokenize.estimate_tokens.
                try:
                    from mullet.core.providers.tokenize import estimate_tokens as _estimate_tokens
                    if prompt is not None:
                        estimated_tokens = _estimate_tokens(prompt)
                    else:
                        estimated_tokens = max(1, prompt_len // 6)
                except Exception as _est_err:
                    logger.warning("swarm: estimate_tokens failed (%s); using char fallback", _est_err)
                    estimated_tokens = max(1, prompt_len // 6)

                start_idx = _TIER_ORDER.index(tier) if tier in _TIER_ORDER else 1
                registry = get_registry()
                for t in _TIER_ORDER[start_idx:]:
                    model_id = registry.get_best_model(
                        "kimi", t, context_tokens=estimated_tokens,
                    )
                    info = get_model_info(model_id)
                    if info.context_length >= estimated_tokens:
                        logger.debug(
                            "swarm: using model %s (tier=%s, prompt_chars=%d, est_tokens=%d)",
                            model_id, t, prompt_len, estimated_tokens,
                        )
                        return KimiAsyncProvider(provider=KimiProvider(api_key=api_key or None, model=model_id))
                # All tiers exhausted — use largest known
                model_id = registry.get_best_model(
                    "kimi", "CRITICAL", context_tokens=estimated_tokens,
                )
                return KimiAsyncProvider(provider=KimiProvider(api_key=api_key or None, model=model_id))
            except Exception as exc:
                logger.warning("swarm: _make_provider(kimi) failed (%s), using fallback model", exc)
                import os
                from mullet.core.providers.kimi import KimiProvider
                api_key = os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY") or ""
                return KimiAsyncProvider(provider=KimiProvider(api_key=api_key or None, model=_FALLBACK_KIMI))

        # --- Generic path for gemini / openrouter / anthropic aliases / local ---
        try:
            # For Anthropic aliases use the hardcoded model; for others try registry first.
            if provider_name in _ANTHROPIC_ALIASES:
                model_id = _ANTHROPIC_ALIASES[provider_name]
            else:
                # Try registry-based selection; fall back to static defaults.
                registry_provider = provider_name  # "gemini", "openrouter", "local"
                try:
                    from mullet.core.providers.model_registry import get_registry
                    registry = get_registry()
                    model_id = registry.get_best_model(registry_provider, tier)
                except Exception:
                    model_id = _PROVIDER_DEFAULT_MODELS.get(provider_name, "default")

            sync_provider = self._build_sync_provider(provider_name, model_id)
            logger.debug(
                "swarm: using %s/%s (tier=%s, prompt_chars=%d)",
                provider_name, model_id, tier, prompt_len,
            )
            return KimiAsyncProvider(provider=sync_provider)
        except Exception as exc:
            logger.warning(
                "swarm: _make_provider(%s) failed (%s); falling back to kimi", provider_name, exc
            )
            # Fall back to kimi when the requested provider fails to build
            return self._make_provider(tier, prompt_len, provider_name="kimi", prompt=prompt)

    async def dispatch(
        self,
        tasks: list[SwarmTask],
        on_progress: Callable[[TaskResult, int, int], None] | None = None,
    ) -> list[TaskResult]:
        """Fan out all tasks concurrently. Never raises — failures are records.

        on_progress: optional callback(result, completed, total) called after each task.
        """
        total = len(tasks)
        completed = 0
        results_map: dict[str, TaskResult] = {}

        async def _tracked(task: SwarmTask) -> TaskResult:
            nonlocal completed
            r = await self._run_task(task)
            completed += 1
            results_map[task.id] = r
            if on_progress is not None:
                try:
                    on_progress(r, completed, total)
                except Exception as _exc:
                    logger.warning("swarm_dispatcher: Error in on_progress callback: %s", _exc)
            return r

        coros = [_tracked(t) for t in tasks]
        raw = await asyncio.gather(*coros, return_exceptions=True)
        results = []
        for i, r in enumerate(raw):
            if isinstance(r, TaskResult):
                results.append(r)
            else:
                results.append(TaskResult(
                    task_id=tasks[i].id,
                    success=False,
                    response="",
                    error=str(r),
                    tokens=TokenUsage(),
                    elapsed_seconds=0.0,
                ))
        return results

    async def _run_task(self, task: SwarmTask) -> TaskResult:
        # Build prompt — file context is unsupported in swarm; pass task text only
        prompt = task.instruction

        # Phase C2: pick provider per-task based on tier and allowlist
        provider_name = self._pick_task_provider(task.tier)
        provider = self._make_provider(task.tier, len(prompt), provider_name=provider_name, prompt=prompt)
        _sync = getattr(provider, "_sync", provider)
        _model = getattr(_sync, "model", "?")
        _name = getattr(_sync, "name", getattr(provider, "name", provider_name))
        logger.info("[swarm] %s → %s/%s (tier=%s, chars=%d)", task.id, _name, _model, task.tier, len(prompt))

        for attempt in range(self._max_retries + 1):
            # Acquire semaphore fresh each attempt — released before sleep so other
            # tasks can proceed while this one backs off.
            async with self._sem:
                start = time.monotonic()
                try:
                    response = await asyncio.wait_for(
                        provider.generate(prompt),
                        timeout=self._task_timeout,
                    )
                    return TaskResult(
                        task_id=task.id,
                        success=True,
                        response=response.text,
                        error=None,
                        tokens=response.usage,
                        elapsed_seconds=time.monotonic() - start,
                        files_changed=[],
                        provider=_name,
                    )
                except asyncio.TimeoutError:
                    elapsed = time.monotonic() - start
                    logger.warning("swarm: task %s timed out (attempt %d)", task.id, attempt + 1)
                    if attempt == self._max_retries:
                        return TaskResult(
                            task_id=task.id, success=False, response="",
                            error="timeout", tokens=TokenUsage(), elapsed_seconds=elapsed,
                            provider=_name,
                        )
                    # Will retry — fall through to sleep outside semaphore
                except RateLimitError:
                    logger.warning("swarm: task %s rate limited (attempt %d)", task.id, attempt + 1)
                    if attempt == self._max_retries:
                        return TaskResult(
                            task_id=task.id, success=False, response="",
                            error="rate_limit", tokens=TokenUsage(),
                            elapsed_seconds=time.monotonic() - start,
                            provider=_name,
                        )
                    # Will retry — fall through to sleep outside semaphore
                except asyncio.CancelledError:
                    raise  # Propagate Ctrl-C cleanly
                except Exception as e:
                    elapsed = time.monotonic() - start
                    logger.warning("swarm: task %s failed: %s (attempt %d)", task.id, e, attempt + 1)
                    if attempt == self._max_retries:
                        return TaskResult(
                            task_id=task.id, success=False, response="",
                            error=str(e), tokens=TokenUsage(), elapsed_seconds=elapsed,
                            provider=_name,
                        )
            # Sleep OUTSIDE the semaphore so queued tasks can start during backoff
            backoff = 10 * (attempt + 1)
            logger.info("swarm: task %s backing off %ds before retry", task.id, backoff)
            await asyncio.sleep(backoff)

        # Unreachable — loop always returns or re-raises above — satisfies type checker
        return TaskResult(  # pragma: no cover
            task_id=task.id, success=False, response="",
            error="exhausted retries", tokens=TokenUsage(), elapsed_seconds=0.0,
        )
