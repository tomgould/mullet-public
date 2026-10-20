"""Dynamic model registry — fetches live model lists from provider APIs and caches them.

Supplements the static MODEL_DATABASE in model_info.py with live data from
Anthropic, Kimi/Moonshot, and Gemini APIs. Results are cached to disk with a
24-hour TTL. Falls back gracefully to static data if APIs are unreachable.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from mullet.core.providers.errors import ModelContextExceeded
from mullet.core.providers.model_info import MODEL_DATABASE
from mullet.core.providers.pricing import get_known_pricing, get_cache_hit_pricing
from mullet.core.providers.value_picker_constants import (
    KIMI_VALUE_PICK_CANDIDATES,
    OPENROUTER_FREE_TIER_CANDIDATES,
    OPENROUTER_VALUE_PICK_CANDIDATES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pricing reference — per 1K tokens (input, output)
# ---------------------------------------------------------------------------

# Safety margin applied to context_tokens when filtering candidate models.
# A 10% buffer absorbs the gap between estimate_tokens() approximation and
# the provider's real tokenizer (they rarely agree exactly). If a candidate's
# max_context < context_tokens * CONTEXT_SAFETY_MARGIN, the candidate is
# rejected by `_filter_candidates_by_context`.
CONTEXT_SAFETY_MARGIN = 1.1

# Canonical tier ladder used by D5 auto-escalation. When ``get_best_model``
# is asked for tier X with context_tokens=N and the picker raises
# ``ModelContextExceeded(all_candidates_exhausted=False)`` (i.e., a bigger
# sibling model exists in the same provider but at a higher tier), D5 walks
# this list from the requested tier upward and retries until either a tier
# succeeds or all tiers refuse — at which point we re-raise with
# ``all_candidates_exhausted=True``.
TIER_ORDER: list[str] = ["LOW", "STANDARD", "ADVANCED", "CRITICAL"]


def _tier_chain_starting_from(tier: str) -> list[str]:
    """Return the tier ladder from ``tier`` upward in escalation order.

    >>> _tier_chain_starting_from("STANDARD")
    ['STANDARD', 'ADVANCED', 'CRITICAL']
    >>> _tier_chain_starting_from("CRITICAL")
    ['CRITICAL']

    Unknown tiers return ``[tier]`` unchanged so a typo or future tier
    string never raises KeyError inside the dispatch path.
    """
    if not tier:
        return [tier]
    upper = tier.upper()
    try:
        idx = TIER_ORDER.index(upper)
    except ValueError:
        return [tier]
    return TIER_ORDER[idx:]


def _filter_candidates_by_context(
    candidates: list[tuple[str, int]],
    context_tokens: int | None,
) -> list[tuple[str, int]]:
    """Filter a list of (model_id, max_context) candidates by prompt size.

    Returns only candidates whose ``max_context >= context_tokens * CONTEXT_SAFETY_MARGIN``.
    When ``context_tokens`` is ``None`` or ``0``, the full list is returned unchanged
    (caller did not provide a prompt-size estimate; don't filter speculatively).

    This helper is shared by ``_best_kimi``, ``_best_gemini``, ``_best_anthropic``
    and the public ``get_best_model`` fallback path so the filter logic lives in
    one place — matching the D2 consolidation requirement.
    """
    if not context_tokens:
        return list(candidates)
    floor = int(context_tokens * CONTEXT_SAFETY_MARGIN)
    return [(mid, ctx) for mid, ctx in candidates if ctx >= floor]


# Pricing reference — per 1K tokens (input, output)
# Loaded from pricing.yaml via mullet.core.providers.pricing.
KNOWN_PRICING: dict[str, tuple[float, float]] = get_known_pricing()

# Cache-hit input pricing (per 1K tokens) — output price is same as cache-miss
# Loaded from pricing.yaml via mullet.core.providers.pricing.
KNOWN_CACHE_HIT_PRICING: dict[str, tuple[float, float]] = get_cache_hit_pricing()

# Preferred default model per provider for STANDARD-complexity tasks.
# These are used by get_best_model() to short-circuit heuristic selection
# and always return a known-good model when it fits the context window.
# Prevents live-fetch from silently picking a cheaper/wrong model (e.g.
# moonshot-v1-128k instead of kimi-k2.5) when the HUD label says otherwise.
PREFERRED_DEFAULTS: dict[str, str] = {
    "kimi": "kimi-k2.7-code",  # preferred default — K2.7-Code over K2.6 (same price, code-first)
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-2.5-flash",  # 3.5-flash is 5x the input cost for STANDARD/LOW; not worth it there
    "openrouter": "deepseek/deepseek-v4-flash",
}


# ---------------------------------------------------------------------------
# ProviderModel dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProviderModel:
    """A single model available from a provider API."""

    id: str
    provider: str              # "anthropic" | "kimi" | "gemini" | "openai"
    display_name: str
    context_length: int
    input_cost_per_1k: float
    output_cost_per_1k: float
    supports_tools: bool = True
    supports_vision: bool = False
    is_reasoning: bool = False  # o1, o3, claude-thinking models
    tier: str = "standard"      # "fast" | "standard" | "powerful" | "reasoning"
    is_available: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the model."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProviderModel:
        """Create a ProviderModel from a dictionary."""
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _pricing_for(model_id: str) -> tuple[float, float]:
    """Return (input, output) cost per 1K tokens for a known model, else (0, 0)."""
    # Exact match first
    if model_id in KNOWN_PRICING:
        return KNOWN_PRICING[model_id]
    # Substring match against known keys
    lower = model_id.lower()
    for key, prices in KNOWN_PRICING.items():
        if key in lower or lower in key:
            return prices
    # Fall back to MODEL_DATABASE
    if model_id in MODEL_DATABASE:
        info = MODEL_DATABASE[model_id]
        return info.input_cost_per_1k, info.output_cost_per_1k
    return (0.0, 0.0)


def _context_for(model_id: str) -> int:
    """Return context length for a known model, else 32_000."""
    if model_id in MODEL_DATABASE:
        return MODEL_DATABASE[model_id].context_length
    lower = model_id.lower()
    for key, info in MODEL_DATABASE.items():
        if key in lower or lower in key:
            return info.context_length
    # Heuristic from model name
    if "128k" in lower:
        return 131_072
    if "32k" in lower:
        return 32_768
    if "8k" in lower:
        return 8_192
    return 32_000


def _tier_for(model_id: str) -> str:
    """Classify a model into a tier bucket."""
    lower = model_id.lower()
    if any(x in lower for x in ("haiku", "flash", "mini", "fast", "lite", "8k")):
        return "fast"
    if any(x in lower for x in ("opus", "pro", "powerful", "ultra")):
        return "powerful"
    if any(x in lower for x in ("o1", "o3", "reasoning", "think", "r1", "distill")):
        return "reasoning"
    return "standard"


def _http_get(url: str, headers: dict[str, str] | None = None, timeout: int = 10) -> dict[str, Any]:
    """Perform a GET request with urllib and return parsed JSON."""
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# Live Moonshot pricing fetch
# ---------------------------------------------------------------------------

_MOONSHOT_PRICING_CACHE_PATH = Path.home() / ".mullet" / "moonshot_pricing_cache.json"
_MOONSHOT_PRICING_TTL_SECONDS = 24 * 3600


def _fetch_moonshot_pricing() -> dict[str, tuple[float, float]]:
    """Fetch live pricing from Moonshot API and cache it for 24h.

    Falls back to an empty dict on any error (network, auth, parse).
    """
    # 1. Check disk cache
    try:
        if _MOONSHOT_PRICING_CACHE_PATH.exists():
            raw = json.loads(_MOONSHOT_PRICING_CACHE_PATH.read_text())
            if time.time() - raw.get("timestamp", 0) < _MOONSHOT_PRICING_TTL_SECONDS:
                pricing = raw.get("pricing", {})
                # JSON round-trips tuples as lists — normalise back to tuples
                return {
                    k: tuple(v) if isinstance(v, list) else v
                    for k, v in pricing.items()
                }
    except Exception:
        pass

    # 2. No fresh cache — call API
    api_key = os.environ.get("MOONSHOT_API_KEY") or os.environ.get("KIMI_API_KEY", "")
    if not api_key:
        return {}

    try:
        data = _http_get(
            "https://api.moonshot.cn/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
    except Exception:
        return {}

    # 3. Extract pricing if present in the response
    pricing: dict[str, tuple[float, float]] = {}
    for item in data.get("data", []):
        model_id = item.get("id", "")
        if not model_id:
            continue
        pr = item.get("pricing") or {}
        if isinstance(pr, dict):
            inp = pr.get("input") or pr.get("input_cost") or pr.get("input_cost_per_1k")
            out = pr.get("output") or pr.get("output_cost") or pr.get("output_cost_per_1k")
            if inp is not None and out is not None:
                try:
                    pricing[model_id] = (float(inp), float(out))
                except (ValueError, TypeError):
                    pass

    # 4. Persist to disk cache (best-effort)
    try:
        _MOONSHOT_PRICING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=_MOONSHOT_PRICING_CACHE_PATH.parent, suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(
                    {"timestamp": time.time(), "pricing": pricing},
                    f,
                    indent=2,
                )
            os.replace(tmp_path, _MOONSHOT_PRICING_CACHE_PATH)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception:
        pass

    return pricing


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Thread-safe registry of provider models with disk caching."""

    CACHE_PATH = Path.home() / ".mullet" / "model_cache.json"
    CACHE_TTL_HOURS = 24

    def __init__(self) -> None:
        self._cache: dict[str, list[ProviderModel]] = {}
        self._cache_ts: dict[str, float] = {}
        self._lock = threading.Lock()
        # Merge live Moonshot pricing into static table on first instantiation
        live_pricing = _fetch_moonshot_pricing()
        if live_pricing:
            KNOWN_PRICING.update(live_pricing)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_models(self, provider: str, refresh: bool = False) -> list[ProviderModel]:
        """Return models for *provider*, fetching from API if cache is stale.

        A background thread performs the fetch so this call returns quickly.
        If the fetch fails, returns static models.
        """
        with self._lock:
            if not refresh and self._is_fresh(provider):
                return list(self._cache[provider])

        # Kick off a background fetch but also return immediately with best data
        fetch_thread = threading.Thread(
            target=self._fetch_and_cache,
            args=(provider,),
            daemon=True,
        )
        fetch_thread.start()
        # Give the thread a short window to complete (keeps first call useful)
        fetch_thread.join(timeout=12)

        with self._lock:
            if provider in self._cache:
                return list(self._cache[provider])

        return self._static_models(provider)

    def _assert_can_serve(
        self,
        provider: str,
        model_id: str,
        complexity: str,
        context_tokens: int | None,
    ) -> str:
        """Post-pick guard: confirm the chosen model's context window fits.

        When ``context_tokens`` is provided, queries the ConstraintRegistry
        and raises ValueError if the picked model's ``max_context`` is below
        ``context_tokens * CONTEXT_SAFETY_MARGIN``. Returns the model id
        unchanged on success. This is the D2 invariant: NO ``get_best_model``
        code path may return a model without first confirming the constraint
        registry agrees it can hold the prompt.

        Only the ``context_tokens`` dimension is post-validated here — tool /
        streaming requirements are the caller's concern (agent or dispatcher)
        because the picker doesn't know which capability the specific task
        needs.
        """
        if not context_tokens or not model_id:
            return model_id
        try:
            from mullet.core.providers.constraints import get_constraint_registry
            cr = get_constraint_registry()
            # Only check context-window fit with the 10% safety margin.
            # requires_tools/streaming are left to the caller since
            # get_best_model doesn't know the task's capability profile.
            needs = {
                "context_tokens": int(context_tokens * CONTEXT_SAFETY_MARGIN),
            }
            if cr.get(model_id).can_serve(needs):
                return model_id
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001 — constraint registry failure is non-fatal
            logger.warning(
                "constraint registry failed while post-validating %s/%s: %s",
                provider, model_id, exc,
            )
            return model_id
        # can_serve returned False — surface a clear error rather than pass
        # a doomed model to the wire.
        logger.warning(
            "get_best_model returned %s but ConstraintRegistry.can_serve=False "
            "for context_tokens=%d (provider=%s, tier=%s)",
            model_id, context_tokens, provider, complexity,
        )
        # Decide exhaustion: if the picked model's declared max_context is
        # still >= context_tokens * 1.1 the rejection came from a non-context
        # dimension (tools/streaming/capability). A bigger sibling won't help
        # — but the caller can still treat this as "escalation possible" and
        # try a different provider. Only flag exhausted when the model's
        # declared max_context literally cannot hold the prompt.
        try:
            max_ctx = int(getattr(cr.get(model_id), "max_context", 0) or 0)
        except Exception:
            max_ctx = 0
        exhausted = max_ctx > 0 and max_ctx < int(context_tokens * CONTEXT_SAFETY_MARGIN)
        raise ModelContextExceeded(
            provider=provider,
            tier=complexity,
            context_tokens=int(context_tokens),
            tried_candidates=[model_id],
            max_candidate_context=max_ctx or None,
            all_candidates_exhausted=exhausted,
        )

    def get_best_model(
        self,
        provider: str,
        complexity: str = "STANDARD",
        context_needed: int = 4096,
        allowed_providers: set[str] | None = None,
        context_tokens: int | None = None,
        prefer_reasoning: bool = False,
        preflight=None,
        speed_mode: str = "balanced",
    ) -> str:
        """Return the model ID best suited for the given complexity and context.

        complexity: "LOW" | "STANDARD" | "ADVANCED" | "CRITICAL"
        context_needed: minimum context window required (tokens)
        allowed_providers: if set and `provider` not in this set, raise ValueError.
        context_tokens: optional prompt size; when provided, preferred model is only
                        returned if ConstraintRegistry.can_serve() accepts it.
                        Every return path is post-validated against
                        ``ConstraintRegistry`` — raising ValueError if the
                        picked model can't serve the request (D2 invariant).
        """
        if allowed_providers is not None and provider not in allowed_providers:
            raise ValueError(
                f"provider {provider!r} not in allowed_providers {allowed_providers!r}"
            )
        complexity = complexity.upper()

        # D5 auto-escalation: walk the tier ladder from the requested tier
        # upward. If the picker at tier X raises MCE(exhausted=False) — a
        # bigger sibling model exists in this provider but at a higher tier
        # — silently bump to the next tier and retry. Stop and re-raise on
        # exhausted=True (the provider's biggest model can't serve the
        # prompt; climbing tiers will hit the same wall) or on any non-MCE
        # error.
        requested_tier = complexity
        chain = _tier_chain_starting_from(requested_tier)
        last_mce: ModelContextExceeded | None = None
        for tier in chain:
            try:
                pick_kwargs = {
                    "provider": provider,
                    "complexity": tier,
                    "context_needed": context_needed,
                    "context_tokens": context_tokens,
                    "preflight": preflight,
                    "speed_mode": speed_mode,
                }
                if prefer_reasoning:
                    pick_kwargs["prefer_reasoning"] = True
                picked = self._pick_model(**pick_kwargs)
                # D2: post-validate every return through ConstraintRegistry.
                validated = self._assert_can_serve(
                    provider, picked, tier, context_tokens,
                )
                if tier != requested_tier:
                    logger.info(
                        "model_registry: %s requested_tier=%s context_tokens=%s "
                        "auto-escalated to tier=%s (picked=%s)",
                        provider, requested_tier, context_tokens, tier, validated,
                    )
                return validated
            except ModelContextExceeded as exc:
                last_mce = exc
                if exc.all_candidates_exhausted:
                    # Provider's biggest model is already too small —
                    # climbing tiers won't help. Re-raise immediately.
                    raise
                # Non-exhausted: a bigger sibling could exist at the next
                # tier; continue the walk.
                continue

        # Every tier in the chain refused with a non-exhausted MCE — the
        # provider has no model that can serve this prompt. Convert the
        # trail of non-exhausted refusals into a single exhausted MCE so
        # callers see "this provider is doomed; move on or split the task".
        max_ctx = (
            last_mce.max_candidate_context if last_mce is not None else None
        )
        tried = list(last_mce.tried_candidates) if last_mce is not None else []
        raise ModelContextExceeded(
            provider=provider,
            tier=requested_tier,
            context_tokens=int(context_tokens or 0),
            tried_candidates=tried,
            max_candidate_context=max_ctx,
            all_candidates_exhausted=True,
        )

    def _pick_model(
        self,
        *,
        provider: str,
        complexity: str,
        context_needed: int,
        context_tokens: int | None,
        prefer_reasoning: bool = False,
        preflight=None,
        speed_mode: str = "balanced",
    ) -> str:
        """Core heuristic that returns a model id without post-validation.

        Kept separate so ``get_best_model`` can funnel every return path
        through ``_assert_can_serve`` exactly once (D2).
        """
        from mullet.core.router.routing_matrix import _tier_gte
        from mullet.core.router.complexity import Tier

        # Local-first: if local is available and tier is LOW or STANDARD, prefer it (free)
        if provider == "local" or (
            provider is None
            and speed_mode in ("economy", "balanced")
            and preflight is not None
            and not _tier_gte(preflight.complexity_tier, Tier.ADVANCED)
        ):
            local_result = self._best_local(preflight=preflight, speed_mode=speed_mode)
            if local_result is not None:
                return local_result.id

        # STANDARD tasks: prefer a known-good default if it fits the context window.
        # This short-circuits heuristic selection and prevents the live-fetch from
        # silently choosing a different model (e.g. a cheap fallback instead of
        # kimi-k2.5), which is the root cause of the HUD/wire desync bug.
        if complexity == "STANDARD":
            preferred = PREFERRED_DEFAULTS.get(provider)
            if preferred:
                # When caller asks for a reasoning model, don't short-circuit
                # on the non-reasoning preferred default.
                if prefer_reasoning and "thinking" not in preferred:
                    pass  # fall through to heuristic so _best_kimi can override
                else:
                    live_models = self._cache.get(provider, [])
                    live_hit = any(
                        m.id == preferred and m.context_length >= context_needed and m.is_available
                        for m in live_models
                    )
                    static_hit = (
                        preferred in MODEL_DATABASE
                        and MODEL_DATABASE[preferred].context_length >= context_needed
                    )
                    if live_hit or static_hit:
                        if context_tokens is not None:
                            try:
                                from mullet.core.providers.constraints import get_constraint_registry
                                cr = get_constraint_registry()
                                needs = {
                                    "context_tokens": int(context_tokens * CONTEXT_SAFETY_MARGIN),
                                }
                                if cr.get(preferred).can_serve(needs):
                                    return preferred
                                # constraint rejected — fall through to heuristic
                            except Exception:
                                return preferred
                        else:
                            return preferred
                    # preferred not available — fall through to heuristic

        # Named providers: context-aware selection with live-model fallback.
        # When the caller supplied an explicit prompt-size estimate, use it as
        # the effective minimum context window so the heuristic doesn't pick
        # a model that literally cannot hold the request.
        effective_ctx = max(context_needed, context_tokens or 0)
        if provider == "anthropic":
            return self._best_anthropic(complexity, effective_ctx, context_tokens=context_tokens)
        if provider == "kimi":
            return self._best_kimi(complexity, effective_ctx, context_tokens=context_tokens, prefer_reasoning=prefer_reasoning, preflight=preflight, speed_mode=speed_mode)
        if provider == "gemini":
            return self._best_gemini(complexity, effective_ctx, context_tokens=context_tokens, preflight=preflight, speed_mode=speed_mode)
        if provider == "openrouter":
            return self._best_openrouter(complexity, effective_ctx, context_tokens=context_tokens, prefer_reasoning=prefer_reasoning, preflight=preflight, speed_mode=speed_mode)

        # Generic: pick from live/static list by tier
        models = self.get_models(provider)
        models = [m for m in models if m.context_length >= context_needed and m.is_available]
        if not models:
            models = self._static_models(provider)

        # Apply D2 safety filter when caller supplied a prompt-size estimate.
        if context_tokens:
            allowed = {
                mid for mid, _ in _filter_candidates_by_context(
                    [(m.id, m.context_length) for m in models],
                    context_tokens,
                )
            }
            models = [m for m in models if m.id in allowed]

        target_tier = {
            "LOW": "fast",
            "STANDARD": "standard",
            "ADVANCED": "powerful",
            "CRITICAL": "powerful",
        }.get(complexity, "standard")

        tier_models = [m for m in models if m.tier == target_tier]
        if not tier_models:
            tier_models = models

        if not tier_models:
            return ""

        # LOW → cheapest; CRITICAL → lowest output cost (proxy for most capable with budget)
        if complexity in ("LOW",):
            return min(tier_models, key=lambda m: m.input_cost_per_1k).id
        return min(tier_models, key=lambda m: m.output_cost_per_1k).id

    def refresh_all(self) -> dict[str, int]:
        """Refresh all providers and return {provider: model_count}."""
        threads = []
        counts: dict[str, int] = {}

        for provider in ("anthropic", "kimi", "gemini", "openrouter", "local"):
            t = threading.Thread(
                target=self._fetch_and_cache,
                args=(provider,),
                daemon=True,
            )
            threads.append((provider, t))
            t.start()

        for provider, t in threads:
            t.join(timeout=15)
            with self._lock:
                counts[provider] = len(self._cache.get(provider, []))

        return counts

    def list_all(self) -> dict[str, list[ProviderModel]]:
        """Return all currently cached (or static) models by provider."""
        result: dict[str, list[ProviderModel]] = {}
        for provider in ("anthropic", "kimi", "gemini", "openrouter", "local"):
            with self._lock:
                if provider in self._cache:
                    result[provider] = list(self._cache[provider])
                else:
                    result[provider] = self._static_models(provider)
        return result

    # ------------------------------------------------------------------
    # Cache persistence
    # ------------------------------------------------------------------

    def _load_cache(self) -> None:
        """Load model cache from disk if it exists and is within TTL."""
        if not self.CACHE_PATH.exists():
            return
        try:
            raw = json.loads(self.CACHE_PATH.read_text())
            now = time.time()
            ttl_seconds = self.CACHE_TTL_HOURS * 3600
            with self._lock:
                for provider, entry in raw.items():
                    ts = entry.get("timestamp", 0)
                    if now - ts < ttl_seconds:
                        self._cache[provider] = [
                            ProviderModel.from_dict(m) for m in entry.get("models", [])
                        ]
                        self._cache_ts[provider] = ts
        except Exception:
            # Corrupt cache — ignore, will be overwritten on next fetch
            pass

    def _save_cache(self) -> None:
        """Atomically save current cache to disk."""
        self.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {
                provider: {
                    "timestamp": self._cache_ts.get(provider, time.time()),
                    "models": [m.to_dict() for m in models],
                }
                for provider, models in self._cache.items()
            }
        # Atomic write: write to tmp then rename
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self.CACHE_PATH.parent, suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, self.CACHE_PATH)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Provider fetch methods
    # ------------------------------------------------------------------

    def _fetch_and_cache(self, provider: str) -> None:
        """Fetch models from provider API and update cache. Never raises."""
        fetcher = {
            "anthropic": self._fetch_anthropic,
            "kimi": self._fetch_kimi,
            "gemini": self._fetch_gemini,
            "openrouter": self._fetch_openrouter,
            "local": self._fetch_local,
        }.get(provider)

        if fetcher is None:
            return

        try:
            models = fetcher()
        except Exception:
            models = []

        if not models:
            models = self._static_models(provider)

        with self._lock:
            self._cache[provider] = models
            self._cache_ts[provider] = time.time()

        self._save_cache()

    def _fetch_anthropic(self) -> list[ProviderModel]:
        """Fetch model list from Anthropic API."""
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return []

        data = _http_get(
            "https://api.anthropic.com/v1/models",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )

        models: list[ProviderModel] = []
        for item in data.get("data", []):
            model_id: str = item.get("id", "")
            if not model_id:
                continue
            inp, out = _pricing_for(model_id)
            ctx = _context_for(model_id)
            models.append(
                ProviderModel(
                    id=model_id,
                    provider="anthropic",
                    display_name=item.get("display_name", model_id),
                    context_length=ctx,
                    input_cost_per_1k=inp,
                    output_cost_per_1k=out,
                    supports_tools=True,
                    supports_vision=True,
                    is_reasoning="think" in model_id.lower(),
                    tier=_tier_for(model_id),
                )
            )
        return models

    def _fetch_kimi(self) -> list[ProviderModel]:
        """Fetch model list from Moonshot/Kimi API (OpenAI-compatible)."""
        api_key = os.environ.get("MOONSHOT_API_KEY", os.environ.get("KIMI_API_KEY", ""))
        if not api_key:
            return []

        data = _http_get(
            "https://api.moonshot.ai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        models: list[ProviderModel] = []
        for item in data.get("data", []):
            model_id: str = item.get("id", "")
            if not model_id:
                continue
            inp, out = _pricing_for(model_id)
            ctx = _context_for(model_id)
            models.append(
                ProviderModel(
                    id=model_id,
                    provider="kimi",
                    display_name=item.get("id", model_id),
                    context_length=ctx,
                    input_cost_per_1k=inp,
                    output_cost_per_1k=out,
                    supports_tools=True,
                    supports_vision=False,
                    is_reasoning=False,
                    tier=_tier_for(model_id),
                )
            )

        # Ensure key Kimi models are always present even if the live API omits them.
        for pinned_id, pinned_name, pinned_tier in (
            ("kimi-k2.6", "Kimi K2.6", "powerful"),
            ("kimi-k2.5", "Kimi K2.5", "standard"),
            ("kimi-k2.7-code", "Kimi K2.7 Code", "powerful"),
            ("kimi-k2.7-code-highspeed", "Kimi K2.7 Code HighSpeed", "powerful"),
        ):
            if not any(m.id == pinned_id for m in models) and pinned_id in MODEL_DATABASE:
                info = MODEL_DATABASE[pinned_id]
                models.append(ProviderModel(
                    id=pinned_id,
                    provider="kimi",
                    display_name=pinned_name,
                    context_length=info.context_length,
                    input_cost_per_1k=info.input_cost_per_1k,
                    output_cost_per_1k=info.output_cost_per_1k,
                    supports_tools=True,
                    supports_vision=True,
                    tier=pinned_tier,
                ))

        return models

    def _fetch_gemini(self) -> list[ProviderModel]:
        """Fetch model list from Google Generative Language API."""
        api_key = os.environ.get("GOOGLE_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
        if not api_key:
            return []

        data = _http_get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        )

        models: list[ProviderModel] = []
        for item in data.get("models", []):
            name: str = item.get("name", "")  # e.g. "models/gemini-2.5-flash"
            if not name.startswith("models/gemini"):
                continue
            # Only include text generation models
            supported = item.get("supportedGenerationMethods", [])
            if "generateContent" not in supported:
                continue
            model_id = name.removeprefix("models/")
            inp, out = _pricing_for(model_id)
            ctx = item.get("inputTokenLimit", _context_for(model_id))
            models.append(
                ProviderModel(
                    id=model_id,
                    provider="gemini",
                    display_name=item.get("displayName", model_id),
                    context_length=ctx,
                    input_cost_per_1k=inp,
                    output_cost_per_1k=out,
                    supports_tools=True,
                    supports_vision=True,
                    is_reasoning="think" in model_id.lower(),
                    tier=_tier_for(model_id),
                )
            )
        return models

    def _fetch_local(self) -> list[ProviderModel]:
        """Fetch model list from LM Studio (http://localhost:1234)."""
        try:
            data = _http_get("http://localhost:1234/api/v0/models", timeout=3)
        except Exception:
            return []

        models: list[ProviderModel] = []
        for item in data.get("data", []):
            if item.get("type") not in ("llm", "vlm"):
                continue
            model_id: str = item.get("id", "")
            if not model_id:
                continue
            ctx = item.get("max_context_length", _context_for(model_id))
            state = item.get("state", "not-loaded")
            display = model_id + (" [loaded]" if state == "loaded" else "")
            models.append(
                ProviderModel(
                    id=model_id,
                    provider="local",
                    display_name=display,
                    context_length=ctx,
                    input_cost_per_1k=0.0,
                    output_cost_per_1k=0.0,
                    supports_tools="tool_use" in item.get("capabilities", []),
                    supports_vision=item.get("type") == "vlm",
                    is_reasoning=False,
                    tier=_tier_for(model_id),
                    is_available=True,
                )
            )
        return models

    def _fetch_openrouter(self) -> list[ProviderModel]:
        """Fetch model list from OpenRouter API (no auth required, but pass key if available)."""
        headers: dict[str, str] = {}
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        data = _http_get("https://openrouter.ai/api/v1/models", headers=headers or None)

        models: list[ProviderModel] = []
        for item in data.get("data", []):
            model_id: str = item.get("id", "")
            if not model_id:
                continue
            pr = item.get("pricing") or {}
            try:
                inp = float(pr.get("prompt", 0)) * 1000
                out = float(pr.get("completion", 0)) * 1000
            except (TypeError, ValueError):
                inp, out = 0.0, 0.0
            ctx = item.get("context_length", _context_for(model_id))
            models.append(
                ProviderModel(
                    id=model_id,
                    provider="openrouter",
                    display_name=item.get("name", model_id),
                    context_length=ctx,
                    input_cost_per_1k=inp,
                    output_cost_per_1k=out,
                    supports_tools=True,
                    supports_vision=False,
                    is_reasoning="reasoning" in model_id.lower() or "r1" in model_id.lower() or "think" in model_id.lower(),
                    tier=_tier_for(model_id),
                )
            )
        return models

    # ------------------------------------------------------------------
    # Static fallback
    # ------------------------------------------------------------------

    def _static_models(self, provider: str) -> list[ProviderModel]:
        """Hardcoded current-known models for offline fallback."""
        if provider == "anthropic":
            return [
                ProviderModel(
                    id="claude-haiku-4-5-20251001",
                    provider="anthropic",
                    display_name="Claude Haiku 4.5",
                    context_length=200_000,
                    input_cost_per_1k=0.001,
                    output_cost_per_1k=0.005,
                    supports_vision=True,
                    tier="fast",
                ),
                ProviderModel(
                    id="claude-sonnet-4-6",
                    provider="anthropic",
                    display_name="Claude Sonnet 4.6",
                    context_length=1_000_000,
                    input_cost_per_1k=0.003,
                    output_cost_per_1k=0.015,
                    supports_vision=True,
                    tier="standard",
                ),
                ProviderModel(
                    id="claude-opus-4-6",
                    provider="anthropic",
                    display_name="Claude Opus 4.6",
                    context_length=1_000_000,
                    input_cost_per_1k=0.005,
                    output_cost_per_1k=0.025,
                    supports_vision=True,
                    tier="powerful",
                ),
                ProviderModel(
                    id="claude-opus-4-7",
                    provider="anthropic",
                    display_name="Claude Opus 4.7",
                    context_length=1_000_000,
                    input_cost_per_1k=0.005,
                    output_cost_per_1k=0.025,
                    supports_vision=True,
                    tier="powerful",
                ),
                ProviderModel(
                    id="claude-opus-4-8",
                    provider="anthropic",
                    display_name="Claude Opus 4.8",
                    context_length=1_000_000,
                    input_cost_per_1k=0.005,
                    output_cost_per_1k=0.025,
                    supports_vision=True,
                    tier="powerful",
                ),
                ProviderModel(
                    id="claude-fable-5",
                    provider="anthropic",
                    display_name="Claude Fable 5 (deactivated)",
                    context_length=1_000_000,
                    input_cost_per_1k=0.010,
                    output_cost_per_1k=0.050,
                    supports_vision=True,
                    tier="powerful",
                    is_available=False,  # deactivated by Anthropic — kept in catalog, not selectable
                ),
            ]

        if provider == "kimi":
            return [
                ProviderModel(
                    id="kimi-k2.6",
                    provider="kimi",
                    display_name="Kimi K2.6",
                    context_length=262_144,
                    input_cost_per_1k=0.00095,
                    output_cost_per_1k=0.00400,
                    supports_tools=True,
                    supports_vision=True,
                    tier="powerful",
                ),
                ProviderModel(
                    id="kimi-k2.5",
                    provider="kimi",
                    display_name="Kimi K2.5",
                    context_length=262_144,
                    input_cost_per_1k=0.00060,
                    output_cost_per_1k=0.00300,
                    supports_tools=True,
                    supports_vision=True,
                    tier="standard",
                ),
                ProviderModel(
                    id="kimi-k2.7-code",
                    provider="kimi",
                    display_name="Kimi K2.7 Code",
                    context_length=262_144,
                    input_cost_per_1k=0.00095,
                    output_cost_per_1k=0.00400,
                    supports_tools=True,
                    supports_vision=True,
                    tier="powerful",
                ),
                ProviderModel(
                    id="kimi-k2.7-code-highspeed",
                    provider="kimi",
                    display_name="Kimi K2.7 Code HighSpeed",
                    context_length=262_144,
                    input_cost_per_1k=0.0019,
                    output_cost_per_1k=0.00800,
                    supports_tools=True,
                    supports_vision=True,
                    tier="powerful",
                ),
                # deprecated 2026-05-25 — kept for backward compatibility only
                ProviderModel(
                    id="kimi-k2-0905-preview",
                    provider="kimi",
                    display_name="Kimi K2 (0905-preview)",
                    context_length=262_144,
                    input_cost_per_1k=0.00060,
                    output_cost_per_1k=0.00250,
                    supports_tools=True,
                    supports_vision=False,
                    tier="standard",
                ),
                # deprecated 2026-05-25 — kept for backward compatibility only
                ProviderModel(
                    id="kimi-k2-0711-preview",
                    provider="kimi",
                    display_name="Kimi K2 (0711-preview)",
                    context_length=131_072,
                    input_cost_per_1k=0.00060,
                    output_cost_per_1k=0.00250,
                    supports_tools=True,
                    supports_vision=False,
                    tier="standard",
                ),
                # deprecated 2026-05-25 — kept for backward compatibility only
                ProviderModel(
                    id="kimi-k2-thinking",
                    provider="kimi",
                    display_name="Kimi K2 Thinking",
                    context_length=262_144,
                    input_cost_per_1k=0.00060,
                    output_cost_per_1k=0.00250,
                    supports_tools=True,
                    supports_vision=False,
                    is_reasoning=True,
                    tier="reasoning",
                ),
                # deprecated 2026-05-25 — kept for backward compatibility only
                ProviderModel(
                    id="kimi-k2-turbo-preview",
                    provider="kimi",
                    display_name="Kimi K2 Turbo",
                    context_length=262_144,
                    input_cost_per_1k=0.00115,
                    output_cost_per_1k=0.00800,
                    supports_tools=True,
                    supports_vision=False,
                    tier="fast",
                ),
                # deprecated 2026-05-25 — kept for backward compatibility only
                ProviderModel(
                    id="kimi-k2-thinking-turbo",
                    provider="kimi",
                    display_name="Kimi K2 Thinking Turbo",
                    context_length=262_144,
                    input_cost_per_1k=0.00115,
                    output_cost_per_1k=0.00800,
                    supports_tools=True,
                    supports_vision=False,
                    is_reasoning=True,
                    tier="reasoning",
                ),
            ]

        if provider == "gemini":
            return [
                ProviderModel(
                    id="gemini-2.5-flash-lite",
                    provider="gemini",
                    display_name="Gemini 2.5 Flash Lite",
                    context_length=1_000_000,
                    input_cost_per_1k=0.000100,
                    output_cost_per_1k=0.000400,
                    supports_vision=True,
                    tier="fast",
                ),
                ProviderModel(
                    id="gemini-2.5-flash",
                    provider="gemini",
                    display_name="Gemini 2.5 Flash",
                    context_length=1_000_000,
                    input_cost_per_1k=0.000300,
                    output_cost_per_1k=0.002500,
                    supports_vision=True,
                    tier="fast",
                ),
                ProviderModel(
                    id="gemini-2.5-pro",
                    provider="gemini",
                    display_name="Gemini 2.5 Pro",
                    context_length=1_000_000,
                    input_cost_per_1k=0.002000,
                    output_cost_per_1k=0.012000,
                    supports_vision=True,
                    tier="powerful",
                ),
            ]

        if provider == "openrouter":
            return [
                ProviderModel(id="deepseek/deepseek-chat", provider="openrouter",
                    display_name="DeepSeek V3", context_length=163_840,
                    input_cost_per_1k=0.00027, output_cost_per_1k=0.0011,
                    supports_tools=True, tier="standard"),
                ProviderModel(id="deepseek/deepseek-v4-flash", provider="openrouter",
                    display_name="DeepSeek V4 Flash", context_length=1_000_000,
                    input_cost_per_1k=0.000098, output_cost_per_1k=0.000196,
                    supports_tools=True, tier="standard"),
                ProviderModel(id="deepseek/deepseek-v4-pro", provider="openrouter",
                    display_name="DeepSeek V4 Pro", context_length=1_000_000,
                    input_cost_per_1k=0.000435, output_cost_per_1k=0.00087,
                    supports_tools=True, tier="powerful"),
                ProviderModel(id="z-ai/glm-5.1", provider="openrouter",
                    display_name="GLM 5.1", context_length=202_752,
                    input_cost_per_1k=0.00098, output_cost_per_1k=0.00308,
                    supports_tools=True, tier="powerful"),
                ProviderModel(id="minimax/minimax-m3", provider="openrouter",
                    display_name="MiniMax M3", context_length=1_048_576,
                    input_cost_per_1k=0.00030, output_cost_per_1k=0.00120,
                    supports_tools=True, supports_vision=True, tier="standard"),
                ProviderModel(id="qwen/qwen3-coder-next", provider="openrouter",
                    display_name="Qwen3 Coder Next", context_length=203_000,
                    input_cost_per_1k=0.00100, output_cost_per_1k=0.00320,
                    supports_tools=True, tier="standard"),
                ProviderModel(id="meta-llama/llama-4-scout", provider="openrouter",
                    display_name="Llama 4 Scout", context_length=512_000,
                    input_cost_per_1k=0.00017, output_cost_per_1k=0.00067,
                    supports_tools=True, tier="fast"),
                ProviderModel(id="meta-llama/llama-4-maverick", provider="openrouter",
                    display_name="Llama 4 Maverick", context_length=1_000_000,
                    input_cost_per_1k=0.0003, output_cost_per_1k=0.0012,
                    supports_tools=True, tier="standard"),
                ProviderModel(id="qwen/qwen-2.5-coder-32b-instruct", provider="openrouter",
                    display_name="Qwen 2.5 Coder 32B", context_length=131_072,
                    input_cost_per_1k=0.0002, output_cost_per_1k=0.0006,
                    supports_tools=True, tier="standard"),
                ProviderModel(id="qwen/qwen3-30b-a3b", provider="openrouter",
                    display_name="Qwen3 30B MoE", context_length=131_072,
                    input_cost_per_1k=0.00015, output_cost_per_1k=0.0006,
                    supports_tools=True, is_reasoning=True, tier="reasoning"),
                ProviderModel(id="mistralai/devstral-small", provider="openrouter",
                    display_name="Devstral Small", context_length=131_072,
                    input_cost_per_1k=0.0001, output_cost_per_1k=0.0003,
                    supports_tools=True, tier="fast"),
                ProviderModel(id="x-ai/grok-3-mini-beta", provider="openrouter",
                    display_name="Grok 3 Mini", context_length=131_072,
                    input_cost_per_1k=0.0003, output_cost_per_1k=0.0005,
                    is_reasoning=True, tier="reasoning"),
            ]

        # local: no static fallback — either LM Studio is reachable or not
        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_fresh(self, provider: str) -> bool:
        """Return True if cached data for *provider* is within TTL."""
        if provider not in self._cache:
            return False
        age = time.time() - self._cache_ts.get(provider, 0)
        return age < self.CACHE_TTL_HOURS * 3600

    def _kimi_candidates(self) -> list[ProviderModel]:
        """Return available Kimi models from cache or static fallback."""
        live = self._cache.get("kimi", [])
        if live:
            return live
        return self._static_models("kimi")

    def _gemini_candidates(self) -> list[ProviderModel]:
        """Return available Gemini models from cache or static fallback."""
        live = self._cache.get("gemini", [])
        if live:
            return live
        return self._static_models("gemini")

    def _openrouter_candidates(self) -> list[ProviderModel]:
        """Return available OpenRouter models from cache or static fallback."""
        live = self._cache.get("openrouter", [])
        if live:
            return live
        return self._static_models("openrouter")

    # ------------------------------------------------------------------
    # Per-provider best-model selectors
    # ------------------------------------------------------------------

    def _best_anthropic(
        self,
        complexity: str,
        context_needed: int = 4096,
        *,
        context_tokens: int | None = None,
        prefer_reasoning: bool = False,
    ) -> str:
        # Try live models first — filter by context_tokens safety margin.
        live = self._cache.get("anthropic", [])
        if live:
            result = self._pick_from_live(
                live, complexity, context_needed, context_tokens=context_tokens,
            )
            if result:
                self._maybe_log_downgrade("anthropic", complexity, result, context_tokens)
                return result

        # routing-v3 Phase 3c (Anthropic): try the value-based picker (cost x
        # complexity x quality across the full candidate set) before falling
        # back to the static mapping/ordered ladder below. Any exception or
        # None result falls through unchanged to the existing static logic --
        # this must never break routing, only improve it. domain is always
        # "coding" here since no Anthropic model is is_reasoning_tuned (same
        # dead-reasoning-gate situation as Gemini) -- prefer_reasoning is
        # accepted for signature parity with the other _best_* methods but is
        # otherwise inert.
        try:
            from mullet.core.providers.value_picker import pick_by_value
            value_candidates: list[tuple[str, int]] = [
                ("claude-opus-4-8",              1_000_000),
                ("claude-opus-4-7",              1_000_000),
                ("claude-opus-4-6",              1_000_000),
                ("claude-sonnet-5",              1_000_000),
                ("claude-sonnet-4-6",            1_000_000),
                ("claude-haiku-4-5-20251001",    200_000),
            ]
            value_pick = pick_by_value(
                value_candidates, complexity, context_needed, "coding",
                context_tokens=context_tokens, prefer_reasoning=False,
            )
            if value_pick is not None:
                logger.debug(
                    "_best_anthropic: value picker selected %s (domain=coding, tier=%s)",
                    value_pick, complexity,
                )
                self._maybe_log_downgrade("anthropic", complexity, value_pick, context_tokens)
                return value_pick
        except Exception as exc:
            logger.warning(
                "_best_anthropic: value picker failed (%s) -- falling back to static ladder",
                exc,
            )

        # Static fallback — haiku has 200K context; escalate to sonnet if larger needed
        mapping = {
            "LOW": "claude-haiku-4-5-20251001",
            "STANDARD": "claude-sonnet-5",
            "ADVANCED": "claude-opus-4-8",
            "CRITICAL": "claude-opus-4-8",
        }
        preferred = mapping.get(complexity, "claude-sonnet-5")
        ordered: list[tuple[str, int]] = [
            ("claude-opus-4-8", 1_000_000),
            ("claude-opus-4-7", 1_000_000),
            ("claude-opus-4-6", 1_000_000),
            ("claude-sonnet-5", 1_000_000),
            ("claude-sonnet-4-6", 1_000_000),
            ("claude-haiku-4-5-20251001", 200_000),
        ]
        actual = self._first_fitting(
            provider="anthropic",
            preferred=preferred,
            candidates=ordered,
            context_needed=context_needed,
            context_tokens=context_tokens,
            complexity=complexity,
        )
        return actual

    def _best_kimi(
        self,
        complexity: str,
        context_needed: int = 4096,
        *,
        context_tokens: int | None = None,
        prefer_reasoning: bool = False,
        preflight=None,  # PreflightResult | None — avoid circular import with string annotation
        speed_mode: str = "balanced",
    ) -> str:
        if preflight is not None:
            from mullet.core.router.routing_matrix import select_kimi_model
            model_id, reason = select_kimi_model(preflight, speed_mode=speed_mode)
            logger.debug("_best_kimi: routing_matrix selected %s (%s)", model_id, reason)
            # Fall through to existing lookup to get the ProviderModel object
            # Find the model by ID in the candidate list
            candidates = self._kimi_candidates()
            for cand in candidates:
                if cand.id == model_id:
                    return cand.id
            # If not found in candidates (model not available from API), fall through to ladder
            logger.debug("_best_kimi: routing_matrix model %s not in candidates, using ladder", model_id)

        # Try live models first — filter by context_tokens safety margin.
        live = self._cache.get("kimi", [])
        chosen: str | None = None
        # Static fallback ladder — ordered by capability.  Hoisted so it is
        # always defined before both the value-picker and tier-rank blocks.
        candidates = [
            # (model_id, context_length, min_complexity)
            # kimi-k2.6: most capable, most expensive — ADVANCED/CRITICAL
            ("kimi-k2.7-code",        262_144, "ADVANCED"),
            # kimi-k2.5: replaces deprecated 0711/0905-preview (discontinued 2026-05-25)
            ("kimi-k2.5",             262_144, "LOW"),
        ]
        if live:
            result = self._pick_from_live(
                live, complexity, context_needed, context_tokens=context_tokens,
            )
            if result:
                chosen = result
        if not chosen:
            # routing-v3 Phase 3b (Kimi only, cautious single-provider pass): try
            # the value-based picker (cost x complexity x quality across the full
            # candidate set) before falling back to the static tier-ladder logic
            # below. Any exception or None result falls through unchanged to the
            # existing static logic -- this must never break routing, only
            # improve it. Static fallback only -- the live-cache path above is
            # untouched. Domain is always "coding" here since a reasoning
            # request would already have been routed via the preflight branch
            # at the top of this method.
            try:
                from mullet.core.providers.value_picker import pick_by_value
                value_candidates = KIMI_VALUE_PICK_CANDIDATES
                domain = "coding"
                value_pick = pick_by_value(
                    value_candidates, complexity, context_needed, domain,
                    context_tokens=context_tokens, prefer_reasoning=False,
                )
                if value_pick is not None:
                    logger.debug(
                        "_best_kimi: value picker selected %s (domain=%s, tier=%s)",
                        value_pick, domain, complexity,
                    )
                    chosen = value_pick
            except Exception as exc:
                logger.warning(
                    "_best_kimi: value picker failed (%s) -- falling back to static ladder",
                    exc,
                )

        if not chosen:
            tier_rank = {"LOW": 0, "STANDARD": 1, "ADVANCED": 2, "CRITICAL": 3}
            req_rank = tier_rank.get(complexity, 1)
            # Tier filter first: only keep candidates whose min_complexity is <= requested.
            tier_fitting = [
                (mid, ctx) for mid, ctx, min_c in candidates
                if ctx >= context_needed and tier_rank.get(min_c, 0) <= req_rank
            ]
            # Then apply the shared context-safety filter (10% margin via CONTEXT_SAFETY_MARGIN).
            fitting = _filter_candidates_by_context(tier_fitting, context_tokens)
            if not fitting:
                # No candidate fits the requested window — surface this loudly
                max_ctx = max((c for _, c, _ in candidates), default=0)
                effective_tokens = int(context_tokens or context_needed)
                # exhausted=True when NO candidate in the provider's full table
                # can serve even after removing tier caps. If tier-filtering
                # left zero fitters but the raw table has a larger candidate
                # that could serve, the caller can still escalate tier — hence
                # exhausted=False in that narrower case.
                raw_fits = _filter_candidates_by_context(
                    [(m, c) for m, c, _ in candidates], effective_tokens,
                )
                raise ModelContextExceeded(
                    provider="kimi",
                    tier=complexity,
                    context_tokens=effective_tokens,
                    tried_candidates=[m for m, _, _ in candidates],
                    max_candidate_context=max_ctx or None,
                    all_candidates_exhausted=not raw_fits,
                )
            # For LOW tier: pick cheapest by output cost, excluding thinking models
            # (thinking is only used via prefer_reasoning flag)
            if complexity in ("ADVANCED", "CRITICAL"):
                chosen = fitting[0][0]  # most capable first
            elif complexity == "LOW":
                non_thinking = [(m, c) for m, c in fitting if "thinking" not in m]
                pool = non_thinking or fitting
                chosen = min(pool, key=lambda x: KNOWN_PRICING.get(x[0], (0, 999))[1])[0]
            else:  # STANDARD
                chosen = min(fitting, key=lambda x: KNOWN_PRICING.get(x[0], (0, 999))[1])[0]

        # Reasoning override: bump to thinking for non-trivial tasks
        if (
            prefer_reasoning
            and complexity not in ("LOW",)
            and chosen
            and "thinking" not in chosen
        ):
            chosen = "kimi-k2.7-code"  # kimi-k2-thinking discontinued 2026-05-25; K2.7 = fewer reasoning tokens

        # Log if we moved off the preferred default
        if chosen:
            self._maybe_log_downgrade("kimi", complexity, chosen, context_tokens)
        return chosen or ""

    def _best_gemini(
        self,
        complexity: str,
        context_needed: int = 4096,
        *,
        context_tokens: int | None = None,
        preflight=None,  # PreflightResult | None
        speed_mode: str = "balanced",
    ) -> str:
        if preflight is not None:
            from mullet.core.router.routing_matrix import select_gemini_model
            model_id, reason = select_gemini_model(preflight, speed_mode=speed_mode)
            logger.debug("_best_gemini: routing_matrix selected %s (%s)", model_id, reason)
            candidates = self._gemini_candidates()
            for cand in candidates:
                if cand.id == model_id:
                    return cand.id
            logger.debug("_best_gemini: routing_matrix model %s not in candidates, using ladder", model_id)

        # Try live models first — filter by context_tokens safety margin.
        live = self._cache.get("gemini", [])
        if live:
            result = self._pick_from_live(
                live, complexity, context_needed, context_tokens=context_tokens,
            )
            if result:
                self._maybe_log_downgrade("gemini", complexity, result, context_tokens)
                return result
        # Static fallback — all current Gemini models expose 1M context.
        # Still apply the shared filter so a 1.5M-token prompt surfaces as
        # ValueError rather than silently picking gemini-2.5-flash.
        candidates: list[tuple[str, int]] = [
            ("gemini-3.1-pro",        2_000_000),
            ("gemini-2.5-pro",        1_000_000),
            ("gemini-3.5-flash",      1_048_576),
            ("gemini-2.5-flash",      1_000_000),
            ("gemini-2.5-flash-lite", 1_000_000),
        ]
        # routing-v3 Phase 3 (Gemini only, cautious single-provider pass): try the
        # value-based picker (cost x complexity x quality across the full candidate
        # set) before falling back to the static preferred_map ladder below. Any
        # exception or None result falls through unchanged to the existing static
        # logic -- this must never break routing, only improve it.
        try:
            from mullet.core.providers.value_picker import pick_by_value
            needs_reasoning = bool(preflight is not None and preflight.needs_reasoning)
            domain = "reasoning" if needs_reasoning else "coding"
            value_pick = pick_by_value(
                candidates, complexity, context_needed, domain,
                context_tokens=context_tokens, prefer_reasoning=needs_reasoning,
            )
            if value_pick is not None:
                logger.debug(
                    "_best_gemini: value picker selected %s (domain=%s, tier=%s)",
                    value_pick, domain, complexity,
                )
                return value_pick
        except Exception as exc:
            logger.warning(
                "_best_gemini: value picker failed (%s) -- falling back to static ladder",
                exc,
            )
        preferred_map = {
            "LOW": "gemini-2.5-flash",
            "STANDARD": "gemini-2.5-flash",
            "ADVANCED": "gemini-3.1-pro",
            "CRITICAL": "gemini-3.1-pro",
        }
        preferred = preferred_map.get(complexity, "gemini-2.5-flash")
        chosen = self._first_fitting(
            provider="gemini",
            preferred=preferred,
            candidates=candidates,
            context_needed=context_needed,
            context_tokens=context_tokens,
            complexity=complexity,
        )
        return chosen

    def _best_openrouter(
        self,
        complexity: str,
        context_needed: int = 4096,
        *,
        context_tokens: int | None = None,
        prefer_reasoning: bool = False,
        preflight=None,  # PreflightResult | None — avoid circular import with string annotation
        speed_mode: str = "balanced",
    ) -> str:
        """Pick the best OpenRouter model for the given complexity."""
        if preflight is not None:
            from mullet.core.router.routing_matrix import select_openrouter_model
            model_id, reason = select_openrouter_model(preflight, speed_mode=speed_mode)
            logger.debug("_best_openrouter: routing_matrix selected %s (%s)", model_id, reason)
            candidates = self._openrouter_candidates()
            for cand in candidates:
                if cand.id == model_id:
                    return cand.id
            logger.debug("_best_openrouter: routing_matrix model %s not in candidates, using ladder", model_id)

        if complexity == "LOW" and not prefer_reasoning:
            try:
                from mullet.core.providers.model_availability import without_unavailable
                from mullet.core.providers.value_picker import pick_by_value
                free_pick = pick_by_value(
                    without_unavailable("openrouter", OPENROUTER_FREE_TIER_CANDIDATES),
                    complexity, context_needed, "coding",
                    context_tokens=context_tokens, prefer_reasoning=False,
                )
                if free_pick is not None:
                    logger.debug("_best_openrouter: free-tier pick %s for LOW complexity", free_pick)
                    return free_pick
            except Exception as exc:
                logger.warning(
                    "_best_openrouter: free-tier pick failed (%s) -- falling back to paid ladder",
                    exc,
                )

        if prefer_reasoning or complexity == "CRITICAL":
            preferred = "deepseek/deepseek-v4-pro"
        else:
            # routing-v3 Phase 3a (OpenRouter): try the value-based picker (cost x
            # complexity x quality across the full candidate set) before falling
            # back to the static ladder below. Any exception or None result falls
            # through unchanged to the existing static logic -- this must never
            # break routing, only improve it. The reasoning/CRITICAL override
            # above always wins first, so domain here is always "coding" and
            # prefer_reasoning is always False.
            value_pick = None
            try:
                from mullet.core.providers.model_availability import without_unavailable
                from mullet.core.providers.value_picker import pick_by_value
                candidates = without_unavailable("openrouter", OPENROUTER_VALUE_PICK_CANDIDATES)
                value_pick = pick_by_value(
                    candidates, complexity, context_needed, "coding",
                    context_tokens=context_tokens, prefer_reasoning=False,
                )
                if value_pick is not None:
                    logger.debug(
                        "_best_openrouter: value picker selected %s (domain=coding, tier=%s)",
                        value_pick, complexity,
                    )
            except Exception as exc:
                logger.warning(
                    "_best_openrouter: value picker failed (%s) -- falling back to static ladder",
                    exc,
                )
                value_pick = None

            if value_pick is not None:
                preferred = value_pick
            elif complexity == "ADVANCED":
                preferred = "deepseek/deepseek-v4-pro"  # ~3x cheaper than qwen3.7-max; benchmark edge didn't justify the cost
            else:
                preferred = "deepseek/deepseek-v4-flash"

        live = self._cache.get("openrouter", [])
        if live:
            live_hit = any(
                m.id == preferred and m.context_length >= context_needed and m.is_available
                for m in live
            )
            if live_hit:
                return preferred
            # Preferred pick is missing/renamed/deprecated upstream — fall back to
            # the live catalog's best tier-fit instead of returning a dead model id.
            fallback = self._pick_from_live(live, complexity, context_needed, context_tokens=context_tokens)
            if fallback:
                logger.warning(
                    "openrouter preferred model %r not found in live catalog (deprecated/renamed?); falling back to %r",
                    preferred, fallback,
                )
                return fallback
        return preferred

    def _best_local(
        self,
        preflight=None,  # PreflightResult | None
        speed_mode: str = "balanced",
    ) -> ProviderModel | None:
        """Pick the best available local model using the routing matrix.

        Queries LM Studio for available models, classifies them, and delegates
        to select_local_model(). Returns None if local is unavailable or no
        suitable model found.
        """
        try:
            from mullet.core.providers.local_detection import detect_local_provider
            provider = detect_local_provider()
            if provider is None:
                return None

            # Get available model IDs from LM Studio
            available = []
            if hasattr(provider, 'list_models'):
                available = [m.id if hasattr(m, 'id') else str(m) for m in provider.list_models()]
            elif hasattr(provider, 'model'):
                available = [provider.model]

            if not available:
                return None

            from mullet.core.router.routing_matrix import select_local_model
            from mullet.core.router.preflight import PreflightResult as _PF

            if preflight is None:
                # No preflight — use a STANDARD default
                from mullet.core.router.complexity import Tier
                preflight = _PF(
                    input_tokens=1000,
                    expected_output_tokens=512,
                    complexity_tier=Tier.STANDARD,
                    needs_reasoning=False,
                    confidence=0.5,
                    analyzer="heuristic",
                    latency_ms=0,
                    cached=False,
                )

            model_id, reason = select_local_model(preflight, available, speed_mode=speed_mode)
            if model_id is None:
                logger.debug("_best_local: select_local_model returned None: %s", reason)
                return None

            logger.debug("_best_local: selected %s (%s)", model_id, reason)
            return ProviderModel(
                id=model_id,
                provider="local",
                display_name=model_id,
                context_length=131072,
                input_cost_per_1k=0.0,
                output_cost_per_1k=0.0,
            )
        except Exception as exc:
            logger.debug("_best_local: failed (%s)", exc)
            return None

    def _first_fitting(
        self,
        *,
        provider: str,
        preferred: str,
        candidates: list[tuple[str, int]],
        context_needed: int,
        context_tokens: int | None,
        complexity: str,
    ) -> str:
        """Return the first candidate whose max_context covers the request.

        Prefers ``preferred`` when it fits, otherwise picks the first larger
        candidate in iteration order. Uses the shared
        ``_filter_candidates_by_context`` helper so the 10% safety-margin
        filter is applied consistently with ``_best_kimi`` and
        ``_best_gemini``. Raises ValueError when no candidate can serve the
        requested context — callers must not silently fall back.
        """
        # Step 1: hard floor from context_needed (legacy contract).
        ctx_fitting = [(n, c) for n, c in candidates if c >= context_needed]
        # Step 2: apply 10% safety margin when caller provided a token estimate.
        fitting = _filter_candidates_by_context(ctx_fitting, context_tokens)
        fitting_ids = {n for n, _ in fitting}

        if preferred in fitting_ids:
            self._maybe_log_downgrade(provider, complexity, preferred, context_tokens)
            return preferred
        for name, _ in fitting:
            if name != preferred:
                logger.info(
                    "model_registry: %s tier=%s downgrade %s → %s "
                    "(reason: context_needed=%d context_tokens=%s exceeds preferred max_context)",
                    provider, complexity, preferred, name,
                    context_needed, context_tokens,
                )
            return name
        max_ctx = max((c for _, c in candidates), default=0)
        effective_tokens = int(context_tokens or context_needed)
        # Every candidate in the full table has ctx < effective * margin →
        # exhausted. If ctx_fitting was non-empty but the safety margin
        # knocked them out, a bigger sibling could still serve with a
        # loosened margin — mark non-exhausted so the caller knows
        # escalation is theoretically possible.
        raw_fits = _filter_candidates_by_context(candidates, effective_tokens)
        raise ModelContextExceeded(
            provider=provider,
            tier=complexity,
            context_tokens=effective_tokens,
            tried_candidates=[m for m, _ in candidates],
            max_candidate_context=max_ctx or None,
            all_candidates_exhausted=not raw_fits,
        )

    def _maybe_log_downgrade(
        self,
        provider: str,
        complexity: str,
        chosen: str,
        context_tokens: int | None,
    ) -> None:
        """Emit an INFO log when the chosen model isn't the provider's preferred default."""
        preferred = PREFERRED_DEFAULTS.get(provider)
        if not preferred or chosen == preferred:
            return
        reason = (
            f"context_tokens={context_tokens}" if context_tokens
            else f"tier={complexity}"
        )
        logger.info(
            "model_registry: %s tier=%s downgrade %s → %s (reason: %s)",
            provider, complexity, preferred, chosen, reason,
        )

    def _pick_from_live(
        self,
        models: list[ProviderModel],
        complexity: str,
        context_needed: int,
        *,
        context_tokens: int | None = None,
    ) -> str | None:
        """Pick the best live model for given complexity + context. Returns None if no match.

        When ``context_tokens`` is provided, the 10% safety margin via
        ``_filter_candidates_by_context`` is applied so models whose
        ``context_length`` is too small to hold the real prompt are excluded.
        """
        fitting = [m for m in models if m.context_length >= context_needed and m.is_available]
        if context_tokens:
            ctx_pairs = _filter_candidates_by_context(
                [(m.id, m.context_length) for m in fitting],
                context_tokens,
            )
            allowed_ids = {mid for mid, _ in ctx_pairs}
            fitting = [m for m in fitting if m.id in allowed_ids]
        if not fitting:
            return None
        target_tier = {
            "LOW": "fast",
            "STANDARD": "standard",
            "ADVANCED": "powerful",
            "CRITICAL": "powerful",
        }.get(complexity, "standard")
        tier_models = [m for m in fitting if m.tier == target_tier]
        pool = tier_models or fitting
        if complexity == "LOW":
            return min(pool, key=lambda m: m.input_cost_per_1k).id
        return min(pool, key=lambda m: m.output_cost_per_1k).id


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: ModelRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> ModelRegistry:
    """Return the module-level ModelRegistry singleton, loading cache on first call."""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = ModelRegistry()
            _registry._load_cache()
    return _registry
