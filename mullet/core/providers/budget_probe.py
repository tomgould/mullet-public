"""Multi-provider budget probe — async parallel balance/quota queries for all providers.

Supports: OpenRouter (real $ balance via API), Kimi/Moonshot (real $ balance via API),
Anthropic/Claude (subscription — uses HUD token pressure), Gemini (subscription — rate-limit based),
Local LLMs (free, always available).

Results are cached in-process for 5 minutes (CACHE_TTL_SECONDS).
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from mullet.core.env import load_dotenv, local_llm_url

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 300  # 5-minute probe cache
_LOW_BALANCE_THRESHOLD_USD = 0.50  # warn when remaining < $0.50


@dataclass
class ProviderBalance:
    """Balance/quota snapshot for a single provider."""

    name: str                               # display name e.g. "OpenRouter"
    key: str                                # internal key e.g. "openrouter"
    credits_remaining_usd: float | None     # None = subscription/unknown
    credits_used_usd: float | None          # None = unknown
    credits_total_usd: float | None         # None = unlimited/subscription
    is_subscription: bool = False           # Claude Max, Gemini Pro
    is_free: bool = False                   # Local LLM
    billing_model: str = "api"              # "free" | "subscription_cli" | "api"
    subscription_pressure: float | None = None  # 0.0–1.0 from HUD guard (Anthropic only)
    rate_limit_rpm: int | None = None       # requests/minute if known
    window_label: str = ""                  # human-readable time-window status e.g. "5h: 14% | weekly: 25%"
    window_resets_at: str = ""              # ISO timestamp of nearest window reset
    extra: dict[str, Any] = field(default_factory=dict)  # provider-specific extras
    error: str | None = None
    probed_at: float = field(default_factory=time.time)

    @property
    def has_budget(self) -> bool:
        """True if the provider has usable capacity."""
        if self.error:
            # Missing API key = definitively unavailable; other errors fail open
            if "not set" in self.error or "not found" in self.error:
                return False
            return True  # transient errors don't block routing
        if self.is_free:
            return True
        if self.is_subscription:
            pressure = self.subscription_pressure or 0.0
            # Also block if either raw window is at/above 90% regardless of pacing
            _5h = self.extra.get("five_hour_pct") if self.extra else None
            _wk = self.extra.get("weekly_pct") if self.extra else None
            if _5h is not None and _5h >= 90.0:
                return False
            if _wk is not None and _wk >= 90.0:
                return False
            return pressure < 0.90
        if self.credits_remaining_usd is None:
            return True  # unknown = assume available
        return self.credits_remaining_usd > 0.001

    @property
    def priority_score(self) -> float:
        """0.0–1.0 routing priority. Higher = prefer this provider.

        Ordering philosophy:
        - subscription_cli at low pressure: 0.85–1.0  (zero cost, prefer first)
        - api with large balance: 0.65–0.85           (pay but budget available)
        - free/local: 0.5                             (free but slower)
        - subscription_cli at high pressure: 0.2–0.5  (near limit, deprioritise)
        - api with low balance: 0.0–0.3               (almost broke)
        - no budget: 0.0
        """
        if not self.has_budget:
            return 0.0
        if self.billing_model == "free":
            return 0.5
        if self.billing_model == "subscription_cli":
            pressure = self.subscription_pressure or 0.0
            # Low pressure = near 1.0, high pressure = drops off sharply
            return max(0.1, 1.0 - (pressure ** 0.7))
        # billing_model == "api"
        if self.credits_remaining_usd is not None:
            import math
            if self.credits_remaining_usd <= 0:  # pragma: no cover
                return 0.0
            # $200 → 0.85, $50 → 0.75, $5 → 0.55, $0.01 → 0.15
            return min(0.85, 0.5 + 0.35 * math.log1p(self.credits_remaining_usd) / math.log1p(200))
        return 0.65  # API with unknown balance — assume it's available but don't over-prioritise

    @property
    def is_low(self) -> bool:
        """True when balance is below the low-balance threshold."""
        if self.credits_remaining_usd is None:
            return False
        return self.credits_remaining_usd < _LOW_BALANCE_THRESHOLD_USD

    def summary_line(self) -> str:
        """One-line status suitable for display in tables."""
        prefix = {"free": "⚡ ", "subscription_cli": "∞ ", "api": "$ "}.get(self.billing_model, "")
        if self.error:
            return f"{prefix}⚠  error: {self.error[:60]}"
        if self.is_free:
            return f"{prefix}free (local GPU) · no limits"
        if self.is_subscription:
            if self.window_label:
                return f"{prefix}{self.window_label}"
            if self.subscription_pressure is not None:
                pct = int(self.subscription_pressure * 100)
                return f"{prefix}subscription — {pct}% window used"
            return f"{prefix}subscription active · window unknown"
        if self.credits_remaining_usd is not None:
            remaining = f"${self.credits_remaining_usd:.2f}"
            total = f" / ${self.credits_total_usd:.2f}" if self.credits_total_usd else ""
            flag = " ⚠ LOW" if self.is_low else ""
            window = f" · {self.window_label}" if self.window_label else ""
            return f"{prefix}{remaining}{total} remaining{flag}{window}"
        return f"{prefix}balance unknown"


# ---------------------------------------------------------------------------
# In-process cache
# ---------------------------------------------------------------------------

class _BudgetCache:
    def __init__(self) -> None:
        self._data: dict[str, ProviderBalance] = {}
        self._ts: dict[str, float] = {}

    def get(self, key: str) -> ProviderBalance | None:
        """Return the cached balance for ``key`` if still within the TTL window."""
        ts = self._ts.get(key)
        if ts and (time.time() - ts) < CACHE_TTL_SECONDS:
            return self._data.get(key)
        return None

    def set(self, balance: ProviderBalance) -> None:
        """Insert or refresh ``balance`` in the cache, stamping the current time."""
        self._data[balance.key] = balance
        self._ts[balance.key] = time.time()

    def invalidate(self, key: str | None = None) -> None:
        """Drop the cached entry for ``key`` (or the whole cache when key is None)."""
        if key:
            self._data.pop(key, None)
            self._ts.pop(key, None)
        else:
            self._data.clear()
            self._ts.clear()


_cache = _BudgetCache()


load_dotenv()  # ensure keys are available before any probe runs


# ---------------------------------------------------------------------------
# Provider probes
# ---------------------------------------------------------------------------

async def probe_openrouter(api_key: str | None = None, timeout: float = 8.0) -> ProviderBalance:
    """Query OpenRouter /api/v1/auth/key for real $ balance."""
    key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        return ProviderBalance(
            name="OpenRouter", key="openrouter",
            credits_remaining_usd=None, credits_used_usd=None, credits_total_usd=None,
            error="OPENROUTER_API_KEY not set",
        )
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {key}"},
            )
            r.raise_for_status()
            data = r.json().get("data", {})
        # OpenRouter usage/limit are in USD (float)
        usage = data.get("usage")   # USD spent
        limit = data.get("limit")   # USD limit (None = unlimited)
        is_free = data.get("is_free_tier", False)
        rate_info = data.get("rate_limit", {})
        rpm = rate_info.get("requests") if isinstance(rate_info, dict) else None

        remaining = (limit - usage) if (limit is not None and usage is not None) else None

        # Build tier-accurate window_label from label + RPM
        _or_label = data.get("label", "") or ("Free" if is_free else "")
        _tier_str = f"{_or_label} tier" if _or_label else "API"
        _rpm_str = f" · {rpm} RPM" if rpm else ""
        _window = f"{_tier_str}{_rpm_str}"

        return ProviderBalance(
            name="OpenRouter", key="openrouter",
            credits_remaining_usd=remaining,
            credits_used_usd=usage,
            credits_total_usd=limit,
            is_free=is_free,
            billing_model="api",
            rate_limit_rpm=rpm,
            window_label=_window,
            extra={"label": data.get("label", ""), "is_free_tier": is_free},
        )
    except Exception as exc:
        logger.debug("OpenRouter probe failed: %s", exc)
        return ProviderBalance(
            name="OpenRouter", key="openrouter",
            credits_remaining_usd=None, credits_used_usd=None, credits_total_usd=None,
            billing_model="api",
            error=str(exc)[:80],
        )


async def probe_kimi(api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0) -> ProviderBalance:
    """Query Moonshot AI /v1/users/me/balance for real $ balance."""
    key = api_key or os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY", "")
    if not key:
        return ProviderBalance(
            name="Kimi K2 (Moonshot)", key="kimi",
            credits_remaining_usd=None, credits_used_usd=None, credits_total_usd=None,
            billing_model="api",
            error="KIMI_API_KEY not set",
        )
    url_base = base_url or os.environ.get("KIMI_BASE_URL") or "https://api.moonshot.ai/v1"
    # Moonshot balance endpoint (international)
    balance_url = url_base.rstrip("/") + "/users/me/balance"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(
                balance_url,
                headers={"Authorization": f"Bearer {key}"},
            )
            r.raise_for_status()
            data = r.json()
        # Moonshot international API (api.moonshot.ai) returns balances in USD.
        # The currency field may be absent; default to USD (not CNY).
        # Response shape: {"data": {"available_balance": 171.67, ...}}
        balance_data = data.get("data") or data.get("balance") or data
        avail = balance_data.get("available_balance") or balance_data.get("remaining_balance")
        currency = balance_data.get("currency", "USD")
        if avail is not None:
            avail_f = float(avail)
            # Only convert if the API explicitly declares CNY
            if currency == "CNY":
                avail_f = avail_f / 7.25  # approx CNY→USD
            return ProviderBalance(
                name="Kimi K2 (Moonshot)", key="kimi",
                credits_remaining_usd=avail_f,
                credits_used_usd=None,
                credits_total_usd=None,
                billing_model="api",
                extra={"currency": currency, "raw": str(avail)},
            )
        # If shape unknown, return with unknown balance but no error
        return ProviderBalance(
            name="Kimi K2 (Moonshot)", key="kimi",
            credits_remaining_usd=None, credits_used_usd=None, credits_total_usd=None,
            billing_model="api",
            extra={"raw_response": str(data)[:200]},
        )
    except Exception as exc:
        logger.debug("Kimi probe failed: %s", exc)
        return ProviderBalance(
            name="Kimi K2 (Moonshot)", key="kimi",
            credits_remaining_usd=None, credits_used_usd=None, credits_total_usd=None,
            billing_model="api",
            error=str(exc)[:80],
        )


async def probe_anthropic() -> ProviderBalance:
    """Anthropic Max subscription — no $ API. Use HUD guard token pressure."""
    billing = "subscription_cli"  # default assumption for Anthropic
    try:
        from mullet.hud.guard import check_budget
        result = check_budget()
        pressure = result.pressure
        _5h = result.five_hour_pct
        _wk = result.weekly_pct
        _reset = result.reset_time or ""
        # Format reset time as HH:MM UTC for display
        _reset_display = ""
        if _reset:
            try:
                from datetime import datetime, timezone
                dt = datetime.fromisoformat(_reset).astimezone(timezone.utc)
                _reset_display = dt.strftime("%H:%M UTC")
            except Exception:
                _reset_display = _reset[:16]
        # Detect plan tier from raw usage data
        _is_max5 = False
        _sonnet_pct: float | None = None
        try:
            from mullet.hud.usage import fetch_usage
            _raw_limits = fetch_usage()
            if _raw_limits is not None:
                _is_max5 = _raw_limits.has_sonnet_limit
                if _raw_limits.sonnet_weekly_pct:
                    _sonnet_pct = _raw_limits.sonnet_weekly_pct
        except Exception:
            pass
        tier_name = "Claude Max ×5" if _is_max5 else "Claude Max ×1"
        _window_label = f"5h: {_5h:.0f}% | weekly: {_wk:.0f}%"
        if _sonnet_pct is not None:
            _window_label += f" | sonnet: {_sonnet_pct:.0f}%"
        if _reset_display:
            _window_label += f" | resets {_reset_display}"
        return ProviderBalance(
            name=f"Anthropic ({tier_name})", key="anthropic",
            credits_remaining_usd=None,
            credits_used_usd=None,
            credits_total_usd=None,
            is_subscription=True,
            billing_model=billing,
            subscription_pressure=pressure,
            window_label=_window_label,
            window_resets_at=_reset,
            extra={
                "five_hour_pct": _5h,
                "weekly_pct": _wk,
                "status": result.status.value,
            },
        )
    except Exception as exc:
        tier_name = "Claude Max ×5"  # assume ×5 on error
        return ProviderBalance(
            name=f"Anthropic ({tier_name})", key="anthropic",
            credits_remaining_usd=None, credits_used_usd=None, credits_total_usd=None,
            is_subscription=True,
            billing_model=billing,
            error=str(exc)[:80],
        )


_GEMINI_TIER_LIMITS = {
    "free":  ("Free tier",                 "1000 RPD / 60 RPM"),
    "pro":   ("Google AI Pro/Standard",    "1500 RPD"),
    "ultra": ("Google AI Ultra",           "2000 RPD"),
}


def _detect_gemini_tier() -> str:
    """Best-effort tier detection — env var override → heuristic → default.

    Detection precedence:
      1. ``GEMINI_TIER`` env var (case-insensitive: free / pro / ultra)
      2. ``MULLET_GEMINI_TIER`` env var (mullet-prefixed alias)
      3. Heuristic — Vertex AI project + GOOGLE_GENAI_USE_VERTEXAI=1 implies
         a paid project; reported as "pro" unless overridden.
      4. Default: "free"

    No API endpoint exists to query the active tier, so heuristics are the
    best we can do. Users on Pro/Ultra should set GEMINI_TIER explicitly
    for accurate budget display.
    """
    raw = (
        os.environ.get("GEMINI_TIER")
        or os.environ.get("MULLET_GEMINI_TIER")
        or ""
    ).strip().lower()
    if raw in _GEMINI_TIER_LIMITS:
        return raw
    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") == "1" and (
        os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("VERTEX_PROJECT_ID")
    ):
        return "pro"
    return "free"


async def probe_gemini(api_key: str | None = None) -> ProviderBalance:
    """Gemini API — no balance API; report the detected tier."""
    billing = "api"
    key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")

    # No API endpoint reports remaining daily requests, so window usage is unknown at runtime.
    tier = _detect_gemini_tier()
    tier_name, _tier_rpd = _GEMINI_TIER_LIMITS.get(tier, _GEMINI_TIER_LIMITS["free"])
    _window = f"{tier_name} · window: unknown"
    if not key:
        return ProviderBalance(
            name=f"Gemini (Google {tier_name})", key="gemini",
            credits_remaining_usd=None, credits_used_usd=None, credits_total_usd=None,
            is_subscription=True,
            billing_model=billing,
            window_label=_window,
            error="GOOGLE_API_KEY / GEMINI_API_KEY not set",
            extra={"tier": tier},
        )
    return ProviderBalance(
        name=f"Gemini (Google {tier_name})", key="gemini",
        credits_remaining_usd=None, credits_used_usd=None, credits_total_usd=None,
        is_subscription=True,
        billing_model=billing,
        window_label=_window,
        extra={"tier": tier},
    )


async def probe_local(base_url: str | None = None) -> ProviderBalance:
    """Local LLM (LM Studio / Ollama) — free, check if server is reachable.

    ``base_url`` is the OpenAI-compatible root ending in ``/v1``; it defaults
    to ``LOCAL_LLM_URL``.
    """
    base_url = (base_url or local_llm_url()).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{base_url}/models")
            online = r.status_code == 200
    except Exception:
        online = False
    return ProviderBalance(
        name="Local LLM (GPU)", key="local",
        credits_remaining_usd=None, credits_used_usd=None, credits_total_usd=None,
        is_free=True,
        billing_model="free",
        window_label="unlimited · no rate limits",
        extra={"online": online, "base_url": base_url},
        error=None if online else "server not reachable",
    )


# ---------------------------------------------------------------------------
# Parallel probe runner
# ---------------------------------------------------------------------------

async def probe_all(
    *,
    include: list[str] | None = None,
    force_refresh: bool = False,
) -> dict[str, ProviderBalance]:
    """Probe all configured providers concurrently. Returns {key: ProviderBalance}.

    Results are cached for CACHE_TTL_SECONDS. Pass force_refresh=True to bypass cache.
    Pass include=["kimi", "openrouter"] to probe only specific providers.
    """
    all_keys = ["openrouter", "kimi", "anthropic", "gemini", "local"]
    keys = [k for k in all_keys if include is None or k in include]

    # Check cache first
    results: dict[str, ProviderBalance] = {}
    to_probe: list[str] = []
    for k in keys:
        if not force_refresh:
            cached = _cache.get(k)
            if cached is not None:
                results[k] = cached
                continue
        to_probe.append(k)

    if not to_probe:
        return results

    # Build coroutines for uncached providers
    coros = []
    for k in to_probe:
        if k == "openrouter":
            coros.append(probe_openrouter())
        elif k == "kimi":
            coros.append(probe_kimi())
        elif k == "anthropic":
            coros.append(probe_anthropic())
        elif k == "gemini":
            coros.append(probe_gemini())
        elif k == "local":
            coros.append(probe_local())

    fetched = await asyncio.gather(*coros, return_exceptions=True)
    for k, result in zip(to_probe, fetched):
        if isinstance(result, ProviderBalance):
            _cache.set(result)
            results[k] = result
        else:
            # Exception from gather — create error balance
            err_bal = ProviderBalance(
                name=k, key=k,
                credits_remaining_usd=None, credits_used_usd=None, credits_total_usd=None,
                error=str(result)[:80],
            )
            _cache.set(err_bal)
            results[k] = err_bal

    return results


def get_all_balances(force_refresh: bool = False) -> dict[str, ProviderBalance]:
    """Synchronous wrapper for probe_all(). Runs in a new event loop."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Can't nest event loops — return cached only
            results = {}
            for k in ["openrouter", "kimi", "anthropic", "gemini", "local"]:
                cached = _cache.get(k)
                if cached:
                    results[k] = cached
            return results
    except RuntimeError:
        pass
    return asyncio.run(probe_all(force_refresh=force_refresh))


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def sort_providers_by_budget(provider_keys: list[str]) -> list[str]:
    """Re-order a provider list by available budget (highest priority first).

    Providers with no budget (credits_remaining == 0) are moved to the end.
    Providers not in the probe results retain their original position.
    """
    balances = get_all_balances()
    def _score(k: str) -> float:
        bal = balances.get(k)
        if bal is None:
            return 0.5  # unknown = middle priority
        return bal.priority_score

    return sorted(provider_keys, key=_score, reverse=True)


def get_available_providers(provider_keys: list[str]) -> list[str]:
    """Filter provider list to only those with available budget."""
    balances = get_all_balances()
    available = []
    for k in provider_keys:
        bal = balances.get(k)
        if bal is None or bal.has_budget:
            available.append(k)
    return available


def parallel_provider_pairs(
    tasks: list[str],
    provider_keys: list[str],
) -> list[tuple[str, str]]:
    """Assign N tasks to M providers for parallel execution.

    Returns list of (task, provider_key) pairs. Tasks are spread across
    providers with available budget, up to one task per provider at a time.

    Example: 4 tasks, [kimi, openrouter, local] with all budget available
    → [(task0, kimi), (task1, openrouter), (task2, local), (task3, kimi)]
    """
    available = get_available_providers(provider_keys)
    if not available:
        available = provider_keys  # fallback: use all even if balance unknown

    pairs = []
    for i, task in enumerate(tasks):
        provider = available[i % len(available)]
        pairs.append((task, provider))
    return pairs
