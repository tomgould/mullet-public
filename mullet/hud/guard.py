"""Token Guard — rate limit awareness for autonomous loops.

Checks Claude API usage and returns whether to continue, throttle, or pause.
When paused, sleeps until the 5h window resets (preserving session context).
"""
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import logging
logger = logging.getLogger(__name__)

from mullet.hud.usage import fetch_usage, RateLimits


class BudgetStatus(Enum):
    """Budget status enumeration."""
    GO = "go"            # Full speed, use Opus freely
    THROTTLE = "throttle" # Use Sonnet, delegate more to Gemini
    PAUSE = "pause"       # Sleep until 5h reset


@dataclass
class BudgetCheck:
    """Budget check result dataclass."""
    status: BudgetStatus
    five_hour_pct: float = 0.0
    weekly_pct: float = 0.0
    message: str = ""
    reset_time: Optional[str] = ""
    daily_budget_pct: float = 14.3  # 100/7 days
    pressure: float = 0.0  # 0.0 (no pressure) to 1.0 (max)
    recommended_provider: str = ""  # "local", "kimi", "gemini", "haiku", "sonnet", or "pause"
    session_pressure: float = 0.0   # paced 5h pressure component (0-1)
    weekly_pressure: float = 0.0    # paced 7d pressure component (0-1)


def _paced_pressure(pct: float, elapsed_fraction: float) -> float:
    """Convert a raw utilisation pct + elapsed window fraction into pressure.

    pace = pct / elapsed_fraction. pace == 1.0 means on linear track.
    - pace <= 1.0 (banking headroom): pressure = pct * pace (drops below raw)
    - pace  > 1.0 (over-spending):    pressure = pct * (1 + (pace-1) * 2)
    """
    elapsed_fraction = max(elapsed_fraction, 0.01)
    pace = pct / elapsed_fraction
    if pace <= 1.0:
        return max(0.0, pct * pace)
    return min(pct * (1.0 + (pace - 1.0) * 2.0), 1.0)


def pressure_components(limits: RateLimits) -> tuple[float, float]:
    """Return (session_pressure, weekly_pressure) as 0-1 floats.

    Each component is the paced pressure for its window — accounts for where
    we are in the burn cycle, not just raw utilisation.
    """
    session_pct = limits.five_hour_pct / 100.0
    session_pressure = session_pct
    if limits.five_hour_resets:
        try:
            reset_dt = datetime.fromisoformat(limits.five_hour_resets)
            now = datetime.now(timezone.utc)
            mins_remaining = max((reset_dt - now).total_seconds() / 60, 1)
            elapsed = max((300 - mins_remaining) / 300, 0.01)
            session_pressure = _paced_pressure(session_pct, elapsed)
        except (ValueError, TypeError) as _exc:
            logger.warning("hud_guard: pressure_components (session): %s", _exc)

    weekly_pct = limits.weekly_pct / 100.0
    weekly_pressure = weekly_pct
    if limits.weekly_resets:
        try:
            reset_dt = datetime.fromisoformat(limits.weekly_resets)
            now = datetime.now(timezone.utc)
            days_remaining = max((reset_dt - now).total_seconds() / 86400, 0.01)
            elapsed = max((7.0 - days_remaining) / 7.0, 0.01)
            weekly_pressure = _paced_pressure(weekly_pct, elapsed)
        except (ValueError, TypeError) as _exc:
            logger.warning("hud_guard: pressure_components (weekly): %s", _exc)

    return session_pressure, weekly_pressure


def compute_pressure(limits: RateLimits) -> float:
    """Compute a 0.0–1.0 pressure score using pacing against reset windows."""
    sp, wp = pressure_components(limits)
    return min(max(sp, wp), 1.0)


def recommend_provider(pressure: float, tier: str = "STANDARD") -> str:
    """Map pressure + tier to the optimal provider name.

    Balanced policy: OpenRouter is the default cloud workhorse (pay-per-token).
    When Anthropic pressure <60%, substitute Haiku/Sonnet — Max plan is prepaid
    so using it avoids unnecessary pay-per-token spend on OpenRouter/Kimi.

    Advisory only — explicit --provider / user override always wins.
    """
    if tier == "CRITICAL":
        # Highest complexity: best Anthropic quality when headroom, else openrouter
        if pressure < 0.55:
            return "opus"
        if pressure < 0.80:
            return "sonnet"
        return "openrouter"
    elif tier == "PREMIUM":
        # Opus-first mode (mullet override on). Auto-downgrade under budget pressure.
        if pressure < 0.30:
            return "opus"
        if pressure < 0.55:
            return "sonnet"
        if pressure < 0.80:
            return "haiku"
        return "openrouter"
    elif tier == "ADVANCED":
        # Complex reasoning: sonnet at low pressure, step down
        if pressure < 0.30:
            return "sonnet"
        if pressure < 0.60:
            return "haiku"
        if pressure < 0.80:
            return "openrouter"
        return "kimi"
    elif tier == "STANDARD":
        # Default: substitute Haiku when Anthropic has headroom (saves pay-per-token)
        if pressure < 0.60:
            return "haiku"
        if pressure < 0.80:
            return "openrouter"
        return "kimi"
    else:  # LOW
        return "local"


def check_budget() -> BudgetCheck:
    """Check current usage and return budget status."""
    limits = fetch_usage()
    if not limits:
        # Can't reach API — assume GO but be cautious
        return BudgetCheck(status=BudgetStatus.GO, message="Usage API unavailable, proceeding cautiously")

    # Calculate daily budget from weekly remaining
    weekly_remaining = 100.0 - limits.weekly_pct
    days_left = _days_until_reset(limits.weekly_resets)
    daily_budget = weekly_remaining / max(days_left, 1)

    sp, wp = pressure_components(limits)
    pressure = min(max(sp, wp), 1.0)
    provider = recommend_provider(pressure)

    common = dict(
        five_hour_pct=limits.five_hour_pct,
        weekly_pct=limits.weekly_pct,
        reset_time=limits.five_hour_resets,
        daily_budget_pct=daily_budget,
        pressure=pressure,
        recommended_provider=provider,
        session_pressure=sp,
        weekly_pressure=wp,
    )

    # Hard PAUSE: both limits critically high (runaway safeguard)
    if limits.five_hour_pct >= 95 and limits.weekly_pct >= 90:
        return BudgetCheck(
            status=BudgetStatus.PAUSE,
            message=f"PAUSE — 5h: {limits.five_hour_pct:.0f}%, weekly: {limits.weekly_pct:.0f}%  (both limits critical)",
            **common,
        )

    # Throttle: degrade to cheaper provider rather than stopping
    throttle_threshold = 70.0
    if limits.weekly_pct >= 85:
        throttle_threshold = 55.0  # Earlier throttle signal when weekly is very high
    if limits.five_hour_pct >= throttle_threshold or daily_budget < 8:
        return BudgetCheck(
            status=BudgetStatus.THROTTLE,
            message=f"5h: {limits.five_hour_pct:.0f}%, weekly: {limits.weekly_pct:.0f}%, pressure: {pressure:.2f} → {provider}",
            **common,
        )

    return BudgetCheck(
        status=BudgetStatus.GO,
        message=f"5h: {limits.five_hour_pct:.0f}%, weekly: {limits.weekly_pct:.0f}%, daily budget: {daily_budget:.1f}%, pressure: {pressure:.2f} → {provider}",
        **common,
    )


def wait_for_reset(check: BudgetCheck) -> None:
    """Sleep until the rate limit resets. Preserves session context."""
    if not check.reset_time:
        # No reset time known, sleep 30 minutes as safe default
        print("[Token Guard] No reset time known, sleeping 30 minutes...")
        time.sleep(30 * 60)
        return

    try:
        reset_dt = datetime.fromisoformat(check.reset_time)
        now = datetime.now(timezone.utc)
        wait_seconds = (reset_dt - now).total_seconds()

        if wait_seconds <= 0:
            # Already past reset, re-check
            return

        # Add 60s buffer after reset
        wait_seconds = min(wait_seconds + 60, 6 * 3600)  # Cap at 6 hours safety
        minutes = int(wait_seconds / 60)
        print(f"[Token Guard] Sleeping {minutes}m until reset at {reset_dt.strftime('%H:%M UTC')}...")
        time.sleep(wait_seconds)
    except (ValueError, TypeError):
        print("[Token Guard] Could not parse reset time, sleeping 30 minutes...")
        time.sleep(30 * 60)


def _days_until_reset(reset_iso: str) -> float:
    """Calculate days remaining until weekly reset."""
    if not reset_iso:
        return 3.5  # Assume mid-week as safe default
    try:
        reset_dt = datetime.fromisoformat(reset_iso)
        now = datetime.now(timezone.utc)
        delta = (reset_dt - now).total_seconds() / 86400
        return max(delta, 0.5)  # At least half a day
    except (ValueError, TypeError):
        return 3.5
