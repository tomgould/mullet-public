"""Parse Claude Code stdin rate_limits + context_window into RateLimits.

Claude Code pipes a JSON payload to the statusline command every tick that
includes the authoritative `rate_limits` and `context_window` data. They match
the values Claude Code itself displays, and they are the only source of usage
figures Mullet has: it never reads Claude's credentials to ask the API.
"""
from datetime import datetime, timezone
from typing import Optional

from mullet.hud.usage import RateLimits


def _clamp_pct(value) -> Optional[float]:
    if not isinstance(value, (int, float)):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return max(0.0, min(100.0, f))


def _unix_to_iso(value) -> str:
    if not isinstance(value, (int, float)) or value <= 0:
        return ""
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except (ValueError, OSError, OverflowError):
        return ""


def parse_stdin_limits(stdin_data: dict) -> Optional[RateLimits]:
    """Return a populated ``RateLimits`` from stdin, or ``None`` when absent.

    ``used_percentage`` values are 0-100 floats (clamped). ``resets_at`` values
    arrive as Unix seconds (number) and are converted to ISO-8601 UTC strings so
    the existing :func:`mullet.hud.usage.parse_resets_at` helper continues to
    work across both the stdin and OAuth code paths.
    """
    if not isinstance(stdin_data, dict):
        return None
    rl = stdin_data.get("rate_limits")
    if not isinstance(rl, dict):
        return None

    five = rl.get("five_hour") if isinstance(rl.get("five_hour"), dict) else {}
    seven = rl.get("seven_day") if isinstance(rl.get("seven_day"), dict) else {}

    five_pct = _clamp_pct(five.get("used_percentage"))
    seven_pct = _clamp_pct(seven.get("used_percentage"))
    if five_pct is None and seven_pct is None:
        return None

    return RateLimits(
        five_hour_pct=five_pct or 0.0,
        five_hour_resets=_unix_to_iso(five.get("resets_at")),
        weekly_pct=seven_pct or 0.0,
        weekly_resets=_unix_to_iso(seven.get("resets_at")),
    )


def _window_present(rl: dict, name: str) -> bool:
    window = rl.get(name)
    return isinstance(window, dict) and _clamp_pct(window.get("used_percentage")) is not None


def fill_missing_windows(
    limits: RateLimits, stdin_data: dict, previous: Optional[RateLimits]
) -> RateLimits:
    """Keep ``previous`` figures for any window this tick left out.

    :func:`parse_stdin_limits` reads a missing window as 0%, which is right for
    display but would wipe the last real figure if recorded as it is.
    """
    if previous is None:
        return limits
    rl = stdin_data.get("rate_limits") if isinstance(stdin_data, dict) else None
    rl = rl if isinstance(rl, dict) else {}
    if not _window_present(rl, "five_hour"):
        limits.five_hour_pct = previous.five_hour_pct
        limits.five_hour_resets = previous.five_hour_resets
    if not _window_present(rl, "seven_day"):
        limits.weekly_pct = previous.weekly_pct
        limits.weekly_resets = previous.weekly_resets
    return limits
