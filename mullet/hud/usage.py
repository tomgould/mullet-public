"""Claude usage limits, as Claude Code itself reports them.

Claude Code sends its current rate limits to the statusline command on every
tick. ``mullet-hud`` stores them with :func:`record_usage`, and the budget
guard, router and swarm read them back with :func:`fetch_usage`. Mullet never
reads Claude's credentials or calls Anthropic's usage API itself.
"""
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RateLimits:
    """Parsed Anthropic usage limits — 5h + weekly percentages, reset times, Sonnet sub-limit."""

    five_hour_pct: float = 0.0
    five_hour_resets: str = ""
    weekly_pct: float = 0.0
    weekly_resets: str = ""
    sonnet_weekly_pct: float = 0.0  # Sonnet-only sub-limit if available
    sonnet_weekly_resets: str = ""  # Sonnet reset time if available
    has_sonnet_limit: bool = False  # True when seven_day_sonnet is present (Max ×5)


def parse_resets_at(iso: str) -> Optional[datetime]:
    """Parse an Anthropic-shaped ``resets_at`` ISO-8601 string into a ``datetime``.

    Accepts either a trailing ``Z`` or an explicit ``+HH:MM`` offset. Returns
    ``None`` for empty input or malformed timestamps so callers can treat the
    result as a simple optional.
    """
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


# Figures older than this are too stale to act on: no Claude Code session has
# drawn its statusline for a while.
_MAX_AGE = 900
# Tests point this at a temporary file; None means usage-cache.json in the
# cache directory.
_CACHE_FILE: Optional[Path] = None


def _cache_file() -> Path:
    from mullet.core.paths import cache_dir
    return _CACHE_FILE or cache_dir() / "usage-cache.json"


def _read_cache() -> Optional[tuple[float, RateLimits]]:
    try:
        cache_file = _cache_file()
        if not cache_file.exists():
            return None
        data = json.loads(cache_file.read_text())
        return (data["ts"], RateLimits(**data["limits"]))
    except Exception as e:
        logging.getLogger(__name__).debug("Cache read failed: %s", e)
        return None


def record_usage(limits: RateLimits) -> None:
    """Store the limits Claude Code just reported, for :func:`fetch_usage`.

    Several Claude Code sessions can draw their statuslines at once, so the
    file is replaced in one step rather than rewritten in place.
    """
    try:
        cache_file = _cache_file()
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=cache_file.parent, prefix=".usage-")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump({"ts": time.time(), "limits": asdict(limits)}, fh)
            os.replace(tmp, cache_file)
        except BaseException:
            os.unlink(tmp)
            raise
    except OSError as _exc:
        logger.warning("usage: cache write failed: %s", _exc)


_STREAM_WINDOWS = {"five_hour": ("five_hour_pct", "five_hour_resets"), "seven_day": ("weekly_pct", "weekly_resets")}


def record_stream_limits(info: object) -> bool:
    """Store the limits from a ``claude -p`` stream's ``rate_limit_event``, as the statusline would.

    ``info`` is the event's ``rate_limit_info``: ``unifiedWindows`` maps
    ``five_hour`` and ``seven_day`` to ``utilization`` (0 to 1) and
    ``resetsAt`` (epoch seconds). An event without the weekly window is
    ignored: storing it would stamp an old or missing weekly figure as
    current, and the ladder's budget level is read from that figure.
    Returns True when something was stored.
    """
    windows = info.get("unifiedWindows") if isinstance(info, dict) else None
    if not isinstance(windows, dict) or not isinstance(windows.get("seven_day"), dict):
        return False
    previous = _read_cache()
    limits = previous[1] if previous else RateLimits()
    found = False
    for name, (pct_field, reset_field) in _STREAM_WINDOWS.items():
        window = windows.get(name)
        if not isinstance(window, dict):
            continue
        try:
            utilization = float(window.get("utilization"))
        except (TypeError, ValueError):
            continue
        if utilization != utilization:  # NaN
            continue
        pct = utilization * 100 if utilization <= 1.0 else utilization
        setattr(limits, pct_field, max(0.0, min(100.0, pct)))
        try:
            resets = float(window.get("resetsAt"))
            setattr(limits, reset_field, datetime.fromtimestamp(resets, tz=timezone.utc).isoformat())
        except (TypeError, ValueError, OverflowError, OSError):
            pass
        found = True
    if found:
        record_usage(limits)
    return found


def fetch_usage() -> Optional[RateLimits]:
    """Return the limits Claude Code last reported, or ``None`` when there are
    none from the last 15 minutes.

    A window whose reset time has passed since then reads as 0%, so a figure
    taken just before a reset doesn't hold work back after it.
    """
    cached = _read_cache()
    now = time.time()
    if cached is None or now - cached[0] > _MAX_AGE:
        return None
    limits = cached[1]
    for pct, resets in (
        ("five_hour_pct", "five_hour_resets"),
        ("weekly_pct", "weekly_resets"),
        ("sonnet_weekly_pct", "sonnet_weekly_resets"),
    ):
        reset_at = parse_resets_at(getattr(limits, resets))
        if reset_at is not None and reset_at.timestamp() <= now:
            setattr(limits, pct, 0.0)
    return limits
