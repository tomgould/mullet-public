"""Fetch Claude usage/rate limits from the Anthropic OAuth API."""
import json
import logging
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
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


_CACHE_TTL = 300  # 5 minutes — prevents 429 from Anthropic usage API
_CACHE_FILE = Path.home() / ".claude" / ".mullet-usage-cache.json"


def _get_access_token() -> Optional[str]:
    creds_path = Path.home() / ".claude" / ".credentials.json"
    if not creds_path.exists():
        return None
    try:
        data = json.loads(creds_path.read_text())
        return data.get("claudeAiOauth", {}).get("accessToken")
    except (json.JSONDecodeError, OSError):
        return None


def _read_cache() -> Optional[tuple[float, RateLimits]]:
    try:
        if not _CACHE_FILE.exists():
            return None
        data = json.loads(_CACHE_FILE.read_text())
        return (data["ts"], RateLimits(**data["limits"]))
    except Exception as e:
        logging.getLogger(__name__).debug("Cache read failed: %s", e)
        return None


def _write_cache(limits: RateLimits) -> None:
    try:
        _CACHE_FILE.write_text(json.dumps({
            "ts": time.time(),
            "limits": {"five_hour_pct": limits.five_hour_pct, "five_hour_resets": limits.five_hour_resets,
                       "weekly_pct": limits.weekly_pct, "weekly_resets": limits.weekly_resets,
                       "sonnet_weekly_pct": limits.sonnet_weekly_pct, "sonnet_weekly_resets": limits.sonnet_weekly_resets,
                       "has_sonnet_limit": limits.has_sonnet_limit},
        }))
        import os
        os.chmod(str(_CACHE_FILE), 0o600)
    except OSError as _exc:
        logger.warning("usage: cache write failed: %s", _exc)


_last_fetch_attempt: float = 0.0
_RETRY_COOLDOWN: float = 60.0  # Wait 60s before retrying after API error


def fetch_usage() -> Optional[RateLimits]:
    """Fetch rate limits from Anthropic API. File-cached for 5 minutes."""
    global _last_fetch_attempt

    cached = _read_cache()
    if cached and time.time() - cached[0] < _CACHE_TTL:
        return cached[1]

    # Don't hammer the API after a failure — wait 60s before retry
    if _last_fetch_attempt and time.time() - _last_fetch_attempt < _RETRY_COOLDOWN:
        return cached[1] if cached else None

    token = _get_access_token()
    if not token:
        return cached[1] if cached else None

    _last_fetch_attempt = time.time()

    req = urllib.request.Request(
        "https://api.anthropic.com/api/oauth/usage",
        headers={
            "Authorization": f"Bearer {token}",
            "anthropic-beta": "oauth-2025-04-20",
            "Content-Type": "application/json",
        },
    )
    try:
        resp = urllib.request.urlopen(req, timeout=5)
        data = json.loads(resp.read())
        sonnet_data = data.get("sonnet_weekly") or data.get("seven_day_sonnet") or {}
        limits = RateLimits(
            five_hour_pct=data.get("five_hour", {}).get("utilization", 0.0),
            five_hour_resets=data.get("five_hour", {}).get("resets_at", ""),
            weekly_pct=data.get("seven_day", {}).get("utilization", 0.0),
            weekly_resets=data.get("seven_day", {}).get("resets_at", ""),
            sonnet_weekly_pct=sonnet_data.get("utilization", 0.0),
            sonnet_weekly_resets=sonnet_data.get("resets_at", ""),
            has_sonnet_limit=data.get("seven_day_sonnet") is not None,
        )
        _write_cache(limits)
        _last_fetch_attempt = 0.0  # Reset on success
        return limits
    except Exception as e:
        logging.getLogger(__name__).debug("Usage API error: %s", e)
        return cached[1] if cached else None
