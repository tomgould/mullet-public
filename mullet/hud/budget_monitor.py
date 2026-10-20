"""Budget monitoring — check if Anthropic tokens have headroom."""
import logging

from mullet.hud.usage import fetch_usage
from mullet.core.router.resource_guard import get_resource_guard

logger = logging.getLogger(__name__)


def check_headroom() -> dict:
    """Check Anthropic usage headroom and system resources."""
    limits = fetch_usage()
    resources = get_resource_guard().get_system_snapshot()

    if not limits or limits.five_hour_pct is None:
        return {
            "can_use_anthropic": True,
            "advice": "UNKNOWN — usage data unavailable",
            "usage_5h_pct": None,
            "usage_7d_pct": None,
            "resources": resources,
        }

    pct = limits.five_hour_pct
    if pct > 80:
        advice = "THROTTLE — delegate to Kimi/Gemini, Anthropic nearly exhausted"
        can_use = False
    elif pct > 60:
        advice = "CAUTION — prefer delegation, save Anthropic for critical tasks"
        can_use = True
    else:
        advice = "OK — Anthropic available, use judiciously"
        can_use = True

    return {
        "can_use_anthropic": can_use,
        "advice": advice,
        "usage_5h_pct": pct,
        "usage_7d_pct": limits.weekly_pct,
        "resources": resources,
    }


def format_advice(result: dict) -> str:
    """One-line summary for display."""
    pct = result.get("usage_5h_pct")
    pct_str = f"{pct:.0f}%" if pct is not None else "?"
    return f"{result['advice']} (5h: {pct_str})"
