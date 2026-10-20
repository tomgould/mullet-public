"""Shared budget gate for the delegation hooks.

Delegation is a last resort, for when the Anthropic budget is too tight to
finish the work in-context. Delegating to OpenRouter, Kimi or the other
third-party providers spends real money per token; the Anthropic subscription
does not, being already paid for and costing nothing extra until the 5-hour or
weekly allowance actually runs short. So these hooks suggest delegation only
once that allowance is genuinely too tight, and never without asking the
operator first.

Both hooks that push toward delegation import this, so the thresholds live in
one place.
"""

from __future__ import annotations

import os

# Suggest delegation only once the paced 5-hour pressure is this high. Pressure
# is paced against the reset window rather than raw utilisation, so 0.80 means
# "spending fast enough to exhaust the window early", not simply "80% used".
SESSION_PRESSURE_THRESHOLD = 0.80

# The weekly window is the dangerous one: it refills once a week rather than
# every five hours, so running it down has a much longer consequence.
WEEKLY_PRESSURE_THRESHOLD = 0.80

# A raw weekly figure this high is dangerous regardless of pacing, because
# there is not enough left for pacing to recover it.
WEEKLY_PCT_DANGER = 85.0

CONFIRM_LINE = (
    "LAST RESORT, AND ASK FIRST. Delegation is not the expected response to budget "
    "pressure, only the option once working in-context is no longer viable. Try the free "
    "routes first: a Claude Code subagent, the local LLM, trawl search. If you still want "
    "to delegate, say what you would delegate and to which provider, then wait for an "
    "explicit go-ahead. Delegation spends third-party tokens, which is real money, so an "
    "unasked-for delegation is a wasted spend."
)


def delegation_turned_off() -> bool:
    """True when MULLET_DELEGATION=off, in the environment or Mullet's .env.

    With delegation off, the MCP tools refuse to dispatch, so the hooks must
    not suggest delegating (or block manual work until Claude delegates).
    """
    try:
        from mullet.core.env import delegation_enabled

        return not delegation_enabled()
    except Exception:
        value = os.environ.get("MULLET_DELEGATION", "").strip().lower()
        return value in {"0", "off", "false", "no", "disabled"}


def pressure_snapshot() -> dict | None:
    """Current budget figures, or None if they can't be read.

    `check_budget()` reads a file cache with a 5-minute TTL, so this is
    effectively free on the hook path; only one call per five minutes reaches
    the network, and a failure there returns the stale cache rather than
    blocking.
    """
    try:
        from mullet.hud.guard import check_budget

        result = check_budget()
        return {
            "status": result.status.value,
            "pressure": float(result.pressure),
            "session_pressure": float(result.session_pressure),
            "weekly_pressure": float(result.weekly_pressure),
            "five_hour_pct": float(getattr(result, "five_hour_pct", 0.0) or 0.0),
            "weekly_pct": float(getattr(result, "weekly_pct", 0.0) or 0.0),
        }
    except Exception:
        # Never let a budget lookup break a hook, and never guess.
        return None


def delegation_warranted(snapshot: dict | None = None) -> tuple[bool, str]:
    """Return (warranted, human-readable reason).

    Fails closed: if the figures can't be read, the answer is no. Staying quiet
    costs nothing, whereas nudging toward a paid provider on a bad guess does.
    Always no while delegation is turned off (MULLET_DELEGATION=off).
    """
    if delegation_turned_off():
        return False, "delegation is turned off (MULLET_DELEGATION=off)"
    snap = pressure_snapshot() if snapshot is None else snapshot
    if snap is None:
        return False, "budget figures unavailable"

    if snap["status"] == "pause":
        return True, (
            f"Anthropic budget is at PAUSE (5h {snap['five_hour_pct']:.0f}%, "
            f"weekly {snap['weekly_pct']:.0f}%)"
        )
    if snap["weekly_pct"] >= WEEKLY_PCT_DANGER:
        return True, (
            f"weekly allowance is at {snap['weekly_pct']:.0f}%, past the "
            f"{WEEKLY_PCT_DANGER:.0f}% danger line"
        )
    if snap["weekly_pressure"] >= WEEKLY_PRESSURE_THRESHOLD:
        return True, (
            f"weekly pressure is {snap['weekly_pressure']:.2f} "
            f"(threshold {WEEKLY_PRESSURE_THRESHOLD:.2f}), and the weekly window "
            "only refills once a week"
        )
    if snap["session_pressure"] >= SESSION_PRESSURE_THRESHOLD:
        return True, (
            f"5-hour pressure is {snap['session_pressure']:.2f} "
            f"(threshold {SESSION_PRESSURE_THRESHOLD:.2f})"
        )

    return False, (
        f"Anthropic budget is comfortable (5h {snap['five_hour_pct']:.0f}%, "
        f"weekly {snap['weekly_pct']:.0f}%, pressure {snap['pressure']:.2f}) — "
        "in-context work is already paid for"
    )
