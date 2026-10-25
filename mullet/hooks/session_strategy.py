#!/usr/bin/env python3
"""Claude Code SessionStart hook: open the session with a budget and delegation question.

Contract: Claude Code passes a JSON object on stdin (ignored).
The hook prints EXACTLY one JSON object of this shape:
    {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "<text>"}}
It reads local facts only (the session strategy, the spending envelope and the
jobs ledger, Claude's usage figures as the statusline last recorded them, the
active preset, night mode) and never touches the network.

With a fresh strategy, the text says so and tells Claude not to ask again.
Otherwise it gives the facts, a recommendation from `recommend()`, and tells
Claude to ask the user one short question, then run `mullet strategy set`.
With ``session.ask_at_start: false`` it prints nothing.

Fail-open: on an error, print nothing on stdout, say what went wrong on stderr
and exit 0, so a session always starts. If Mullet can't be imported at all,
exit 1, which Claude Code shows as a non-blocking error.

Run as `mullet hook session-strategy` (SessionStart).
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

WINDOW_WORDS = {"daily": "today", "weekly": "this week"}


@dataclass
class Facts:
    """What the hook knows at session start; None means unknown or not set."""

    now: float
    ttl_hours: float
    claude_low: float  # the ladder's thresholds and default job cap (ladder.*)
    claude_high: float
    max_spend: float
    strategy: Any = None  # mullet.core.strategy.Strategy
    envelope: dict[str, tuple[float, float]] = field(default_factory=dict)  # window -> (limit, left)
    envelope_known: bool = True
    claude_weekly_pct: float | None = None
    claude_pressure: float | None = None
    claude_level: str | None = None
    preset: str | None = None
    night_on: bool | None = None

    @property
    def envelope_left(self) -> float | None:
        """Room left under the tightest envelope window, or None when no envelope is set."""
        return min((left for _, left in self.envelope.values()), default=None)


def recommend(envelope_left: float | None, claude_pressure: float | None,
              claude_low: float, claude_high: float) -> tuple[str, str]:
    """The delegation choice to suggest, and why.

    The rules, in order:

    1. An envelope is set and nothing is left under it: paid rungs can't run.
       If Claude's paced weekly pressure is also above ``claude_high``,
       recommend ``off``; otherwise ``claude-only``.
    2. Claude's paced weekly pressure is above ``claude_high``: recommend
       ``ladder``, so cheaper rungs take the mechanical steps.
    3. Otherwise ``ladder``; while Claude's pressure is below ``claude_low``
       the ladder puts Claude first anyway.

    The thresholds are the ladder's own (``ladder.claude_low`` and
    ``ladder.claude_high``); an unknown pressure never triggers a rule.
    """
    if envelope_left is not None and envelope_left <= 0:
        if claude_pressure is not None and claude_pressure > claude_high:
            return "off", (f"the spending envelope is used up and Claude's weekly pressure "
                           f"{claude_pressure:.2f} is above claude_high {claude_high:.2f}")
        return "claude-only", "the spending envelope is used up, so paid rungs can't run"
    if claude_pressure is not None and claude_pressure > claude_high:
        return "ladder", (f"Claude's weekly pressure {claude_pressure:.2f} is above claude_high "
                          f"{claude_high:.2f}, so cheaper rungs should take the mechanical steps")
    if claude_pressure is not None and claude_pressure < claude_low:
        return "ladder", (f"Claude's weekly pressure {claude_pressure:.2f} is below claude_low "
                          f"{claude_low:.2f}, so the ladder puts Claude first anyway")
    return "ladder", "nothing argues against the default ladder"


def suggest_budget(envelope_left: float | None, max_spend: float) -> float:
    """The suggested session budget: ``ladder.max_spend``, capped at the envelope room left."""
    if envelope_left is None:
        return max(max_spend, 0.0)
    return max(min(max_spend, envelope_left), 0.0)


def _envelope(facts: Facts) -> None:
    """Fill in the envelope's limits and room left, from the jobs ledger when it exists."""
    from mullet.jobs.budget import BudgetConfig, Limits, earliest_start, headroom, load_budget
    from mullet.jobs.ladder import load_config
    from mullet.jobs.ledger import Ledger
    from mullet.jobs.paths import jobs_home

    budget = load_budget()
    windows = {w: budget.envelope.get(w) for w in WINDOW_WORDS if budget.envelope.get(w) is not None}
    if not windows:
        return
    config = load_config()
    records: list[tuple[float, dict[str, Any]]] = []
    ledger_path = jobs_home() / "ledger.db"
    since = earliest_start(budget, facts.now)
    if since is not None and ledger_path.is_file():
        with Ledger(ledger_path) as ledger:
            records = ledger.attempt_usages_since(since)
    for window, limit in windows.items():
        only = BudgetConfig(envelope=Limits(**{window: limit}))
        left = headroom(only, records, config, facts.now).left
        facts.envelope[window] = (float(limit), float(limit if left is None else left))


def _claude(facts: Facts, config: Any) -> None:
    """Claude's weekly use from the usage figures the ladder reads, and its budget level."""
    from mullet.jobs.ladder import Ladder, current_usage

    limits = current_usage()
    if limits is not None:
        facts.claude_weekly_pct = float(getattr(limits, "weekly_pct", 0.0) or 0.0)
    state = Ladder(config=config, usage=lambda: limits, now=lambda: facts.now).budget()
    if state.weekly_pressure is not None:
        facts.claude_pressure = float(state.weekly_pressure)
        facts.claude_level = state.level


def gather(now: float | None = None) -> Facts:
    """Collect the local facts; a part that can't be read is left unknown."""
    from mullet.core import strategy
    from mullet.jobs.ladder import load_config

    settings = strategy.load_settings()
    config = load_config()
    facts = Facts(now=time.time() if now is None else now, ttl_hours=settings.strategy_ttl_hours,
                  claude_low=config.claude_low, claude_high=config.claude_high, max_spend=config.max_spend)
    facts.strategy = strategy.load(facts.now, ttl_hours=settings.strategy_ttl_hours)
    try:
        _envelope(facts)
    except Exception as exc:
        facts.envelope_known = False
        print(f"[hook:session_strategy] envelope unknown: {exc}", file=sys.stderr)
    try:
        _claude(facts, config)
    except Exception as exc:
        print(f"[hook:session_strategy] Claude's usage unknown: {exc}", file=sys.stderr)
    try:
        from mullet.core.presets import PresetManager

        facts.preset = PresetManager().active().name
    except Exception:
        facts.preset = None
    try:
        from mullet.jobs import night

        facts.night_on = night.is_on(facts.now)
    except Exception:
        facts.night_on = None
    return facts


def _envelope_text(facts: Facts) -> str:
    if not facts.envelope_known:
        return "spending envelope: unknown"
    if not facts.envelope:
        return "spending envelope: no envelope set (budget.envelope)"
    parts = [f"${left:.2f} of ${limit:.2f} left {WINDOW_WORDS[w]}" for w, (limit, left) in facts.envelope.items()]
    return "spending envelope: " + ", ".join(parts)


def _claude_text(facts: Facts) -> str:
    if facts.claude_pressure is None and facts.claude_weekly_pct is None:
        return "Claude's week: unknown"
    used = f"{facts.claude_weekly_pct:.0f}% used, " if facts.claude_weekly_pct is not None else ""
    if facts.claude_pressure is None:
        return f"Claude's week: {used}paced pressure unknown"
    return (f"Claude's week: {used}paced pressure {facts.claude_pressure:.2f} ({facts.claude_level}; "
            f"ladder thresholds claude_low {facts.claude_low:.2f}, claude_high {facts.claude_high:.2f})")


def facts_line(facts: Facts) -> str:
    """The facts as one line."""
    night = {True: "on", False: "off", None: "unknown"}[facts.night_on]
    return "; ".join([_envelope_text(facts), _claude_text(facts),
                      f"preset: {facts.preset or 'unknown'}", f"night mode: {night}"])


def build_context(facts: Facts) -> str:
    """The text Claude gets at session start."""
    from mullet.core.strategy import DELEGATION_CHOICES, DELEGATION_MEANINGS

    if facts.strategy is not None:
        hours = max(facts.now - facts.strategy.set_at, 0.0) / 3600
        return (f"[Mullet session strategy] Strategy already set {hours:.1f} hours ago: "
                f"{facts.strategy.describe()}. It lasts {facts.ttl_hours:g} hours from when it was set. "
                f"Do not ask about budget or delegation again unless the user wants to change it "
                f"(`mullet strategy set --budget <usd> --delegation <choice>`). Facts: {facts_line(facts)}.")
    choice, why = recommend(facts.envelope_left, facts.claude_pressure, facts.claude_low, facts.claude_high)
    budget = suggest_budget(facts.envelope_left, facts.max_spend)
    options = "; ".join(f"{c} = {DELEGATION_MEANINGS[c]}" for c in DELEGATION_CHOICES)
    return (
        f"[Mullet session strategy] Facts: {facts_line(facts)}. "
        f"In your first reply, ask the user ONE short question, as one message (if the preset hook also "
        f"asks which preset to use, put that in the same message, not a second one), covering: "
        f"(1) this session's budget in dollars, suggesting ${budget:.2f}; "
        f"(2) the delegation strategy: {options}. Recommend {choice}, because {why}. "
        f"After the answer, run `mullet strategy set --budget <usd> --delegation <choice>`. "
        f"If the user's first message already states a budget and a strategy, don't ask: run that "
        f"command with their values."
    )


def main(gather_facts: Callable[[], Facts] | None = None) -> int:
    """Print the session-start context, or nothing when ``session.ask_at_start`` is false."""
    try:
        from mullet.core import strategy
    except ImportError as exc:
        print(f"[hook:session_strategy] can't import Mullet: {exc}", file=sys.stderr)
        return 1
    try:
        sys.stdin.read()
        if not strategy.load_settings().ask_at_start:
            return 0
        facts = (gather_facts or gather)()
        output = {"hookSpecificOutput": {"hookEventName": "SessionStart",
                                         "additionalContext": build_context(facts)}}
        print(json.dumps(output))
    except Exception as exc:
        print(f"[hook:session_strategy] error: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
