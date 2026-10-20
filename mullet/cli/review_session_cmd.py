"""CLI backing ``mullet review-session``.

Produces a single retrospective report over a session window with 6 sections:
  1. Session window header
  2. Activity digest
  3. Decisions surfaced
  4. Routing summary
  5. Anti-pattern flags
  6. Next-suggested lane

Usage::

    mullet review-session [--session SID] [--window MINUTES]
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Color helpers — optional, plain-text fallback
# ---------------------------------------------------------------------------

def bold(t: str) -> str: return t
def cyan(t: str) -> str: return t
def yellow(t: str) -> str: return t
def red(t: str) -> str: return t
def green(t: str) -> str: return t
def dim(t: str) -> str: return t
RESET = ""
_HAS_COLORS = False

try:
    from mullet.hud.colors import bold, cyan, yellow, red, green, dim, RESET
    _HAS_COLORS = True
except ImportError:  # pragma: no cover — colors always available in test env
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(ts_str: str) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp string (with or without timezone)."""
    if not ts_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S+00:00"):
        try:
            dt = datetime.strptime(ts_str, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _section(title: str) -> str:
    bar = "─" * 60
    return f"\n{bold(cyan(title))}\n{dim(bar)}"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(argv: list[str] | None = None, base: Optional[Path] = None) -> int:
    """Run ``mullet review-session`` and print the report. Returns exit code."""
    parser = argparse.ArgumentParser(
        prog="mullet review-session",
        description="Retrospective session report across all 4 recording pipelines.",
    )
    parser.add_argument("--session", "-s", default=None,
                        help="Restrict to a specific session ID.")
    parser.add_argument("--window", "-w", type=int, default=120,
                        help="Look-back window in minutes (default 120).")
    args = parser.parse_args(argv or [])

    window_minutes: int = args.window
    session_id: Optional[str] = args.session
    now = _utcnow()
    cutoff = now - timedelta(minutes=window_minutes)

    lines: list[str] = []

    # -----------------------------------------------------------------------
    # Load observations (fail-open)
    # -----------------------------------------------------------------------
    from mullet.core.observations import iter_session, iter_recent, Observation

    obs_list: list[Observation] = []
    warnings: list[str] = []
    try:
        if session_id:
            raw_obs = list(iter_session(session_id, base=base))
        else:
            raw_obs = list(iter_recent(limit=2000, base=base))
        for o in raw_obs:
            ts = _parse_ts(o.timestamp)
            if ts and ts >= cutoff:
                obs_list.append(o)
    except Exception as exc:
        warnings.append(f"observations load error: {exc}")

    # -----------------------------------------------------------------------
    # Section 1 — Session window header
    # -----------------------------------------------------------------------
    lines.append(_section("1. SESSION WINDOW"))
    if session_id:
        lines.append(f"  Session : {bold(session_id)}")
    else:
        lines.append(f"  Session : {bold('(all sessions)')}")
    lines.append(f"  Window  : last {window_minutes} min  ({cutoff.strftime('%H:%M:%S')} → {now.strftime('%H:%M:%S')} UTC)")

    if obs_list:
        first_ts = min((o.timestamp for o in obs_list), default="")
        last_ts = max((o.timestamp for o in obs_list), default="")
        dt_first = _parse_ts(first_ts)
        dt_last = _parse_ts(last_ts)
        if dt_first and dt_last:
            duration_s = int((dt_last - dt_first).total_seconds())
            lines.append(f"  Active  : {dt_first.strftime('%H:%M:%S')} → {dt_last.strftime('%H:%M:%S')} UTC  ({duration_s}s)")
        lines.append(f"  Events  : {len(obs_list)} observations")
    else:
        lines.append(f"  {yellow('No activity in this window.')}")

    # -----------------------------------------------------------------------
    # Section 2 — Activity digest
    # -----------------------------------------------------------------------
    lines.append(_section("2. ACTIVITY DIGEST"))
    if not obs_list:
        lines.append(f"  {dim('(empty)')}")
    else:
        tool_counts: dict[str, int] = {}
        files_touched: set[str] = set()
        for o in obs_list:
            tool_counts[o.tool_name] = tool_counts.get(o.tool_name, 0) + 1
            if o.cwd:
                files_touched.add(o.cwd)
        sorted_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)
        for tool, count in sorted_tools[:8]:
            lines.append(f"  {tool:<30} {count:>4} call{'s' if count != 1 else ''}")
        if len(files_touched) > 0:
            lines.append(f"  Distinct working dirs: {len(files_touched)}")

    # -----------------------------------------------------------------------
    # Section 3 — Decisions surfaced
    # -----------------------------------------------------------------------
    lines.append(_section("3. DECISIONS SURFACED"))
    decisions_in_window: list = []
    try:
        from mullet.core.decisions import load
        all_decisions = load(base=base)
        for d in all_decisions:
            if d.timestamp:
                ts = _parse_ts(d.timestamp)
                if ts and ts >= cutoff:
                    decisions_in_window.append(d)
            # Also include decisions without timestamps (from docs/progress)
        if not decisions_in_window:
            # Fall back to most-recent without timestamp filter
            no_ts = [d for d in all_decisions if not d.timestamp][:5]
            decisions_in_window = no_ts
    except Exception as exc:
        warnings.append(f"decisions load error: {exc}")

    if not decisions_in_window:
        lines.append(f"  {dim('No decisions found in window.')}")
    else:
        for d in decisions_in_window[:10]:
            kind_tag = f"[{d.kind}]"
            excerpt = d.excerpt[:72] + ("…" if len(d.excerpt) > 72 else "")
            lines.append(f"  {yellow(kind_tag):<18} {excerpt}")

    # -----------------------------------------------------------------------
    # Section 4 — Routing summary
    # -----------------------------------------------------------------------
    lines.append(_section("4. ROUTING SUMMARY"))
    routing_summary: dict[str, dict] = {}
    try:
        from mullet.core.routing_history import iter_recent as rh_iter_recent
        outcomes_in_window = []
        for outcome in rh_iter_recent(limit=2000, base=base):
            ts = _parse_ts(outcome.timestamp)
            if ts and ts >= cutoff:
                if session_id is None or session_id in (outcome.tags or []):
                    outcomes_in_window.append(outcome)

        if outcomes_in_window:
            tmp: dict[str, dict] = {}
            for o in outcomes_in_window:
                p = o.provider
                if p not in tmp:
                    tmp[p] = {"calls": 0, "successes": 0, "total_cost": 0.0, "total_wall_ms": 0}
                tmp[p]["calls"] += 1
                if o.success:
                    tmp[p]["successes"] += 1
                tmp[p]["total_cost"] += o.cost_usd
                tmp[p]["total_wall_ms"] += o.wall_time_ms
            for p, s in tmp.items():
                c = s["calls"]
                routing_summary[p] = {
                    "calls": c,
                    "success_rate": round(s["successes"] / c, 2) if c else 0.0,
                    "avg_cost": round(s["total_cost"] / c, 6) if c else 0.0,
                    "avg_wall_ms": round(s["total_wall_ms"] / c) if c else 0,
                }
    except Exception as exc:
        warnings.append(f"routing_history load error: {exc}")

    if not routing_summary:
        lines.append(f"  {dim('No routing outcomes in window.')}")
    else:
        lines.append(f"  {'Provider':<14} {'Calls':>6}  {'Success':>7}  {'Avg cost':>10}  {'Avg ms':>8}")
        lines.append(f"  {dim('─'*55)}")
        for provider, stats in sorted(routing_summary.items()):
            sr_str = f"{stats['success_rate']:.0%}"
            cost_str = f"${stats['avg_cost']:.5f}"
            lines.append(
                f"  {provider:<14} {stats['calls']:>6}  {sr_str:>7}  {cost_str:>10}  {stats['avg_wall_ms']:>7}ms"
            )

    # -----------------------------------------------------------------------
    # Section 5 — Anti-pattern flags
    # -----------------------------------------------------------------------
    lines.append(_section("5. ANTI-PATTERN FLAGS"))
    flags: list[str] = []
    for provider, stats in routing_summary.items():
        if stats["calls"] >= 3 and stats["success_rate"] < 0.5:
            flags.append(
                f"  {red('WARNING')} {provider}: {stats['calls']} calls, "
                f"success rate {stats['success_rate']:.0%} (<50% threshold)"
            )
    if warnings:
        for w in warnings:
            flags.append(f"  {yellow('WARN')} {w}")
    if not flags:
        lines.append(f"  {green('No anti-patterns detected.')}")
    else:
        lines.extend(flags)

    # -----------------------------------------------------------------------
    # Section 6 — Next-suggested lane
    # -----------------------------------------------------------------------
    lines.append(_section("6. NEXT-SUGGESTED LANE"))
    try:
        from mullet.hud.guard import check_budget
        budget = check_budget()
        pressure = budget.pressure
        if pressure < 0.30:
            lane = "premium"
            rationale = f"pressure={pressure:.2f} — Anthropic quota healthy, capable models available"
        elif pressure < 0.65:
            lane = "balanced"
            rationale = f"pressure={pressure:.2f} — moderate load, use Kimi/Gemini for bulk work"
        else:
            lane = "economy"
            rationale = f"pressure={pressure:.2f} — high budget pressure, prefer local LLM"

        # Override toward economy if anti-pattern flags exist
        if flags and lane == "premium":
            lane = "balanced"
            rationale += "; anti-pattern flags present → downgrade to balanced"

        lane_color = {"premium": green, "balanced": yellow, "economy": red}.get(lane, cyan)
        lines.append(f"  Suggested lane : {lane_color(bold(lane.upper()))}")
        lines.append(f"  Rationale      : {rationale}")
    except Exception as exc:
        lines.append(f"  {dim('(budget check unavailable)')}")
        lines.append(f"  Suggested lane : {yellow(bold('BALANCED'))}  (default fallback)")
        lines.append(f"  Rationale      : budget API error ({exc})")

    # -----------------------------------------------------------------------
    # Print report
    # -----------------------------------------------------------------------
    report = "\n".join(lines)
    print(report)
    return 0
