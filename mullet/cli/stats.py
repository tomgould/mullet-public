"""Usage statistics — show how mullet has been used.

Aggregates session data to show provider usage, tool calls,
tokens consumed, and estimated savings.
"""

import datetime
import logging
from pathlib import Path

from mullet.core.agent.session import get_session_aggregates

logger = logging.getLogger(__name__)

_STATS_WINDOW_DAYS = 30


def get_stats(*, sessions_dir: Path | None = None) -> dict:
    """Gather usage statistics from session files via the canonical aggregate helper."""
    agg = get_session_aggregates(days=_STATS_WINDOW_DAYS, sessions_dir=sessions_dir)

    avg_msgs = agg.total_messages / agg.total_sessions if agg.total_sessions else 0

    # Convert ISO window strings back to epoch floats for callers that use them
    def _iso_to_epoch(iso: str | None) -> float | None:
        if iso is None:
            return None
        try:
            return datetime.datetime.fromisoformat(iso).timestamp()
        except ValueError:
            return None

    return {
        "sessions": agg.total_sessions,
        "messages": agg.total_messages,
        "tool_calls": agg.total_tool_calls,
        "avg_messages_per_session": round(avg_msgs, 1),
        "providers": {p: v["calls"] for p, v in agg.by_provider.items()},
        "earliest": _iso_to_epoch(agg.window_start),
        "latest": _iso_to_epoch(agg.window_end),
    }


def format_stats() -> str:
    """Format stats for display."""
    stats = get_stats()

    # Check for any data sources (sessions, analytics, or routing history)
    has_analytics = False
    has_routing = False
    try:
        from mullet.core.agent.analytics import ToolAnalytics
        has_analytics = len(ToolAnalytics.load_history(days=30)) > 0
    except Exception as exc:
        logger.warning("stats: failed to load tool analytics: %s", exc)
    try:
        from mullet.core.router.learning import _load_history
        has_routing = len(_load_history()) > 0
    except Exception as exc:
        logger.warning("stats: failed to load routing history: %s", exc)

    if stats["sessions"] == 0 and not has_analytics and not has_routing:
        return "No usage data yet. Start a conversation with mullet to generate stats."

    lines = ["\033[1mMullet Usage Statistics\033[0m\n"]

    if stats["sessions"] == 0:
        lines.append("  Sessions:    0 (use REPL for session tracking)")
    else:
        lines.append(f"  Sessions:    {stats['sessions']}")
        lines.append(f"  Messages:    {stats['messages']} ({stats['avg_messages_per_session']} avg/session)")
        if stats["tool_calls"]:
            lines.append(f"  Tool Calls:  {stats['tool_calls']}")

    if stats["earliest"]:
        first = datetime.datetime.fromtimestamp(stats["earliest"]).strftime("%Y-%m-%d")
        last = datetime.datetime.fromtimestamp(stats["latest"]).strftime("%Y-%m-%d")
        lines.append(f"  Period:    {first} to {last}")

    if stats["providers"]:
        lines.append("\n  Providers:")
        for provider, count in sorted(stats["providers"].items(), key=lambda x: -x[1]):
            lines.append(f"    {provider:15s} {count} sessions")

    # Check database for more detailed usage
    try:
        from mullet.server.db import get_db_path
        db_path = Path(get_db_path())
        if db_path.exists():
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            row = conn.execute("SELECT COUNT(*), SUM(input_tokens), SUM(output_tokens), SUM(estimated_cost) FROM usage_log").fetchone()
            if row and row[0]:
                lines.append(f"\n  API Calls:  {row[0]:,}")
                if row[1]:
                    lines.append(f"  Tokens In:  {row[1]:,}")
                if row[2]:
                    lines.append(f"  Tokens Out: {row[2]:,}")
                if row[3]:
                    lines.append(f"  Total Cost: ${row[3]:.4f}")
            conn.close()
    except Exception as exc:
        logger.warning("stats: failed to query usage_log database: %s", exc)

    # Tool analytics
    try:
        from mullet.core.agent.analytics import ToolAnalytics
        history = ToolAnalytics.aggregate_history(days=30)
        if history.get("sessions", 0) > 0:
            lines.append(f"\n  Tool Analytics ({history['sessions']} sessions):")
            for tool_name, tool_stats in list(history["tools"].items())[:10]:
                rate = f"{tool_stats['success_rate']*100:.0f}%"
                lines.append(f"    {tool_name:20s} {tool_stats['total_calls']:>5} calls  {rate:>4} success")
    except Exception as exc:
        logger.warning("stats: failed to load tool analytics history: %s", exc)

    # Routing learning stats
    try:
        from mullet.core.router.learning import _load_history
        routing_history = _load_history()
        if routing_history:
            total = len(routing_history)
            successes = sum(1 for r in routing_history if r.get("success"))
            tiers = {}
            for r in routing_history:
                t = r.get("tier", "?")
                tiers[t] = tiers.get(t, 0) + 1
            lines.append(f"\n  Routing Learning ({total} outcomes):")
            lines.append(f"    Success rate: {successes/total*100:.0f}%")
            for tier, count in sorted(tiers.items()):
                lines.append(f"    {tier:8s} {count:>5} tasks")
    except Exception as exc:
        logger.warning("stats: failed to load routing learning history: %s", exc)

    # Routing history stats (from routing_history.jsonl)
    try:
        from mullet.core.routing_history import summary_by_provider
        rh_summary = summary_by_provider()
        if rh_summary:
            lines.append("\n  Routing History (from routing_history.jsonl):")
            lines.append(f"  {'Provider':<14} {'Calls':>6}  {'Success':>7}  {'Total $':>10}  {'p95 ms':>8}")
            lines.append(f"  {'─'*50}")
            for provider, s in sorted(rh_summary.items(), key=lambda x: -x[1]['total_cost']):
                if s['calls'] == 0:
                    continue
                sr = f"{s['success_rate']:.0%}"
                cost = f"${s['total_cost']:.4f}"
                lines.append(f"  {provider:<14} {s['calls']:>6}  {sr:>7}  {cost:>10}  {s['p95_wall_ms']:>7}ms")
        else:
            lines.append("\n  Routing History: No data on disk yet.")
    except Exception as exc:
        logger.warning("stats: failed to load routing history summary: %s", exc)

    # Daily cost distribution
    try:
        from mullet.core.routing_history import daily_cost_summary
        dcs = daily_cost_summary()
        if dcs and dcs.get("count", 0) >= 2:
            lines.append(
                f"\n  Daily cost ({dcs['count']} days):  "
                f"p50=${dcs['p50']:.2f}  p95=${dcs['p95']:.2f}  max=${dcs['max']:.2f}"
            )
    except Exception as exc:
        logger.warning("stats: failed to compute daily cost summary: %s", exc)

    return "\n".join(lines)
