"""Tool usage analytics — track success rates, common failures, patterns.

Records tool call outcomes to enable:
- Smart routing improvement (learn from past task outcomes)
- Tool recommendation (suggest tools based on task type)
- Failure pattern detection (identify broken tools early)
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from mullet.core.paths import state_dir

logger = logging.getLogger(__name__)

# Tests point this at a temporary directory; None means analytics/ in the
# state directory.
ANALYTICS_DIR: Path | None = None


def _analytics_dir() -> Path:
    return ANALYTICS_DIR or state_dir() / "analytics"


@dataclass
class ToolCallRecord:
    """A single tool call and its outcome."""
    tool_name: str
    success: bool
    duration_ms: float = 0.0
    error: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolStats:
    """Aggregated statistics for a single tool."""
    name: str
    total_calls: int = 0
    successes: int = 0
    failures: int = 0
    avg_duration_ms: float = 0.0
    common_errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Return successes / total_calls as a 0-1 ratio (0 when no calls yet)."""
        return self.successes / self.total_calls if self.total_calls > 0 else 0.0


class ToolAnalytics:
    """Track and analyze tool usage patterns."""

    def __init__(self):
        self._records: list[ToolCallRecord] = []
        self._session_start = time.time()

    def record(self, tool_name: str, success: bool, duration_ms: float = 0.0, error: str = ""):
        """Record a tool call outcome."""
        self._records.append(ToolCallRecord(
            tool_name=tool_name,
            success=success,
            duration_ms=duration_ms,
            error=error[:200] if error else "",
        ))

    def get_stats(self) -> dict[str, ToolStats]:
        """Compute statistics per tool."""
        stats: dict[str, ToolStats] = {}

        for rec in self._records:
            if rec.tool_name not in stats:
                stats[rec.tool_name] = ToolStats(name=rec.tool_name)
            s = stats[rec.tool_name]
            s.total_calls += 1
            if rec.success:
                s.successes += 1
            else:
                s.failures += 1
                if rec.error and rec.error not in s.common_errors:
                    s.common_errors.append(rec.error)
                s.common_errors = s.common_errors[:5]  # Keep top 5

        # Compute averages
        for name, s in stats.items():
            durations = [r.duration_ms for r in self._records if r.tool_name == name and r.duration_ms > 0]
            if durations:
                s.avg_duration_ms = sum(durations) / len(durations)

        return stats

    def get_summary(self) -> dict:
        """Get a summary suitable for display."""
        stats = self.get_stats()
        duration = time.time() - self._session_start

        return {
            "session_duration_s": round(duration, 1),
            "total_tool_calls": len(self._records),
            "unique_tools_used": len(stats),
            "tools": {
                name: {
                    "calls": s.total_calls,
                    "success_rate": round(s.success_rate, 3),
                    "avg_ms": round(s.avg_duration_ms, 1),
                    "failures": s.failures,
                }
                for name, s in sorted(stats.items(), key=lambda x: -x[1].total_calls)
            },
        }

    def save(self):
        """Persist analytics to disk."""
        analytics_dir = _analytics_dir()
        analytics_dir.mkdir(parents=True, exist_ok=True)
        path = analytics_dir / f"session_{int(self._session_start)}.json"
        try:
            path.write_text(json.dumps(self.get_summary(), indent=2))
        except Exception as e:
            logger.debug("Failed to save analytics: %s", e)

    @staticmethod
    def load_history(days: int = 30) -> list[dict]:
        """Load analytics from past sessions."""
        if not _analytics_dir().exists():
            return []

        cutoff = time.time() - (days * 86400)
        sessions = []
        for path in sorted(_analytics_dir().glob("session_*.json"), reverse=True):
            # Extract timestamp from filename (session_<timestamp>.json)
            try:
                ts_str = path.stem.split("_", 1)[1]
                file_ts = float(ts_str)
                if file_ts > 1_000_000_000 and file_ts < cutoff:
                    # Only apply cutoff for plausible Unix timestamps (post-2001)
                    continue
            except (IndexError, ValueError):
                pass  # Can't parse timestamp — include the file anyway

            try:
                data = json.loads(path.read_text())
                sessions.append(data)
            except Exception as e:
                logger.debug("Failed to load analytics session from %s: %s", path, e)
            if len(sessions) >= 100:
                break

        return sessions

    @staticmethod
    def aggregate_history(days: int = 30) -> dict:
        """Aggregate tool stats across all sessions."""
        sessions = ToolAnalytics.load_history(days)
        if not sessions:
            return {"sessions": 0, "tools": {}}

        tool_totals: dict[str, dict] = defaultdict(lambda: {"calls": 0, "failures": 0})

        for session in sessions:
            for name, stats in session.get("tools", {}).items():
                tool_totals[name]["calls"] += stats.get("calls", 0)
                tool_totals[name]["failures"] += stats.get("failures", 0)

        tools = {}
        for name, t in sorted(tool_totals.items(), key=lambda x: -x[1]["calls"]):
            calls = t["calls"]
            failures = t["failures"]
            tools[name] = {
                "total_calls": calls,
                "success_rate": round((calls - failures) / calls, 3) if calls > 0 else 0,
                "failures": failures,
            }

        return {"sessions": len(sessions), "tools": tools}
