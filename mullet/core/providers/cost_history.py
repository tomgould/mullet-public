"""Persistent session cost history for p50/p95 distribution tracking.

Appends one JSON line per completed session to cost_history.jsonl in the
state directory (see ``mullet.core.paths``).
Provides percentile accessors for the HUD and budget display.

Format of each line:
  {"date": "YYYY-MM-DD", "session_usd": 0.1234, "providers": {"kimi": 0.05, "openrouter": 0.07}}
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from mullet.core.paths import state_dir

logger = logging.getLogger(__name__)

# Tests point this at a temporary file; None means cost_history.jsonl in the
# state directory.
COST_HISTORY_PATH: Path | None = None


def _cost_history_path() -> Path:
    return COST_HISTORY_PATH or state_dir() / "cost_history.jsonl"


def record_session(session_usd: float, providers: dict[str, float] | None = None) -> None:
    """Append today's session cost to the history file.

    Safe to call multiple times per day — each call appends a new line.
    Silently skips on write errors.
    """
    entry = {
        "date": str(date.today()),
        "session_usd": round(session_usd, 6),
        "providers": providers or {},
    }
    try:
        path = _cost_history_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except OSError as exc:
        logger.warning("cost_history: failed to write entry: %s", exc)


def load_history(window_days: int = 90) -> list[dict[str, Any]]:
    """Load cost history entries within the last window_days days.

    Returns a list of dicts with keys: date, session_usd, providers.
    Returns [] if the file doesn't exist or is malformed.
    """
    if not _cost_history_path().exists():
        return []

    cutoff = date.today() - timedelta(days=window_days)
    entries: list[dict[str, Any]] = []

    try:
        with open(_cost_history_path(), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("cost_history: skipping malformed line: %s", exc)
                    continue
                if not isinstance(obj, dict):
                    continue
                entry_date_str = obj.get("date", "")
                try:
                    entry_date = date.fromisoformat(entry_date_str)
                except (TypeError, ValueError):
                    logger.warning("cost_history: skipping line with invalid date: %r", entry_date_str)
                    continue
                if entry_date < cutoff:
                    continue
                entries.append(obj)
    except OSError as exc:
        logger.warning("cost_history: failed to read history: %s", exc)
        return []

    entries.sort(key=lambda e: e.get("date", ""))
    return entries


def get_percentiles(window_days: int = 30) -> dict[str, float]:
    """Compute cost percentiles from the last window_days of history.

    Returns:
        {"p50": float, "p95": float, "mean": float, "n": int, "max": float}
    Returns all-zero dict if fewer than 2 data points.
    """
    entries = load_history(window_days=window_days)
    vals = [e["session_usd"] for e in entries if isinstance(e.get("session_usd"), (int, float))]
    n = len(vals)
    if n < 2:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0, "n": 0, "max": 0.0}

    s = sorted(vals)
    total = sum(s)
    mean = total / n
    max_val = s[-1]

    def _percentile(data: list[float], p: float) -> float:
        """Linear interpolation percentile."""
        if n == 1:
            return data[0]
        # Use nearest-rank variant: index = p * (n - 1)
        idx = p * (n - 1)
        lower = int(idx)
        upper = lower + 1
        if upper >= n:
            return data[-1]
        frac = idx - lower
        return data[lower] + frac * (data[upper] - data[lower])

    p50 = _percentile(s, 0.50)
    p95 = _percentile(s, 0.95)

    return {
        "p50": round(p50, 6),
        "p95": round(p95, 6),
        "mean": round(mean, 6),
        "n": n,
        "max": round(max_val, 6),
    }
