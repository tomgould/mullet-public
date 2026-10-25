"""Routing history recorder — append-only JSONL log of provider call outcomes.

This module is the data substrate for a future Q-learning router.
It does NOT wire into routing decisions; it only records and aggregates.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from mullet.core.paths import state_dir

logger = logging.getLogger(__name__)

def _get_pricing():
    from mullet.core.providers.model_registry import KNOWN_PRICING
    return KNOWN_PRICING

_HISTORY_FILENAME = "routing_history.jsonl"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RoutingOutcome:
    """A single recorded provider call outcome."""

    timestamp: str
    task_signature: str
    task_preview: str
    provider: str
    model: str
    tier: str
    success: bool
    error_class: str | None
    tokens_in: int
    tokens_out: int
    wall_time_ms: int
    cost_usd: float
    tags: list[str] = field(default_factory=list)
    # Truncated failure message (distinct from error_class, which is just
    # the exception class name, e.g. "RuntimeError"). Added so a failure's
    # actual cause is diagnosable from the log alone, without live
    # reproduction — see the opencode id=3-error-swallowing incident this
    # was added to prevent a recurrence of.
    error_message: str | None = None

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_json_line(self) -> str:
        """Serialise to a single JSON line (no trailing newline)."""
        d = asdict(self)
        return json.dumps(d, separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict) -> RoutingOutcome:
        """Deserialise from a dict (as parsed from JSONL)."""
        return cls(
            timestamp=d.get("timestamp", ""),
            task_signature=d.get("task_signature", ""),
            task_preview=d.get("task_preview", ""),
            provider=d.get("provider", "unknown"),
            model=d.get("model", ""),
            tier=d.get("tier", "STANDARD"),
            success=bool(d.get("success", False)),
            error_class=d.get("error_class"),
            tokens_in=int(d.get("tokens_in", 0)),
            tokens_out=int(d.get("tokens_out", 0)),
            wall_time_ms=int(d.get("wall_time_ms", 0)),
            cost_usd=float(d.get("cost_usd", 0.0)),
            tags=list(d.get("tags", [])),
            error_message=d.get("error_message"),
        )


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def history_path(base: Optional[Path] = None) -> Path:
    """Return path to routing_history.jsonl under *.mullet* in *base*, or Mullet's state directory by default."""
    if base is not None:
        return base / ".mullet" / _HISTORY_FILENAME
    return state_dir() / _HISTORY_FILENAME


def _decision_log_path(base: Optional[Path] = None) -> Path:
    if base is None:
        from mullet.core.paths import state_dir
        log_dir = state_dir()
    else:
        log_dir = base / ".mullet"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "routing.log"


def append_decision_log(
    provider: str,
    model: str,
    tier: str,
    reasons: list[str],
    task_preview: str,
    base: Optional[Path] = None,
) -> None:
    """Append a human-readable routing decision line to routing.log in the state directory.

    Fail-open — IO errors are logged as DEBUG and never raised.
    Format: ISO-timestamp  provider/model  TIER  reason; reason  |  task[:80]
    """
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    reason_str = "; ".join(reasons) if reasons else "—"
    task_str = task_preview.strip()[:80].replace("\n", " ")
    line = f"{ts}  {provider}/{model}  {tier}  {reason_str}  |  {task_str}\n"
    path = _decision_log_path(base)
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.write(line)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
    except Exception as exc:
        logger.debug("routing_history: append_decision_log failed: %s", exc)


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------

def make_signature(
    task: str,
    tool_family: str = "run",
    tier: str = "STANDARD",
) -> str:
    """Return a stable 16-char hex signature for *(task_prefix, tool_family, tier)*.

    Normalises whitespace and lowercases the first 80 chars of *task* before
    hashing so that minor rephrasing of the same conceptual task maps to the
    same bucket.
    """
    prefix = re.sub(r"\s+", " ", task.strip().lower())[:80]
    payload = f"{prefix}|{tool_family.lower()}|{tier.upper()}"
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def estimate_cost(provider: str, tokens_in: int, tokens_out: int, model: str = "") -> float:
    """Return estimated USD cost given provider name, optional model, and token counts."""
    lower = provider.lower()
    if lower in ("local", "lmstudio", "ollama"):
        return 0.0

    pricing = _get_pricing()

    # Exact model match first — most accurate when caller passes the model name.
    if model:
        model_lower = model.lower()
        if model_lower in pricing:
            inp, out = pricing[model_lower]
            return (tokens_in * inp * 1000 + tokens_out * out * 1000) / 1_000_000
        # Substring match on model name
        for key, (inp, out) in pricing.items():
            if model_lower in key or key in model_lower:
                return (tokens_in * inp * 1000 + tokens_out * out * 1000) / 1_000_000

    # Fall back to provider-name substring match
    rates = None
    for key, (inp, out) in pricing.items():
        if lower in key or key in lower:
            rates = (inp, out)
            break

    if rates is None:
        return 0.0

    return (tokens_in * rates[0] * 1000 + tokens_out * rates[1] * 1000) / 1_000_000


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

def record(outcome: RoutingOutcome, base: Optional[Path] = None) -> None:
    """Atomically append *outcome* as a JSON line to the history file.

    Fail-open: logs a WARNING on IO error rather than raising.
    """
    path = history_path(base)
    line = outcome.to_json_line() + "\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.write(line)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
    except OSError as exc:
        logger.warning("routing_history: failed to record outcome: %s", exc)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def iter_recent(
    limit: int = 200,
    base: Optional[Path] = None,
) -> Iterator[RoutingOutcome]:
    """Yield up to *limit* outcomes, newest first."""
    path = history_path(base)
    if not path.exists():
        return

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("routing_history: failed to read history: %s", exc)
        return

    # Reverse so newest first; skip blanks
    for raw in reversed(lines):
        raw = raw.strip()
        if not raw:
            continue
        if limit <= 0:
            break
        try:
            yield RoutingOutcome.from_dict(json.loads(raw))
            limit -= 1
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("routing_history: skipping malformed line: %s", exc)


def success_rate(
    signature: str,
    provider: str,
    base: Optional[Path] = None,
    min_samples: int = 3,
) -> float | None:
    """Return success rate for *(signature, provider)*, or None if < *min_samples* records."""
    path = history_path(base)
    if not path.exists():
        return None

    total = 0
    successes = 0
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("task_signature") == signature and d.get("provider") == provider:
                total += 1
                if d.get("success"):
                    successes += 1
    except OSError as exc:
        logger.warning("routing_history: failed to read for success_rate: %s", exc)
        return None

    if total < min_samples:
        return None
    return successes / total


def _percentile(values: list[float], pct: float) -> float:
    """Return the pct-th percentile of a sorted list (linear interpolation)."""
    n = len(values)
    if n == 0:
        return 0.0
    if n == 1:
        return values[0]
    idx = (pct / 100) * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return values[-1]
    return values[lo] + (idx - lo) * (values[hi] - values[lo])


def summary_by_provider(
    base: Optional[Path] = None,
    limit: int = 2000,
) -> dict[str, dict]:
    """Return per-provider aggregated stats across the last *limit* records.

    Returns ``{provider: {calls, successes, success_rate, total_cost, avg_cost,
    avg_wall_ms, p50_cost, p95_cost, p50_wall_ms, p95_wall_ms}}``.
    """
    path = history_path(base)
    if not path.exists():
        return {}

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("routing_history: failed to read for summary: %s", exc)
        return {}

    # Take last `limit` non-blank lines
    valid_lines = [ln.strip() for ln in lines if ln.strip()][-limit:]

    stats: dict[str, dict] = {}
    for raw in valid_lines:
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        p = d.get("provider", "unknown")
        if p not in stats:
            stats[p] = {
                "calls": 0,
                "successes": 0,
                "rate_limit_errors": 0,
                "total_cost": 0.0,
                "total_wall_ms": 0,
                "costs": [],
                "wall_times": [],
            }
        stats[p]["calls"] += 1
        if d.get("success"):
            stats[p]["successes"] += 1
        if d.get("error_class") == "rate_limit":
            stats[p]["rate_limit_errors"] += 1
        cost = float(d.get("cost_usd", 0.0))
        wall = int(d.get("wall_time_ms", 0))
        stats[p]["total_cost"] += cost
        stats[p]["costs"].append(cost)
        stats[p]["total_wall_ms"] += wall
        stats[p]["wall_times"].append(wall)

    result = {}
    for p, s in stats.items():
        calls = s["calls"]
        sorted_costs = sorted(s["costs"])
        sorted_walls = sorted(s["wall_times"])
        result[p] = {
            "calls": calls,
            "successes": s["successes"],
            "success_rate": round(s["successes"] / calls, 4) if calls else 0.0,
            "rate_limit_rate": round(s["rate_limit_errors"] / calls, 4) if calls else 0.0,
            "total_cost": round(s["total_cost"], 6),
            "avg_cost": round(s["total_cost"] / calls, 6) if calls else 0.0,
            "avg_wall_ms": round(s["total_wall_ms"] / calls) if calls else 0,
            "p50_cost": round(_percentile(sorted_costs, 50), 8),
            "p95_cost": round(_percentile(sorted_costs, 95), 8),
            "p50_wall_ms": round(_percentile(sorted_walls, 50)),
            "p95_wall_ms": round(_percentile(sorted_walls, 95)),
        }
    return result


# ---------------------------------------------------------------------------
# Per-provider spend — HUD helper
# ---------------------------------------------------------------------------

def provider_spend_5h(base: Optional[Path] = None) -> dict[str, float]:
    """Return {provider: cost_usd} for calls in the last 5 hours.

    Reads the tail of routing_history.jsonl (up to 5000 lines) and sums
    cost_usd per provider within the 5h window.  Returns {} when the file
    is absent or unreadable.  Free providers (cost == 0.0) are included so
    callers can decide whether to display them.
    """
    import time as _time

    cutoff = _time.time() - 5 * 3600
    path = history_path(base)
    if not path.exists():
        return {}

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    spend: dict[str, float] = {}
    for raw in lines[-5000:]:
        raw = raw.strip()
        if not raw:
            continue
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        ts_str = d.get("timestamp", "")
        if not ts_str:
            continue
        try:
            from datetime import datetime as _dt
            ts = _dt.fromisoformat(ts_str).timestamp()
        except ValueError:
            continue
        if ts < cutoff:
            continue
        p = d.get("provider", "unknown")
        spend[p] = spend.get(p, 0.0) + float(d.get("cost_usd", 0.0))

    return spend


# ---------------------------------------------------------------------------
# Daily cost summary
# ---------------------------------------------------------------------------

def daily_cost_summary(
    base: Optional[Path] = None,
    limit_days: int = 30,
) -> dict:
    """Return per-day cost totals and distribution statistics.

    Reads routing_history.jsonl, groups cost_usd by UTC calendar date,
    and computes p50/p95/max across the last *limit_days* days.

    Returns::

        {
            "days": {"2026-06-14": 3.12, "2026-06-15": 1.88, ...},  # date→total cost
            "p50": 2.50,
            "p95": 5.10,
            "max": 7.23,
            "count": 14,  # number of days with activity
        }

    Returns an empty dict when the history file is absent or unreadable.
    """
    from datetime import timedelta

    path = history_path(base)
    if not path.exists():
        return {}

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    cutoff = datetime.now(timezone.utc).date() - timedelta(days=limit_days)
    daily_totals: dict[str, float] = {}

    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        ts_str = d.get("timestamp", "")
        if not ts_str:
            continue
        date_str = ts_str[:10]
        try:
            entry_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        if entry_date < cutoff:
            continue
        daily_totals[date_str] = daily_totals.get(date_str, 0.0) + float(d.get("cost_usd", 0.0))

    if not daily_totals:
        return {}

    sorted_values = sorted(daily_totals.values())
    return {
        "days": daily_totals,
        "p50": _percentile(sorted_values, 50),
        "p95": _percentile(sorted_values, 95),
        "max": sorted_values[-1],
        "count": len(daily_totals),
    }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_outcome(
    task: str,
    provider: str,
    model: str,
    tier: str,
    success: bool,
    tokens_in: int,
    tokens_out: int,
    wall_time_ms: int,
    error_class: str | None = None,
    error_message: str | None = None,
    tool_family: str = "run",
    tags: list[str] | None = None,
) -> RoutingOutcome:
    """Build a :class:`RoutingOutcome` with auto-computed signature and cost."""
    return RoutingOutcome(
        timestamp=datetime.now(timezone.utc).isoformat(),
        task_signature=make_signature(task, tool_family, tier),
        # 2000 chars covers the complexity scorer's word-count bands (up to the
        # 500-word tier) for most real prompts, so `mullet routing simulate`
        # replays something close to the original complexity signal instead of
        # a truncated stub.
        task_preview=task.strip()[:2000],
        provider=provider,
        model=model,
        tier=tier,
        success=success,
        error_class=error_class,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        wall_time_ms=wall_time_ms,
        cost_usd=estimate_cost(provider, tokens_in, tokens_out, model=model),
        tags=tags or [],
        # Truncated (not the full traceback) -- this is a log line, not an
        # error report; 500 chars is enough to identify the cause (compare
        # task_preview's 2000-char budget, which serves a different purpose).
        error_message=(error_message or "").strip()[:500] or None,
    )
