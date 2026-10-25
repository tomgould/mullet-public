"""Spending limits shared by every job: an envelope over a time window, and limits per rung.

The ``budget:`` block of the Mullet config file::

    budget:
      envelope:            # every paid rung, every job together
        daily: 25.0        # dollars per UTC day
        weekly: 100.0      # dollars per week, Monday 00:00 UTC
      per_rung:            # the same windows for one paid rung (kimi, openrouter)
        kimi: {daily: 15.0}
        openrouter: {weekly: 40.0}

Nothing is limited until a value is set. A job's own cap (``ladder.max_spend``)
still applies on top. The window spend is every paid attempt in the ledger that
started inside the window, running ones included, priced as the ladder prices it.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mullet.jobs.ladder import PAID_RUNGS, LadderConfig, attempt_cost, config_path

logger = logging.getLogger(__name__)

WINDOWS = ("daily", "weekly")


@dataclass(frozen=True)
class Limits:
    """Dollar limits per window; None means no limit for that window."""

    daily: float | None = None
    weekly: float | None = None

    def get(self, window: str) -> float | None:
        return getattr(self, window)


@dataclass
class BudgetConfig:
    """The ``budget:`` limits the job runner enforces."""

    envelope: Limits = field(default_factory=Limits)
    per_rung: dict[str, Limits] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return bool(self.envelope.daily is not None or self.envelope.weekly is not None
                    or any(v.daily is not None or v.weekly is not None for v in self.per_rung.values()))


def _dollars(value: Any, where: str, problems: list[str]) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        problems.append(f"{where}: expected a number of dollars, 0 or more; ignored")
        return None
    return float(value)


def _limits(raw: Any, where: str, problems: list[str]) -> Limits:
    if raw is None:
        return Limits()
    if not isinstance(raw, dict):
        problems.append(f"{where}: expected a mapping with daily and/or weekly; ignored")
        return Limits()
    for key in raw:
        if key not in WINDOWS:
            problems.append(f"{where}.{key}: unknown window (use daily or weekly); ignored")
    return Limits(_dollars(raw.get("daily"), f"{where}.daily", problems),
                  _dollars(raw.get("weekly"), f"{where}.weekly", problems))


def parse_budget(data: Any, source: str = "budget") -> BudgetConfig:
    """Read the envelope and per-rung limits out of the raw ``budget:`` mapping.

    The older ``per_task``, ``per_session`` and ``daily`` keys belong to the
    legacy delegation path and are not read here.
    """
    config = BudgetConfig()
    if not isinstance(data, dict):
        return config
    config.envelope = _limits(data.get("envelope"), f"{source}.envelope", config.problems)
    raw_rungs = data.get("per_rung")
    if raw_rungs is not None and not isinstance(raw_rungs, dict):
        config.problems.append(f"{source}.per_rung: expected a mapping of rung name to limits; ignored")
        raw_rungs = None
    for name, raw in (raw_rungs or {}).items():
        if name not in PAID_RUNGS:
            config.problems.append(f"{source}.per_rung.{name}: not a paid rung "
                                   f"({', '.join(sorted(PAID_RUNGS))}); ignored")
            continue
        config.per_rung[name] = _limits(raw, f"{source}.per_rung.{name}", config.problems)
    return config


def load_budget(path: Path | None = None) -> BudgetConfig:
    """The budget limits from the user's Mullet config file; no limits when there are none."""
    target = path or config_path()
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return BudgetConfig()
    except (OSError, yaml.YAMLError) as exc:
        config = BudgetConfig()
        config.problems.append(f"{target}: unreadable ({exc}); no budget limits are applied")
        return config
    return parse_budget(data.get("budget") if isinstance(data, dict) else None, f"{target} budget")


def window_start(window: str, now: float) -> float:
    """When the current window began: 00:00 UTC today, or Monday 00:00 UTC."""
    moment = dt.datetime.fromtimestamp(now, dt.timezone.utc)
    day = moment.replace(hour=0, minute=0, second=0, microsecond=0)
    if window == "weekly":
        day -= dt.timedelta(days=day.weekday())
    return day.timestamp()


def earliest_start(config: BudgetConfig, now: float) -> float | None:
    """The oldest attempt start that matters for these limits, or None when nothing is limited."""
    used = [w for w in WINDOWS
            if config.envelope.get(w) is not None or any(v.get(w) is not None for v in config.per_rung.values())]
    return min(window_start(w, now) for w in used) if used else None


@dataclass
class Headroom:
    """What is left under the limits right now."""

    left: float | None = None  # under the envelope; None means unlimited
    reason: str = ""  # which envelope window is the tightest
    rung_left: dict[str, float] = field(default_factory=dict)
    rung_reasons: dict[str, str] = field(default_factory=dict)


def headroom(config: BudgetConfig, records: Iterable[tuple[float, dict[str, Any] | None]],
             ladder: LadderConfig, now: float) -> Headroom:
    """The room left under each limit, from `records` of (attempt start, usage) in the ledger."""
    spent: dict[str, float] = {w: 0.0 for w in WINDOWS}
    by_rung: dict[str, dict[str, float]] = {}
    starts = {w: window_start(w, now) for w in WINDOWS}
    for started, usage in records:
        if not usage:
            continue
        dollars, _ = attempt_cost(usage, ladder)
        if dollars <= 0:
            continue
        rung = str(usage.get("rung") or "")
        for window in WINDOWS:
            if started >= starts[window]:
                spent[window] += dollars
                by_rung.setdefault(rung, {w: 0.0 for w in WINDOWS})[window] += dollars
    result = Headroom()
    for window in WINDOWS:
        limit = config.envelope.get(window)
        if limit is None:
            continue
        room = limit - spent[window]
        if result.left is None or room < result.left:
            result.left = room
            result.reason = (f"the {window} spending envelope is reached "
                             f"(${spent[window]:.2f} of ${limit:.2f}, budget.envelope.{window})")
    for name, limits in config.per_rung.items():
        for window in WINDOWS:
            limit = limits.get(window)
            if limit is None:
                continue
            room = limit - by_rung.get(name, {}).get(window, 0.0)
            if name not in result.rung_left or room < result.rung_left[name]:
                result.rung_left[name] = room
                result.rung_reasons[name] = (
                    f"the {name} {window} limit is reached "
                    f"(${limit - room:.2f} of ${limit:.2f}, budget.per_rung.{name}.{window})")
    return result
