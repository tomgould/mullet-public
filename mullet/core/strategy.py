"""The session strategy: this session's budget and how much work Mullet may delegate.

A strategy is set at the start of a Claude Code session (the
``session-strategy`` hook asks for it) with ``mullet strategy set``, and lasts
``session.strategy_ttl_hours``. While it is fresh:

- ``mullet job plan`` uses its budget as the job's spending cap when
  ``--max-spend`` is not given, and plans per its delegation choice;
- ``delegation: off`` turns delegation off exactly as ``MULLET_DELEGATION=off``
  does (see ``mullet.core.env.delegation_enabled``).

It is stored as JSON in the state directory, written atomically with mode
0600. The ``session:`` block of the Mullet config file holds its settings::

    session:
      ask_at_start: true        # false silences the session-strategy hook
      strategy_ttl_hours: 12    # how long a strategy stays in force

Bad values are reported in ``problems`` and ignored, as the other blocks are.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mullet.core import paths

DELEGATION_CHOICES = ("ladder", "claude-only", "off")

DELEGATION_MEANINGS = {
    "ladder": "cheap and free models first, Claude last (the job ladder)",
    "claude-only": "jobs run on Claude Code alone, no cheaper rungs",
    "off": "nothing is delegated",
}

DEFAULT_ASK_AT_START = True
DEFAULT_TTL_HOURS = 12.0

STRATEGY_OFF_MESSAGE = (
    "Delegation is turned off for this session (mullet strategy: delegation off). "
    "Run `mullet strategy set --delegation ladder` to change it."
)

# A strategy set a little in the future (clock adjustments) still counts.
_CLOCK_SLACK_S = 60.0


@dataclass(frozen=True)
class Strategy:
    """One session's budget in dollars (None for no budget) and delegation choice."""

    budget_usd: float | None
    delegation: str
    set_at: float
    note: str = ""

    def expires_at(self, ttl_hours: float) -> float:
        """When this strategy stops counting."""
        return self.set_at + ttl_hours * 3600.0

    def describe(self) -> str:
        """One line for people: budget, delegation and note."""
        budget = "no budget" if self.budget_usd is None else f"budget ${self.budget_usd:.2f}"
        text = f"{budget}, delegation {self.delegation} ({DELEGATION_MEANINGS[self.delegation]})"
        return text + (f"; note: {self.note}" if self.note else "")


@dataclass
class SessionSettings:
    """The ``session:`` settings, with defaults filled in."""

    ask_at_start: bool = DEFAULT_ASK_AT_START
    strategy_ttl_hours: float = DEFAULT_TTL_HOURS
    problems: list[str] = field(default_factory=list)


def parse_settings(data: Any, source: str = "session") -> SessionSettings:
    """Read the raw ``session:`` mapping; a bad value is reported and its default kept."""
    settings = SessionSettings()
    if data is None:
        return settings
    if not isinstance(data, dict):
        settings.problems.append(f"{source}: expected a mapping; ignored")
        return settings
    for key in data:
        if key not in ("ask_at_start", "strategy_ttl_hours"):
            settings.problems.append(f"{source}.{key}: unknown setting; ignored")
    ask = data.get("ask_at_start")
    if ask is not None:
        if isinstance(ask, bool):
            settings.ask_at_start = ask
        else:
            settings.problems.append(f"{source}.ask_at_start: expected true or false; ignored")
    ttl = data.get("strategy_ttl_hours")
    if ttl is not None:
        if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or not 0 < ttl <= 24 * 30:
            settings.problems.append(f"{source}.strategy_ttl_hours: expected hours above 0 "
                                     "and at most 720; ignored")
        else:
            settings.strategy_ttl_hours = float(ttl)
    return settings


def settings_path() -> Path:
    """The user's Mullet config file, which holds the ``session:`` block."""
    return paths.config_file("config.yaml")


def load_settings(path: Path | None = None) -> SessionSettings:
    """The ``session:`` settings from the user's config file; defaults when there are none."""
    target = path or settings_path()
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return SessionSettings()
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        settings = SessionSettings()
        settings.problems.append(f"{target}: unreadable ({exc}); the session settings use their defaults")
        return settings
    return parse_settings(data.get("session") if isinstance(data, dict) else None, f"{target} session")


def strategy_path() -> Path:
    """Where the strategy is kept: ``strategy.json`` in the state directory."""
    return paths.state_dir() / "strategy.json"


def _from_dict(data: Any) -> Strategy | None:
    if not isinstance(data, dict):
        return None
    delegation = data.get("delegation")
    budget = data.get("budget_usd")
    set_at = data.get("set_at")
    note = data.get("note", "")
    if delegation not in DELEGATION_CHOICES:
        return None
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, (int, float)) or budget < 0):
        return None
    if isinstance(set_at, bool) or not isinstance(set_at, (int, float)):
        return None
    return Strategy(None if budget is None else float(budget), delegation, float(set_at),
                    note if isinstance(note, str) else "")


def read(path: Path | None = None) -> Strategy | None:
    """The stored strategy whatever its age, or None when missing or unreadable."""
    try:
        data = json.loads((path or strategy_path()).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return _from_dict(data)


def is_fresh(strategy: Strategy, now: float, ttl_hours: float) -> bool:
    """True while `strategy` is younger than `ttl_hours`."""
    return strategy.set_at - _CLOCK_SLACK_S <= now < strategy.expires_at(ttl_hours)


def load(now: float | None = None, *, ttl_hours: float | None = None,
         path: Path | None = None) -> Strategy | None:
    """The strategy in force, or None when missing, unreadable or older than the TTL."""
    strategy = read(path)
    if strategy is None:
        return None
    ttl = load_settings().strategy_ttl_hours if ttl_hours is None else ttl_hours
    return strategy if is_fresh(strategy, time.time() if now is None else now, ttl) else None


def save(strategy: Strategy, path: Path | None = None) -> Path:
    """Write `strategy` atomically with mode 0600 and return its path."""
    if strategy.delegation not in DELEGATION_CHOICES:
        raise ValueError(f"delegation must be one of {', '.join(DELEGATION_CHOICES)}")
    if strategy.budget_usd is not None and strategy.budget_usd < 0:
        raise ValueError("the budget must be 0 or more")
    target = path or strategy_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".strategy-", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(asdict(strategy), handle)
        os.chmod(tmp, 0o600)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


def clear(path: Path | None = None) -> bool:
    """Remove the stored strategy; True when there was one."""
    try:
        (path or strategy_path()).unlink()
    except FileNotFoundError:
        return False
    return True


def delegation_off(now: float | None = None) -> bool:
    """True when a fresh strategy says ``delegation: off``; any error counts as no."""
    try:
        strategy = load(now)
    except Exception:
        return False
    return strategy is not None and strategy.delegation == "off"
