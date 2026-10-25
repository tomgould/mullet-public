"""The ``jobs:`` settings in the Mullet config file: how a job run treats slow or silent steps.

::

    jobs:
      reask_attempts: 1      # times to ask again, in the same session, for a reply that lacks its
                             # structured part (0 turns it off)
      idle_minutes: null     # minutes of silence from a worker before it is stopped; null derives
                             # it from the profile's longest check timeout
      grace_seconds: 60      # seconds a worker may take to exit after it reports it is finished
                             # (at least 5, so the CLI can save its session)
      git_timeout_seconds: 600  # longest a single git command on a job clone may run
      commit_body: full      # full: a "Mullet step" line and the quoted worker claim under the
                             # subject; minimal: the subject alone

A bad value (including `.inf` and `.nan`) is reported and its default kept.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mullet.core import paths

DEFAULT_REASK_ATTEMPTS = 1
DEFAULT_GRACE_SECONDS = 60.0
# The CLI saves its session file after its final reply; a shorter grace could
# kill it before it does, and a resumed attempt would then have nothing to resume.
MIN_GRACE_SECONDS = 5
DEFAULT_GIT_TIMEOUT_SECONDS = 600.0
# full: the subject, a "Mullet step ... passed gates" line and the quoted worker
# claim; minimal: the subject alone, for repos whose rules forbid tool lines.
COMMIT_BODIES = ("full", "minimal")
KEYS = ("reask_attempts", "idle_minutes", "grace_seconds", "git_timeout_seconds", "commit_body")


@dataclass
class JobSettings:
    """The ``jobs:`` settings, with defaults filled in."""

    reask_attempts: int = DEFAULT_REASK_ATTEMPTS
    idle_minutes: float | None = None  # None: derived from the profile at run time
    grace_seconds: float = DEFAULT_GRACE_SECONDS
    git_timeout_seconds: float = DEFAULT_GIT_TIMEOUT_SECONDS
    commit_body: str = "full"
    problems: list[str] = field(default_factory=list)


def _number(value: Any) -> float | None:
    """A finite number, or None: YAML's `.inf` and `.nan` are floats, and `int()` raises on them."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:  # an int too large for a float
        return None
    return number if math.isfinite(number) else None


def parse_settings(data: Any, source: str = "jobs") -> JobSettings:
    """Read the raw ``jobs:`` mapping; a bad value is reported and its default kept."""
    settings = JobSettings()
    if data is None:
        return settings
    if not isinstance(data, dict):
        settings.problems.append(f"{source}: expected a mapping; ignored")
        return settings
    for key in data:
        if key not in KEYS:
            settings.problems.append(f"{source}.{key}: unknown setting; ignored")
    value = data.get("reask_attempts")
    if value is not None:
        number = _number(value)
        if number is None or number != int(number) or not 0 <= number <= 5:
            settings.problems.append(f"{source}.reask_attempts: expected a whole number from 0 to 5; ignored")
        else:
            settings.reask_attempts = int(number)
    if data.get("idle_minutes") is not None:
        number = _number(data["idle_minutes"])
        if number is None or not 0 < number <= 24 * 60:
            settings.problems.append(f"{source}.idle_minutes: expected minutes above 0 and at most 1440, "
                                     "or null; ignored")
        else:
            settings.idle_minutes = number
    for key, low, high in (("grace_seconds", MIN_GRACE_SECONDS, 3600), ("git_timeout_seconds", 1, 86400)):
        if data.get(key) is None:
            continue
        number = _number(data[key])
        if number is None or not low <= number <= high:
            settings.problems.append(f"{source}.{key}: expected seconds from {low} to {high}; ignored")
        else:
            setattr(settings, key, number)
    value = data.get("commit_body")
    if value is not None:
        if value not in COMMIT_BODIES:
            settings.problems.append(f"{source}.commit_body: expected {' or '.join(COMMIT_BODIES)}; ignored")
        else:
            settings.commit_body = value
    return settings


def settings_path() -> Path:
    """The user's Mullet config file, which holds the ``jobs:`` block."""
    return paths.config_file("config.yaml")


def load_settings(path: Path | None = None) -> JobSettings:
    """The ``jobs:`` settings from the user's config file; defaults when there are none.

    It never raises: every git command on a job clone calls it, so a value it
    can't use must leave the defaults in place, not stop the job.
    """
    target = path or settings_path()
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return JobSettings()
    except Exception as exc:  # unreadable, undecodable or bad YAML
        settings = JobSettings()
        settings.problems.append(f"{target}: unreadable ({exc}); the jobs settings use their defaults")
        return settings
    try:
        return parse_settings(data.get("jobs") if isinstance(data, dict) else None, f"{target} jobs")
    except Exception as exc:
        settings = JobSettings()
        settings.problems.append(f"{target} jobs: couldn't be read ({type(exc).__name__}); defaults used")
        return settings
