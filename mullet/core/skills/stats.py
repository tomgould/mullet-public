"""Skill invocation and outcome tracking.

Stats persisted to skill_stats.json in the state directory.
Invocations are tracked automatically via SkillRegistry.invoke().
Outcomes require explicit record_outcome() calls at the use site.
"""

import json
import logging
import time
from pathlib import Path

from mullet.core.paths import state_dir

logger = logging.getLogger(__name__)

# Tests point this at a temporary file; None means skill_stats.json in the
# state directory.
_STATS_PATH: Path | None = None


def _stats_path() -> Path:
    return _STATS_PATH or state_dir() / "skill_stats.json"


def _load() -> dict:
    try:
        return json.loads(_stats_path().read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(data: dict) -> None:
    path = _stats_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _entry(data: dict, skill_name: str) -> dict:
    if skill_name not in data:
        data[skill_name] = {
            "invocations": 0,
            "successes": 0,
            "failures": 0,
            "last_invoked": None,
            "last_outcome": None,
        }
    return data[skill_name]


def record_invocation(skill_name: str) -> None:
    """Record that a skill was invoked. Called automatically by SkillRegistry."""
    try:
        data = _load()
        e = _entry(data, skill_name)
        e["invocations"] += 1
        e["last_invoked"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _save(data)
    except Exception as exc:
        logger.debug("skill_stats: failed to record invocation for %s: %s", skill_name, exc)


def record_outcome(skill_name: str, success: bool) -> None:
    """Record the outcome of a skill invocation. Called by the use site after execution."""
    try:
        data = _load()
        e = _entry(data, skill_name)
        if success:
            e["successes"] += 1
        else:
            e["failures"] += 1
        e["last_outcome"] = "success" if success else "failure"
        _save(data)
    except Exception as exc:
        logger.debug("skill_stats: failed to record outcome for %s: %s", skill_name, exc)


def get_stats(skill_name: str | None = None) -> dict:
    """Return stats for one skill or all skills."""
    data = _load()
    if skill_name:
        return data.get(skill_name, {})
    return data


def success_rate(skill_name: str) -> float | None:
    """Return success rate [0.0, 1.0] for a skill, or None if no outcome data."""
    e = get_stats(skill_name)
    total_outcomes = e.get("successes", 0) + e.get("failures", 0)
    if total_outcomes == 0:
        return None
    return e["successes"] / total_outcomes


def top_skills(n: int = 10) -> list[dict]:
    """Return the N most-invoked skills, sorted by invocation count."""
    data = _load()
    rows = [
        {
            "name": name,
            "invocations": e.get("invocations", 0),
            "success_rate": success_rate(name),
            "last_invoked": e.get("last_invoked"),
        }
        for name, e in data.items()
    ]
    return sorted(rows, key=lambda r: r["invocations"], reverse=True)[:n]
