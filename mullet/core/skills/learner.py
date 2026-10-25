"""Learned skills — auto-extract reusable patterns from successful tasks."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mullet.core.paths import data_dir

logger = logging.getLogger(__name__)

# Tests point this at a temporary directory; None means learned_skills/ in
# the data directory.
SKILLS_DIR: Path | None = None


def _skills_dir() -> Path:
    return SKILLS_DIR or data_dir() / "learned_skills"


@dataclass
class LearnedSkill:
    """Represents a reusable skill extracted from successful tasks."""
    name: str
    pattern: str  # What the task looked like
    solution: str  # How it was solved (tool sequence, approach)
    success_count: int = 1
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize this skill to a dictionary."""
        return {
            "name": self.name, "pattern": self.pattern,
            "solution": self.solution, "success_count": self.success_count,
            "tags": self.tags or [],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LearnedSkill":
        """Deserialize a skill from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def extract_skill(task: str, tool_calls: list[str], result_summary: str) -> Optional[LearnedSkill]:
    """Extract a reusable skill pattern from a successful task execution."""
    if len(tool_calls) < 2:
        return None  # Too simple to be a pattern

    name = task[:40].replace(" ", "_").lower()
    pattern = task[:200]
    solution = " → ".join(tool_calls[:10])
    tags = []
    if any("file" in t for t in tool_calls):
        tags.append("file-ops")
    if any("terminal" in t for t in tool_calls):
        tags.append("terminal")
    if any("web" in t for t in tool_calls):
        tags.append("web")

    return LearnedSkill(name=name, pattern=pattern, solution=solution, tags=tags)


def save_skill(skill: LearnedSkill) -> Path:
    """Save a learned skill to disk."""
    skills_dir = _skills_dir()
    skills_dir.mkdir(parents=True, exist_ok=True)
    path = skills_dir / f"{skill.name}.json"

    # If exists, increment success count
    if path.exists():
        try:
            existing = LearnedSkill.from_dict(json.loads(path.read_text()))
            skill.success_count = existing.success_count + 1
        except Exception as _exc:
            logger.warning("learner: Error loading existing skill for update %s: %s", path, _exc)
            pass

    path.write_text(json.dumps(skill.to_dict(), indent=2))
    return path


def find_similar_skill(task: str) -> Optional[LearnedSkill]:
    """Find a previously learned skill that matches the current task."""
    if not _skills_dir().exists():
        return None

    task_words = set(task.lower().split())
    best_match = None
    best_score = 0

    for f in _skills_dir().glob("*.json"):
        try:
            skill = LearnedSkill.from_dict(json.loads(f.read_text()))
            pattern_words = set(skill.pattern.lower().split())
            if not pattern_words:
                continue
            overlap = len(task_words & pattern_words) / len(pattern_words)
            if overlap > best_score and overlap > 0.5:
                best_score = overlap
                best_match = skill
        except Exception:
            continue

    return best_match


def list_skills() -> list[LearnedSkill]:
    """List all learned skills."""
    if not _skills_dir().exists():
        return []
    skills = []
    for f in sorted(_skills_dir().glob("*.json")):
        try:
            skills.append(LearnedSkill.from_dict(json.loads(f.read_text())))
        except Exception:
            continue
    return skills
