"""Learned skills — lightweight pattern-based skills persisted as JSON.

A learned skill is extracted from observed usage patterns and matched
against user text via trigger phrases.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LearnedSkill:
    """A skill learned from observed patterns."""
    name: str
    description: str
    triggers: list[str] = field(default_factory=list)
    pattern: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    use_count: int = 0

    def to_dict(self) -> dict:
        """Serialize the skill to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "triggers": self.triggers,
            "pattern": self.pattern,
            "created_at": self.created_at.isoformat(),
            "use_count": self.use_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LearnedSkill:
        """Deserialize a skill from a dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            triggers=data.get("triggers", []),
            pattern=data.get("pattern", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            use_count=data.get("use_count", 0),
        )


class LearnedSkillManager:
    """Manage learned skills stored in a single JSON file."""

    def __init__(self, skills_dir: Path | None = None) -> None:
        self._skills_dir = skills_dir or Path.home() / ".mullet"
        self._path = self._skills_dir / "learned_skills.json"
        self._skills: list[LearnedSkill] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._skills = [LearnedSkill.from_dict(item) for item in data.get("skills", [])]
        except (json.JSONDecodeError, OSError, KeyError, ValueError) as e:
            logger.warning("Failed to load learned skills from %s: %s", self._path, e)
            self._skills = []

    def _save(self) -> None:
        """Persist skills to JSON."""
        self._skills_dir.mkdir(parents=True, exist_ok=True)
        payload = {"skills": [s.to_dict() for s in self._skills]}
        try:
            self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as e:
            logger.warning("Failed to save learned skills to %s: %s", self._path, e)

    def record(self, pattern: str, context: str) -> LearnedSkill:
        """Extract a skill from an observed pattern and persist it."""
        name = pattern.strip().replace(" ", "_").lower()[:64]
        skill = LearnedSkill(
            name=name,
            description=context[:200],
            triggers=[pattern.lower()],
            pattern=pattern,
            created_at=datetime.utcnow(),
            use_count=0,
        )
        self._skills.append(skill)
        self._save()
        return skill

    def match(self, text: str) -> LearnedSkill | None:
        """Check text against learned triggers. Returns first match or None."""
        lower = text.lower()
        for skill in self._skills:
            for trigger in skill.triggers:
                if trigger in lower:
                    skill.use_count += 1
                    self._save()
                    return skill
        return None

    def list(self) -> list[LearnedSkill]:
        """Return all learned skills."""
        return list(self._skills)
