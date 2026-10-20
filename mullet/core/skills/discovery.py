"""Legacy skill discovery walker — superseded by SkillLoader.

Kept for backwards compatibility with tests. Do not use in new code;
use mullet.core.skills.loader.SkillLoader instead.
"""
import os
from typing import List, Dict


class SkillDiscovery:
    """Walk a directory tree and return one entry per ``SKILL.md`` found."""

    def discover(self, root_dir: str) -> List[Dict[str, str]]:
        """Discover skills in the given root directory."""
        skills = []
        for dirpath, _dirnames, filenames in os.walk(root_dir):
            if "SKILL.md" in filenames:
                skill_path = os.path.join(dirpath, "SKILL.md")
                skill_name = os.path.basename(dirpath)
                triggers = self._extract_triggers(skill_path)
                skills.append({"name": skill_name, "path": skill_path, "triggers": triggers})
        return skills

    def _extract_triggers(self, skill_md_path: str) -> List[str]:
        triggers = []
        with open(skill_md_path) as f:
            for line in f:
                if line.startswith("Triggers:"):
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        triggers.extend([t.strip() for t in parts[1].split(",") if t.strip()])
        return triggers
