"""Skill installation, discovery, and lifecycle management for the agent runtime."""
import logging
import re
import shutil
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class SkillManager:
    """Owns the on-disk skills tree (``skills/`` in the data directory) and ID lifecycle."""

    def __init__(self):
        self.skills_dir = self.get_skill_dir()

    def get_skill_dir(self) -> Path:
        """
        Returns the path to the user skills directory (skills/ in the data
        directory). Creates the directory if it doesn't exist.
        """
        from mullet.core.paths import data_dir
        skills_dir = data_dir() / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        return skills_dir

    def _validate_name(self, name: str) -> bool:
        """
        Validates the skill name (alphanumeric + hyphens only).
        """
        return bool(re.fullmatch(r"^[a-zA-Z0-9-]+$", name))

    def create(
        self,
        name: str,
        description: str,
        body: str,
        triggers: list = None,
        tags: list = None,
        mode: str = None,
    ) -> dict:
        """
        Creates a new user skill.
        """
        if not self._validate_name(name):
            return {"success": False, "message": "Invalid skill name. Use alphanumeric characters and hyphens only."}

        skill_path = self.skills_dir / name
        skill_path.mkdir(exist_ok=True)
        skill_file = skill_path / "SKILL.md"

        if skill_file.exists():
            return {"success": False, "message": f"Skill '{name}' already exists."}

        frontmatter = {
            "name": name,
            "description": description,
            "version": "1.0.0",
            "triggers": triggers if triggers is not None else [],
            "tags": tags if tags is not None else [],
            "mode": mode,
        }

        content = "---\n" + yaml.dump(frontmatter, sort_keys=False) + "---\n" + body
        skill_file.write_text(content)
        logger.info("Skill '%s' created at %s", name, skill_file)
        return {"success": True, "path": str(skill_file), "message": f"Skill '{name}' created."}

    def edit(
        self,
        name: str,
        body: str = None,
        description: str = None,
        triggers: list = None,
        tags: list = None,
    ) -> dict:
        """
        Edits an existing user skill.
        """
        skill_path = self.skills_dir / name
        skill_file = skill_path / "SKILL.md"

        if not skill_file.exists():
            return {"success": False, "message": f"Skill '{name}' not found."}

        current_content = skill_file.read_text()
        parts = current_content.split("---\n", 2)
        if len(parts) < 3:
            return {"success": False, "message": "Invalid SKILL.md format."}

        frontmatter = yaml.safe_load(parts[1])
        current_body = parts[2]

        if description is not None:
            frontmatter["description"] = description
        if triggers is not None:
            frontmatter["triggers"] = triggers
        if tags is not None:
            frontmatter["tags"] = tags
        if body is not None:
            current_body = body

        new_content = "---\n" + yaml.dump(frontmatter, sort_keys=False) + "---\n" + current_body
        skill_file.write_text(new_content)
        logger.info("Skill '%s' edited at %s", name, skill_file)
        return {"success": True, "path": str(skill_file), "message": f"Skill '{name}' updated."}

    def delete(self, name: str) -> dict:
        """
        Deletes a user skill.
        """
        skill_path = self.skills_dir / name

        if not skill_path.exists():
            return {"success": False, "message": f"Skill '{name}' not found."}

        try:
            shutil.rmtree(skill_path)
            logger.info("Skill '%s' deleted from %s", name, skill_path)
            return {"success": True, "message": f"Skill '{name}' deleted."}
        except OSError as e:
            return {"success": False, "message": f"Error deleting skill '{name}': {e}"}

    def list_user(self) -> list[dict]:
        """
        Lists all user-created skills.
        """
        skills = []
        for skill_dir in self.skills_dir.iterdir():
            if skill_dir.is_dir():
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    try:
                        content = skill_file.read_text()
                        parts = content.split("---\n", 2)
                        if len(parts) >= 3:
                            frontmatter = yaml.safe_load(parts[1])
                            skills.append(
                                {
                                    "name": frontmatter.get("name", skill_dir.name),
                                    "description": frontmatter.get("description", ""),
                                    "path": str(skill_file),
                                    "tags": frontmatter.get("tags", []),
                                }
                            )
                    except Exception as e:
                        logger.warning(f"Error reading skill file {skill_file}: {e}")
        return skills
