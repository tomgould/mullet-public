"""Skill loader — discovers and parses SKILL.md files from directories."""

import logging
import sys
import yaml
from pathlib import Path

from mullet.core.skills.types import Skill, SkillMetadata

logger = logging.getLogger(__name__)

EXCLUDE_DIRS = {".git", ".github", "__pycache__", "node_modules", ".venv"}


class SkillLoader:
    """Discover and load SKILL.md files from multiple directories.

    Scan order (highest priority first):
    1. Project skills: .mullet/skills/
    2. User skills: ~/.mullet/skills/
    3. External dirs from config
    4. Built-in skills: package builtin/
    """

    def __init__(
        self,
        builtin_dir: Path | None = None,
        user_dir: Path | None = None,
        project_dir: Path | None = None,
        external_dirs: list[Path] | None = None,
        disabled: list[str] | None = None,
    ):
        self._builtin_dir = builtin_dir or (Path(__file__).parent / "builtin")
        self._user_dir = user_dir or (Path.home() / ".mullet" / "skills")
        self._project_dir = project_dir or Path(".mullet") / "skills"
        self._external_dirs = external_dirs or []
        self._disabled = set(disabled or [])

    def scan(self) -> list[Skill]:
        """Scan all directories and return loaded skills.

        Priority: project > user > external > builtin.
        Dedup: first occurrence of a name wins.
        """
        seen_names: set[str] = set()
        skills: list[Skill] = []

        scan_order = [
            (self._project_dir, "project"),
            (self._user_dir, "user"),
        ]
        for d in self._external_dirs:
            scan_order.append((Path(d), "external"))
        scan_order.append((self._builtin_dir, "builtin"))

        for directory, scope in scan_order:
            if not directory.exists():
                continue
            for skill_path in self._find_skill_files(directory):
                skill = self._load_skill(skill_path, scope)
                if skill and skill.metadata.name not in seen_names:
                    if skill.metadata.name in self._disabled:
                        logger.debug("Skipping disabled skill: %s", skill.metadata.name)
                        continue
                    if not self._matches_platform(skill.metadata.platforms):
                        continue
                    seen_names.add(skill.metadata.name)
                    skills.append(skill)

        return skills

    def _find_skill_files(self, directory: Path) -> list[Path]:
        """Recursively find SKILL.md files, sorted by path."""
        results = []
        for path in sorted(directory.rglob("SKILL.md")):
            if not any(part in EXCLUDE_DIRS for part in path.parts):
                results.append(path)
        return results

    def _load_skill(self, path: Path, scope: str) -> Skill | None:
        """Parse a single SKILL.md file."""
        # Security scan for non-builtin skills
        if scope != "builtin":
            try:
                from mullet.core.skills.guard import scan_skill
                result = scan_skill(path.parent)
                if not result["safe"]:
                    logger.warning(
                        "Skill %s failed security scan: %s",
                        path, [t["description"] for t in result["threats"]],
                    )
                    return None
            except Exception as e:
                logger.debug("Skill security scan skipped for %s: %s", path, e)
        try:
            content = path.read_text(encoding="utf-8")
            metadata_dict, body = parse_frontmatter(content)
            if not metadata_dict.get("name"):
                metadata_dict["name"] = path.parent.name

            metadata = SkillMetadata(
                name=metadata_dict.get("name", ""),
                description=metadata_dict.get("description", ""),
                version=metadata_dict.get("version", "0.0.0"),
                triggers=metadata_dict.get("triggers", []),
                aliases=metadata_dict.get("aliases", []),
                mode=metadata_dict.get("mode"),
                tier_override=metadata_dict.get("tier_override"),
                tags=metadata_dict.get("tags", []),
                platforms=metadata_dict.get("platforms", []),
                required_env_vars=metadata_dict.get("required_env_vars", []),
                required_tools=metadata_dict.get("required_tools", []),
                pipeline=metadata_dict.get("pipeline", []),
                next_skill=metadata_dict.get("next_skill"),
                handoff=metadata_dict.get("handoff"),
            )

            return Skill(metadata=metadata, body=body.strip(), path=path, scope=scope)
        except Exception as e:
            logger.warning("Failed to load skill %s: %s", path, e)
            return None

    def _matches_platform(self, platforms: list[str]) -> bool:
        """Check if skill is compatible with current OS."""
        if not platforms:
            return True
        current = sys.platform
        platform_map = {"macos": "darwin", "linux": "linux", "windows": "win32"}
        return any(platform_map.get(p, p) == current for p in platforms)


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from SKILL.md content.

    Returns (metadata_dict, body_str).
    """
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    try:
        metadata = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as e:
        logger.warning("Malformed SKILL.md YAML frontmatter: %s", e)
        metadata = {}

    body = parts[2]
    return metadata, body
