"""Skill system types."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SkillMetadata:
    """Parsed SKILL.md frontmatter."""
    name: str = ""
    description: str = ""
    version: str = "0.0.0"
    triggers: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    mode: str | None = None
    tier_override: str | None = None
    tags: list[str] = field(default_factory=list)
    platforms: list[str] = field(default_factory=list)
    required_env_vars: list[dict] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    pipeline: list[str] = field(default_factory=list)
    next_skill: str | None = None
    handoff: str | None = None


@dataclass
class Skill:
    """A loaded skill ready for injection."""
    metadata: SkillMetadata
    body: str
    path: Path
    scope: str = "builtin"


@dataclass
class SkillMatch:
    """Result of matching user input against skills."""
    skill: Skill
    confidence: float
    trigger: str
    match_type: str  # "slash_command" | "keyword" | "fuzzy"


@dataclass
class SkillInvocation:
    """Ready-to-inject skill content."""
    skill: Skill
    content: str
    mode: str | None = None
    tier_override: str | None = None
    unmet_requirements: list[str] = field(default_factory=list)
    next_skill: str | None = None
