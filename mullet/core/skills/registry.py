"""Skill registry — lookup, trigger matching, index building."""

import logging
import re
from mullet.core.skills.types import Skill, SkillMatch, SkillInvocation
from mullet.core.skills.loader import SkillLoader

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Central registry for skill lookup and invocation.

    Integrates builtin skills, user skills, and learned skills.
    Uses SkillInvoker for full pipeline and requirement support.
    """

    def __init__(self, loader: SkillLoader | None = None):
        self._skills: dict[str, Skill] = {}
        self._aliases: dict[str, str] = {}
        self._loader = loader
        self._learned_manager = None

        # Initialize learned skills manager
        try:
            from mullet.core.skills.learned import LearnedSkillManager
            self._learned_manager = LearnedSkillManager()
        except Exception as e:
            logger.debug("Learned skill manager unavailable: %s", e)

    def load_all(self):
        """Load all skills from the loader and build indexes."""
        if not self._loader:
            return
        for skill in self._loader.scan():
            self.register(skill)

        # Also load learned skills
        if self._learned_manager:
            try:
                self._learned_manager.load()
            except Exception as e:
                logger.debug("Failed to load learned skills: %s", e)

    def register(self, skill: Skill):
        """Register a skill and build its trigger rules."""
        name = skill.metadata.name
        self._skills[name] = skill
        for alias in skill.metadata.aliases:
            self._aliases[alias] = name

    def get(self, name: str) -> Skill | None:
        """Look up a skill by name or alias."""
        canonical = self._aliases.get(name, name)
        return self._skills.get(canonical)

    def match(self, text: str) -> SkillMatch | None:
        """Match user input against registered triggers.

        Priority:
        1. Slash commands (/skill-name)
        2. Trigger keywords
        3. Name/alias substring match
        """
        stripped = text.strip()

        # Slash command
        if stripped.startswith("/"):
            cmd = stripped.split()[0][1:]  # Remove /
            skill = self.get(cmd)
            if skill:
                return SkillMatch(skill=skill, confidence=1.0, trigger=f"/{cmd}", match_type="slash_command")

        # Trigger keywords
        lower = stripped.lower()
        # Remove code blocks and URLs to avoid false positives
        clean = re.sub(r'```[\s\S]*?```', '', lower)
        clean = re.sub(r'https?://\S+', '', clean)

        # Filter informational intent — "what is autopilot?" should NOT trigger autopilot
        informational_patterns = [
            r'\bwhat\s+is\b', r'\bwhat\s+does\b', r'\bhow\s+does\b',
            r'\bexplain\b', r'\bdescribe\b', r'\btell\s+me\s+about\b',
        ]
        is_informational = any(re.search(p, clean) for p in informational_patterns)

        if not is_informational:
            # Sort skills by trigger length (longer = more specific = higher priority)
            all_matches = []
            for skill in self._skills.values():
                for trigger in skill.metadata.triggers:
                    if trigger.lower() in clean:
                        all_matches.append((len(trigger), trigger, skill))

            if all_matches:
                # Return the most specific (longest) match
                all_matches.sort(key=lambda x: -x[0])
                _, trigger, skill = all_matches[0]
                return SkillMatch(skill=skill, confidence=0.9, trigger=trigger, match_type="keyword")

        # Check learned skills as fallback
        if self._learned_manager:
            try:
                learned_matches = self._learned_manager.match(text)
                if learned_matches and learned_matches[0].confidence >= 70:
                    lm = learned_matches[0]
                    return SkillMatch(
                        skill=lm.skill.skill,
                        confidence=lm.confidence / 100,
                        trigger=", ".join(lm.matched_triggers),
                        match_type="learned",
                    )
            except Exception as e:
                logger.debug("Learned skill match failed: %s", e)

        return None

    def invoke(self, skill_name: str, user_args: str = "") -> SkillInvocation:
        """Prepare a skill for injection into the agent.

        Uses SkillInvoker for full pipeline support and requirement checking.
        """
        skill = self.get(skill_name)
        if not skill:
            placeholder = Skill(
                metadata=SkillMetadata(name=skill_name, description="NOT FOUND"),
                body="",
                path=Path("."),
            )
            return SkillInvocation(
                skill=placeholder,
                content=f"Error: skill '{skill_name}' not found. "
                        f"Available skills: {', '.join(sorted(self._skills.keys())) or 'none'}",
                unmet_requirements=[f"Skill '{skill_name}' is not registered"],
            )

        # Track invocation
        try:
            from mullet.core.skills.stats import record_invocation
            record_invocation(skill_name)
        except Exception as _exc:
            logger.warning("skills/registry.py: Failed to record skill invocation: %s", _exc)

        # Use the full SkillInvoker for pipeline and requirement support
        try:
            from mullet.core.skills.invoker import SkillInvoker
            invoker = SkillInvoker()
            return invoker.invoke(skill, user_args=user_args)
        except Exception as e:
            logger.debug("SkillInvoker failed, using simple invocation: %s", e)

        # Fallback: simple invocation
        # Check requirements
        import os
        unmet = []
        for req in skill.metadata.required_env_vars:
            var_name = req.get("name", "") if isinstance(req, dict) else req
            if not os.environ.get(var_name):
                unmet.append(f"Missing env var: {var_name}")

        # Build content
        parts = [skill.body]
        if user_args:
            parts.append(f"\n## User Arguments\n{user_args}")

        return SkillInvocation(
            skill=skill,
            content="\n\n".join(parts),
            mode=skill.metadata.mode,
            tier_override=skill.metadata.tier_override,
            unmet_requirements=unmet,
        )

    def list_skills(self, tags: list[str] | None = None) -> list[dict]:
        """List available skills with optional tag filtering."""
        results = []
        for skill in self._skills.values():
            if tags and not any(t in skill.metadata.tags for t in tags):
                continue
            results.append({
                "name": skill.metadata.name,
                "description": skill.metadata.description,
                "scope": skill.scope,
                "tags": skill.metadata.tags,
                "mode": skill.metadata.mode,
            })
        return results

    def build_index_prompt(self) -> str:
        """Build a skills index for system prompt injection."""
        if not self._skills:
            return ""

        # Group by tags
        groups: dict[str, list[Skill]] = {}
        for skill in self._skills.values():
            tag = skill.metadata.tags[0] if skill.metadata.tags else "general"
            groups.setdefault(tag, []).append(skill)

        lines = ["## Available Skills", ""]
        for tag, skills in sorted(groups.items()):
            lines.append(f"**{tag}:**")
            for s in skills:
                lines.append(f"  - /{s.metadata.name}: {s.metadata.description}")
            lines.append("")

        return "\n".join(lines)


# Need this import for the invoke method fallback
from pathlib import Path
from mullet.core.skills.types import SkillMetadata
