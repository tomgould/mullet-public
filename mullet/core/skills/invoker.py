"""Skill invoker — prepare skills for injection with pipeline support.

Handles requirement checking, pipeline context building, and
handoff artifact tracking.
"""

import logging
import os

from mullet.core.skills.types import Skill, SkillInvocation

logger = logging.getLogger(__name__)


class SkillInvoker:
    """Invoke skills with full pipeline and requirement support."""

    def invoke(
        self,
        skill: Skill,
        user_args: str = "",
        available_tools: list[str] | None = None,
    ) -> SkillInvocation:
        """Prepare a skill for injection into the agent."""
        # Check requirements
        unmet = self.check_requirements(skill)

        # Build content
        parts = [skill.body]

        # Add pipeline context if applicable
        pipeline_ctx = self.build_pipeline_context(skill)
        if pipeline_ctx:
            parts.append(pipeline_ctx)

        # Add runtime guidance
        parts.append(generate_runtime_guidance(skill))

        # Add user arguments
        if user_args:
            parts.append(f"\n## User Arguments\n{user_args}")

        # Add requirement warnings
        if unmet:
            warnings = "\n".join(f"  - {r}" for r in unmet)
            parts.append(f"\n## Warnings\nSome requirements are not met:\n{warnings}")

        return SkillInvocation(
            skill=skill,
            content="\n\n".join(parts),
            mode=skill.metadata.mode,
            tier_override=skill.metadata.tier_override,
            unmet_requirements=unmet,
            next_skill=skill.metadata.next_skill,
        )

    def check_requirements(self, skill: Skill) -> list[str]:
        """Check if skill requirements are met. Returns list of unmet requirements."""
        unmet = []

        for req in skill.metadata.required_env_vars:
            var_name = req.get("name", "") if isinstance(req, dict) else req
            if not os.environ.get(var_name):
                help_text = req.get("help", "") if isinstance(req, dict) else ""
                unmet.append(f"Missing env var: {var_name}" + (f" ({help_text})" if help_text else ""))

        # Tool availability is checked at runtime, not here
        return unmet

    def build_pipeline_context(self, skill: Skill) -> str | None:
        """Build pipeline guidance if skill is part of a pipeline."""
        meta = skill.metadata
        if not meta.pipeline and not meta.next_skill:
            return None

        lines = ["## Skill Pipeline"]

        if meta.pipeline:
            pipeline_str = " → ".join(meta.pipeline)
            # Find current position
            current_idx = -1
            for i, step in enumerate(meta.pipeline):
                if step == meta.name:
                    current_idx = i
                    break
            lines.append(f"Pipeline: `{pipeline_str}`")
            if current_idx >= 0:
                lines.append(f"Current stage: `{meta.name}` ({current_idx + 1}/{len(meta.pipeline)})")

        if meta.next_skill:
            lines.append(f"Next skill: `{meta.next_skill}`")
            lines.append("")
            lines.append("When this stage completes:")
            if meta.handoff:
                lines.append(f"1. Write or update the handoff artifact at `{meta.handoff}`")
            lines.append("2. Carry forward decisions made and remaining risks")
            lines.append(f"3. The next skill (`{meta.next_skill}`) will be invoked automatically")

        return "\n".join(lines)


def resolve_handoff_artifact(pattern: str) -> str | None:
    """Find the most recent file matching a handoff glob pattern."""
    from glob import glob
    matches = sorted(glob(pattern), key=os.path.getmtime, reverse=True)
    return matches[0] if matches else None


def generate_runtime_guidance(skill: Skill) -> str:
    """Generate a guidance string from the skill's metadata."""
    meta = skill.metadata
    lines = ["## Skill Runtime Guidance"]

    if meta.tier_override:
        lines.append(f"Tier: {meta.tier_override}")
    if meta.required_tools:
        lines.append(f"Required tools: {', '.join(meta.required_tools)}")
    if meta.required_env_vars:
        env_vars = ", ".join([e.get("name", "") if isinstance(e, dict) else e for e in meta.required_env_vars])
        lines.append(f"Required environment variables: {env_vars}")
    if meta.platforms:
        lines.append(f"Platforms: {', '.join(meta.platforms)}")

    if len(lines) == 1: # Only the header
        return ""

    return "\n".join(lines)
