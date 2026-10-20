"""Skill invocation tool."""

import json
import logging
from typing import Any, Literal

from mullet.core.skills.registry import SkillRegistry
from mullet.core.skills.loader import SkillLoader

logger = logging.getLogger(__name__)

SKILL_INVOKE_SCHEMA = {
    "description": (
        "Invoke, list, or match skills. "
        "Use 'invoke' to run a skill, 'list' to see available skills, "
        "or 'match' to find a skill that matches given text."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["invoke", "list", "match"],
                "description": "Action to perform: invoke, list, or match.",
            },
            "skill_name": {
                "type": "string",
                "description": "Name of the skill to invoke.",
            },
            "user_args": {
                "type": "string",
                "description": "Arguments for the skill invocation.",
            },
            "text": {
                "type": "string",
                "description": "Text to match against skill triggers.",
            },
        },
        "required": ["action"],
    },
}


def skill_invoke_handler(
    action: Literal["invoke", "list", "match"],
    skill_name: str | None = None,
    user_args: str | None = None,
    text: str | None = None,
    **kwargs: Any,
) -> str:
    """Handle skill invocation, listing, and matching."""
    loader = SkillLoader()
    registry = SkillRegistry(loader=loader)
    registry.load_all()

    if action == "invoke":
        if not skill_name:
            return json.dumps({"error": "skill_name is required for 'invoke' action."})
        invocation = registry.invoke(skill_name, user_args or "")
        return json.dumps(
            {
                "skill_name": invocation.skill.metadata.name,
                "content": invocation.content,
                "unmet_requirements": invocation.unmet_requirements,
                "mode": invocation.mode,
                "tier_override": invocation.tier_override,
            }
        )
    elif action == "list":
        skills = registry.list_skills()
        return json.dumps({"skills": skills})
    elif action == "match":
        if not text:
            return json.dumps({"error": "text is required for 'match' action."})
        match_result = registry.match(text)
        if match_result:
            return json.dumps(
                {
                    "matched": True,
                    "skill_name": match_result.skill.metadata.name,
                    "confidence": match_result.confidence,
                    "trigger": match_result.trigger,
                    "match_type": match_result.match_type,
                }
            )
        else:
            return json.dumps({"matched": False})
    else:
        return json.dumps({"error": f"Invalid action: {action}"})


def register(registry: Any):
    """Register skill_invoke tool into the tool registry."""
    registry.register(
        name="skill_invoke",
        func=skill_invoke_handler,
        schema=SKILL_INVOKE_SCHEMA,
        toolset="skills",
    )
