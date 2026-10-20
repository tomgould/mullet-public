"""Agent tool that lets the model install, list, and inspect skills at runtime."""
import json
import logging

from mullet.core.skills.manager import SkillManager

logger = logging.getLogger(__name__)

SKILL_MGR_SCHEMA = {
    "description": "Manage user-defined skills (create, edit, delete, list).",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "edit", "delete", "list"],
                "description": "Action to perform: create, edit, delete, or list skills.",
            },
            "name": {
                "type": "string",
                "description": "Name of the skill.",
            },
            "description": {
                "type": "string",
                "description": "Description of the skill.",
            },
            "body": {
                "type": "string",
                "description": "The content/code of the skill.",
            },
            "triggers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of trigger phrases for the skill.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of tags for the skill.",
            },
        },
        "required": ["action"],
    },
}

_skill_manager = SkillManager()


def skill_mgr_handler(**kwargs) -> str:
    """Tool handler for SkillManager operations."""
    action = kwargs.pop("action")
    name = kwargs.pop("name", None)
    description = kwargs.pop("description", None)
    body = kwargs.pop("body", None)
    triggers = kwargs.pop("triggers", None)
    tags = kwargs.pop("tags", None)

    result = {"success": False, "message": "Invalid action or parameters."}

    if action == "create":
        if name and description and body:
            result = _skill_manager.create(name, description, body, triggers, tags)
        else:
            result["message"] = "Name, description, and body are required for creating a skill."
    elif action == "edit":
        if name and (body or description or triggers or tags):
            result = _skill_manager.edit(name, body, description, triggers, tags)
        else:
            result["message"] = "Name and at least one field (body, description, triggers, tags) are required for editing a skill."
    elif action == "delete":
        if name:
            result = _skill_manager.delete(name)
        else:
            result["message"] = "Name is required for deleting a skill."
    elif action == "list":
        result = {"skills": _skill_manager.list_user(), "success": True}
    
    return json.dumps(result)


def register(registry):
    """Register skill_mgr tool into the tool registry."""
    registry.register(
        name="skill_mgr",
        func=skill_mgr_handler,
        schema=SKILL_MGR_SCHEMA,
        toolset="skills",
    )
