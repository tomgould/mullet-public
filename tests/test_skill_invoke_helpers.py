"""Tests for mullet.core.tools.skill_invoke pure helpers and schema."""

import json

from mullet.core.tools.skill_invoke import (
    SKILL_INVOKE_SCHEMA,
    skill_invoke_handler,
)


# ---------------------------------------------------------------------------
# SKILL_INVOKE_SCHEMA
# ---------------------------------------------------------------------------

class TestSkillInvokeSchema:
    def test_has_description(self):
        assert "description" in SKILL_INVOKE_SCHEMA
        assert len(SKILL_INVOKE_SCHEMA["description"]) > 0

    def test_has_parameters(self):
        assert "parameters" in SKILL_INVOKE_SCHEMA

    def test_action_is_required(self):
        required = SKILL_INVOKE_SCHEMA["parameters"].get("required", [])
        assert "action" in required

    def test_action_has_enum(self):
        props = SKILL_INVOKE_SCHEMA["parameters"]["properties"]
        assert "action" in props
        assert "enum" in props["action"]

    def test_enum_contains_expected_actions(self):
        enum = SKILL_INVOKE_SCHEMA["parameters"]["properties"]["action"]["enum"]
        assert "invoke" in enum
        assert "list" in enum
        assert "match" in enum

    def test_schema_is_dict(self):
        assert isinstance(SKILL_INVOKE_SCHEMA, dict)


# ---------------------------------------------------------------------------
# skill_invoke_handler — error paths (no real skill loading needed)
# ---------------------------------------------------------------------------

class TestSkillInvokeHandlerErrors:
    def test_invoke_without_skill_name_returns_error(self):
        result = skill_invoke_handler(action="invoke", skill_name=None)
        data = json.loads(result)
        assert "error" in data
        assert "skill_name" in data["error"]

    def test_match_without_text_returns_error(self):
        result = skill_invoke_handler(action="match", text=None)
        data = json.loads(result)
        assert "error" in data
        assert "text" in data["error"]

    def test_invalid_action_returns_error(self):
        result = skill_invoke_handler(action="bad_action")
        data = json.loads(result)
        assert "error" in data
        assert "bad_action" in data["error"]

    def test_returns_valid_json_string(self):
        result = skill_invoke_handler(action="invoke", skill_name=None)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_list_action_returns_skills_key(self):
        result = skill_invoke_handler(action="list")
        data = json.loads(result)
        assert "skills" in data

    def test_list_returns_list_type(self):
        result = skill_invoke_handler(action="list")
        data = json.loads(result)
        assert isinstance(data["skills"], list)
