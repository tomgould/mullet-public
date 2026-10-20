import json
import pytest
from unittest.mock import Mock, patch

from mullet.core.tools import skill_invoke
from mullet.core.skills.registry import SkillRegistry
from mullet.core.skills.loader import SkillLoader


@pytest.fixture
def mock_registry():
    return Mock()


@pytest.fixture
def mock_skill_loader():
    return Mock(spec=SkillLoader)


@pytest.fixture
def mock_skill_registry(mock_skill_loader):
    # Patch the SkillRegistry and SkillLoader instantiation within the handler
    with (
        patch('mullet.core.tools.skill_invoke.SkillLoader', return_value=mock_skill_loader),
        patch('mullet.core.tools.skill_invoke.SkillRegistry') as MockRegistryClass
    ):
        mock_registry_instance = Mock(spec=SkillRegistry)
        mock_registry_instance.load_all.return_value = None # Mock load_all to do nothing
        MockRegistryClass.return_value = mock_registry_instance
        yield mock_registry_instance


def test_skill_invoke_schema_has_required_fields():
    schema = skill_invoke.SKILL_INVOKE_SCHEMA
    assert "action" in schema["parameters"]["properties"]
    assert "action" in schema["parameters"]["required"]
    assert schema["parameters"]["properties"]["action"]["enum"] == ["invoke", "list", "match"]


def test_skill_invoke_register_with_mock_registry(mock_registry):
    skill_invoke.register(mock_registry)
    mock_registry.register.assert_called_once_with(
        name="skill_invoke",
        func=skill_invoke.skill_invoke_handler,
        schema=skill_invoke.SKILL_INVOKE_SCHEMA,
        toolset="skills",
    )


def test_skill_invoke_list_action_returns_json(mock_skill_registry):
    mock_skill_registry.list_skills.return_value = [{"name": "test_skill", "description": "A test skill"}]
    result = skill_invoke.skill_invoke_handler(action="list")
    data = json.loads(result)
    assert "skills" in data
    assert isinstance(data["skills"], list)
    assert data["skills"][0]["name"] == "test_skill"


def test_skill_invoke_match_action_with_no_match_returns_null_empty(mock_skill_registry):
    mock_skill_registry.match.return_value = None
    result = skill_invoke.skill_invoke_handler(action="match", text="no matching skill")
    data = json.loads(result)
    assert "matched" in data
    assert data["matched"] is False


def test_skill_invoke_match_action_with_match_returns_json(mock_skill_registry):
    mock_match_result = Mock()
    mock_match_result.skill.metadata.name = "matched_skill"
    mock_match_result.confidence = 0.9
    mock_match_result.trigger = "trigger phrase"
    mock_match_result.match_type = "exact"
    mock_skill_registry.match.return_value = mock_match_result

    result = skill_invoke.skill_invoke_handler(action="match", text="matching skill")
    data = json.loads(result)
    assert data["matched"] is True
    assert data["skill_name"] == "matched_skill"
    assert data["confidence"] == 0.9
    assert data["trigger"] == "trigger phrase"
    assert data["match_type"] == "exact"


def test_skill_invoke_invoke_action_returns_json(mock_skill_registry):
    mock_invocation_result = Mock()
    mock_invocation_result.skill.metadata.name = "invoked_skill"
    mock_invocation_result.content = "skill output"
    mock_invocation_result.unmet_requirements = []
    mock_invocation_result.mode = "sync"
    mock_invocation_result.tier_override = None
    mock_skill_registry.invoke.return_value = mock_invocation_result

    result = skill_invoke.skill_invoke_handler(action="invoke", skill_name="test_skill", user_args="arg1")
    data = json.loads(result)
    assert data["skill_name"] == "invoked_skill"
    assert data["content"] == "skill output"
    assert data["unmet_requirements"] == []
    assert data["mode"] == "sync"
    assert data["tier_override"] is None
    mock_skill_registry.invoke.assert_called_once_with("test_skill", "arg1")


def test_skill_invoke_invoke_action_missing_skill_name():
    result = skill_invoke.skill_invoke_handler(action="invoke")
    data = json.loads(result)
    assert "error" in data
    assert data["error"] == "skill_name is required for 'invoke' action."
