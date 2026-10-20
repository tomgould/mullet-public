"""Tests for skill manager tool."""
import json
from unittest.mock import patch
from mullet.core.tools.skill_mgr import skill_mgr_handler


@patch("mullet.core.tools.skill_mgr._skill_manager")
def test_create_skill(mock_mgr):
    mock_mgr.create.return_value = {"success": True}
    result = json.loads(skill_mgr_handler(action="create", name="test", description="desc", body="code"))
    assert result["success"] is True
    mock_mgr.create.assert_called_once()


@patch("mullet.core.tools.skill_mgr._skill_manager")
def test_list_skills(mock_mgr):
    mock_mgr.list_user.return_value = ["skill1", "skill2"]
    result = json.loads(skill_mgr_handler(action="list"))
    assert result["success"] is True
    assert len(result["skills"]) == 2


def test_create_missing_fields():
    result = json.loads(skill_mgr_handler(action="create", name="x"))
    assert result["success"] is False


def test_delete_missing_name():
    result = json.loads(skill_mgr_handler(action="delete"))
    assert result["success"] is False
