
import json
import os
import pytest
from unittest.mock import Mock, patch

# Mock SkillManager for skill_mgr tests
class MockSkillManager:
    def __init__(self):
        self.skills = {}

    def create(self, name, description, body, triggers, tags):
        if name in self.skills:
            return {"success": False, "message": "Skill already exists."}
        self.skills[name] = {"description": description, "body": body, "triggers": triggers, "tags": tags}
        return {"success": True, "message": f"Skill '{name}' created."}

    def edit(self, name, body=None, description=None, triggers=None, tags=None):
        if name not in self.skills:
            return {"success": False, "message": "Skill not found."}
        if body:
            self.skills[name]["body"] = body
        if description:
            self.skills[name]["description"] = description
        if triggers:
            self.skills[name]["triggers"] = triggers
        if tags:
            self.skills[name]["tags"] = tags
        return {"success": True, "message": f"Skill '{name}' updated."}

    def delete(self, name):
        if name not in self.skills:
            return {"success": False, "message": "Skill not found."}
        del self.skills[name]
        return {"success": True, "message": f"Skill '{name}' deleted."}

    def list_user(self):
        return [{"name": name, **data} for name, data in self.skills.items()]

@pytest.fixture
def mock_skill_manager_instance():
    with patch('mullet.core.tools.skill_mgr._skill_manager', new_callable=MockSkillManager) as mock_mgr:
        yield mock_mgr

# Skill Manager Tests
def test_skill_mgr_schema_required_action():
    from mullet.core.tools.skill_mgr import SKILL_MGR_SCHEMA
    assert "action" in SKILL_MGR_SCHEMA["parameters"]["required"]

def test_skill_mgr_register(mock_skill_manager_instance):
    from mullet.core.tools import skill_mgr
    mock_registry = Mock()
    skill_mgr.register(mock_registry)
    mock_registry.register.assert_called_once()
    args, kwargs = mock_registry.register.call_args
    assert kwargs["name"] == "skill_mgr"
    assert kwargs["func"] == skill_mgr.skill_mgr_handler
    assert "schema" in kwargs

def test_skill_mgr_list_action(mock_skill_manager_instance):
    from mullet.core.tools.skill_mgr import skill_mgr_handler
    mock_skill_manager_instance.create("test_skill", "desc", "body", [], [])
    result = skill_mgr_handler(action="list")
    data = json.loads(result)
    assert data["success"] is True
    assert isinstance(data["skills"], list)
    assert len(data["skills"]) == 1
    assert data["skills"][0]["name"] == "test_skill"

def test_skill_mgr_create_action(tmp_path, mock_skill_manager_instance):
    from mullet.core.tools.skill_mgr import skill_mgr_handler
    # Temporarily override skills_dir for testing
    original_skills_dir = mock_skill_manager_instance.skills_dir if hasattr(mock_skill_manager_instance, 'skills_dir') else None
    mock_skill_manager_instance.skills_dir = tmp_path
    
    skill_name = "test_create_skill"
    skill_description = "A skill for testing create."
    skill_body = "print('Hello from test skill!')"
    result = skill_mgr_handler(action="create", name=skill_name, description=skill_description, body=skill_body)
    data = json.loads(result)
    assert data["success"] is True
    assert data["message"] == f"Skill '{skill_name}' created."
    assert skill_name in mock_skill_manager_instance.skills
    assert mock_skill_manager_instance.skills[skill_name]["description"] == skill_description
    assert mock_skill_manager_instance.skills[skill_name]["body"] == skill_body

    # Restore original skills_dir if it existed
    if original_skills_dir:
        mock_skill_manager_instance.skills_dir = original_skills_dir


# Notepad Tests
@pytest.fixture
def notepad_setup(tmp_path):
    # Temporarily override NOTEPAD_DIR for testing
    with patch('mullet.core.tools.notepad.NOTEPAD_DIR', str(tmp_path)):
        # Ensure the directory exists (mullet's notepad tool creates it)
        os.makedirs(tmp_path, exist_ok=True)
        yield
        # Clean up files created during tests
        for f in os.listdir(tmp_path):
            os.remove(os.path.join(tmp_path, f))


def test_notepad_write_read_roundtrip(notepad_setup):
    from mullet.core.tools.notepad import notepad_handler
    key = "test_note_1"
    content = "This is a test note."
    
    write_result = notepad_handler(action="write", key=key, content=content)
    write_data = json.loads(write_result)
    assert write_data["status"] == "success"
    assert write_data["action"] == "write"
    assert write_data["key"] == key

    read_result = notepad_handler(action="read", key=key)
    read_data = json.loads(read_result)
    assert read_data["status"] == "success"
    assert read_data["action"] == "read"
    assert read_data["key"] == key
    assert read_data["content"] == content

def test_notepad_append(notepad_setup):
    from mullet.core.tools.notepad import notepad_handler
    key = "test_note_append"
    initial_content = "First line.\n"
    append_content = "Second line."

    notepad_handler(action="write", key=key, content=initial_content)
    append_result = notepad_handler(action="append", key=key, content=append_content)
    append_data = json.loads(append_result)
    assert append_data["status"] == "success"
    assert append_data["action"] == "append"

    read_result = notepad_handler(action="read", key=key)
    read_data = json.loads(read_result)
    assert read_data["content"] == initial_content + append_content

def test_notepad_list_keys(notepad_setup):
    from mullet.core.tools.notepad import notepad_handler
    notepad_handler(action="write", key="key_a", content="content_a")
    notepad_handler(action="write", key="key_b", content="content_b")

    list_result = notepad_handler(action="list")
    list_data = json.loads(list_result)
    assert list_data["status"] == "success"
    assert list_data["action"] == "list"
    assert "key_a" in list_data["keys"]
    assert "key_b" in list_data["keys"]
    assert len(list_data["keys"]) >= 2 # May contain default if not cleared

def test_notepad_clear(notepad_setup):
    from mullet.core.tools.notepad import notepad_handler
    key = "test_note_clear"
    notepad_handler(action="write", key=key, content="content to clear")

    clear_result = notepad_handler(action="clear", key=key)
    clear_data = json.loads(clear_result)
    assert clear_data["status"] == "success"
    assert clear_data["action"] == "clear"

    read_result = notepad_handler(action="read", key=key)
    read_data = json.loads(read_result)
    assert read_data["content"] == ""

def test_notepad_read_nonexistent(notepad_setup):
    from mullet.core.tools.notepad import notepad_handler
    key = "nonexistent_key"
    read_result = notepad_handler(action="read", key=key)
    read_data = json.loads(read_result)
    assert read_data["status"] == "success"
    assert read_data["action"] == "read"
    assert read_data["key"] == key
    assert read_data["content"] == ""

def test_notepad_write_no_content_error(notepad_setup):
    from mullet.core.tools.notepad import notepad_handler
    key = "error_key"
    result = notepad_handler(action="write", key=key)
    data = json.loads(result)
    assert "error" in data
    assert "Content is required" in data["error"]

def test_notepad_append_no_content_error(notepad_setup):
    from mullet.core.tools.notepad import notepad_handler
    key = "error_append_key"
    result = notepad_handler(action="append", key=key)
    data = json.loads(result)
    assert "error" in data
    assert "Content is required" in data["error"]
