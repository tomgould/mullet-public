"""Tests for notepad tool."""
import json
import os
import pytest
from unittest.mock import patch

from mullet.core.tools.notepad import notepad_handler


@patch("mullet.core.tools.notepad.NOTEPAD_DIR", "/tmp/mullet-test-notepad")
def test_write_and_read():
    os.makedirs("/tmp/mullet-test-notepad", exist_ok=True)
    notepad_handler("write", content="hello", key="test1")
    result = json.loads(notepad_handler("read", key="test1"))
    assert result["content"] == "hello"
    os.remove("/tmp/mullet-test-notepad/test1.md")


@patch("mullet.core.tools.notepad.NOTEPAD_DIR", "/tmp/mullet-test-notepad")
def test_append():
    os.makedirs("/tmp/mullet-test-notepad", exist_ok=True)
    notepad_handler("write", content="a", key="test2")
    notepad_handler("append", content="b", key="test2")
    result = json.loads(notepad_handler("read", key="test2"))
    assert result["content"] == "ab"
    os.remove("/tmp/mullet-test-notepad/test2.md")


@patch("mullet.core.tools.notepad.NOTEPAD_DIR", "/tmp/mullet-test-notepad")
def test_clear():
    os.makedirs("/tmp/mullet-test-notepad", exist_ok=True)
    notepad_handler("write", content="x", key="test3")
    notepad_handler("clear", key="test3")
    result = json.loads(notepad_handler("read", key="test3"))
    assert result["content"] == ""


def test_write_no_content():
    result = json.loads(notepad_handler("write", content=None, key="x"))
    assert "error" in result


class TestNotepadMissingLines:
    @patch("mullet.core.tools.notepad.NOTEPAD_DIR", "/tmp/mullet-test-notepad")
    @patch("mullet.core.tools.notepad._sanitize_key", return_value="../../outside")
    def test_path_traversal_blocked(self, mock_sanitize):
        with pytest.raises(ValueError, match="Path traversal blocked"):
            from mullet.core.tools.notepad import _get_notepad_path
            _get_notepad_path("ignored")

    @patch("mullet.core.tools.notepad.NOTEPAD_DIR", "/tmp/mullet-test-notepad")
    def test_clear_already_empty(self, tmp_path):
        import json
        result = json.loads(notepad_handler("clear", key="nonexistent_key_xyz"))
        assert "already empty" in result.get("message", "")

    @patch("mullet.core.tools.notepad.NOTEPAD_DIR", "/tmp/mullet-test-notepad")
    def test_invalid_action(self):
        import json
        result = json.loads(notepad_handler("frobnicate", key="x"))
        assert "Invalid action" in result["error"]
