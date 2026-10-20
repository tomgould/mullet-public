"""Tests for file_ops patch action."""
import json
import pytest
from mullet.core.tools.file_ops import file_ops_handler


@pytest.fixture
def temp_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Hello World\nAnother line\nHello World again")
    return f


def test_patch_replaces_text(temp_file):
    result = json.loads(file_ops_handler(action="patch", path=str(temp_file), old_text="World", new_text="Python"))
    assert result["status"] == "patched"
    assert temp_file.read_text() == "Hello Python\nAnother line\nHello World again"


def test_patch_file_not_found(tmp_path):
    result = json.loads(file_ops_handler(action="patch", path=str(tmp_path / "nope.txt"), old_text="x", new_text="y"))
    assert "error" in result


def test_patch_old_text_not_found(temp_file):
    result = json.loads(file_ops_handler(action="patch", path=str(temp_file), old_text="MISSING", new_text="x"))
    assert "error" in result
    assert "not found" in result["error"]


def test_patch_empty_old_text(temp_file):
    result = json.loads(file_ops_handler(action="patch", path=str(temp_file), old_text="", new_text="x"))
    assert "error" in result


def test_patch_first_occurrence_only(temp_file):
    result = json.loads(file_ops_handler(action="patch", path=str(temp_file), old_text="Hello", new_text="Hi"))
    assert result["status"] == "patched"
    content = temp_file.read_text()
    assert content.startswith("Hi World")
    assert "Hello World again" in content


def test_patch_multiline(tmp_path):
    f = tmp_path / "multi.txt"
    f.write_text("Line 1\nLine 2\nLine 3")
    result = json.loads(file_ops_handler(action="patch", path=str(f), old_text="Line 1\nLine 2", new_text="New 1\nNew 2"))
    assert result["status"] == "patched"
    assert f.read_text() == "New 1\nNew 2\nLine 3"


def test_patch_deletion(temp_file):
    result = json.loads(file_ops_handler(action="patch", path=str(temp_file), old_text=" World", new_text=""))
    assert result["status"] == "patched"
    assert temp_file.read_text().startswith("Hello\n")


def test_patch_unicode(tmp_path):
    f = tmp_path / "unicode.txt"
    f.write_text("Héllö Wörld")
    result = json.loads(file_ops_handler(action="patch", path=str(f), old_text="Wörld", new_text="Python"))
    assert result["status"] == "patched"
    assert f.read_text() == "Héllö Python"
