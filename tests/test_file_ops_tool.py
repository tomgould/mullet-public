"""Tests for file_ops tool missing lines."""

import json
from pathlib import Path

from mullet.core.tools.file_ops import file_ops_handler


class TestFileOpsMissingLines:
    def test_path_traversal_blocked(self):
        result = json.loads(file_ops_handler(action="read_file", path="/etc/passwd"))
        assert "traversal blocked" in result["error"].lower()

    def test_read_file_exception(self, tmp_path, monkeypatch):
        f = tmp_path / "secret.txt"
        f.write_text("hello")
        def raise_exc(*args, **kwargs):
            raise PermissionError("denied")
        monkeypatch.setattr(Path, "read_text", raise_exc)
        result = json.loads(file_ops_handler(action="read_file", path=str(f)))
        assert "denied" in result["error"]

    def test_write_file_exception(self, tmp_path, monkeypatch):
        path = str(tmp_path / "out.txt")
        def raise_exc(*args, **kwargs):
            raise PermissionError("denied")
        monkeypatch.setattr(Path, "write_text", raise_exc)
        result = json.loads(file_ops_handler(action="write_file", path=path, content="hi"))
        assert "denied" in result["error"]

    def test_patch_exception(self, tmp_path, monkeypatch):
        f = tmp_path / "code.py"
        f.write_text("hello")
        def raise_exc(*args, **kwargs):
            raise PermissionError("denied")
        monkeypatch.setattr(Path, "write_text", raise_exc)
        result = json.loads(file_ops_handler(action="patch", path=str(f), old_text="hello", new_text="bye"))
        assert "denied" in result["error"]

    def test_list_directory_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "a.txt").write_text("a")
        result = json.loads(file_ops_handler(action="list_directory", path=str(tmp_path), recursive=True))
        assert result["count"] >= 1

    def test_list_directory_exception(self, tmp_path, monkeypatch):
        def raise_exc(*args, **kwargs):
            raise PermissionError("denied")
        monkeypatch.setattr(Path, "iterdir", raise_exc)
        result = json.loads(file_ops_handler(action="list_directory", path=str(tmp_path)))
        assert "denied" in result["error"]

    def test_search_files_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "a.py").write_text("a")
        result = json.loads(file_ops_handler(action="search_files", path=str(tmp_path), pattern="*.py", recursive=True))
        assert result["count"] == 1
        assert "sub/a.py" in result["matches"]

    def test_search_files_exception(self, tmp_path, monkeypatch):
        def raise_exc(*args, **kwargs):
            raise PermissionError("denied")
        monkeypatch.setattr(Path, "glob", raise_exc)
        result = json.loads(file_ops_handler(action="search_files", path=str(tmp_path), pattern="*.py"))
        assert "denied" in result["error"]

    def test_register(self):
        class FakeRegistry:
            def __init__(self):
                self.calls = []
            def register(self, **kwargs):
                self.calls.append(kwargs)
        from mullet.core.tools.file_ops import register, FILE_OPS_SCHEMA, file_ops_handler
        reg = FakeRegistry()
        register(reg)
        assert len(reg.calls) == 1
        assert reg.calls[0]["name"] == "file_ops"
        assert reg.calls[0]["func"] == file_ops_handler
        assert reg.calls[0]["schema"] == FILE_OPS_SCHEMA


class TestFileOpsHandlerBranches:
    def test_empty_path_returns_error(self):
        result = json.loads(file_ops_handler(action="read_file", path=""))
        assert "path is required" in result["error"]

    def test_read_file_not_found(self, tmp_path):
        result = json.loads(file_ops_handler(action="read_file", path=str(tmp_path / "missing.txt")))
        assert "File not found" in result["error"]

    def test_read_file_with_offset_and_limit(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("\n".join(str(i) for i in range(20)))
        result = json.loads(file_ops_handler(action="read_file", path=str(f), offset=5, limit=3))
        assert result["offset"] == 5
        assert len(result["lines"]) == 3
        assert result["lines"][0] == "5"

    def test_write_file_success(self, tmp_path):
        dest = tmp_path / "out.txt"
        result = json.loads(file_ops_handler(action="write_file", path=str(dest), content="hello"))
        assert result["bytes_written"] == 5

    def test_patch_file_not_found(self, tmp_path):
        result = json.loads(file_ops_handler(action="patch", path=str(tmp_path / "no.txt"), old_text="x", new_text="y"))
        assert "File not found" in result["error"]

    def test_patch_no_old_text(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("hello")
        result = json.loads(file_ops_handler(action="patch", path=str(f), old_text="", new_text="world"))
        assert "old_text is required" in result["error"]

    def test_patch_old_text_not_found(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("hello world")
        result = json.loads(file_ops_handler(action="patch", path=str(f), old_text="NOTHERE", new_text="x"))
        assert "old_text not found" in result["error"]

    def test_patch_success(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("hello world")
        result = json.loads(file_ops_handler(action="patch", path=str(f), old_text="world", new_text="there"))
        assert result["status"] == "patched"
        assert f.read_text() == "hello there"

    def test_list_directory_not_a_dir(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        result = json.loads(file_ops_handler(action="list_directory", path=str(f)))
        assert "Directory not found" in result["error"]

    def test_list_directory_non_recursive(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        result = json.loads(file_ops_handler(action="list_directory", path=str(tmp_path)))
        names = [e["path"] for e in result["entries"]]
        assert "a.txt" in names and "b.txt" in names

    def test_unknown_action(self, tmp_path):
        result = json.loads(file_ops_handler(action="frobnicate", path=str(tmp_path)))
        assert "Unknown action" in result["error"]
