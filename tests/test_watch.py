"""Tests for file watcher mode."""

import os
import time
from mullet.cli.watch import _scan_files, _detect_changes, _WATCH_EXTENSIONS, _IGNORE_DIRS


def test_scan_finds_python_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "main.py").write_text("print('hello')")
    (tmp_path / "utils.py").write_text("def util(): pass")
    files = _scan_files(str(tmp_path))
    assert len(files) == 2
    assert any("main.py" in p for p in files)


def test_scan_finds_js_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.js").write_text("console.log('hi')")
    (tmp_path / "style.css").write_text("body {}")
    files = _scan_files(str(tmp_path))
    assert len(files) == 2


def test_scan_ignores_pycache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache_dir = tmp_path / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "module.cpython-310.pyc").write_bytes(b"\x00")
    (tmp_path / "main.py").write_text("x = 1")
    files = _scan_files(str(tmp_path))
    assert len(files) == 1  # Only main.py


def test_scan_ignores_git(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]")
    (tmp_path / "app.py").write_text("x = 1")
    files = _scan_files(str(tmp_path))
    assert len(files) == 1


def test_scan_ignores_node_modules(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    nm = tmp_path / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("module.exports = {}")
    (tmp_path / "app.js").write_text("require('pkg')")
    files = _scan_files(str(tmp_path))
    assert len(files) == 1


def test_scan_ignores_non_watched_extensions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "data.bin").write_bytes(b"\x00\x01")
    (tmp_path / "main.py").write_text("x = 1")
    files = _scan_files(str(tmp_path))
    assert len(files) == 1


def test_detect_changes_new_file(tmp_path):
    before = {}
    (tmp_path / "new.py").write_text("x = 1")
    after = _scan_files(str(tmp_path))
    changes = _detect_changes(before, after)
    assert len(changes) == 1
    assert "new.py" in changes[0]


def test_detect_changes_modified_file(tmp_path):
    (tmp_path / "main.py").write_text("x = 1")
    before = _scan_files(str(tmp_path))
    time.sleep(0.05)
    (tmp_path / "main.py").write_text("x = 2")
    after = _scan_files(str(tmp_path))
    changes = _detect_changes(before, after)
    assert len(changes) == 1


def test_detect_changes_deleted_file(tmp_path):
    (tmp_path / "temp.py").write_text("x = 1")
    before = _scan_files(str(tmp_path))
    os.unlink(tmp_path / "temp.py")
    after = _scan_files(str(tmp_path))
    changes = _detect_changes(before, after)
    assert len(changes) == 1
    assert "deleted" in changes[0]


def test_detect_no_changes(tmp_path):
    (tmp_path / "stable.py").write_text("x = 1")
    snapshot = _scan_files(str(tmp_path))
    changes = _detect_changes(snapshot, snapshot)
    assert changes == []


def test_detect_deleted_file():
    """Line 62: deleted file (in before but not after) is reported."""
    before = {"a.py": 1.0, "b.py": 2.0}
    after = {"a.py": 1.0}
    changes = _detect_changes(before, after)
    assert any("b.py" in c and "deleted" in c for c in changes)


def test_watch_extensions_include_common_types():
    assert ".py" in _WATCH_EXTENSIONS
    assert ".js" in _WATCH_EXTENSIONS
    assert ".ts" in _WATCH_EXTENSIONS
    assert ".rs" in _WATCH_EXTENSIONS
    assert ".go" in _WATCH_EXTENSIONS


def test_ignore_dirs_include_common_patterns():
    assert "__pycache__" in _IGNORE_DIRS
    assert ".git" in _IGNORE_DIRS
    assert "node_modules" in _IGNORE_DIRS
    assert ".venv" in _IGNORE_DIRS
