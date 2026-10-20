"""Tests for diff.py missing coverage lines."""

from pathlib import Path
from unittest.mock import patch

import pytest

from mullet.core.agent.diff import ChangeTracker, _fmt_size


class TestDiffMissingLines:
    def test_format_changes_created(self, tmp_path):
        """Line 92: ANSI formatting for created files."""
        tracker = ChangeTracker(watch_dirs=[str(tmp_path)])
        tracker.snapshot_before()
        (tmp_path / "new.txt").write_text("hello world")
        tracker.snapshot_after()
        output = tracker.format_changes()
        assert "\033[32m+ " in output
        assert "new.txt" in output

    def test_format_changes_deleted(self, tmp_path):
        """Line 93: ANSI formatting for deleted files."""
        f = tmp_path / "gone.txt"
        f.write_text("bye")
        tracker = ChangeTracker(watch_dirs=[str(tmp_path)])
        tracker.snapshot_before()
        f.unlink()
        tracker.snapshot_after()
        output = tracker.format_changes()
        assert "\033[31m- " in output
        assert "gone.txt" in output

    def test_format_changes_modified(self, tmp_path):
        """Lines 95-97: ANSI formatting for modified files with delta."""
        f = tmp_path / "mod.txt"
        f.write_text("short")
        tracker = ChangeTracker(watch_dirs=[str(tmp_path)])
        tracker.snapshot_before()
        f.write_text("this is a much longer text")
        tracker.snapshot_after()
        output = tracker.format_changes()
        assert "\033[33m~ " in output
        assert "mod.txt" in output
        assert "bytes" in output

    def test_scan_missing_watch_dir(self, tmp_path):
        """Line 107: missing watch_dir is skipped."""
        tracker = ChangeTracker(watch_dirs=[str(tmp_path / "nonexistent")])
        result = tracker._scan()
        assert result == {}

    def test_scan_skips_non_files(self, tmp_path):
        """Line 110: non-file entries are skipped."""
        (tmp_path / "subdir").mkdir()
        tracker = ChangeTracker(watch_dirs=[str(tmp_path)])
        result = tracker._scan()
        assert "subdir" not in result

    def test_scan_skips_hidden(self, tmp_path):
        """Line 112: hidden directories are skipped."""
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "file.txt").write_text("secret")
        tracker = ChangeTracker(watch_dirs=[str(tmp_path)])
        result = tracker._scan()
        assert str(hidden / "file.txt") not in result

    def test_scan_skips_pycache(self, tmp_path):
        """Line 114: __pycache__ is skipped."""
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "foo.cpython-311.pyc").write_text("bytecode")
        tracker = ChangeTracker(watch_dirs=[str(tmp_path)])
        result = tracker._scan()
        assert str(pycache / "foo.cpython-311.pyc") not in result

    def test_scan_permission_error(self, tmp_path, caplog):
        """Lines 121-122: PermissionError/OSError is logged and skipped."""
        f = tmp_path / "secret.txt"
        f.write_text("secret")
        tracker = ChangeTracker(watch_dirs=[str(tmp_path)])
        with patch.object(Path, "read_bytes", side_effect=PermissionError("denied")):
            with caplog.at_level("WARNING", logger="mullet.core.agent.diff"):
                result = tracker._scan()
        assert str(f) not in result
        assert "failed to scan file" in caplog.text

    def test_fmt_size_kb(self):
        """Lines 129-130: KB formatting."""
        assert _fmt_size(1536) == "1.5KB"

    def test_fmt_size_mb(self):
        """Line 131: MB formatting."""
        assert _fmt_size(2 * 1024 * 1024) == "2.0MB"

    def test_fmt_size_bytes(self):
        """Line 128: bytes formatting."""
        assert _fmt_size(512) == "512B"

    def test_format_changes_no_changes(self, tmp_path):
        """Line 85: format_changes returns 'No files changed.' when empty."""
        tracker = ChangeTracker(watch_dirs=[str(tmp_path)])
        tracker.snapshot_before()
        tracker.snapshot_after()
        assert tracker.format_changes() == "No files changed."

    def test_scan_extension_filter_continue(self, tmp_path):
        """Line 116: _scan skips files not matching extension filter."""
        (tmp_path / "code.py").write_text("x = 1")
        (tmp_path / "data.json").write_text("{}")
        tracker = ChangeTracker(watch_dirs=[str(tmp_path)], extensions=[".py"])
        result = tracker._scan()
        assert str(tmp_path / "code.py") in result
        assert str(tmp_path / "data.json") not in result
