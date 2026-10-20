"""Tests for watch pure helpers and license pure functions."""

import base64
import json
import pytest

from mullet.cli.watch import (
    _detect_changes,
    _scan_files,
    _WATCH_EXTENSIONS,
    _IGNORE_DIRS,
    MIN_INTERVAL,
)
from mullet.config.license import (
    LicenseChecker,
    LicenseError,
    TRIAL_DAYS,
)


# ---------------------------------------------------------------------------
# _detect_changes
# ---------------------------------------------------------------------------

class TestDetectChanges:
    def test_no_changes_returns_empty(self):
        files = {"a.py": 1.0, "b.py": 2.0}
        assert _detect_changes(files, files.copy()) == []

    def test_modified_file_detected(self):
        before = {"a.py": 1.0}
        after = {"a.py": 2.0}
        changes = _detect_changes(before, after)
        assert "a.py" in changes

    def test_new_file_detected(self):
        before = {"a.py": 1.0}
        after = {"a.py": 1.0, "b.py": 2.0}
        changes = _detect_changes(before, after)
        assert "b.py" in changes

    def test_deleted_file_detected(self):
        before = {"a.py": 1.0, "b.py": 2.0}
        after = {"a.py": 1.0}
        changes = _detect_changes(before, after)
        assert any("b.py" in c for c in changes)

    def test_deleted_file_has_deleted_suffix(self):
        before = {"a.py": 1.0}
        after = {}
        changes = _detect_changes(before, after)
        assert "a.py (deleted)" in changes

    def test_empty_before_and_after_returns_empty(self):
        assert _detect_changes({}, {}) == []

    def test_returns_list(self):
        assert isinstance(_detect_changes({}, {}), list)

    def test_multiple_changes(self):
        before = {"a.py": 1.0, "b.py": 2.0, "c.py": 3.0}
        after = {"a.py": 9.0, "b.py": 2.0}  # a.py modified, c.py deleted
        changes = _detect_changes(before, after)
        assert "a.py" in changes
        assert any("c.py" in c for c in changes)
        assert "b.py" not in changes


# ---------------------------------------------------------------------------
# _scan_files
# ---------------------------------------------------------------------------

class TestScanFiles:
    def test_empty_dir_returns_empty(self, tmp_path):
        result = _scan_files(str(tmp_path))
        assert result == {}

    def test_py_file_included(self, tmp_path):
        f = tmp_path / "main.py"
        f.write_text("x = 1")
        result = _scan_files(str(tmp_path))
        assert any("main.py" in k for k in result)

    def test_txt_file_excluded(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("text")
        result = _scan_files(str(tmp_path))
        assert not any("notes.txt" in k for k in result)

    def test_ignored_dirs_skipped(self, tmp_path):
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "module.py").write_text("x = 1")
        result = _scan_files(str(tmp_path))
        assert not any("__pycache__" in k for k in result)

    def test_node_modules_skipped(self, tmp_path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "index.js").write_text("const x = 1")
        result = _scan_files(str(tmp_path))
        assert not any("node_modules" in k for k in result)

    def test_returns_dict_with_float_mtimes(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1")
        result = _scan_files(str(tmp_path))
        for v in result.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# Watch constants
# ---------------------------------------------------------------------------

class TestWatchConstants:
    def test_py_in_watch_extensions(self):
        assert ".py" in _WATCH_EXTENSIONS

    def test_js_in_watch_extensions(self):
        assert ".js" in _WATCH_EXTENSIONS

    def test_ts_in_watch_extensions(self):
        assert ".ts" in _WATCH_EXTENSIONS

    def test_pycache_in_ignore_dirs(self):
        assert "__pycache__" in _IGNORE_DIRS

    def test_git_in_ignore_dirs(self):
        assert ".git" in _IGNORE_DIRS

    def test_node_modules_in_ignore_dirs(self):
        assert "node_modules" in _IGNORE_DIRS

    def test_min_interval_positive(self):
        assert MIN_INTERVAL > 0


# ---------------------------------------------------------------------------
# LicenseChecker._validate_key
# ---------------------------------------------------------------------------

def _make_key(payload: dict) -> str:
    return base64.b64encode(json.dumps(payload).encode()).decode()


class TestValidateKey:
    def setup_method(self):
        self.checker = LicenseChecker()

    def test_valid_key_returns_claims(self):
        key = _make_key({"customer": "Acme", "max_users": 10})
        claims = self.checker._validate_key(key)
        assert claims is not None
        assert claims["customer"] == "Acme"

    def test_missing_customer_returns_none(self):
        key = _make_key({"max_users": 10})
        assert self.checker._validate_key(key) is None

    def test_invalid_base64_returns_none(self):
        assert self.checker._validate_key("not-valid-base64!!!") is None

    def test_empty_string_returns_none(self):
        assert self.checker._validate_key("") is None

    def test_invalid_json_returns_none(self):
        bad = base64.b64encode(b"not json {").decode()
        assert self.checker._validate_key(bad) is None

    def test_claims_include_max_users(self):
        key = _make_key({"customer": "Corp", "max_users": 5})
        claims = self.checker._validate_key(key)
        assert claims["max_users"] == 5

    def test_returns_none_for_none_input(self):
        assert self.checker._validate_key(None) is None


# ---------------------------------------------------------------------------
# LicenseError
# ---------------------------------------------------------------------------

class TestLicenseError:
    def test_is_exception(self):
        assert issubclass(LicenseError, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(LicenseError):
            raise LicenseError("Invalid license")

    def test_message_preserved(self):
        try:
            raise LicenseError("custom message")
        except LicenseError as e:
            assert "custom message" in str(e)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestLicenseConstants:
    def test_trial_days_positive(self):
        assert TRIAL_DAYS > 0

    def test_trial_days_is_int(self):
        assert isinstance(TRIAL_DAYS, int)
