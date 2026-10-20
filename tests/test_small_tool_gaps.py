"""Targeted tests for missing lines in small core tool modules."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── registry.py ────────────────────────────────────────────────────────────

class TestRegistryMissingLines:
    def _make_registry(self):
        from mullet.core.tools.registry import ToolRegistry
        return ToolRegistry()

    def test_dispatch_bad_json_arguments_falls_back_to_empty(self):
        reg = self._make_registry()
        calls = []
        reg.register(name="t", func=lambda **kw: json.dumps({"ok": True}), schema={})
        result = json.loads(reg.dispatch("t", "NOT VALID JSON"))
        assert "ok" in result

    def test_dispatch_returns_non_string_result_as_json(self):
        reg = self._make_registry()
        # Function that does NOT accept task_id but returns a non-string (triggers line 82)
        def no_task_id_func():
            return {"key": "val"}
        reg.register(name="t", func=no_task_id_func, schema={})
        result = json.loads(reg.dispatch("t", {}))
        assert result["key"] == "val"


# ── skill_mgr.py ───────────────────────────────────────────────────────────

class TestSkillMgrMissingLines:
    def test_edit_action_calls_manager(self):
        from mullet.core.tools.skill_mgr import skill_mgr_handler
        mock_result = {"success": True, "message": "edited"}
        with patch("mullet.core.tools.skill_mgr._skill_manager") as mock_mgr:
            mock_mgr.edit.return_value = mock_result
            result = json.loads(skill_mgr_handler(action="edit", name="myskill", body="new body"))
        assert result["success"] is True

    def test_delete_action_calls_manager(self):
        from mullet.core.tools.skill_mgr import skill_mgr_handler
        mock_result = {"success": True, "message": "deleted"}
        with patch("mullet.core.tools.skill_mgr._skill_manager") as mock_mgr:
            mock_mgr.delete.return_value = mock_result
            result = json.loads(skill_mgr_handler(action="delete", name="myskill"))
        assert result["success"] is True


# ── url_safety.py ──────────────────────────────────────────────────────────

class TestUrlSafetyMissingLines:
    def test_exception_returns_false(self):
        from mullet.core.tools.url_safety import is_safe_url
        with patch("mullet.core.tools.url_safety.urlparse", side_effect=Exception("bad")):
            result = is_safe_url("http://example.com")
        assert result is False


# ── vision.py ──────────────────────────────────────────────────────────────

class TestVisionMissingLines:
    def test_not_a_file_returns_error(self, tmp_path):
        from mullet.core.tools.vision import _encode_image
        result_data, err = _encode_image(str(tmp_path))  # tmp_path is a dir
        assert result_data is None
        assert "Not a file" in err

    def test_read_error_returns_message(self, tmp_path):
        from mullet.core.tools.vision import _encode_image
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)  # valid PNG header
        with patch.object(Path, "read_bytes", side_effect=OSError("disk error")):
            result_data, err = _encode_image(str(img))
        assert result_data is None
        assert "Failed to read image" in err


# ── web_fetch.py ───────────────────────────────────────────────────────────

class TestWebFetchMissingLines:
    def test_blocked_url_raises_os_error(self):
        from mullet.core.tools.web_fetch import fetch_url
        # is_safe_url is imported inside fetch_url, so patch it at source
        with patch("mullet.core.tools.url_safety.is_safe_url", return_value=False):
            try:
                fetch_url("file:///etc/passwd")
                assert False, "should have raised"
            except OSError as e:
                assert "blocked" in str(e)


# ── website_policy.py ──────────────────────────────────────────────────────

class TestWebsitePolicyMissingLines:
    def test_cache_evicts_oldest_when_full(self):
        from mullet.core.tools.website_policy import RobotsTxtChecker, _MAX_CACHE_SIZE
        checker = RobotsTxtChecker()
        # Fill the cache to the limit with fake parsers
        for i in range(_MAX_CACHE_SIZE):
            fake_rp = MagicMock()
            fake_rp.can_fetch.return_value = True
            fake_rp.crawl_delay.return_value = None
            checker._cache[f"http://site{i}.com"] = fake_rp
        # One more entry should evict the oldest
        assert len(checker._cache) == _MAX_CACHE_SIZE
        with patch("urllib.robotparser.RobotFileParser") as mock_rp_cls:
            mock_rp = MagicMock()
            mock_rp.can_fetch.return_value = True
            mock_rp.crawl_delay.return_value = None
            mock_rp_cls.return_value = mock_rp
            checker.check_robots_txt("http://newsite.com/page")
        assert len(checker._cache) == _MAX_CACHE_SIZE  # evicted one, added one


# ── credential_guard.py ────────────────────────────────────────────────────

class TestCredentialGuardMissingLines:
    def test_credential_filename_pattern_match(self):
        from mullet.core.tools.credential_guard import CredentialGuard, CREDENTIAL_FILENAMES
        guard = CredentialGuard()
        # Find a path that contains a CREDENTIAL_FILENAMES pattern
        # (e.g. "credentials" is a common pattern)
        # Use a path where basename isn't in CREDENTIAL_FILENAMES and ext isn't in
        # CREDENTIAL_EXTENSIONS, but basename CONTAINS a pattern (triggers line 71)
        test_path = "/home/user/my_token_config.json"
        result = guard.is_credential_file(test_path)
        assert result is True


# ── python_repl.py ─────────────────────────────────────────────────────────

class TestPythonReplMissingLines:
    def test_generic_exception_returns_error(self):
        from mullet.core.tools.python_repl import python_repl_handler
        with patch("subprocess.run", side_effect=RuntimeError("unexpected error")):
            result = json.loads(python_repl_handler(code="print('hi')"))
        assert result["success"] is False
        assert "unexpected error" in result["stderr"]
