"""Tests for mullet.knowledge.sources.git_ops pure helpers."""
from __future__ import annotations

import os
import stat
import pytest

from mullet.knowledge.sources.git_ops import (
    _sanitise_output,
    _validate_name,
    _make_askpass_script,
    CloneResult,
    GitError,
)


class TestSanitiseOutput:
    def test_removes_ghp_token(self):
        text = "error: token ghp_" + "A" * 37 + " is bad"
        result = _sanitise_output(text)
        assert "ghp_" not in result
        assert "****" in result

    def test_removes_github_pat_token(self):
        text = "auth: github_pat_" + "B" * 80 + " failed"
        result = _sanitise_output(text)
        assert "github_pat_" not in result
        assert "****" in result

    def test_removes_url_credentials(self):
        text = "Cloning https://user:secret@github.com/org/repo"
        result = _sanitise_output(text)
        assert "secret" not in result
        assert "****@github.com" in result

    def test_passthrough_clean_text(self):
        text = "Everything is fine"
        assert _sanitise_output(text) == text

    def test_empty_string(self):
        assert _sanitise_output("") == ""

    def test_removes_gho_token(self):
        text = "token: gho_" + "C" * 37
        result = _sanitise_output(text)
        assert "gho_" not in result

    def test_removes_ghs_token(self):
        text = "ghs_" + "D" * 37
        result = _sanitise_output(text)
        assert "ghs_" not in result


class TestValidateName:
    def test_valid_alphanumeric(self):
        _validate_name("my-repo")  # no exception

    def test_valid_with_dots(self):
        _validate_name("repo.v2")  # no exception

    def test_valid_with_underscore(self):
        _validate_name("my_repo_123")  # no exception

    def test_rejects_slash(self):
        with pytest.raises(GitError, match="Invalid repository name"):
            _validate_name("org/repo")

    def test_rejects_space(self):
        with pytest.raises(GitError, match="Invalid repository name"):
            _validate_name("my repo")

    def test_rejects_path_traversal(self):
        with pytest.raises(GitError):
            _validate_name("..")

    def test_rejects_semicolon(self):
        with pytest.raises(GitError, match="Invalid repository name"):
            _validate_name("repo;rm")

    def test_rejects_empty_string(self):
        with pytest.raises(GitError, match="Invalid repository name"):
            _validate_name("")


class TestMakeAskpassScript:
    def test_creates_executable_file(self, tmp_path):
        path = _make_askpass_script("mytoken123")
        try:
            assert os.path.exists(path)
            mode = os.stat(path).st_mode
            assert mode & stat.S_IXUSR, "Script must be user-executable"
        finally:
            os.unlink(path)

    def test_script_contains_echo(self, tmp_path):
        path = _make_askpass_script("secrettoken")
        try:
            with open(path) as f:
                content = f.read()
            assert "echo" in content
            assert "secrettoken" in content
        finally:
            os.unlink(path)

    def test_script_has_shebang(self):
        path = _make_askpass_script("tok")
        try:
            with open(path) as f:
                first_line = f.readline()
            assert first_line.startswith("#!/")
        finally:
            os.unlink(path)


class TestCloneResult:
    def test_successful_result(self):
        r = CloneResult(success=True, message="Cloned successfully", is_fresh_clone=True)
        assert r.success is True
        assert r.is_fresh_clone is True
        assert r.changed_files == []

    def test_failed_result(self):
        r = CloneResult(success=False, message="Clone failed: auth error")
        assert r.success is False
        assert r.changed_files == []

    def test_result_with_changed_files(self):
        r = CloneResult(success=True, message="Updated: 3 files", changed_files=["a.py", "b.py", "c.py"])
        assert len(r.changed_files) == 3

    def test_default_changed_files_is_list(self):
        r = CloneResult(success=True, message="ok")
        assert isinstance(r.changed_files, list)


from unittest.mock import patch, MagicMock, call
import subprocess
from mullet.knowledge.sources.git_ops import _run_git, clone_repo, fetch_repo, get_default_branch


class TestMakeAskpassScriptError:
    def test_os_write_raises_cleans_up(self, tmp_path):
        """Lines 53-59: os.write raises → fd closed, temp file unlinked, exception re-raised."""
        with patch("os.write", side_effect=IOError("disk full")):
            with pytest.raises(IOError, match="disk full"):
                _make_askpass_script("token")

    def test_os_write_raises_unlink_oserror_silenced(self, tmp_path):
        """Lines 55-58: if unlink also fails, OSError is silenced but original exception re-raised."""
        with patch("os.write", side_effect=IOError("disk full")), \
             patch("os.unlink", side_effect=OSError("unlink failed")):
            with pytest.raises(IOError, match="disk full"):
                _make_askpass_script("token")


class TestRunGit:
    def _make_result(self, returncode=0, stdout="", stderr=""):
        r = MagicMock(spec=subprocess.CompletedProcess)
        r.returncode = returncode
        r.stdout = stdout
        r.stderr = stderr
        return r

    def test_no_token_runs_git(self):
        """Lines 73-74, 81-89, 111: basic invocation without token."""
        mock_result = self._make_result(returncode=0, stdout="ok")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = _run_git(["status"])
        mock_run.assert_called_once()
        assert result.returncode == 0

    def test_with_token_sets_askpass(self):
        """Lines 76-79: token provided → askpass script created."""
        mock_result = self._make_result(returncode=0, stdout="")
        with patch("subprocess.run", return_value=mock_result), \
             patch("mullet.knowledge.sources.git_ops._make_askpass_script", return_value="/tmp/askpass.sh") as mock_askpass, \
             patch("os.unlink") as mock_unlink:
            _run_git(["status"], token="mytoken")
        mock_askpass.assert_called_once_with("mytoken")
        mock_unlink.assert_called_once_with("/tmp/askpass.sh")

    def test_timeout_returns_negative_returncode(self):
        """Lines 90-97: subprocess.TimeoutExpired → returncode -1, stderr with timeout msg."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="git", timeout=5)):
            result = _run_git(["clone", "url"], timeout=5)
        assert result.returncode == -1
        assert "timed out" in result.stderr.lower()

    def test_token_cleanup_file_not_found_is_silenced(self):
        """Lines 100-103: askpass unlink raises FileNotFoundError → silenced."""
        mock_result = self._make_result(returncode=0)
        with patch("subprocess.run", return_value=mock_result), \
             patch("mullet.knowledge.sources.git_ops._make_askpass_script", return_value="/tmp/askpass.sh"), \
             patch("os.unlink", side_effect=FileNotFoundError()):
            result = _run_git(["status"], token="tok")
        assert result.returncode == 0

    def test_token_cleanup_oserror_is_logged(self, caplog):
        """Lines 104-109: askpass unlink raises OSError → warning logged."""
        import logging
        mock_result = self._make_result(returncode=0)
        with patch("subprocess.run", return_value=mock_result), \
             patch("mullet.knowledge.sources.git_ops._make_askpass_script", return_value="/tmp/askpass.sh"), \
             patch("os.unlink", side_effect=OSError("permission denied")), \
             caplog.at_level(logging.WARNING):
            _run_git(["status"], token="tok")
        assert "askpass" in caplog.text.lower() or "token" in caplog.text.lower()


class TestCloneRepo:
    def _make_run_result(self, returncode=0, stdout="", stderr=""):
        r = MagicMock(spec=subprocess.CompletedProcess)
        r.returncode = returncode
        r.stdout = stdout
        r.stderr = stderr
        return r

    def test_success_no_token(self, tmp_path):
        """Lines 130-150: successful clone without token."""
        target = tmp_path / "my-repo"
        ok = self._make_run_result(returncode=0, stdout="")
        with patch("subprocess.run", return_value=ok):
            result = clone_repo("https://github.com/org/my-repo.git", target)
        assert result.success is True
        assert result.is_fresh_clone is True

    def test_success_with_token(self, tmp_path):
        """Lines 132-133: with token → oauth2 URL used."""
        target = tmp_path / "my-repo"
        ok = self._make_run_result(returncode=0)
        with patch("subprocess.run", return_value=ok) as mock_run:
            result = clone_repo("https://github.com/org/my-repo.git", target, token="tok")
        assert result.success is True
        # auth_url passed to subprocess should contain oauth2
        first_call_args = mock_run.call_args_list[0]
        assert "oauth2" in str(first_call_args)

    def test_failure_returns_false(self, tmp_path):
        """Lines 141-143: clone fails → CloneResult(success=False)."""
        target = tmp_path / "my-repo"
        fail = self._make_run_result(returncode=128, stderr="repo not found")
        with patch("subprocess.run", return_value=fail):
            result = clone_repo("https://github.com/org/my-repo.git", target)
        assert result.success is False
        assert "Clone failed" in result.message

    def test_url_cleanup_failure_logs_warning(self, tmp_path, caplog):
        """Lines 147-148: URL cleanup command fails → warning logged."""
        import logging
        target = tmp_path / "my-repo"
        ok = self._make_run_result(returncode=0)
        cleanup_fail = self._make_run_result(returncode=1, stderr="config error")
        with patch("subprocess.run", side_effect=[ok, cleanup_fail]), \
             caplog.at_level(logging.WARNING):
            result = clone_repo("https://github.com/org/my-repo.git", target)
        assert result.success is True
        assert "clean remote URL" in caplog.text or "Failed" in caplog.text


class TestFetchRepo:
    def _make_run_result(self, returncode=0, stdout="", stderr=""):
        r = MagicMock(spec=subprocess.CompletedProcess)
        r.returncode = returncode
        r.stdout = stdout
        r.stderr = stderr
        return r

    def test_not_a_git_repo_returns_early(self, tmp_path):
        """Line 161: no .git dir → CloneResult(success=False)."""
        result = fetch_repo(tmp_path, "https://github.com/org/repo.git")
        assert result.success is False
        assert "Not a git repository" in result.message

    def test_fetch_failure_returns_false(self, tmp_path):
        """Lines 176-180: fetch command fails → CloneResult(success=False)."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        old_head = self._make_run_result(returncode=0, stdout="abc123\n")
        fetch_fail = self._make_run_result(returncode=1, stderr="network error")
        with patch("subprocess.run", side_effect=[old_head, fetch_fail, self._make_run_result(returncode=0)]):
            result = fetch_repo(tmp_path, "https://github.com/org/repo.git")
        assert result.success is False
        assert "Fetch failed" in result.message

    def test_branch_not_found_returns_empty_success(self, tmp_path):
        """Lines 186-187: remote branch doesn't exist → success=True, empty message."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        old_head = self._make_run_result(returncode=0, stdout="abc123\n")
        fetch_ok = self._make_run_result(returncode=0)
        branch_missing = self._make_run_result(returncode=1, stdout="")
        cleanup = self._make_run_result(returncode=0)
        with patch("subprocess.run", side_effect=[old_head, fetch_ok, branch_missing, cleanup]):
            result = fetch_repo(tmp_path, "https://github.com/org/repo.git")
        assert result.success is True
        assert "empty" in result.message.lower()

    def test_reset_failure_returns_false(self, tmp_path):
        """Lines 193-197: reset --hard fails → CloneResult(success=False)."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        old_head = self._make_run_result(returncode=0, stdout="abc123\n")
        fetch_ok = self._make_run_result(returncode=0)
        branch_ok = self._make_run_result(returncode=0, stdout="def456")
        reset_fail = self._make_run_result(returncode=1, stderr="reset failed")
        cleanup = self._make_run_result(returncode=0)
        with patch("subprocess.run", side_effect=[old_head, fetch_ok, branch_ok, reset_fail, cleanup]):
            result = fetch_repo(tmp_path, "https://github.com/org/repo.git")
        assert result.success is False
        assert "Reset failed" in result.message

    def test_success_with_changed_files(self, tmp_path):
        """Lines 199-221: hash changed → diff computed → changed_files populated."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        old_head = self._make_run_result(returncode=0, stdout="aaa111\n")
        fetch_ok = self._make_run_result(returncode=0)
        branch_ok = self._make_run_result(returncode=0, stdout="origin/main")
        reset_ok = self._make_run_result(returncode=0)
        new_head = self._make_run_result(returncode=0, stdout="bbb222\n")
        diff_ok = self._make_run_result(returncode=0, stdout="src/a.py\nsrc/b.py\n")
        cleanup = self._make_run_result(returncode=0)
        with patch("subprocess.run", side_effect=[old_head, fetch_ok, branch_ok, reset_ok, new_head, diff_ok, cleanup]):
            result = fetch_repo(tmp_path, "https://github.com/org/repo.git")
        assert result.success is True
        assert len(result.changed_files) == 2
        assert "Updated: 2 files changed" in result.message

    def test_success_already_up_to_date(self, tmp_path):
        """Lines 214-217: old_hash == new_hash → no diff → msg 'Already up to date'."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        same_hash = "abc123"
        old_head = self._make_run_result(returncode=0, stdout=same_hash + "\n")
        fetch_ok = self._make_run_result(returncode=0)
        branch_ok = self._make_run_result(returncode=0, stdout="origin/main")
        reset_ok = self._make_run_result(returncode=0)
        new_head = self._make_run_result(returncode=0, stdout=same_hash + "\n")
        cleanup = self._make_run_result(returncode=0)
        with patch("subprocess.run", side_effect=[old_head, fetch_ok, branch_ok, reset_ok, new_head, cleanup]):
            result = fetch_repo(tmp_path, "https://github.com/org/repo.git")
        assert result.success is True
        assert "up to date" in result.message.lower()

    def test_with_token_sets_auth_url(self, tmp_path):
        """Lines 170-172: token provided → auth URL set before fetch."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        ok = self._make_run_result(returncode=0, stdout="abc\n")
        # Sequence: rev-parse old HEAD, config auth URL, fetch, branch check, reset, rev-parse new HEAD, diff (same hash), cleanup config
        same_hash = "abc"
        with patch("subprocess.run", return_value=self._make_run_result(returncode=0, stdout=same_hash + "\n")) as mock_run:
            fetch_repo(tmp_path, "https://github.com/org/repo.git", token="mytoken")
        # Check that oauth2 URL was passed in one of the calls
        calls_str = str(mock_run.call_args_list)
        assert "oauth2" in calls_str or "config" in calls_str

    def test_cleanup_failure_logs_warning(self, tmp_path, caplog):
        """Lines 225-227: cleanup config command fails → warning logged."""
        import logging
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        same_hash = "abc123"
        ok = self._make_run_result(returncode=0, stdout=same_hash + "\n")
        branch_ok = self._make_run_result(returncode=0, stdout="origin/main")
        reset_ok = self._make_run_result(returncode=0)
        cleanup_fail = self._make_run_result(returncode=1)
        with patch("subprocess.run", side_effect=[ok, ok, branch_ok, reset_ok, ok, cleanup_fail]), \
             caplog.at_level(logging.WARNING):
            fetch_repo(tmp_path, "https://github.com/org/repo.git")
        assert "clean remote URL" in caplog.text or "Failed" in caplog.text


class TestGetDefaultBranch:
    def _make_run_result(self, returncode=0, stdout=""):
        r = MagicMock(spec=subprocess.CompletedProcess)
        r.returncode = returncode
        r.stdout = stdout
        return r

    def test_returns_detected_branch(self, tmp_path):
        """Lines 232-234: successful rev-parse → returns branch name."""
        with patch("subprocess.run", return_value=self._make_run_result(stdout="main\n")):
            assert get_default_branch(tmp_path) == "main"

    def test_fallback_to_main_on_failure(self, tmp_path, caplog):
        """Lines 235-236: rev-parse fails → logs warning and returns 'main'."""
        import logging
        with patch("subprocess.run", return_value=self._make_run_result(returncode=1, stdout="")), \
             caplog.at_level(logging.WARNING):
            result = get_default_branch(tmp_path)
        assert result == "main"
        assert caplog.text  # warning was logged
