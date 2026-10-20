"""Tests for git status memoisation in hud/state.py."""
import time
from unittest.mock import patch, MagicMock
import pytest
from mullet.hud.state import get_git_info, _GIT_CACHE


@pytest.fixture(autouse=True)
def clear_git_cache():
    """Clear git cache before and after each test."""
    _GIT_CACHE.clear()
    yield
    _GIT_CACHE.clear()


def test_two_calls_within_ttl_use_cache(tmp_path):
    """Two calls within 2s return cached result without invoking subprocess twice."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    head_file = git_dir / "HEAD"
    head_file.write_text("ref: refs/heads/main\n")

    subprocess_call_count = 0

    def mock_subprocess_run(*args, **kwargs):
        nonlocal subprocess_call_count
        subprocess_call_count += 1
        mock_result = MagicMock()
        if "rev-parse" in args[0]:
            mock_result.stdout = "main\n"
        else:
            mock_result.stdout = ""
        return mock_result

    with patch("mullet.hud.state.subprocess.run", side_effect=mock_subprocess_run):
        result1 = get_git_info(str(tmp_path))
        assert result1 == ("main", False)
        assert subprocess_call_count == 2  # 2 subprocess calls for first invocation

        # Second call immediately after — should use cache
        result2 = get_git_info(str(tmp_path))
        assert result2 == ("main", False)
        assert subprocess_call_count == 2  # No new subprocess calls


def test_after_head_mtime_change_re_invokes(tmp_path):
    """After .git/HEAD mtime changes, next call re-invokes subprocess."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    head_file = git_dir / "HEAD"
    head_file.write_text("ref: refs/heads/main\n")

    subprocess_call_count = 0

    def mock_subprocess_run(*args, **kwargs):
        nonlocal subprocess_call_count
        subprocess_call_count += 1
        mock_result = MagicMock()
        if "rev-parse" in args[0]:
            # Return different branch on second invocation
            mock_result.stdout = "develop\n" if subprocess_call_count > 2 else "main\n"
        else:
            # Just return clean
            mock_result.stdout = ""
        return mock_result

    with patch("mullet.hud.state.subprocess.run", side_effect=mock_subprocess_run):
        result1 = get_git_info(str(tmp_path))
        assert result1 == ("main", False)
        assert subprocess_call_count == 2

        # Sleep 1+ second to ensure filesystem mtime changes
        time.sleep(1.1)
        # Rewrite HEAD file to force mtime change
        head_file.write_text("ref: refs/heads/develop\n")

        # Next call should detect new mtime and re-invoke
        result2 = get_git_info(str(tmp_path))
        assert result2 == ("develop", False)
        assert subprocess_call_count == 4  # 2 more subprocess calls


def test_after_ttl_expires_re_invokes(tmp_path):
    """After 2s TTL expires, next call re-invokes even without mtime change."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    head_file = git_dir / "HEAD"
    head_file.write_text("ref: refs/heads/main\n")

    subprocess_call_count = 0

    def mock_subprocess_run(*args, **kwargs):
        nonlocal subprocess_call_count
        subprocess_call_count += 1
        mock_result = MagicMock()
        if "rev-parse" in args[0]:
            mock_result.stdout = "main\n"
        else:
            mock_result.stdout = ""
        return mock_result

    with patch("mullet.hud.state.subprocess.run", side_effect=mock_subprocess_run):
        result1 = get_git_info(str(tmp_path))
        assert result1 == ("main", False)
        assert subprocess_call_count == 2

        # Sleep past TTL (2.0 seconds)
        time.sleep(2.1)

        # Call again — should re-invoke despite same mtime
        result2 = get_git_info(str(tmp_path))
        assert result2 == ("main", False)
        assert subprocess_call_count == 4  # 2 more subprocess calls


def test_missing_git_dir_returns_none_false(tmp_path):
    """When .git/HEAD doesn't exist, return (None, False)."""
    # tmp_path has no .git directory
    # Since .git/HEAD doesn't exist, subprocess should still be called
    # and will fail, returning (None, False)
    with patch("mullet.hud.state.subprocess.run", side_effect=FileNotFoundError("git not found")):
        result = get_git_info(str(tmp_path))
        assert result == (None, False)


def test_subprocess_error_returns_none_false(tmp_path):
    """When subprocess raises, gracefully return (None, False)."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    head_file = git_dir / "HEAD"
    head_file.write_text("ref: refs/heads/main\n")

    with patch("mullet.hud.state.subprocess.run", side_effect=FileNotFoundError("git not found")):
        result = get_git_info(str(tmp_path))
        assert result == (None, False)
