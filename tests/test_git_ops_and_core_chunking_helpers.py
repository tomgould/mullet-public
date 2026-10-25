"""Tests for git_ops pure helpers."""

import pytest
from mullet.knowledge.sources.git_ops import (
    GitError,
    CloneResult,
    _sanitise_output,
    _validate_name,
)


# ---------------------------------------------------------------------------
# GitError
# ---------------------------------------------------------------------------

class TestGitError:
    def test_is_exception(self):
        assert issubclass(GitError, Exception)

    def test_message_stored(self):
        e = GitError("clone failed")
        assert "clone failed" in str(e)

    def test_can_raise_and_catch(self):
        with pytest.raises(GitError):
            raise GitError("error")

    def test_catchable_as_exception(self):
        with pytest.raises(Exception):
            raise GitError("error")


# ---------------------------------------------------------------------------
# CloneResult
# ---------------------------------------------------------------------------

class TestCloneResult:
    def test_success_stored(self):
        cr = CloneResult(success=True, message="ok")
        assert cr.success is True

    def test_message_stored(self):
        cr = CloneResult(success=True, message="cloned ok")
        assert cr.message == "cloned ok"

    def test_changed_files_default_empty(self):
        cr = CloneResult(success=True, message="ok")
        assert cr.changed_files == []

    def test_is_fresh_clone_default_false(self):
        cr = CloneResult(success=True, message="ok")
        assert cr.is_fresh_clone is False

    def test_custom_is_fresh_clone(self):
        cr = CloneResult(success=True, message="ok", is_fresh_clone=True)
        assert cr.is_fresh_clone is True


# ---------------------------------------------------------------------------
# _sanitise_output
# ---------------------------------------------------------------------------

class TestSanitiseOutput:
    def test_clean_text_unchanged(self):
        assert _sanitise_output("no tokens here") == "no tokens here"

    def test_url_token_masked(self):
        url = "https://user:mysecret@github.com/org/repo"
        result = _sanitise_output(url)
        assert "mysecret" not in result
        assert "****" in result

    def test_url_host_preserved(self):
        url = "https://user:secret@github.com/org/repo"
        result = _sanitise_output(url)
        assert "github.com" in result

    def test_ghp_token_masked(self):
        token = "ghp_" + "A" * 36
        result = _sanitise_output(f"token: {token}")
        assert token not in result
        assert "****" in result

    def test_empty_string_unchanged(self):
        assert _sanitise_output("") == ""

    def test_returns_string(self):
        assert isinstance(_sanitise_output("test"), str)


# ---------------------------------------------------------------------------
# _validate_name
# ---------------------------------------------------------------------------

class TestValidateName:
    def test_valid_alphanumeric_ok(self):
        _validate_name("my-repo")  # should not raise

    def test_valid_with_dots(self):
        _validate_name("repo.name")  # should not raise

    def test_valid_with_underscores(self):
        _validate_name("my_repo_123")  # should not raise

    def test_empty_raises_git_error(self):
        with pytest.raises(GitError):
            _validate_name("")

    def test_path_traversal_raises(self):
        with pytest.raises(GitError):
            _validate_name("../escape")

    def test_space_raises(self):
        with pytest.raises(GitError):
            _validate_name("my repo")

    def test_slash_raises(self):
        with pytest.raises(GitError):
            _validate_name("my/repo")
