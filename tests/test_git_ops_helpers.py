"""Tests for mullet.knowledge.sources.git_ops pure helpers."""

import pytest

from mullet.knowledge.sources.git_ops import (
    _sanitise_output,
    GitError,
    _validate_name,
    CloneResult,
)


# ---------------------------------------------------------------------------
# _sanitise_output
# ---------------------------------------------------------------------------

class TestSanitiseOutput:
    def test_plain_text_unchanged(self):
        result = _sanitise_output("no tokens here")
        assert result == "no tokens here"

    def test_strips_github_personal_access_token(self):
        token = "ghp_" + "a" * 36
        result = _sanitise_output(f"Using token {token}")
        assert token not in result
        assert "****" in result

    def test_strips_fine_grained_pat(self):
        token = "github_pat_" + "a" * 80
        result = _sanitise_output(f"Token: {token}")
        assert token not in result
        assert "****" in result

    def test_strips_url_credentials(self):
        url = "https://user:secret_token@github.com/org/repo"
        result = _sanitise_output(url)
        assert "secret_token" not in result
        assert "****@github.com" in result

    def test_empty_string_unchanged(self):
        assert _sanitise_output("") == ""

    def test_returns_string(self):
        assert isinstance(_sanitise_output("test"), str)

    def test_multiple_tokens_all_stripped(self):
        t1 = "ghp_" + "b" * 36
        t2 = "ghp_" + "c" * 36
        result = _sanitise_output(f"{t1} and {t2}")
        assert t1 not in result
        assert t2 not in result


# ---------------------------------------------------------------------------
# GitError
# ---------------------------------------------------------------------------

class TestGitError:
    def test_is_exception(self):
        err = GitError("oops")
        assert isinstance(err, Exception)

    def test_message_preserved(self):
        err = GitError("clone failed")
        assert "clone failed" in str(err)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(GitError):
            raise GitError("test")


# ---------------------------------------------------------------------------
# _validate_name
# ---------------------------------------------------------------------------

class TestValidateName:
    def test_valid_alphanumeric_passes(self):
        _validate_name("myrepo")  # Should not raise

    def test_valid_with_dashes(self):
        _validate_name("my-repo")  # Should not raise

    def test_valid_with_dots(self):
        _validate_name("my.repo")  # Should not raise

    def test_valid_with_underscores(self):
        _validate_name("my_repo")  # Should not raise

    def test_slash_raises_git_error(self):
        with pytest.raises(GitError):
            _validate_name("org/repo")

    def test_space_raises_git_error(self):
        with pytest.raises(GitError):
            _validate_name("my repo")

    def test_path_traversal_raises_git_error(self):
        with pytest.raises(GitError):
            _validate_name("../evil")

    def test_dotdot_in_name_raises(self):
        with pytest.raises(GitError):
            _validate_name("re..po")

    def test_empty_string_raises(self):
        with pytest.raises(GitError):
            _validate_name("")


# ---------------------------------------------------------------------------
# CloneResult dataclass
# ---------------------------------------------------------------------------

class TestCloneResult:
    def test_success_stored(self):
        r = CloneResult(success=True, message="ok")
        assert r.success is True

    def test_failure_stored(self):
        r = CloneResult(success=False, message="failed")
        assert r.success is False

    def test_message_stored(self):
        r = CloneResult(success=True, message="cloned 42 files")
        assert r.message == "cloned 42 files"

    def test_changed_files_defaults_empty(self):
        r = CloneResult(success=True, message="ok")
        assert r.changed_files == []

    def test_is_fresh_clone_defaults_false(self):
        r = CloneResult(success=True, message="ok")
        assert r.is_fresh_clone is False

    def test_custom_changed_files(self):
        r = CloneResult(success=True, message="ok", changed_files=["a.py", "b.py"])
        assert "a.py" in r.changed_files

    def test_is_fresh_clone_can_be_true(self):
        r = CloneResult(success=True, message="fresh", is_fresh_clone=True)
        assert r.is_fresh_clone is True
