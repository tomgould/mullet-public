"""Tests for github_client pure helpers and autopilot_init pure functions."""

import pytest
from pathlib import Path

from mullet.knowledge.sources.github_client import (
    GitHubAPIError,
    AuthenticationError,
    RateLimitError,
    _mask_token,
)
from mullet.cli.autopilot_init import (
    _target_dir,
    require_goal,
)


# ---------------------------------------------------------------------------
# GitHubAPIError
# ---------------------------------------------------------------------------

class TestGitHubAPIError:
    def test_is_exception(self):
        assert issubclass(GitHubAPIError, Exception)

    def test_message_stored(self):
        try:
            raise GitHubAPIError("Not found")
        except GitHubAPIError as e:
            assert "Not found" in str(e)

    def test_status_code_stored(self):
        err = GitHubAPIError("Not found", status_code=404)
        assert err.status_code == 404

    def test_status_code_default_none(self):
        err = GitHubAPIError("Error")
        assert err.status_code is None

    def test_can_be_caught_as_exception(self):
        with pytest.raises(Exception):
            raise GitHubAPIError("fail")


# ---------------------------------------------------------------------------
# AuthenticationError
# ---------------------------------------------------------------------------

class TestAuthenticationError:
    def test_is_github_api_error(self):
        assert issubclass(AuthenticationError, GitHubAPIError)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(AuthenticationError):
            raise AuthenticationError("Unauthorized", status_code=401)

    def test_catchable_as_github_api_error(self):
        with pytest.raises(GitHubAPIError):
            raise AuthenticationError("Unauthorized", status_code=401)

    def test_status_code_stored(self):
        err = AuthenticationError("Unauthorized", status_code=401)
        assert err.status_code == 401


# ---------------------------------------------------------------------------
# RateLimitError
# ---------------------------------------------------------------------------

class TestRateLimitError:
    def test_is_github_api_error(self):
        assert issubclass(RateLimitError, GitHubAPIError)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(RateLimitError):
            raise RateLimitError("Rate limited", status_code=429)

    def test_catchable_as_github_api_error(self):
        with pytest.raises(GitHubAPIError):
            raise RateLimitError("Rate limited", status_code=429)


# ---------------------------------------------------------------------------
# _mask_token
# ---------------------------------------------------------------------------

class TestMaskToken:
    def test_none_returns_none_string(self):
        assert _mask_token(None) == "(none)"

    def test_empty_string_returns_none_string(self):
        assert _mask_token("") == "(none)"

    def test_short_token_masked(self):
        assert _mask_token("abcd") == "****"

    def test_exactly_eight_chars_masked(self):
        assert _mask_token("12345678") == "****"

    def test_long_token_shows_first_and_last(self):
        result = _mask_token("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        assert result.startswith("ghp_")
        assert result.endswith("WXYZ")
        assert "..." in result

    def test_long_token_hides_middle(self):
        result = _mask_token("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        assert "ABCDEFGHIJKLMNOPQRSTU" not in result

    def test_returns_string(self):
        assert isinstance(_mask_token("token"), str)


# ---------------------------------------------------------------------------
# _target_dir
# ---------------------------------------------------------------------------

class TestTargetDir:
    def test_returns_path_in_mullet_autopilot(self):
        p = Path("/home/user/myproject")
        result = _target_dir(p)
        assert result == p / ".mullet" / "autopilot"

    def test_returns_path_object(self):
        assert isinstance(_target_dir(Path("/tmp")), Path)


# ---------------------------------------------------------------------------
# require_goal
# ---------------------------------------------------------------------------

class TestRequireGoal:
    def test_missing_goal_file_returns_false(self, tmp_path):
        ok, msg = require_goal(tmp_path)
        assert ok is False

    def test_missing_goal_file_message_mentions_file(self, tmp_path):
        ok, msg = require_goal(tmp_path)
        assert "goal.md" in msg

    def test_empty_goal_returns_false(self, tmp_path):
        mullet_dir = tmp_path / ".mullet" / "autopilot"
        mullet_dir.mkdir(parents=True)
        (mullet_dir / "goal.md").write_text("")
        ok, msg = require_goal(tmp_path)
        assert ok is False

    def test_sparse_goal_returns_false(self, tmp_path):
        mullet_dir = tmp_path / ".mullet" / "autopilot"
        mullet_dir.mkdir(parents=True)
        (mullet_dir / "goal.md").write_text("# Goal\n\nSomething.\n")
        ok, msg = require_goal(tmp_path)
        assert ok is False

    def test_filled_goal_returns_true(self, tmp_path):
        mullet_dir = tmp_path / ".mullet" / "autopilot"
        mullet_dir.mkdir(parents=True)
        content = (
            "## Problem\n\n"
            "Need automated deployment pipeline for our microservices.\n\n"
            "## Acceptance Criteria\n\n"
            "- CI pipeline passes all tests\n"
            "- Deploys automatically on push to main\n"
            "- Rollback mechanism works within 2 minutes\n"
        )
        (mullet_dir / "goal.md").write_text(content)
        ok, msg = require_goal(tmp_path)
        assert ok is True

    def test_filled_goal_message_is_string(self, tmp_path):
        mullet_dir = tmp_path / ".mullet" / "autopilot"
        mullet_dir.mkdir(parents=True)
        content = (
            "## Problem\n\nNeed an API gateway.\n\n"
            "## Outcome\n\nSingle entry point for all services.\n\n"
            "## Acceptance Criteria\n\n"
            "- Auth works\n- Rate limiting works\n- Logging works\n"
        )
        (mullet_dir / "goal.md").write_text(content)
        ok, msg = require_goal(tmp_path)
        assert isinstance(msg, str)

    def test_returns_tuple(self, tmp_path):
        result = require_goal(tmp_path)
        assert isinstance(result, tuple)
        assert len(result) == 2
