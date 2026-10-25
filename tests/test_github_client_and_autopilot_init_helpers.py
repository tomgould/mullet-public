"""Tests for github_client pure helpers."""

import pytest

from mullet.knowledge.sources.github_client import (
    GitHubAPIError,
    AuthenticationError,
    RateLimitError,
    _mask_token,
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
