"""Tests for mullet.knowledge.sources.github_client pure helpers."""

import pytest

from mullet.knowledge.sources.github_client import (
    GitHubAPIError,
    AuthenticationError,
    RateLimitError,
    _mask_token,
    GitHubClient,
)


# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------

class TestGitHubAPIError:
    def test_is_exception(self):
        err = GitHubAPIError("api error")
        assert isinstance(err, Exception)

    def test_message_preserved(self):
        err = GitHubAPIError("not found")
        assert "not found" in str(err)

    def test_status_code_stored(self):
        err = GitHubAPIError("not found", status_code=404)
        assert err.status_code == 404

    def test_status_code_defaults_none(self):
        err = GitHubAPIError("error")
        assert err.status_code is None

    def test_can_be_raised(self):
        with pytest.raises(GitHubAPIError):
            raise GitHubAPIError("test error")


class TestAuthenticationError:
    def test_is_github_api_error(self):
        err = AuthenticationError("unauthorized")
        assert isinstance(err, GitHubAPIError)

    def test_can_be_raised(self):
        with pytest.raises(AuthenticationError):
            raise AuthenticationError("bad creds")

    def test_caught_as_github_api_error(self):
        with pytest.raises(GitHubAPIError):
            raise AuthenticationError("unauthorized")


class TestRateLimitError:
    def test_is_github_api_error(self):
        err = RateLimitError("rate limited")
        assert isinstance(err, GitHubAPIError)

    def test_can_be_raised(self):
        with pytest.raises(RateLimitError):
            raise RateLimitError("slow down")

    def test_caught_as_github_api_error(self):
        with pytest.raises(GitHubAPIError):
            raise RateLimitError("too fast")


# ---------------------------------------------------------------------------
# _mask_token
# ---------------------------------------------------------------------------

class TestMaskToken:
    def test_none_returns_none_string(self):
        assert _mask_token(None) == "(none)"

    def test_empty_string_returns_none_string(self):
        assert _mask_token("") == "(none)"

    def test_short_token_fully_masked(self):
        assert _mask_token("short") == "****"

    def test_exactly_8_chars_fully_masked(self):
        assert _mask_token("12345678") == "****"

    def test_long_token_shows_prefix_suffix(self):
        result = _mask_token("abcdefghijklmnop")
        assert result.startswith("abcd")
        assert result.endswith("mnop")

    def test_long_token_has_ellipsis(self):
        result = _mask_token("abcdefghijklmnop")
        assert "..." in result

    def test_returns_string(self):
        assert isinstance(_mask_token("sometoken"), str)

    def test_9_char_token_shows_both_ends(self):
        # len=9 > 8 → shows first 4 and last 4 with ...
        result = _mask_token("123456789")
        assert result.startswith("1234")
        assert result.endswith("6789")


# ---------------------------------------------------------------------------
# GitHubClient constants and instantiation
# ---------------------------------------------------------------------------

class TestGitHubClientConstants:
    def test_base_url_is_github_api(self):
        assert "github.com" in GitHubClient.BASE_URL

    def test_max_retries_positive(self):
        assert GitHubClient.MAX_RETRIES > 0

    def test_retry_backoff_positive(self):
        assert GitHubClient.RETRY_BACKOFF >= 0

    def test_max_pages_absolute_large(self):
        assert GitHubClient.MAX_PAGES_ABSOLUTE >= 100


class TestGitHubClientInit:
    def test_can_instantiate_without_token(self):
        client = GitHubClient()
        assert client is not None

    def test_can_instantiate_with_token(self):
        client = GitHubClient(token="test-token")
        assert client is not None

    def test_token_stored(self):
        client = GitHubClient(token="my-token")
        assert client._token == "my-token"

    def test_no_token_is_none(self):
        client = GitHubClient()
        assert client._token is None
