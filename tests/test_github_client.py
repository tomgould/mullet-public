"""Tests for GitHub API client rate limit and error handling."""

from unittest.mock import MagicMock, patch
import pytest
from mullet.knowledge.sources.github_client import (
    GitHubClient, GitHubAPIError, AuthenticationError, RateLimitError, _mask_token,
)


def _make_response(status_code: int, json_data=None, headers=None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.headers = headers or {}
    resp.raise_for_status = MagicMock()
    return resp


def test_mask_token_short():
    assert _mask_token("abc") == "****"


def test_mask_token_long():
    result = _mask_token("ghp_abcdefghijklmn")
    assert result.startswith("ghp_")
    assert result.endswith("lmn")
    assert "..." in result


def test_mask_token_none():
    assert _mask_token(None) == "(none)"


def test_rate_limit_error_raised_on_403_empty_remaining():
    client = GitHubClient(token="test")
    resp = _make_response(403, headers={
        "x-ratelimit-remaining": "0",
        "x-ratelimit-reset": "9999999999",
    })
    with patch.object(client._client, "request", return_value=resp):
        with pytest.raises(RateLimitError):
            client._request("GET", "/repos")


def test_auth_error_raised_on_401():
    client = GitHubClient(token="bad-token")
    resp = _make_response(401, headers={})
    with patch.object(client._client, "request", return_value=resp):
        with pytest.raises(AuthenticationError):
            client._request("GET", "/user")


def test_server_error_retries_then_raises():
    client = GitHubClient(token="test")
    call_count = 0
    resp = _make_response(500, headers={})

    def fake_request(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return resp

    with patch.object(client._client, "request", side_effect=fake_request):
        with patch("time.sleep"):
            with pytest.raises(GitHubAPIError):
                client._request("GET", "/repos")
    assert call_count == GitHubClient.MAX_RETRIES


def test_network_error_retries_then_raises():
    import httpx
    client = GitHubClient(token="test")
    client.RETRY_BACKOFF = 0
    with patch.object(client._client, "request", side_effect=httpx.TransportError("timeout")):
        with patch("time.sleep"):
            with pytest.raises(GitHubAPIError, match="Network error"):
                client._request("GET", "/repos")


def test_403_non_rate_limit_raises_generic_error():
    client = GitHubClient(token="test")
    resp = _make_response(403, headers={"x-ratelimit-remaining": "50"})
    with patch.object(client._client, "request", return_value=resp):
        with pytest.raises(GitHubAPIError):
            client._request("GET", "/repos")
        # Should NOT be a RateLimitError (remaining > 0)
        try:
            client._request("GET", "/repos")
        except RateLimitError:
            pytest.fail("Should not raise RateLimitError when remaining > 0")
        except GitHubAPIError:
            pass  # expected


def test_successful_request_returns_response():
    client = GitHubClient(token="test")
    resp = _make_response(200, json_data={"name": "testrepo"}, headers={})
    with patch.object(client._client, "request", return_value=resp):
        result = client._request("GET", "/repos/owner/repo")
    assert result.status_code == 200


class TestContextManager:
    def test_enter_returns_self(self):
        """Lines 76-77: __enter__ returns the client instance."""
        client = GitHubClient(token="tok")
        with patch.object(client._client, "close"):
            result = client.__enter__()
        assert result is client

    def test_exit_calls_close(self):
        """Line 80: __exit__ calls self.close()."""
        client = GitHubClient(token="tok")
        with patch.object(client, "close") as mock_close:
            client.__exit__(None, None, None)
        mock_close.assert_called_once()

    def test_context_manager_usage(self):
        """Lines 74, 77, 80: full with-statement usage."""
        with GitHubClient(token="tok") as client:
            assert isinstance(client, GitHubClient)
        # close() was called on exit — client._client.close() should have been invoked


class TestGetPaginated:
    def _make_response(self, status_code=200, json_data=None, link_header=""):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data if json_data is not None else []
        resp.headers = {"link": link_header}
        resp.raise_for_status = MagicMock()
        return resp

    def test_returns_all_items_single_page(self):
        """Lines 131-161: single page of results returned correctly."""
        client = GitHubClient(token="tok")
        items = [{"id": 1}, {"id": 2}]
        resp = self._make_response(json_data=items)
        with patch.object(client, "_request", return_value=resp):
            result = client._get_paginated("/repos")
        assert result == items

    def test_empty_response_returns_empty_list(self):
        """Line 148-149: empty JSON → break immediately → empty list."""
        client = GitHubClient(token="tok")
        resp = self._make_response(json_data=[])
        with patch.object(client, "_request", return_value=resp):
            result = client._get_paginated("/repos")
        assert result == []

    def test_404_raises_github_api_error(self):
        """Lines 140-144: 404 response → GitHubAPIError raised."""
        client = GitHubClient(token="tok")
        resp = self._make_response(status_code=404)
        with patch.object(client, "_request", return_value=resp):
            with pytest.raises(GitHubAPIError, match="Not found"):
                client._get_paginated("/repos/missing")

    def test_pagination_with_next_link(self):
        """Lines 153-157: link header with rel=next → fetches page 2."""
        client = GitHubClient(token="tok")
        page1 = self._make_response(json_data=[{"id": 1}], link_header='<...>; rel="next"')
        page2 = self._make_response(json_data=[{"id": 2}])
        with patch.object(client, "_request", side_effect=[page1, page2]):
            result = client._get_paginated("/repos")
        assert len(result) == 2

    def test_max_pages_stops_pagination(self):
        """Lines 158-159: max_pages=1 stops after first page even with next link."""
        client = GitHubClient(token="tok")
        page1 = self._make_response(json_data=[{"id": 1}], link_header='<...>; rel="next"')
        with patch.object(client, "_request", return_value=page1):
            result = client._get_paginated("/repos", max_pages=1)
        assert len(result) == 1


class TestGetAuthenticatedLogin:
    def _make_response(self, status_code=200, json_data=None):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data or {}
        resp.raise_for_status = MagicMock()
        return resp

    def test_cached_login_returned_immediately(self):
        """Line 165-166: cached login → no request made."""
        client = GitHubClient(token="tok")
        client._authenticated_login = "cached_user"
        with patch.object(client, "_request") as mock_req:
            result = client._get_authenticated_login()
        assert result == "cached_user"
        mock_req.assert_not_called()

    def test_no_token_returns_none(self):
        """Lines 167-168: no token → returns None without request."""
        client = GitHubClient(token=None)
        assert client._get_authenticated_login() is None

    def test_success_returns_login(self):
        """Lines 169-173: successful /user fetch → returns login."""
        client = GitHubClient(token="tok")
        resp = self._make_response(json_data={"login": "myuser"})
        with patch.object(client, "_request", return_value=resp):
            result = client._get_authenticated_login()
        assert result == "myuser"
        assert client._authenticated_login == "myuser"

    def test_auth_error_propagates(self):
        """Lines 174-175: AuthenticationError is re-raised."""
        client = GitHubClient(token="bad")
        with patch.object(client, "_request", side_effect=AuthenticationError("401", 401)):
            with pytest.raises(AuthenticationError):
                client._get_authenticated_login()

    def test_api_error_returns_none(self):
        """Lines 176-178: other GitHubAPIError → warning logged, returns None."""
        client = GitHubClient(token="tok")
        with patch.object(client, "_request", side_effect=GitHubAPIError("500 error", 500)):
            result = client._get_authenticated_login()
        assert result is None


class TestIsOrg:
    def _make_response(self, status_code=200):
        resp = MagicMock()
        resp.status_code = status_code
        return resp

    def test_returns_true_for_org(self):
        """Lines 182-184: 200 response → True."""
        client = GitHubClient(token="tok")
        with patch.object(client, "_request", return_value=self._make_response(200)):
            assert client.is_org("myorg") is True

    def test_returns_false_for_user(self):
        """Line 184: non-200 → False."""
        client = GitHubClient(token="tok")
        with patch.object(client, "_request", return_value=self._make_response(404)):
            assert client.is_org("myuser") is False

    def test_auth_error_propagates(self):
        """Lines 185-186: AuthenticationError is re-raised."""
        client = GitHubClient(token="bad")
        with patch.object(client, "_request", side_effect=AuthenticationError("401", 401)):
            with pytest.raises(AuthenticationError):
                client.is_org("myorg")

    def test_api_error_returns_false(self):
        """Lines 187-189: GitHubAPIError → warning logged, returns False."""
        client = GitHubClient(token="tok")
        with patch.object(client, "_request", side_effect=GitHubAPIError("network error", None)):
            result = client.is_org("myorg")
        assert result is False


class TestListRepos:
    def _make_paginated_response(self, items):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = items
        resp.headers = {"link": ""}
        resp.raise_for_status = MagicMock()
        return resp

    def test_org_account_type_uses_orgs_url(self):
        """Lines 199-201: account_type='org' → /orgs/{owner}/repos URL."""
        client = GitHubClient(token="tok")
        repos = [{"id": 1, "private": False, "fork": False}]
        with patch.object(client, "_get_paginated", return_value=repos) as mock_pag:
            result = client.list_repos("myorg", account_type="org")
        call_url = mock_pag.call_args[0][0]
        assert "/orgs/myorg/repos" in call_url
        assert len(result) == 1

    def test_authenticated_owner_uses_user_repos(self):
        """Lines 202-206: token+authenticated login matches owner → /user/repos."""
        client = GitHubClient(token="tok")
        client._authenticated_login = "myuser"
        repos = [{"id": 1, "private": False, "fork": False}]
        with patch.object(client, "_get_authenticated_login", return_value="myuser"), \
             patch.object(client, "_get_paginated", return_value=repos) as mock_pag:
            result = client.list_repos("myuser")
        call_url = mock_pag.call_args[0][0]
        assert call_url == "/user/repos"

    def test_authenticated_different_owner_uses_users_url(self):
        """Lines 207-209: token+authenticated login does NOT match owner → /users/{owner}/repos."""
        client = GitHubClient(token="tok")
        client._authenticated_login = "otheruser"
        repos = [{"id": 1, "private": False, "fork": False}]
        with patch.object(client, "_get_authenticated_login", return_value="otheruser"), \
             patch.object(client, "_get_paginated", return_value=repos) as mock_pag:
            result = client.list_repos("someotherowner")
        call_url = mock_pag.call_args[0][0]
        assert "/users/someotherowner/repos" in call_url

    def test_no_token_uses_public_users_url(self):
        """Lines 210-212: no token → /users/{owner}/repos."""
        client = GitHubClient(token=None)
        repos = [{"id": 1, "private": False, "fork": False}]
        with patch.object(client, "_get_paginated", return_value=repos) as mock_pag:
            result = client.list_repos("publicuser")
        call_url = mock_pag.call_args[0][0]
        assert "/users/publicuser/repos" in call_url

    def test_exclude_private_repos(self):
        """Lines 217-218: include_private=False filters out private repos."""
        client = GitHubClient(token=None)
        repos = [
            {"id": 1, "private": True, "fork": False},
            {"id": 2, "private": False, "fork": False},
        ]
        with patch.object(client, "_get_paginated", return_value=repos):
            result = client.list_repos("user", include_private=False)
        assert len(result) == 1
        assert result[0]["id"] == 2

    def test_exclude_forks(self):
        """Lines 219-220: include_forks=False filters out forked repos."""
        client = GitHubClient(token=None)
        repos = [
            {"id": 1, "private": False, "fork": True},
            {"id": 2, "private": False, "fork": False},
        ]
        with patch.object(client, "_get_paginated", return_value=repos):
            result = client.list_repos("user", include_forks=False)
        assert len(result) == 1
        assert result[0]["id"] == 2


class TestGetRepo:
    def _make_response(self, status_code=200, json_data=None):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data or {}
        resp.raise_for_status = MagicMock()
        return resp

    def test_returns_repo_data_on_success(self):
        """Lines 226-230: 200 response → repo dict returned."""
        client = GitHubClient(token="tok")
        repo_data = {"name": "myrepo", "full_name": "owner/myrepo"}
        with patch.object(client, "_request", return_value=self._make_response(200, repo_data)):
            result = client.get_repo("owner", "myrepo")
        assert result == repo_data

    def test_returns_none_on_404(self):
        """Line 227-228: 404 response → None returned."""
        client = GitHubClient(token="tok")
        with patch.object(client, "_request", return_value=self._make_response(404)):
            result = client.get_repo("owner", "missing")
        assert result is None
