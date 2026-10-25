"""Tests for web_cache, github_client, and github_connector modules."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from mullet.knowledge.sources.git_ops import CloneResult
from mullet.knowledge.sources.github_client import (
    AuthenticationError,
    GitHubAPIError,
    GitHubClient,
    RateLimitError,
    _mask_token,
)
from mullet.knowledge.sources.github_connector import GitHubSourceConnector
from mullet.knowledge.sources.github_models import GitHubSourceConfig, RepoInfo
from mullet.knowledge.sources.web_cache import CACHE_FILENAME, CACHE_VERSION, CrawlCache, _get_header


# =============================================================================
# web_cache.py
# =============================================================================


class TestGetHeader:
    def test_exact_key(self):
        assert _get_header({"etag": "abc"}, "etag") == "abc"

    def test_case_insensitive_match(self):
        assert _get_header({"ETag": "abc"}, "etag") == "abc"
        assert _get_header({"etag": "abc"}, "ETag") == "abc"
        assert _get_header({"Etag": "abc"}, "ETAG") == "abc"

    def test_missing_returns_empty(self):
        assert _get_header({}, "etag") == ""
        assert _get_header({"other": "val"}, "etag") == ""


class TestCrawlCache:
    def test_init_sets_path_and_data(self, tmp_path: Path):
        cache = CrawlCache(str(tmp_path))
        assert cache.cache_path == os.path.join(str(tmp_path), CACHE_FILENAME)
        assert cache._data == {"version": CACHE_VERSION, "pages": {}}
        assert cache.page_count == 0

    def test_load_missing_file(self, tmp_path: Path):
        cache = CrawlCache(str(tmp_path))
        cache.load()
        assert cache._data == {"version": CACHE_VERSION, "pages": {}}

    def test_load_valid_file(self, tmp_path: Path):
        cache = CrawlCache(str(tmp_path))
        cache._data["pages"]["/foo"] = {"etag": "abc"}
        cache.save()
        cache2 = CrawlCache(str(tmp_path))
        cache2.load()
        assert cache2._data["pages"]["/foo"]["etag"] == "abc"

    def test_load_version_mismatch(self, tmp_path: Path, caplog):
        path = os.path.join(str(tmp_path), CACHE_FILENAME)
        with open(path, "w") as f:
            json.dump({"version": 999, "pages": {}}, f)
        cache = CrawlCache(str(tmp_path))
        with caplog.at_level("INFO", logger="mullet.knowledge.sources.web_cache"):
            cache.load()
        assert cache._data == {"version": CACHE_VERSION, "pages": {}}
        assert "version mismatch" in caplog.text.lower()

    def test_load_json_decode_error(self, tmp_path: Path, caplog):
        path = os.path.join(str(tmp_path), CACHE_FILENAME)
        with open(path, "w") as f:
            f.write("not json")
        cache = CrawlCache(str(tmp_path))
        with caplog.at_level("WARNING", logger="mullet.knowledge.sources.web_cache"):
            cache.load()
        assert cache._data == {"version": CACHE_VERSION, "pages": {}}
        assert "could not read cache" in caplog.text.lower()

    def test_load_oserror(self, tmp_path: Path, caplog):
        cache = CrawlCache(str(tmp_path))
        cache.cache_path = "/nonexistent/path/cache.json"
        with caplog.at_level("WARNING", logger="mullet.knowledge.sources.web_cache"):
            cache.load()
        assert cache._data == {"version": CACHE_VERSION, "pages": {}}

    def test_save_success(self, tmp_path: Path):
        cache = CrawlCache(str(tmp_path))
        cache._data["pages"]["/page"] = {"etag": "e"}
        cache.save()
        with open(cache.cache_path) as f:
            data = json.load(f)
        assert data["version"] == CACHE_VERSION
        assert data["pages"]["/page"]["etag"] == "e"

    def test_save_oserror_on_makedirs(self, tmp_path: Path, caplog):
        cache = CrawlCache(str(tmp_path))
        # Make cache_path a file so makedirs for its parent fails if parent is a file
        cache.cache_path = str(tmp_path / "file" / "cache.json")
        # Actually makedirs won't fail on existing dir, but open will if parent is file
        # Let's mock os.makedirs to raise
        with caplog.at_level("WARNING", logger="mullet.knowledge.sources.web_cache"):
            with patch("mullet.knowledge.sources.web_cache.os.makedirs", side_effect=OSError("boom")):
                cache.save()
        assert "could not save cache" in caplog.text.lower()

    def test_save_cleanup_on_replace_failure(self, tmp_path: Path, caplog):
        cache = CrawlCache(str(tmp_path))
        with caplog.at_level("WARNING", logger="mullet.knowledge.sources.web_cache"):
            with patch("mullet.knowledge.sources.web_cache.os.replace", side_effect=OSError("boom")):
                cache.save()
        assert "could not save cache" in caplog.text.lower()

    def test_get_page_headers_missing(self):
        cache = CrawlCache("/tmp")
        assert cache.get_page_headers("/missing") == {}

    def test_get_page_headers_etag_only(self):
        cache = CrawlCache("/tmp")
        cache._data["pages"]["/p"] = {"etag": "abc", "last_modified": ""}
        assert cache.get_page_headers("/p") == {"If-None-Match": "abc"}

    def test_get_page_headers_last_modified_only(self):
        cache = CrawlCache("/tmp")
        cache._data["pages"]["/p"] = {"etag": "", "last_modified": "Wed"}
        assert cache.get_page_headers("/p") == {"If-Modified-Since": "Wed"}

    def test_get_page_headers_both(self):
        cache = CrawlCache("/tmp")
        cache._data["pages"]["/p"] = {"etag": "abc", "last_modified": "Wed"}
        assert cache.get_page_headers("/p") == {
            "If-None-Match": "abc",
            "If-Modified-Since": "Wed",
        }

    def test_update_page(self):
        cache = CrawlCache("/tmp")
        cache.update_page("/url", {"etag": "e", "last-modified": "lm"}, "content", ["/a", "/b"])
        entry = cache._data["pages"]["/url"]
        assert entry["etag"] == "e"
        assert entry["last_modified"] == "lm"
        assert entry["links"] == ["/a", "/b"]
        expected_hash = f"sha256:{hashlib.sha256('content'.encode('utf-8')).hexdigest()}"
        assert entry["content_hash"] == expected_hash
        assert "timestamp" in entry

    def test_update_page_links_default(self):
        cache = CrawlCache("/tmp")
        cache.update_page("/url", {}, "c")
        assert cache._data["pages"]["/url"]["links"] == []

    def test_get_page_links_missing(self):
        cache = CrawlCache("/tmp")
        assert cache.get_page_links("/missing") == []

    def test_get_page_links_empty(self):
        cache = CrawlCache("/tmp")
        cache._data["pages"]["/p"] = {"links": []}
        assert cache.get_page_links("/p") == []

    def test_get_page_links_has_links(self):
        cache = CrawlCache("/tmp")
        cache._data["pages"]["/p"] = {"links": ["/a"]}
        assert cache.get_page_links("/p") == ["/a"]

    def test_is_page_unchanged_missing(self):
        cache = CrawlCache("/tmp")
        assert cache.is_page_unchanged("/p", "c") is False

    def test_is_page_unchanged_no_content_hash(self):
        cache = CrawlCache("/tmp")
        cache._data["pages"]["/p"] = {}
        assert cache.is_page_unchanged("/p", "c") is False

    def test_is_page_unchanged_changed(self):
        cache = CrawlCache("/tmp")
        cache.update_page("/p", {}, "old")
        assert cache.is_page_unchanged("/p", "new") is False

    def test_is_page_unchanged_unchanged(self):
        cache = CrawlCache("/tmp")
        cache.update_page("/p", {}, "same")
        assert cache.is_page_unchanged("/p", "same") is True

    def test_page_count(self):
        cache = CrawlCache("/tmp")
        assert cache.page_count == 0
        cache.update_page("/a", {}, "a")
        assert cache.page_count == 1
        cache.update_page("/b", {}, "b")
        assert cache.page_count == 2


# =============================================================================
# github_client.py
# =============================================================================


class TestMaskToken:
    def test_none(self):
        assert _mask_token(None) == "(none)"

    def test_short(self):
        assert _mask_token("abc") == "****"
        assert _mask_token("12345678") == "****"

    def test_long(self):
        assert _mask_token("1234567890ab") == "1234...90ab"


class TestGitHubClientInit:
    def test_no_token(self):
        client = GitHubClient()
        assert client._token is None
        assert "Authorization" not in client._client.headers

    def test_with_token(self):
        client = GitHubClient(token="ghp_xxx")
        assert client._client.headers["Authorization"] == "Bearer ghp_xxx"

    def test_context_manager(self):
        with GitHubClient() as client:
            assert isinstance(client, GitHubClient)

    def test_close(self):
        client = GitHubClient()
        with patch.object(client._client, "close") as mock_close:
            client.close()
            mock_close.assert_called_once()


class TestGitHubClientRequest:
    def _make_response(self, status_code: int, headers: dict | None = None, json_data=None):
        response = MagicMock()
        response.status_code = status_code
        response.headers = headers or {}
        response.json.return_value = json_data or {}
        return response

    def test_success(self):
        client = GitHubClient()
        resp = self._make_response(200, json_data={"ok": True})
        with patch.object(client._client, "request", return_value=resp):
            result = client._request("GET", "/test")
            assert result.status_code == 200

    def test_transport_error_then_success(self):
        client = GitHubClient()
        resp = self._make_response(200)
        with patch.object(
            client._client,
            "request",
            side_effect=[httpx.TransportError("boom"), resp],
        ):
            with patch("mullet.knowledge.sources.github_client.time.sleep"):
                result = client._request("GET", "/test")
            assert result.status_code == 200

    def test_transport_error_all_retries(self):
        client = GitHubClient()
        with patch.object(
            client._client,
            "request",
            side_effect=httpx.TransportError("boom"),
        ):
            with patch("mullet.knowledge.sources.github_client.time.sleep"):
                with pytest.raises(GitHubAPIError, match="Network error"):
                    client._request("GET", "/test")

    def test_401_raises_authentication_error(self):
        client = GitHubClient(token="ghp_1234567890")
        resp = self._make_response(401)
        with patch.object(client._client, "request", return_value=resp):
            with pytest.raises(AuthenticationError, match="Authentication failed"):
                client._request("GET", "/test")

    def test_403_rate_limit(self):
        client = GitHubClient()
        reset_at = int(time.time()) + 60
        resp = self._make_response(403, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(reset_at)})
        with patch.object(client._client, "request", return_value=resp):
            with pytest.raises(RateLimitError, match="Rate limit exceeded"):
                client._request("GET", "/test")

    def test_403_forbidden_not_rate_limit(self):
        client = GitHubClient()
        resp = self._make_response(403, headers={"x-ratelimit-remaining": "1"})
        with patch.object(client._client, "request", return_value=resp):
            with pytest.raises(GitHubAPIError, match="Forbidden"):
                client._request("GET", "/test")

    def test_500_then_success(self):
        client = GitHubClient()
        resp = self._make_response(200)
        with patch.object(client._client, "request", side_effect=[self._make_response(500), resp]):
            with patch("mullet.knowledge.sources.github_client.time.sleep"):
                result = client._request("GET", "/test")
            assert result.status_code == 200

    def test_500_all_retries(self):
        client = GitHubClient()
        with patch.object(client._client, "request", return_value=self._make_response(500)):
            with patch("mullet.knowledge.sources.github_client.time.sleep"):
                with pytest.raises(GitHubAPIError, match="Server error"):
                    client._request("GET", "/test")

    def test_transport_error_exhausts_retries(self):
        client = GitHubClient()
        with patch.object(
            client._client,
            "request",
            side_effect=httpx.TransportError("boom"),
        ):
            with patch("mullet.knowledge.sources.github_client.time.sleep"):
                with pytest.raises(GitHubAPIError, match="Network error"):
                    client._request("GET", "/test")


class TestGitHubClientPaginated:
    def _make_response(self, status_code: int, headers: dict | None = None, json_data=None):
        response = MagicMock()
        response.status_code = status_code
        response.headers = headers or {}
        response.json.return_value = json_data or []
        return response

    def test_single_page(self):
        client = GitHubClient()
        resp = self._make_response(200, json_data=[{"id": 1}])
        with patch.object(client._client, "request", return_value=resp):
            items = client._get_paginated("/repos")
        assert items == [{"id": 1}]

    def test_multiple_pages(self):
        client = GitHubClient()
        resp1 = self._make_response(
            200,
            headers={"link": '<https://api.github.com/repos?page=2>; rel="next"'},
            json_data=[{"id": 1}],
        )
        resp2 = self._make_response(200, json_data=[{"id": 2}])
        with patch.object(client._client, "request", side_effect=[resp1, resp2]):
            items = client._get_paginated("/repos")
        assert items == [{"id": 1}, {"id": 2}]

    def test_empty_result(self):
        client = GitHubClient()
        resp = self._make_response(200, json_data=[])
        with patch.object(client._client, "request", return_value=resp):
            items = client._get_paginated("/repos")
        assert items == []

    def test_404_raises(self):
        client = GitHubClient()
        resp = self._make_response(404)
        with patch.object(client._client, "request", return_value=resp):
            with pytest.raises(GitHubAPIError, match="Not found"):
                client._get_paginated("/repos")

    def test_max_pages_limit(self):
        client = GitHubClient()
        resp = self._make_response(
            200,
            headers={"link": '<https://api.github.com/repos?page=2>; rel="next"'},
            json_data=[{"id": 1}],
        )
        with patch.object(client._client, "request", return_value=resp):
            items = client._get_paginated("/repos", max_pages=1)
        assert items == [{"id": 1}]

    def test_params_per_page_default(self):
        client = GitHubClient()
        resp = self._make_response(200, json_data=[{"id": 1}])
        with patch.object(client._client, "request", return_value=resp) as mock_req:
            client._get_paginated("/repos")
            call_kwargs = mock_req.call_args[1]
            assert call_kwargs["params"]["per_page"] == 100


class TestGitHubClientAuthLogin:
    def test_cached(self):
        client = GitHubClient()
        client._authenticated_login = "alice"
        assert client._get_authenticated_login() == "alice"

    def test_no_token(self):
        client = GitHubClient()
        assert client._get_authenticated_login() is None

    def test_success(self):
        client = GitHubClient(token="ghp_xxx")
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"login": "alice"}
        with patch.object(client._client, "request", return_value=resp):
            assert client._get_authenticated_login() == "alice"

    def test_auth_error_reraised(self):
        client = GitHubClient(token="ghp_xxx")
        resp = MagicMock()
        resp.status_code = 401
        with patch.object(client._client, "request", return_value=resp):
            with pytest.raises(AuthenticationError):
                client._get_authenticated_login()

    def test_other_error_returns_none(self, caplog):
        client = GitHubClient(token="ghp_xxx")
        resp = MagicMock()
        resp.status_code = 500
        with patch.object(client._client, "request", return_value=resp):
            with caplog.at_level("WARNING", logger="mullet.knowledge.sources.github_client"):
                assert client._get_authenticated_login() is None
        assert "could not fetch authenticated user" in caplog.text.lower()


class TestGitHubClientIsOrg:
    def test_true(self):
        client = GitHubClient()
        resp = MagicMock()
        resp.status_code = 200
        with patch.object(client._client, "request", return_value=resp):
            assert client.is_org("myorg") is True

    def test_false(self):
        client = GitHubClient()
        resp = MagicMock()
        resp.status_code = 404
        with patch.object(client._client, "request", return_value=resp):
            assert client.is_org("myuser") is False

    def test_auth_error_reraised(self):
        client = GitHubClient()
        resp = MagicMock()
        resp.status_code = 401
        with patch.object(client._client, "request", return_value=resp):
            with pytest.raises(AuthenticationError):
                client.is_org("myorg")


class TestGitHubClientListRepos:
    def test_org(self):
        client = GitHubClient()
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}
        resp.json.return_value = [{"name": "r1", "private": False, "fork": False}]
        with patch.object(client._client, "request", return_value=resp):
            repos = client.list_repos("myorg", account_type="org")
        assert len(repos) == 1
        assert repos[0]["name"] == "r1"

    def test_user_with_token_matching_owner(self):
        client = GitHubClient(token="ghp_xxx")
        client._authenticated_login = "alice"
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}
        resp.json.return_value = [{"name": "r1", "private": False, "fork": False}]
        with patch.object(client._client, "request", return_value=resp) as mock_req:
            repos = client.list_repos("alice")
            assert "/user/repos" in mock_req.call_args[0][1]
        assert len(repos) == 1

    def test_user_with_token_not_matching_owner(self):
        client = GitHubClient(token="ghp_xxx")
        client._authenticated_login = "alice"
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}
        resp.json.return_value = [{"name": "r1", "private": False, "fork": False}]
        with patch.object(client._client, "request", return_value=resp) as mock_req:
            repos = client.list_repos("bob")
            assert "/users/bob/repos" in mock_req.call_args[0][1]
        assert len(repos) == 1

    def test_user_without_token(self):
        client = GitHubClient()
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}
        resp.json.return_value = [{"name": "r1", "private": False, "fork": False}]
        with patch.object(client._client, "request", return_value=resp) as mock_req:
            repos = client.list_repos("alice")
            assert "/users/alice/repos" in mock_req.call_args[0][1]
        assert len(repos) == 1

    def test_exclude_private(self):
        client = GitHubClient()
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}
        resp.json.return_value = [
            {"name": "pub", "private": False, "fork": False},
            {"name": "priv", "private": True, "fork": False},
        ]
        with patch.object(client._client, "request", return_value=resp):
            repos = client.list_repos("alice", include_private=False)
        assert [r["name"] for r in repos] == ["pub"]

    def test_exclude_forks(self):
        client = GitHubClient()
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}
        resp.json.return_value = [
            {"name": "orig", "private": False, "fork": False},
            {"name": "forked", "private": False, "fork": True},
        ]
        with patch.object(client._client, "request", return_value=resp):
            repos = client.list_repos("alice", include_forks=False)
        assert [r["name"] for r in repos] == ["orig"]


class TestGitHubClientGetRepo:
    def test_success(self):
        client = GitHubClient()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"name": "repo"}
        with patch.object(client._client, "request", return_value=resp):
            data = client.get_repo("owner", "repo")
        assert data == {"name": "repo"}

    def test_404_returns_none(self):
        client = GitHubClient()
        resp = MagicMock()
        resp.status_code = 404
        with patch.object(client._client, "request", return_value=resp):
            assert client.get_repo("owner", "repo") is None

    def test_other_error_raises(self):
        client = GitHubClient()
        resp = MagicMock()
        resp.status_code = 500
        with patch.object(client._client, "request", return_value=resp):
            with pytest.raises(GitHubAPIError, match="Server error"):
                client.get_repo("owner", "repo")


# =============================================================================
# github_connector.py
# =============================================================================


def _make_config(**kwargs):
    defaults = {
        "owner": "alice",
        "repo": "",
        "token_env": "",
        "include_repos": [],
        "exclude_repos": [],
        "include_forks": False,
        "include_private": True,
        "sync_on_ingest": True,
    }
    defaults.update(kwargs)
    return GitHubSourceConfig(**defaults)


def _make_repo_info(name="repo", owner="alice", clone_url="https://github.com/alice/repo.git"):
    return RepoInfo(
        name=name,
        owner=owner,
        url=f"https://github.com/{owner}/{name}",
        clone_url=clone_url,
        private=False,
        description="",
        default_branch="main",
        size_kb=0,
        is_fork=False,
    )


class TestGitHubSourceConnectorInit:
    def test_stores_config(self):
        cfg = _make_config()
        conn = GitHubSourceConnector(cfg)
        assert conn.config is cfg


class TestGitHubSourceConnectorStatus:
    def test_status(self):
        cfg = _make_config(owner="alice", repo="r1", token_env="GH_TOKEN")
        with patch.dict(os.environ, {"GH_TOKEN": "t"}):
            conn = GitHubSourceConnector(cfg)
            status = conn.status()
        assert status["source_type"] == "github"
        assert status["owner"] == "alice"
        assert status["repo"] == "r1"
        assert status["has_token"] is True
        assert status["sync_on_ingest"] is True


class TestGitHubSourceConnectorSync:
    def test_api_error_during_discover(self):
        cfg = _make_config()
        conn = GitHubSourceConnector(cfg)
        with patch.object(conn, "_discover_repos", side_effect=GitHubAPIError("boom")):
            result = conn.sync("/tmp/target")
        assert result.errors == ["GitHub API error: boom"]

    def test_no_repos(self):
        cfg = _make_config()
        conn = GitHubSourceConnector(cfg)
        with patch.object(conn, "_discover_repos", return_value=[]):
            result = conn.sync("/tmp/target")
        assert result.errors == []
        assert result.files_added == 0
        assert result.files_updated == 0

    def test_repos_with_changed_files(self, tmp_path: Path):
        cfg = _make_config()
        conn = GitHubSourceConnector(cfg)
        repo = _make_repo_info()
        clone_result = CloneResult(success=True, message="ok", changed_files=["a.py", "b.py"])
        with patch.object(conn, "_discover_repos", return_value=[repo]):
            with patch.object(conn, "_sync_repo", return_value=clone_result):
                with patch.object(conn, "_save_sync_metadata"):
                    result = conn.sync(str(tmp_path))
        assert result.files_updated == 2
        assert result.files_added == 0
        assert result.errors == []

    def test_repos_fresh_clone(self, tmp_path: Path):
        cfg = _make_config()
        conn = GitHubSourceConnector(cfg)
        repo = _make_repo_info()
        clone_result = CloneResult(success=True, message="ok", is_fresh_clone=True)
        with patch.object(conn, "_discover_repos", return_value=[repo]):
            with patch.object(conn, "_sync_repo", return_value=clone_result):
                with patch.object(conn, "_save_sync_metadata"):
                    result = conn.sync(str(tmp_path))
        assert result.files_updated == 0
        assert result.files_added == 1
        assert result.errors == []

    def test_repo_sync_failure(self, tmp_path: Path):
        cfg = _make_config()
        conn = GitHubSourceConnector(cfg)
        repo = _make_repo_info()
        clone_result = CloneResult(success=False, message="auth fail")
        with patch.object(conn, "_discover_repos", return_value=[repo]):
            with patch.object(conn, "_sync_repo", return_value=clone_result):
                with patch.object(conn, "_save_sync_metadata"):
                    result = conn.sync(str(tmp_path))
        assert result.errors == ["alice/repo: auth fail"]

    def test_repo_sync_exception(self, tmp_path: Path):
        cfg = _make_config()
        conn = GitHubSourceConnector(cfg)
        repo = _make_repo_info()
        with patch.object(conn, "_discover_repos", return_value=[repo]):
            with patch.object(conn, "_sync_repo", side_effect=RuntimeError("boom")):
                with patch.object(conn, "_save_sync_metadata"):
                    result = conn.sync(str(tmp_path))
        assert len(result.errors) == 1
        assert "alice/repo: boom" in result.errors[0]

    def test_metadata_save_failure(self, tmp_path: Path):
        cfg = _make_config()
        conn = GitHubSourceConnector(cfg)
        repo = _make_repo_info()
        clone_result = CloneResult(success=True, message="ok", is_fresh_clone=True)
        with patch.object(conn, "_discover_repos", return_value=[repo]):
            with patch.object(conn, "_sync_repo", return_value=clone_result):
                with patch.object(conn, "_save_sync_metadata", side_effect=OSError("disk full")):
                    result = conn.sync(str(tmp_path))
        assert any("metadata" in e.lower() for e in result.errors)


class TestGitHubSourceConnectorDiscoverRepos:
    def test_single_repo_found(self):
        cfg = _make_config(owner="alice", repo="r1")
        conn = GitHubSourceConnector(cfg)
        mock_client = MagicMock()
        mock_client.get_repo.return_value = {
            "name": "r1",
            "owner": {"login": "alice"},
            "html_url": "https://github.com/alice/r1",
            "clone_url": "https://github.com/alice/r1.git",
            "private": False,
            "description": "",
            "default_branch": "main",
            "size": 0,
            "fork": False,
        }
        mock_client.close = MagicMock()
        with patch("mullet.knowledge.sources.github_connector.GitHubClient", return_value=mock_client):
            repos = conn._discover_repos()
        assert len(repos) == 1
        assert repos[0].name == "r1"
        mock_client.close.assert_called_once()

    def test_single_repo_not_found(self):
        cfg = _make_config(owner="alice", repo="r1")
        conn = GitHubSourceConnector(cfg)
        mock_client = MagicMock()
        mock_client.get_repo.return_value = None
        mock_client.close = MagicMock()
        with patch("mullet.knowledge.sources.github_connector.GitHubClient", return_value=mock_client):
            repos = conn._discover_repos()
        assert repos == []

    def test_no_owner(self):
        cfg = _make_config(owner="")
        conn = GitHubSourceConnector(cfg)
        assert conn._discover_repos() == []

    def test_org_mode(self):
        cfg = _make_config(owner="myorg")
        conn = GitHubSourceConnector(cfg)
        mock_client = MagicMock()
        mock_client.is_org.return_value = True
        mock_client.list_repos.return_value = [
            {"name": "r1", "owner": {"login": "myorg"}, "html_url": "", "clone_url": "", "private": False, "description": "", "default_branch": "main", "size": 0, "fork": False}
        ]
        mock_client.__enter__.return_value = mock_client
        with patch("mullet.knowledge.sources.github_connector.GitHubClient", return_value=mock_client):
            repos = conn._discover_repos()
        assert len(repos) == 1
        mock_client.list_repos.assert_called_once_with("myorg", account_type="org", include_private=True, include_forks=False)

    def test_user_mode(self):
        cfg = _make_config(owner="alice")
        conn = GitHubSourceConnector(cfg)
        mock_client = MagicMock()
        mock_client.is_org.return_value = False
        mock_client.list_repos.return_value = [
            {"name": "r1", "owner": {"login": "alice"}, "html_url": "", "clone_url": "", "private": False, "description": "", "default_branch": "main", "size": 0, "fork": False}
        ]
        mock_client.__enter__.return_value = mock_client
        with patch("mullet.knowledge.sources.github_connector.GitHubClient", return_value=mock_client):
            repos = conn._discover_repos()
        assert len(repos) == 1
        mock_client.list_repos.assert_called_once_with("alice", account_type="user", include_private=True, include_forks=False)


class TestGitHubSourceConnectorFilterRepos:
    def test_include_only(self):
        cfg = _make_config(include_repos=["r1", "R2"])
        conn = GitHubSourceConnector(cfg)
        repos = [_make_repo_info("r1"), _make_repo_info("r2"), _make_repo_info("r3")]
        filtered = conn._filter_repos(repos)
        assert [r.name for r in filtered] == ["r1", "r2"]

    def test_exclude_only(self):
        cfg = _make_config(exclude_repos=["r2"])
        conn = GitHubSourceConnector(cfg)
        repos = [_make_repo_info("r1"), _make_repo_info("r2"), _make_repo_info("r3")]
        filtered = conn._filter_repos(repos)
        assert [r.name for r in filtered] == ["r1", "r3"]

    def test_include_and_exclude(self):
        cfg = _make_config(include_repos=["r1", "r2"], exclude_repos=["r2"])
        conn = GitHubSourceConnector(cfg)
        repos = [_make_repo_info("r1"), _make_repo_info("r2"), _make_repo_info("r3")]
        filtered = conn._filter_repos(repos)
        assert [r.name for r in filtered] == ["r1"]

    def test_no_filters(self):
        cfg = _make_config()
        conn = GitHubSourceConnector(cfg)
        repos = [_make_repo_info("r1"), _make_repo_info("r2")]
        filtered = conn._filter_repos(repos)
        assert [r.name for r in filtered] == ["r1", "r2"]


class TestGitHubSourceConnectorSyncRepo:
    def test_existing_repo_calls_fetch(self):
        cfg = _make_config()
        conn = GitHubSourceConnector(cfg)
        repo = _make_repo_info()
        repo_dir = Path("/tmp/fake/repo")
        fetch_result = CloneResult(success=True, message="ok")
        with patch("mullet.knowledge.sources.github_connector.fetch_repo", return_value=fetch_result) as mock_fetch:
            with patch("mullet.knowledge.sources.github_connector.clone_repo") as mock_clone:
                with patch.object(Path, "exists", return_value=True):
                    with patch.object(Path, "is_dir", return_value=True):
                        result = conn._sync_repo(repo, repo_dir)
        mock_fetch.assert_called_once_with(repo_dir, repo.clone_url, default_branch=repo.default_branch, token=None)
        mock_clone.assert_not_called()
        assert result.success is True

    def test_new_repo_calls_clone(self):
        cfg = _make_config()
        conn = GitHubSourceConnector(cfg)
        repo = _make_repo_info()
        repo_dir = Path("/tmp/fake/repo")
        clone_result = CloneResult(success=True, message="ok")
        with patch("mullet.knowledge.sources.github_connector.clone_repo", return_value=clone_result) as mock_clone:
            with patch("mullet.knowledge.sources.github_connector.fetch_repo") as mock_fetch:
                with patch.object(Path, "exists", return_value=False):
                    result = conn._sync_repo(repo, repo_dir)
        mock_clone.assert_called_once_with(repo.clone_url, repo_dir, token=None)
        mock_fetch.assert_not_called()
        assert result.success is True


class TestGitHubSourceConnectorSaveSyncMetadata:
    def test_writes_json(self, tmp_path: Path):
        cfg = _make_config(owner="alice")
        conn = GitHubSourceConnector(cfg)
        repos = [_make_repo_info("r1")]
        github_dir = tmp_path / "github"
        conn._save_sync_metadata(github_dir, repos)
        metadata_path = github_dir / "alice" / ".sync_metadata.json"
        assert metadata_path.exists()
        data = json.loads(metadata_path.read_text())
        assert data["owner"] == "alice"
        assert len(data["repos"]) == 1
        assert data["repos"][0]["name"] == "r1"
        assert "last_sync" in data
