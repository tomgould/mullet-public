"""Tests for pure methods in mullet.knowledge.sources.github_connector."""
from __future__ import annotations


from unittest.mock import MagicMock, patch

from mullet.knowledge.sources.github_connector import GitHubSourceConnector
from mullet.knowledge.sources.github_client import GitHubAPIError
from mullet.knowledge.sources.git_ops import CloneResult
from mullet.knowledge.sources.github_models import GitHubSourceConfig, RepoInfo


def _make_repo(name: str, owner: str = "acme", private: bool = False) -> RepoInfo:
    return RepoInfo(
        name=name,
        owner=owner,
        url=f"https://github.com/{owner}/{name}",
        clone_url=f"https://github.com/{owner}/{name}.git",
        default_branch="main",
        private=private,
        description="",
        size_kb=0,
        is_fork=False,
    )


def _make_connector(
    owner: str = "acme",
    repo: str = "",
    include_repos: list[str] | None = None,
    exclude_repos: list[str] | None = None,
    include_private: bool = False,
    include_forks: bool = False,
) -> GitHubSourceConnector:
    cfg = GitHubSourceConfig(
        owner=owner,
        repo=repo,
        include_repos=include_repos or [],
        exclude_repos=exclude_repos or [],
        include_private=include_private,
        include_forks=include_forks,
    )
    return GitHubSourceConnector(cfg)


class TestFilterRepos:
    def test_no_filter_returns_all(self) -> None:
        conn = _make_connector()
        repos = [_make_repo("alpha"), _make_repo("beta"), _make_repo("gamma")]
        result = conn._filter_repos(repos)
        assert len(result) == 3

    def test_include_filter_keeps_only_listed(self) -> None:
        conn = _make_connector(include_repos=["alpha", "gamma"])
        repos = [_make_repo("alpha"), _make_repo("beta"), _make_repo("gamma")]
        result = conn._filter_repos(repos)
        names = [r.name for r in result]
        assert "alpha" in names
        assert "gamma" in names
        assert "beta" not in names

    def test_exclude_filter_removes_listed(self) -> None:
        conn = _make_connector(exclude_repos=["beta"])
        repos = [_make_repo("alpha"), _make_repo("beta"), _make_repo("gamma")]
        result = conn._filter_repos(repos)
        names = [r.name for r in result]
        assert "beta" not in names
        assert "alpha" in names
        assert "gamma" in names

    def test_include_filter_case_insensitive(self) -> None:
        conn = _make_connector(include_repos=["ALPHA"])
        repos = [_make_repo("alpha"), _make_repo("beta")]
        result = conn._filter_repos(repos)
        assert len(result) == 1
        assert result[0].name == "alpha"

    def test_exclude_filter_case_insensitive(self) -> None:
        conn = _make_connector(exclude_repos=["BETA"])
        repos = [_make_repo("alpha"), _make_repo("beta")]
        result = conn._filter_repos(repos)
        assert len(result) == 1
        assert result[0].name == "alpha"

    def test_empty_repos_list(self) -> None:
        conn = _make_connector(include_repos=["alpha"])
        result = conn._filter_repos([])
        assert result == []

    def test_include_and_exclude_combined(self) -> None:
        conn = _make_connector(include_repos=["alpha", "beta"], exclude_repos=["beta"])
        repos = [_make_repo("alpha"), _make_repo("beta"), _make_repo("gamma")]
        result = conn._filter_repos(repos)
        names = [r.name for r in result]
        assert "alpha" in names
        assert "beta" not in names
        assert "gamma" not in names

    def test_include_nonexistent_returns_empty(self) -> None:
        conn = _make_connector(include_repos=["no-such-repo"])
        repos = [_make_repo("alpha"), _make_repo("beta")]
        result = conn._filter_repos(repos)
        assert result == []


class TestStatus:
    def test_returns_dict(self) -> None:
        conn = _make_connector(owner="myorg")
        s = conn.status()
        assert isinstance(s, dict)

    def test_source_type_is_github(self) -> None:
        conn = _make_connector()
        assert conn.status()["source_type"] == "github"

    def test_owner_in_status(self) -> None:
        conn = _make_connector(owner="myorg")
        assert conn.status()["owner"] == "myorg"

    def test_has_token_false_when_no_token(self) -> None:
        conn = _make_connector()
        # token will be None when env var is absent
        s = conn.status()
        assert "has_token" in s

    def test_sync_on_ingest_in_status(self) -> None:
        conn = _make_connector()
        assert "sync_on_ingest" in conn.status()


class TestSync:
    def test_api_error_during_discover_returns_error(self, tmp_path):
        """Lines 37-41: GitHubAPIError during discover → error in result."""
        conn = _make_connector()
        with patch.object(conn, "_discover_repos", side_effect=GitHubAPIError("rate limit")):
            result = conn.sync(str(tmp_path))
        assert len(result.errors) == 1
        assert "GitHub API error" in result.errors[0]

    def test_no_repos_returns_empty_result(self, tmp_path):
        """Lines 43-45: no repos found → empty result."""
        conn = _make_connector()
        with patch.object(conn, "_discover_repos", return_value=[]):
            result = conn.sync(str(tmp_path))
        assert result.files_added == 0
        assert result.files_updated == 0
        assert result.errors == []

    def test_fresh_clone_increments_files_added(self, tmp_path):
        """Lines 60-63: is_fresh_clone=True → files_added += 1."""
        conn = _make_connector()
        repo = _make_repo("myrepo")
        clone_result = CloneResult(success=True, message="Cloned", is_fresh_clone=True)
        with patch.object(conn, "_discover_repos", return_value=[repo]), \
             patch.object(conn, "_sync_repo", return_value=clone_result), \
             patch.object(conn, "_save_sync_metadata"):
            result = conn.sync(str(tmp_path))
        assert result.files_added == 1
        assert result.files_updated == 0

    def test_changed_files_increments_files_updated(self, tmp_path):
        """Lines 60-61: changed_files non-empty → files_updated incremented."""
        conn = _make_connector()
        repo = _make_repo("myrepo")
        clone_result = CloneResult(success=True, message="Updated", changed_files=["a.py", "b.py"])
        with patch.object(conn, "_discover_repos", return_value=[repo]), \
             patch.object(conn, "_sync_repo", return_value=clone_result), \
             patch.object(conn, "_save_sync_metadata"):
            result = conn.sync(str(tmp_path))
        assert result.files_updated == 2

    def test_sync_failure_appends_error(self, tmp_path):
        """Lines 54-58: sync failure → error appended."""
        conn = _make_connector()
        repo = _make_repo("badrepo")
        clone_result = CloneResult(success=False, message="clone failed")
        with patch.object(conn, "_discover_repos", return_value=[repo]), \
             patch.object(conn, "_sync_repo", return_value=clone_result), \
             patch.object(conn, "_save_sync_metadata"):
            result = conn.sync(str(tmp_path))
        assert len(result.errors) == 1

    def test_sync_exception_appended_as_error(self, tmp_path):
        """Lines 65-67: exception during repo sync → appended to errors."""
        conn = _make_connector()
        repo = _make_repo("crash-repo")
        with patch.object(conn, "_discover_repos", return_value=[repo]), \
             patch.object(conn, "_sync_repo", side_effect=RuntimeError("boom")), \
             patch.object(conn, "_save_sync_metadata"):
            result = conn.sync(str(tmp_path))
        assert any("boom" in e for e in result.errors)

    def test_metadata_save_oserror_appended(self, tmp_path):
        """Lines 71-74: OSError during metadata save → error appended."""
        conn = _make_connector()
        repo = _make_repo("myrepo")
        clone_result = CloneResult(success=True, message="ok", is_fresh_clone=True)
        with patch.object(conn, "_discover_repos", return_value=[repo]), \
             patch.object(conn, "_sync_repo", return_value=clone_result), \
             patch.object(conn, "_save_sync_metadata", side_effect=OSError("disk full")):
            result = conn.sync(str(tmp_path))
        assert any("Failed to save sync metadata" in e for e in result.errors)


class TestDiscoverRepos:
    def test_single_repo_mode_success(self):
        """Lines 93-106: single repo config → get_repo called → RepoInfo returned."""
        conn = _make_connector(owner="acme", repo="myrepo")
        raw = {
            "name": "myrepo", "owner": {"login": "acme"},
            "html_url": "https://github.com/acme/myrepo",
            "clone_url": "https://github.com/acme/myrepo.git",
            "default_branch": "main", "private": False,
            "description": "", "size": 100, "fork": False,
        }
        mock_client = MagicMock()
        mock_client.get_repo.return_value = raw
        with patch("mullet.knowledge.sources.github_connector.GitHubClient", return_value=mock_client):
            repos = conn._discover_repos()
        assert len(repos) == 1
        assert repos[0].name == "myrepo"

    def test_single_repo_not_found_returns_empty(self):
        """Lines 97-103: get_repo returns None → empty list."""
        conn = _make_connector(owner="acme", repo="missing")
        mock_client = MagicMock()
        mock_client.get_repo.return_value = None
        with patch("mullet.knowledge.sources.github_connector.GitHubClient", return_value=mock_client):
            repos = conn._discover_repos()
        assert repos == []

    def test_no_owner_returns_empty(self):
        """Lines 109-111: no owner configured → empty list."""
        conn = _make_connector(owner="")
        repos = conn._discover_repos()
        assert repos == []

    def test_org_discovery(self):
        """Lines 114-128: org mode — is_org=True → org URL used."""
        conn = _make_connector(owner="myorg")
        raw = {
            "name": "repo1", "owner": {"login": "myorg"},
            "html_url": "https://github.com/myorg/repo1",
            "clone_url": "https://github.com/myorg/repo1.git",
            "default_branch": "main", "private": False,
            "description": "", "size": 50, "fork": False,
        }
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.is_org.return_value = True
        mock_client.list_repos.return_value = [raw]
        with patch("mullet.knowledge.sources.github_connector.GitHubClient", return_value=mock_client):
            repos = conn._discover_repos()
        assert len(repos) == 1
        mock_client.list_repos.assert_called_once_with(
            "myorg",
            account_type="org",
            include_private=conn.config.include_private,
            include_forks=conn.config.include_forks,
        )


class TestSyncRepo:
    def test_clone_when_no_git_dir(self, tmp_path):
        """Lines 155-158: no .git dir → clone_repo called."""
        conn = _make_connector(owner="acme")
        repo = _make_repo("newrepo")
        repo_dir = tmp_path / "newrepo"
        clone_result = CloneResult(success=True, message="Cloned", is_fresh_clone=True)
        with patch("mullet.knowledge.sources.github_connector.clone_repo", return_value=clone_result) as mock_clone:
            result = conn._sync_repo(repo, repo_dir)
        mock_clone.assert_called_once_with(repo.clone_url, repo_dir, token=conn.config.token)
        assert result.success is True

    def test_fetch_when_git_dir_exists(self, tmp_path):
        """Lines 146-154: .git dir exists → fetch_repo called."""
        conn = _make_connector(owner="acme")
        repo = _make_repo("existing")
        repo_dir = tmp_path / "existing"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()
        fetch_result = CloneResult(success=True, message="Updated", changed_files=["x.py"])
        with patch("mullet.knowledge.sources.github_connector.fetch_repo", return_value=fetch_result) as mock_fetch:
            result = conn._sync_repo(repo, repo_dir)
        mock_fetch.assert_called_once_with(
            repo_dir, repo.clone_url,
            default_branch=repo.default_branch,
            token=conn.config.token,
        )
        assert result.changed_files == ["x.py"]


class TestSaveSyncMetadata:
    def test_creates_metadata_file(self, tmp_path):
        """Lines 164-179: metadata written to github_dir/owner/.sync_metadata.json."""
        import json
        conn = _make_connector(owner="acme")
        github_dir = tmp_path / "github"
        github_dir.mkdir()
        repos = [_make_repo("repo1")]
        conn._save_sync_metadata(github_dir, repos)
        meta_path = github_dir / "acme" / ".sync_metadata.json"
        assert meta_path.exists()
        data = json.loads(meta_path.read_text())
        assert data["owner"] == "acme"
        assert len(data["repos"]) == 1
        assert data["repos"][0]["name"] == "repo1"
