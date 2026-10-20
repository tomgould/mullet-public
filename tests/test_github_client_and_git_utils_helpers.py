"""Tests for github_client pure helpers and git_utils pure helpers."""

import pytest
from pathlib import Path

from mullet.knowledge.sources.github_client import (
    _mask_token,
    GitHubAPIError,
    AuthenticationError,
    RateLimitError,
)
from mullet.knowledge.code.git_utils import (
    RepoInfo,
    extract_repo_name,
    find_repo_for_file,
    should_skip_directory,
    GLOBAL_EXCLUDES,
)


# ---------------------------------------------------------------------------
# _mask_token
# ---------------------------------------------------------------------------

class TestMaskToken:
    def test_none_returns_none_label(self):
        assert _mask_token(None) == "(none)"

    def test_empty_returns_none_label(self):
        assert _mask_token("") == "(none)"

    def test_short_token_masked_fully(self):
        assert _mask_token("abcdefgh") == "****"

    def test_long_token_shows_ends(self):
        result = _mask_token("abcdefghij")
        assert result.startswith("abcd")
        assert result.endswith("ghij")
        assert "..." in result

    def test_github_pat_masked(self):
        result = _mask_token("ghp_" + "A" * 36)
        assert result.startswith("ghp_")
        assert "..." in result

    def test_returns_string(self):
        assert isinstance(_mask_token("test12345"), str)


# ---------------------------------------------------------------------------
# GitHubAPIError
# ---------------------------------------------------------------------------

class TestGitHubAPIError:
    def test_message_stored(self):
        e = GitHubAPIError("Request failed")
        assert "Request failed" in str(e)

    def test_status_code_stored(self):
        e = GitHubAPIError("Not found", status_code=404)
        assert e.status_code == 404

    def test_status_code_defaults_none(self):
        e = GitHubAPIError("Error")
        assert e.status_code is None

    def test_is_exception(self):
        assert isinstance(GitHubAPIError("e"), Exception)


class TestAuthenticationError:
    def test_is_github_api_error(self):
        assert isinstance(AuthenticationError("auth fail"), GitHubAPIError)

    def test_status_code_stored(self):
        e = AuthenticationError("auth fail", status_code=401)
        assert e.status_code == 401

    def test_can_be_caught_as_github_error(self):
        with pytest.raises(GitHubAPIError):
            raise AuthenticationError("auth fail")


class TestRateLimitError:
    def test_is_github_api_error(self):
        assert isinstance(RateLimitError("rate limit"), GitHubAPIError)

    def test_status_code_stored(self):
        e = RateLimitError("rate limit", status_code=403)
        assert e.status_code == 403

    def test_can_be_caught_as_github_error(self):
        with pytest.raises(GitHubAPIError):
            raise RateLimitError("rate limit")


# ---------------------------------------------------------------------------
# RepoInfo
# ---------------------------------------------------------------------------

class TestRepoInfo:
    def _make(self, root="/src/repo"):
        return RepoInfo(
            repo_root=Path(root),
            repo_name="myrepo",
            remote_url="https://github.com/owner/myrepo.git",
            current_branch="main",
        )

    def test_repo_root_stored(self):
        ri = self._make()
        assert ri.repo_root == Path("/src/repo")

    def test_repo_name_stored(self):
        ri = self._make()
        assert ri.repo_name == "myrepo"

    def test_remote_url_stored(self):
        ri = self._make()
        assert ri.remote_url == "https://github.com/owner/myrepo.git"

    def test_current_branch_stored(self):
        ri = self._make()
        assert ri.current_branch == "main"

    def test_get_relative_path_inside_repo(self):
        ri = self._make("/src/repo")
        rel = ri.get_relative_path(Path("/src/repo/src/app.py"))
        assert rel == "src/app.py"

    def test_get_relative_path_outside_repo(self):
        ri = self._make("/src/repo")
        rel = ri.get_relative_path(Path("/other/file.py"))
        assert "/other/file.py" in rel

    def test_get_relative_path_returns_string(self):
        ri = self._make("/src/repo")
        assert isinstance(ri.get_relative_path(Path("/src/repo/file.py")), str)


# ---------------------------------------------------------------------------
# extract_repo_name
# ---------------------------------------------------------------------------

class TestExtractRepoName:
    def test_https_url_with_git_suffix(self):
        result = extract_repo_name("https://github.com/owner/myrepo.git", Path("/p"))
        assert result == "myrepo"

    def test_https_url_without_git_suffix(self):
        result = extract_repo_name("https://github.com/owner/myrepo", Path("/p"))
        assert result == "myrepo"

    def test_ssh_url(self):
        result = extract_repo_name("git@github.com:owner/myrepo.git", Path("/p"))
        assert result == "myrepo"

    def test_none_url_uses_dirname(self):
        result = extract_repo_name(None, Path("/path/to/my-project"))
        assert result == "my-project"

    def test_empty_url_uses_dirname(self):
        result = extract_repo_name("", Path("/path/to/fallback-repo"))
        assert result == "fallback-repo"

    def test_returns_string(self):
        assert isinstance(extract_repo_name(None, Path("/p/repo")), str)


# ---------------------------------------------------------------------------
# should_skip_directory
# ---------------------------------------------------------------------------

class TestShouldSkipDirectory:
    def test_node_modules_skipped(self):
        assert should_skip_directory("node_modules") is True

    def test_git_dir_skipped(self):
        assert should_skip_directory(".git") is True

    def test_pycache_skipped(self):
        assert should_skip_directory("__pycache__") is True

    def test_venv_skipped(self):
        assert should_skip_directory("venv") is True

    def test_src_not_skipped(self):
        assert should_skip_directory("src") is False

    def test_tests_not_skipped(self):
        assert should_skip_directory("tests") is False

    def test_egg_info_skipped(self):
        assert should_skip_directory("mypackage.egg-info") is True

    def test_any_egg_info_skipped(self):
        assert should_skip_directory("other_pkg.egg-info") is True

    def test_returns_bool(self):
        assert isinstance(should_skip_directory("src"), bool)


# ---------------------------------------------------------------------------
# GLOBAL_EXCLUDES
# ---------------------------------------------------------------------------

class TestGlobalExcludes:
    def test_is_set(self):
        assert isinstance(GLOBAL_EXCLUDES, (set, frozenset))

    def test_nonempty(self):
        assert len(GLOBAL_EXCLUDES) > 0

    def test_node_modules_in_excludes(self):
        assert "node_modules" in GLOBAL_EXCLUDES

    def test_git_in_excludes(self):
        assert ".git" in GLOBAL_EXCLUDES

    def test_pycache_in_excludes(self):
        assert "__pycache__" in GLOBAL_EXCLUDES


# ---------------------------------------------------------------------------
# find_repo_for_file
# ---------------------------------------------------------------------------

class TestFindRepoForFile:
    def _repos(self):
        return [
            RepoInfo(Path("/src/repo1"), "r1", None, None),
            RepoInfo(Path("/src/repo2"), "r2", None, None),
        ]

    def test_returns_matching_repo(self):
        result = find_repo_for_file(Path("/src/repo2/some/file.py"), self._repos())
        assert result is not None
        assert result.repo_name == "r2"

    def test_returns_none_when_no_match(self):
        result = find_repo_for_file(Path("/other/file.py"), self._repos())
        assert result is None

    def test_returns_first_match(self):
        repos = [
            RepoInfo(Path("/src"), "root", None, None),
            RepoInfo(Path("/src/repo1"), "r1", None, None),
        ]
        result = find_repo_for_file(Path("/src/repo1/file.py"), repos)
        assert result is not None

    def test_empty_repos_returns_none(self):
        result = find_repo_for_file(Path("/any/file.py"), [])
        assert result is None
