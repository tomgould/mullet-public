"""Tests for pure functions in mullet.knowledge.code.git_utils."""
from __future__ import annotations

from pathlib import Path


from mullet.knowledge.code.git_utils import RepoInfo, extract_repo_name


# ---------------------------------------------------------------------------
# extract_repo_name
# ---------------------------------------------------------------------------

class TestExtractRepoName:
    def test_github_ssh_url(self) -> None:
        url = "git@github.com:anthropics/claude-code.git"
        name = extract_repo_name(url, Path("/some/path"))
        assert name == "claude-code"

    def test_github_https_url(self) -> None:
        url = "https://github.com/anthropics/anthropic-sdk-python.git"
        name = extract_repo_name(url, Path("/some/path"))
        assert name == "anthropic-sdk-python"

    def test_https_url_without_git_suffix(self) -> None:
        url = "https://github.com/org/my-repo"
        name = extract_repo_name(url, Path("/some/path"))
        assert name == "my-repo"

    def test_gitlab_ssh_url(self) -> None:
        url = "git@gitlab.com:company/data-pipeline.git"
        name = extract_repo_name(url, Path("/some/path"))
        assert name == "data-pipeline"

    def test_no_remote_url_uses_directory(self) -> None:
        name = extract_repo_name(None, Path("/home/user/my-project"))
        assert name == "my-project"

    def test_empty_string_remote_url_uses_directory(self) -> None:
        name = extract_repo_name("", Path("/repos/backend"))
        # Empty string is falsy, should fall back to directory
        assert name == "backend"

    def test_nested_directory_uses_basename(self) -> None:
        name = extract_repo_name(None, Path("/workspace/projects/api-service"))
        assert name == "api-service"


# ---------------------------------------------------------------------------
# RepoInfo.get_relative_path
# ---------------------------------------------------------------------------

class TestRepoInfoGetRelativePath:
    def test_relative_path_within_repo(self) -> None:
        repo = RepoInfo(
            repo_root=Path("/repos/myapp"),
            repo_name="myapp",
            remote_url=None,
            current_branch="main",
        )
        result = repo.get_relative_path(Path("/repos/myapp/src/models.py"))
        assert result == "src/models.py"

    def test_file_at_repo_root(self) -> None:
        repo = RepoInfo(
            repo_root=Path("/repos/myapp"),
            repo_name="myapp",
            remote_url=None,
            current_branch="main",
        )
        result = repo.get_relative_path(Path("/repos/myapp/README.md"))
        assert result == "README.md"

    def test_file_outside_repo_returns_absolute(self) -> None:
        repo = RepoInfo(
            repo_root=Path("/repos/myapp"),
            repo_name="myapp",
            remote_url=None,
            current_branch="main",
        )
        result = repo.get_relative_path(Path("/other/place/file.py"))
        assert result == "/other/place/file.py"

    def test_deep_nested_path(self) -> None:
        repo = RepoInfo(
            repo_root=Path("/workspace/project"),
            repo_name="project",
            remote_url=None,
            current_branch="develop",
        )
        path = Path("/workspace/project/src/auth/handlers/login.py")
        result = repo.get_relative_path(path)
        assert result == "src/auth/handlers/login.py"
