"""Tests for mullet.knowledge.code.git_utils — pure helpers."""

from pathlib import Path

from mullet.knowledge.code.git_utils import (
    RepoInfo,
    extract_repo_name,
    find_repo_for_file,
    should_skip_directory,
    GLOBAL_EXCLUDES,
    _is_child_repo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo(root: Path, name: str = "myrepo") -> RepoInfo:
    return RepoInfo(repo_root=root, repo_name=name, remote_url=None, current_branch=None)


# ---------------------------------------------------------------------------
# RepoInfo.get_relative_path
# ---------------------------------------------------------------------------

class TestRepoInfoGetRelativePath:
    def test_child_path(self, tmp_path):
        repo = _repo(tmp_path)
        child = tmp_path / "src" / "main.py"
        assert repo.get_relative_path(child) == "src/main.py"

    def test_exact_root(self, tmp_path):
        repo = _repo(tmp_path)
        assert repo.get_relative_path(tmp_path) == "."

    def test_unrelated_path(self, tmp_path):
        repo = _repo(tmp_path / "repo_a")
        other = tmp_path / "repo_b" / "file.py"
        # ValueError → falls back to str(file_path)
        result = repo.get_relative_path(other)
        assert "repo_b" in result


# ---------------------------------------------------------------------------
# extract_repo_name
# ---------------------------------------------------------------------------

class TestExtractRepoName:
    def test_https_url_with_git_suffix(self, tmp_path):
        url = "https://github.com/anthropics/anthropic-sdk-python.git"
        assert extract_repo_name(url, tmp_path) == "anthropic-sdk-python"

    def test_ssh_url_with_git_suffix(self, tmp_path):
        url = "git@github.com:example-org/Mullet.git"
        assert extract_repo_name(url, tmp_path) == "Mullet"

    def test_https_url_without_git_suffix(self, tmp_path):
        url = "https://github.com/org/myproject"
        assert extract_repo_name(url, tmp_path) == "myproject"

    def test_ssh_url_without_git_suffix(self, tmp_path):
        url = "git@gitlab.com:team/service"
        assert extract_repo_name(url, tmp_path) == "service"

    def test_none_url_falls_back_to_dirname(self, tmp_path):
        target = tmp_path / "my-cool-project"
        assert extract_repo_name(None, target) == "my-cool-project"

    def test_empty_string_url_falls_back_to_dirname(self, tmp_path):
        target = tmp_path / "fallback-name"
        assert extract_repo_name("", target) == "fallback-name"

    def test_nested_url_takes_last_segment(self, tmp_path):
        url = "https://bitbucket.org/org/sub/repo.git"
        assert extract_repo_name(url, tmp_path) == "repo"


# ---------------------------------------------------------------------------
# should_skip_directory
# ---------------------------------------------------------------------------

class TestShouldSkipDirectory:
    def test_node_modules_skipped(self):
        assert should_skip_directory("node_modules")

    def test_pycache_skipped(self):
        assert should_skip_directory("__pycache__")

    def test_git_skipped(self):
        assert should_skip_directory(".git")

    def test_venv_skipped(self):
        assert should_skip_directory("venv")
        assert should_skip_directory(".venv")

    def test_dist_build_skipped(self):
        assert should_skip_directory("dist")
        assert should_skip_directory("build")

    def test_egg_info_suffix_skipped(self):
        assert should_skip_directory("mypackage.egg-info")
        assert should_skip_directory("some_lib.egg-info")

    def test_regular_dir_not_skipped(self):
        assert not should_skip_directory("src")
        assert not should_skip_directory("tests")
        assert not should_skip_directory("mullet")

    def test_hidden_non_excluded_not_skipped(self):
        assert not should_skip_directory(".hidden_custom")


# ---------------------------------------------------------------------------
# GLOBAL_EXCLUDES constant
# ---------------------------------------------------------------------------

class TestGlobalExcludes:
    def test_is_a_set(self):
        assert isinstance(GLOBAL_EXCLUDES, set)

    def test_contains_common_excludes(self):
        expected = {".git", "node_modules", "__pycache__", "venv", ".venv"}
        assert expected.issubset(GLOBAL_EXCLUDES)

    def test_non_empty(self):
        assert len(GLOBAL_EXCLUDES) >= 10


# ---------------------------------------------------------------------------
# find_repo_for_file
# ---------------------------------------------------------------------------

class TestFindRepoForFile:
    def test_file_in_repo(self, tmp_path):
        repo_root = tmp_path / "myrepo"
        repo_root.mkdir()
        repo = _repo(repo_root)
        file_path = repo_root / "src" / "main.py"
        file_path.parent.mkdir()
        file_path.touch()
        result = find_repo_for_file(file_path, [repo])
        assert result is repo

    def test_file_not_in_any_repo(self, tmp_path):
        repo_root = tmp_path / "repoA"
        repo_root.mkdir()
        repo = _repo(repo_root)
        other_file = tmp_path / "other" / "file.py"
        other_file.parent.mkdir()
        other_file.touch()
        assert find_repo_for_file(other_file, [repo]) is None

    def test_empty_repo_list(self, tmp_path):
        file_path = tmp_path / "file.py"
        file_path.touch()
        assert find_repo_for_file(file_path, []) is None

    def test_most_specific_repo_selected(self, tmp_path):
        outer = tmp_path / "outer"
        inner = tmp_path / "outer" / "inner"
        outer.mkdir()
        inner.mkdir()
        repo_outer = _repo(outer, "outer")
        repo_inner = _repo(inner, "inner")
        file_path = inner / "file.py"
        file_path.touch()
        result = find_repo_for_file(file_path, [repo_outer, repo_inner])
        assert result is repo_inner

    def test_multiple_repos_correct_match(self, tmp_path):
        repo_a = tmp_path / "repoA"
        repo_b = tmp_path / "repoB"
        repo_a.mkdir()
        repo_b.mkdir()
        repos = [_repo(repo_a, "A"), _repo(repo_b, "B")]
        file_in_b = repo_b / "code.py"
        file_in_b.touch()
        result = find_repo_for_file(file_in_b, repos)
        assert result is not None
        assert result.repo_name == "B"


# ---------------------------------------------------------------------------
# _is_child_repo
# ---------------------------------------------------------------------------

class TestIsChildRepo:
    def test_child_repo_is_child(self, tmp_path):
        base = tmp_path / "base"
        child_root = base / "subrepo"
        base.mkdir()
        child_root.mkdir()
        repo = _repo(child_root)
        assert _is_child_repo(repo, base) is True

    def test_same_path_is_not_child(self, tmp_path):
        repo = _repo(tmp_path)
        assert _is_child_repo(repo, tmp_path) is False

    def test_parent_is_not_child(self, tmp_path):
        child = tmp_path / "child"
        child.mkdir()
        repo = _repo(tmp_path)
        assert _is_child_repo(repo, child) is False

    def test_sibling_is_not_child(self, tmp_path):
        sibling_a = tmp_path / "a"
        sibling_b = tmp_path / "b"
        sibling_a.mkdir()
        sibling_b.mkdir()
        repo = _repo(sibling_a)
        assert _is_child_repo(repo, sibling_b) is False
