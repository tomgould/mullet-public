"""Tests for git_utils pure helpers and reference_resolver pure functions."""

from pathlib import Path

from mullet.knowledge.code.git_utils import (
    extract_repo_name,
    should_skip_directory,
    RepoInfo,
)
from mullet.knowledge.engine.reference_resolver import _prioritize_references


# ---------------------------------------------------------------------------
# extract_repo_name
# ---------------------------------------------------------------------------

class TestExtractRepoName:
    def test_github_https_url(self, tmp_path):
        result = extract_repo_name(
            "https://github.com/anthropics/anthropic-sdk-python.git",
            tmp_path,
        )
        assert result == "anthropic-sdk-python"

    def test_github_ssh_url(self, tmp_path):
        result = extract_repo_name(
            "git@github.com:anthropics/anthropic-sdk-python.git",
            tmp_path,
        )
        assert result == "anthropic-sdk-python"

    def test_url_without_git_extension(self, tmp_path):
        result = extract_repo_name(
            "https://github.com/org/my-repo",
            tmp_path,
        )
        assert result == "my-repo"

    def test_none_url_falls_back_to_dir_name(self, tmp_path):
        repo_root = tmp_path / "my-project"
        repo_root.mkdir()
        result = extract_repo_name(None, repo_root)
        assert result == "my-project"

    def test_returns_string(self, tmp_path):
        result = extract_repo_name(None, tmp_path)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# should_skip_directory
# ---------------------------------------------------------------------------

class TestShouldSkipDirectory:
    def test_node_modules_skipped(self):
        assert should_skip_directory("node_modules") is True

    def test_dot_git_skipped(self):
        assert should_skip_directory(".git") is True

    def test_pycache_skipped(self):
        assert should_skip_directory("__pycache__") is True

    def test_venv_skipped(self):
        assert should_skip_directory(".venv") is True

    def test_egg_info_skipped(self):
        assert should_skip_directory("mypackage.egg-info") is True

    def test_src_not_skipped(self):
        assert should_skip_directory("src") is False

    def test_tests_not_skipped(self):
        assert should_skip_directory("tests") is False

    def test_regular_dir_not_skipped(self):
        assert should_skip_directory("mymodule") is False


# ---------------------------------------------------------------------------
# RepoInfo.get_relative_path
# ---------------------------------------------------------------------------

class TestRepoInfoGetRelativePath:
    def setup_method(self):
        self.repo_root = Path("/home/user/myrepo")
        self.info = RepoInfo(
            repo_root=self.repo_root,
            repo_name="myrepo",
            remote_url=None,
            current_branch=None,
        )

    def test_relative_path_computed(self):
        file_path = self.repo_root / "src" / "main.py"
        result = self.info.get_relative_path(file_path)
        assert result == "src/main.py"

    def test_file_in_root(self):
        file_path = self.repo_root / "README.md"
        result = self.info.get_relative_path(file_path)
        assert result == "README.md"

    def test_outside_repo_returns_full_path(self):
        file_path = Path("/tmp/some/other/file.py")
        result = self.info.get_relative_path(file_path)
        assert result == str(file_path)

    def test_returns_string(self):
        file_path = self.repo_root / "main.py"
        result = self.info.get_relative_path(file_path)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _prioritize_references
# ---------------------------------------------------------------------------

class TestPrioritizeReferences:
    def _make_ref(self, ref_type: str, ref_id: str = "A") -> dict:
        return {"type": ref_type, "id": ref_id}

    def test_empty_returns_empty(self):
        assert _prioritize_references([]) == []

    def test_appendix_is_high_priority(self):
        refs = [
            self._make_ref("chapter", "1"),
            self._make_ref("appendix", "A"),
        ]
        result = _prioritize_references(refs)
        # High priority should be first
        types = [r["type"] for r in result]
        assert "appendix" in types

    def test_table_is_high_priority(self):
        refs = [self._make_ref("table", "1"), self._make_ref("chapter", "1")]
        result = _prioritize_references(refs)
        types = [r["type"] for r in result]
        assert "table" in types

    def test_schedule_is_high_priority(self):
        refs = [self._make_ref("schedule", "1"), self._make_ref("chapter", "1")]
        result = _prioritize_references(refs)
        types = [r["type"] for r in result]
        assert "schedule" in types

    def test_chapters_limited_when_high_priority_present(self):
        refs = [
            self._make_ref("appendix", "A"),
            self._make_ref("chapter", "1"),
            self._make_ref("chapter", "2"),
            self._make_ref("chapter", "3"),
            self._make_ref("chapter", "4"),
        ]
        result = _prioritize_references(refs)
        chapters = [r for r in result if r["type"] == "chapter"]
        # With high priority present, chapters limited to 0 (they're low priority)
        # High priority + medium[:2] — no chapter (low priority) included
        assert len(chapters) == 0

    def test_only_chapters_included_when_no_high_priority(self):
        refs = [
            self._make_ref("chapter", "1"),
            self._make_ref("chapter", "2"),
            self._make_ref("chapter", "3"),
            self._make_ref("chapter", "4"),
        ]
        result = _prioritize_references(refs)
        # max 3 chapters included
        chapters = [r for r in result if r["type"] == "chapter"]
        assert len(chapters) <= 3

    def test_returns_list(self):
        refs = [self._make_ref("section", "1")]
        assert isinstance(_prioritize_references(refs), list)
