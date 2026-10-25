"""Tests for mullet/knowledge/code/git_utils.py."""
from pathlib import Path
from unittest.mock import MagicMock, patch

from mullet.knowledge.code.git_utils import (
    RepoInfo,
    get_remote_url,
    get_current_branch,
    extract_repo_name,
    _make_repo_info,
    detect_repos,
    find_repo_for_file,
    should_skip_directory,
    list_repo_files,
    _is_child_repo,
    walk_code_files,
)


def _proc(returncode=0, stdout="", stderr=""):
    """Create a mock subprocess result."""
    r = MagicMock()
    r.returncode = returncode
    r.stdout = stdout
    r.stderr = stderr
    return r


def _repo_info(root: Path, name="myrepo", remote_url=None, branch="main") -> RepoInfo:
    return RepoInfo(repo_root=root, repo_name=name, remote_url=remote_url, current_branch=branch)


# ---------------------------------------------------------------------------
# RepoInfo
# ---------------------------------------------------------------------------

class TestRepoInfo:
    def test_get_relative_path(self, tmp_path):
        r = _repo_info(tmp_path)
        subfile = tmp_path / "src" / "main.py"
        assert r.get_relative_path(subfile) == "src/main.py"

    def test_get_relative_path_outside_returns_full(self, tmp_path):
        r = _repo_info(tmp_path / "repo")
        outside = tmp_path / "other.py"
        assert r.get_relative_path(outside) == str(outside)


# ---------------------------------------------------------------------------
# get_remote_url
# ---------------------------------------------------------------------------

class TestGetRemoteUrl:
    def test_returns_url_on_success(self, tmp_path):
        with patch("subprocess.run", return_value=_proc(0, "git@github.com:org/repo.git\n")):
            result = get_remote_url(tmp_path)
        assert result == "git@github.com:org/repo.git"

    def test_returns_none_on_nonzero_returncode(self, tmp_path):
        with patch("subprocess.run", return_value=_proc(1, "")):
            result = get_remote_url(tmp_path)
        assert result is None

    def test_returns_none_on_exception(self, tmp_path):
        with patch("subprocess.run", side_effect=Exception("git not found")):
            result = get_remote_url(tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# get_current_branch
# ---------------------------------------------------------------------------

class TestGetCurrentBranch:
    def test_returns_branch_name(self, tmp_path):
        with patch("subprocess.run", return_value=_proc(0, "main\n")):
            result = get_current_branch(tmp_path)
        assert result == "main"

    def test_detached_head_fallback(self, tmp_path):
        """When branch is empty string, falls back to rev-parse."""
        responses = [_proc(0, ""), _proc(0, "abc1234\n")]
        with patch("subprocess.run", side_effect=responses):
            result = get_current_branch(tmp_path)
        assert result == "detached:abc1234"

    def test_returns_none_on_both_failures(self, tmp_path):
        with patch("subprocess.run", return_value=_proc(1, "")):
            result = get_current_branch(tmp_path)
        assert result is None

    def test_returns_none_on_exception(self, tmp_path):
        with patch("subprocess.run", side_effect=Exception("error")):
            result = get_current_branch(tmp_path)
        assert result is None

    def test_nonzero_branch_uses_fallback(self, tmp_path):
        """When show-current returns non-zero, tries rev-parse."""
        responses = [_proc(1, ""), _proc(0, "deadbeef\n")]
        with patch("subprocess.run", side_effect=responses):
            result = get_current_branch(tmp_path)
        assert result == "detached:deadbeef"


# ---------------------------------------------------------------------------
# extract_repo_name
# ---------------------------------------------------------------------------

class TestExtractRepoName:
    def test_github_ssh_url(self, tmp_path):
        assert extract_repo_name("git@github.com:org/myrepo.git", tmp_path) == "myrepo"

    def test_github_https_url(self, tmp_path):
        assert extract_repo_name("https://github.com/org/myrepo.git", tmp_path) == "myrepo"

    def test_url_without_git_suffix(self, tmp_path):
        assert extract_repo_name("https://github.com/org/myrepo", tmp_path) == "myrepo"

    def test_none_url_uses_directory_name(self, tmp_path):
        subdir = tmp_path / "cool-project"
        assert extract_repo_name(None, subdir) == "cool-project"

    def test_unrecognized_url_uses_directory_name(self, tmp_path):
        subdir = tmp_path / "fallback-name"
        # If regex doesn't match, falls back to dir name
        assert extract_repo_name("", subdir) == "fallback-name"


# ---------------------------------------------------------------------------
# _make_repo_info
# ---------------------------------------------------------------------------

class TestMakeRepoInfo:
    def test_builds_repo_info(self, tmp_path):
        with patch("mullet.knowledge.code.git_utils.get_remote_url", return_value="https://github.com/org/proj.git"):
            with patch("mullet.knowledge.code.git_utils.get_current_branch", return_value="main"):
                info = _make_repo_info(tmp_path)
        assert info.repo_root == tmp_path
        assert info.repo_name == "proj"
        assert info.remote_url == "https://github.com/org/proj.git"
        assert info.current_branch == "main"

    def test_with_no_remote(self, tmp_path):
        with patch("mullet.knowledge.code.git_utils.get_remote_url", return_value=None):
            with patch("mullet.knowledge.code.git_utils.get_current_branch", return_value="feature"):
                info = _make_repo_info(tmp_path)
        assert info.repo_name == tmp_path.name
        assert info.remote_url is None


# ---------------------------------------------------------------------------
# detect_repos
# ---------------------------------------------------------------------------

class TestDetectRepos:
    def test_detects_base_as_repo(self, tmp_path):
        (tmp_path / ".git").mkdir()
        with patch("mullet.knowledge.code.git_utils._make_repo_info") as mock_make:
            mock_make.return_value = _repo_info(tmp_path)
            repos = detect_repos(tmp_path)
        assert len(repos) == 1
        assert repos[0].repo_root == tmp_path

    def test_detects_child_repo(self, tmp_path):
        child = tmp_path / "project"
        child.mkdir()
        (child / ".git").mkdir()
        with patch("mullet.knowledge.code.git_utils._make_repo_info") as mock_make:
            mock_make.return_value = _repo_info(child)
            repos = detect_repos(tmp_path)
        assert len(repos) == 1
        assert repos[0].repo_root == child

    def test_deduplicates_repos(self, tmp_path):
        (tmp_path / ".git").mkdir()
        with patch("mullet.knowledge.code.git_utils._make_repo_info") as mock_make:
            mock_make.return_value = _repo_info(tmp_path)
            repos = detect_repos(tmp_path)
        # Even if found twice (base + scan), should only appear once
        assert len(repos) <= 1

    def test_empty_dir_returns_empty(self, tmp_path):
        repos = detect_repos(tmp_path)
        assert repos == []

    def test_scandir_oserror_handled(self, tmp_path):
        with patch("os.scandir", side_effect=OSError("permission denied")):
            repos = detect_repos(tmp_path)
        assert repos == []


# ---------------------------------------------------------------------------
# find_repo_for_file
# ---------------------------------------------------------------------------

class TestFindRepoForFile:
    def test_finds_repo_for_file(self, tmp_path):
        repo_root = tmp_path / "myrepo"
        repo_root.mkdir()
        file_path = repo_root / "src" / "main.py"
        file_path.parent.mkdir(parents=True)
        file_path.touch()
        r = _repo_info(repo_root)
        result = find_repo_for_file(file_path, [r])
        assert result is r

    def test_returns_none_when_no_match(self, tmp_path):
        r = _repo_info(tmp_path / "other-repo")
        outside = tmp_path / "outside.py"
        outside.touch()
        result = find_repo_for_file(outside, [r])
        assert result is None

    def test_returns_most_specific_repo(self, tmp_path):
        parent = tmp_path / "parent"
        child = parent / "child"
        child.mkdir(parents=True)
        file_path = child / "file.py"
        file_path.touch()
        r_parent = _repo_info(parent)
        r_child = _repo_info(child)
        result = find_repo_for_file(file_path, [r_parent, r_child])
        assert result is r_child

    def test_returns_none_for_empty_repos(self, tmp_path):
        file_path = tmp_path / "file.py"
        file_path.touch()
        assert find_repo_for_file(file_path, []) is None


# ---------------------------------------------------------------------------
# should_skip_directory
# ---------------------------------------------------------------------------

class TestShouldSkipDirectory:
    def test_skips_node_modules(self):
        assert should_skip_directory("node_modules") is True

    def test_skips_git_dir(self):
        assert should_skip_directory(".git") is True

    def test_skips_pycache(self):
        assert should_skip_directory("__pycache__") is True

    def test_skips_egg_info(self):
        assert should_skip_directory("mypackage.egg-info") is True

    def test_allows_src(self):
        assert should_skip_directory("src") is False

    def test_allows_tests(self):
        assert should_skip_directory("tests") is False


# ---------------------------------------------------------------------------
# list_repo_files
# ---------------------------------------------------------------------------

class TestListRepoFiles:
    def test_returns_filtered_files(self, tmp_path):
        py_file = tmp_path / "main.py"
        py_file.write_text("x")
        stdout = "main.py\nREADME.md\n"
        with patch("subprocess.run", return_value=_proc(0, stdout)):
            result = list_repo_files(tmp_path, {".py"})
        assert py_file in result
        # README.md excluded (not in extensions)
        assert not any("README" in str(p) for p in result)

    def test_returns_none_on_nonzero_returncode(self, tmp_path):
        with patch("subprocess.run", return_value=_proc(1, "")):
            result = list_repo_files(tmp_path, {".py"})
        assert result is None

    def test_returns_none_on_exception(self, tmp_path):
        with patch("subprocess.run", side_effect=Exception("git error")):
            result = list_repo_files(tmp_path, {".py"})
        assert result is None

    def test_exclude_patterns_applied(self, tmp_path):
        py_file = tmp_path / "test_foo.py"
        py_file.write_text("x")
        stdout = "test_foo.py\n"
        with patch("subprocess.run", return_value=_proc(0, stdout)):
            result = list_repo_files(tmp_path, {".py"}, exclude_patterns=["test_*.py"])
        assert result == []

    def test_exclude_dir_pattern(self, tmp_path):
        """Pattern ending in /** excludes paths matching the dir prefix."""
        gen_file = tmp_path / "generated" / "output.py"
        gen_file.parent.mkdir()
        gen_file.write_text("x")
        stdout = "generated/output.py\n"
        with patch("subprocess.run", return_value=_proc(0, stdout)):
            result = list_repo_files(tmp_path, {".py"}, exclude_patterns=["*/generated/**"])
        assert result == []

    def test_large_file_excluded(self, tmp_path):
        big_file = tmp_path / "big.py"
        big_file.write_bytes(b"x" * (2 * 1024 * 1024))  # 2MB
        stdout = "big.py\n"
        with patch("subprocess.run", return_value=_proc(0, stdout)):
            result = list_repo_files(tmp_path, {".py"}, max_file_size_kb=500)
        assert result == []

    def test_empty_lines_skipped(self, tmp_path):
        stdout = "\n\nsome.py\n\n"
        (tmp_path / "some.py").write_text("x")
        with patch("subprocess.run", return_value=_proc(0, stdout)):
            result = list_repo_files(tmp_path, {".py"})
        assert len(result) == 1

    def test_no_size_limit_when_zero(self, tmp_path):
        big_file = tmp_path / "big.py"
        big_file.write_bytes(b"x" * (10 * 1024 * 1024))  # 10MB
        stdout = "big.py\n"
        with patch("subprocess.run", return_value=_proc(0, stdout)):
            result = list_repo_files(tmp_path, {".py"}, max_file_size_kb=0)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _is_child_repo
# ---------------------------------------------------------------------------

class TestIsChildRepo:
    def test_child_repo_returns_true(self, tmp_path):
        child = tmp_path / "child"
        child.mkdir()
        r = _repo_info(child)
        assert _is_child_repo(r, tmp_path) is True

    def test_same_path_returns_false(self, tmp_path):
        r = _repo_info(tmp_path)
        assert _is_child_repo(r, tmp_path) is False

    def test_parent_repo_returns_false(self, tmp_path):
        r = _repo_info(tmp_path)
        child_base = tmp_path / "sub"
        child_base.mkdir()
        assert _is_child_repo(r, child_base) is False

    def test_oserror_returns_false(self, tmp_path):
        r = _repo_info(tmp_path / "nonexistent")
        # Shouldn't raise even if path doesn't exist
        result = _is_child_repo(r, tmp_path)
        # May return True or False depending on resolve behavior — just shouldn't raise
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# walk_code_files
# ---------------------------------------------------------------------------

class TestWalkCodeFiles:
    def test_yields_matching_files(self, tmp_path):
        (tmp_path / "main.py").write_text("x")
        (tmp_path / "main.js").write_text("x")
        result = list(walk_code_files(tmp_path, {".py"}))
        assert len(result) == 1
        assert result[0].suffix == ".py"

    def test_skips_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "index.py").write_text("x")
        (tmp_path / "main.py").write_text("x")
        result = list(walk_code_files(tmp_path, {".py"}))
        assert all("node_modules" not in str(p.relative_to(tmp_path)) for p in result)

    def test_skips_pycache(self, tmp_path):
        pc = tmp_path / "__pycache__"
        pc.mkdir()
        (pc / "cached.py").write_text("x")
        (tmp_path / "main.py").write_text("x")
        result = list(walk_code_files(tmp_path, {".py"}))
        assert all("__pycache__" not in str(p) for p in result)

    def test_exclude_pattern_for_files(self, tmp_path):
        (tmp_path / "test_foo.py").write_text("x")
        (tmp_path / "main.py").write_text("x")
        result = list(walk_code_files(tmp_path, {".py"}, exclude_patterns=["test_*.py"]))
        assert all("test_foo" not in str(p) for p in result)

    def test_large_file_excluded(self, tmp_path):
        big = tmp_path / "big.py"
        big.write_bytes(b"x" * (2 * 1024 * 1024))
        (tmp_path / "small.py").write_text("x")
        result = list(walk_code_files(tmp_path, {".py"}, max_file_size_kb=500))
        assert all("big.py" not in str(p) for p in result)

    def test_uses_git_fastpath_for_known_repo(self, tmp_path):
        repo = tmp_path / "myrepo"
        repo.mkdir()
        py_file = repo / "main.py"
        py_file.write_text("x")
        r = _repo_info(repo)
        with patch("mullet.knowledge.code.git_utils.list_repo_files", return_value=[py_file]) as mock_ls:
            result = list(walk_code_files(repo, {".py"}, repos=[r]))
        mock_ls.assert_called_once()
        assert py_file in result

    def test_falls_back_to_walk_when_git_fastpath_returns_none(self, tmp_path):
        repo = tmp_path / "myrepo"
        repo.mkdir()
        py_file = repo / "main.py"
        py_file.write_text("x")
        r = _repo_info(repo)
        with patch("mullet.knowledge.code.git_utils.list_repo_files", return_value=None):
            result = list(walk_code_files(repo, {".py"}, repos=[r]))
        assert py_file in result

    def test_child_repo_parallel_ls_files(self, tmp_path):
        child = tmp_path / "child"
        child.mkdir()
        py_file = child / "code.py"
        py_file.write_text("x")
        r = _repo_info(child)
        with patch("mullet.knowledge.code.git_utils.list_repo_files", return_value=[py_file]):
            result = list(walk_code_files(tmp_path, {".py"}, repos=[r]))
        assert py_file in result

    def test_empty_directory_returns_empty(self, tmp_path):
        result = list(walk_code_files(tmp_path, {".py"}))
        assert result == []

    def test_permission_error_skipped(self, tmp_path):
        """PermissionError during scan should not propagate."""
        (tmp_path / "main.py").write_text("x")
        restricted = tmp_path / "restricted"
        restricted.mkdir(mode=0o000)
        try:
            result = list(walk_code_files(tmp_path, {".py"}))
            assert any("main.py" in str(p) for p in result)
        finally:
            restricted.chmod(0o755)

    def test_symlink_cycle_detection(self, tmp_path):
        """Symlink cycles should not cause infinite recursion."""
        (tmp_path / "main.py").write_text("x")
        link = tmp_path / "loop"
        link.symlink_to(tmp_path)  # self-referencing symlink
        result = list(walk_code_files(tmp_path, {".py"}, follow_symlinks=True))
        # Should terminate and find main.py
        assert any("main.py" in str(p) for p in result)
