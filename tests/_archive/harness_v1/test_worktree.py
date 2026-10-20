"""Tests for mullet.harness.coordinator.worktree."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mullet.harness.coordinator.worktree import WorktreeManager

pytestmark = pytest.mark.harness


@pytest.fixture
def tmp_git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repo with an initial commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    (repo / "README.md").write_text("# hello")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return repo


class TestWorktreeManager:
    def test_create_worktree(self, tmp_git_repo: Path, tmp_path: Path):
        base = tmp_path / "worktrees"
        mgr = WorktreeManager(repo_root=tmp_git_repo, base_dir=base)
        path = mgr.create("int-1", branch="swarm/test-1")
        assert path.exists()
        assert (path / "README.md").exists()
        assert (path / ".mullet_worktree.json").exists()

    def test_remove_worktree(self, tmp_git_repo: Path, tmp_path: Path):
        base = tmp_path / "worktrees"
        mgr = WorktreeManager(repo_root=tmp_git_repo, base_dir=base)
        path = mgr.create("int-2", branch="swarm/test-2")
        assert path.exists()
        mgr.remove("int-2")
        # Git removes the directory on worktree remove
        assert not path.exists()

    def test_list_returns_dicts(self, tmp_git_repo: Path, tmp_path: Path):
        base = tmp_path / "worktrees"
        mgr = WorktreeManager(repo_root=tmp_git_repo, base_dir=base)
        mgr.create("int-3", branch="swarm/test-3")
        worktrees = mgr.list()
        assert isinstance(worktrees, list)
        assert len(worktrees) >= 1
        # Each entry should have at least a "path" key
        for wt in worktrees:
            assert "path" in wt

    def test_integrate_merge(self, tmp_git_repo: Path, tmp_path: Path):
        base = tmp_path / "worktrees"
        mgr = WorktreeManager(repo_root=tmp_git_repo, base_dir=base)
        path = mgr.create("int-4", branch="swarm/test-4")
        # Make a change in the worktree
        (path / "new.txt").write_text("hello")
        subprocess.run(
            ["git", "add", "."],
            cwd=path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "wt change"],
            cwd=path,
            check=True,
            capture_output=True,
        )
        status = mgr.integrate("int-4", strategy="merge")
        # Should succeed with no conflicts
        assert status["success"] is True
        assert (tmp_git_repo / "new.txt").exists()
