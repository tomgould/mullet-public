"""Tests for mullet.jobs.handback.

These use real git repositories and a hand-made JobRecord, with no models,
sandbox or network.
"""

import subprocess
from pathlib import Path

import pytest

from mullet.jobs.handback import HandBack, apply_job_diff
from mullet.jobs.ledger import JobRecord
from mullet.jobs.types import JobStatus
from tests.jobs_helpers import git, good_change, make_source_repo


def _job_clone(source: Path, tmp_path: Path, *, change: bool = True,
               extra_change: dict[str, str] | None = None) -> JobRecord:
    """Create a plain job clone with a branch carrying one commit over ``source``'s HEAD."""
    base_sha = git(source, "rev-parse", "HEAD")
    clone_root = tmp_path / "job_clone"
    work = clone_root / "work"
    git_dir = clone_root / "git"
    subprocess.run(
        ["git", "clone", "-q", "--separate-git-dir", str(git_dir), str(source), str(work)],
        check=True,
    )
    git(work, "checkout", "-qb", "job", base_sha)
    if change:
        good_change(work)
    if extra_change:
        for rel, content in extra_change.items():
            path = work / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    if change or extra_change:
        git(work, "add", ".")
        git(work, "commit", "-qm", "job change")
    return JobRecord(
        id="20261025-000000-000000",
        repo_path=str(source),
        goal="add subtraction",
        base_ref="main",
        base_sha=base_sha,
        branch="job",
        status=JobStatus.DONE,
        max_attempts=3,
        profile={},
        budget=None,
        clone_dir=str(work),
        git_dir=str(git_dir),
        error=None,
        created_at=0.0,
        updated_at=0.0,
        approved_at=0.0,
    )


def _file_hashes(repo: Path) -> dict[str, str]:
    return {p: (repo / p).read_bytes() for p in ("app.py", "tests/test_app.py", "uv.lock") if (repo / p).exists()}


def _exit_code(repo: Path, *args: str) -> int:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False,
    ).returncode


@pytest.fixture
def source(tmp_path: Path) -> Path:
    return make_source_repo(tmp_path / "source")


def test_clean_apply_changes_files_and_leaves_head_and_index(source: Path, tmp_path: Path):
    job = _job_clone(source, tmp_path)
    head_before = git(source, "rev-parse", "HEAD")

    result = apply_job_diff(job, source)

    assert isinstance(result, HandBack)
    assert result.applied is True
    assert result.reason is None
    assert set(result.files) == {"app.py", "tests/test_app.py"}

    assert "def sub(a, b):" in (source / "app.py").read_text()
    assert "def test_sub():" in (source / "tests" / "test_app.py").read_text()
    assert git(source, "rev-parse", "HEAD") == head_before
    assert _exit_code(source, "diff", "--cached", "--quiet") == 0  # index unchanged
    assert _exit_code(source, "diff", "--quiet") != 0  # working tree has unstaged changes


def test_dirty_target_file_refuses_and_leaves_tree_intact(source: Path, tmp_path: Path):
    job = _job_clone(source, tmp_path)
    (source / "app.py").write_text("def add(a, b):\n    return 0\n")
    hashes_before = _file_hashes(source)
    head_before = git(source, "rev-parse", "HEAD")

    result = apply_job_diff(job, source)

    assert result.applied is False
    assert "uncommitted or untracked changes" in result.reason
    assert "app.py" in result.reason
    assert _file_hashes(source) == hashes_before
    assert git(source, "rev-parse", "HEAD") == head_before


def test_untracked_file_at_target_path_refuses(source: Path, tmp_path: Path):
    job = _job_clone(source, tmp_path, extra_change={"feature.py": "x = 1\n"})
    (source / "feature.py").write_text("caller untracked\n")
    hashes_before = _file_hashes(source)
    head_before = git(source, "rev-parse", "HEAD")

    result = apply_job_diff(job, source)

    assert result.applied is False
    assert "uncommitted or untracked changes" in result.reason
    assert "feature.py" in result.reason
    assert (source / "feature.py").read_text() == "caller untracked\n"
    assert _file_hashes(source) == hashes_before
    assert git(source, "rev-parse", "HEAD") == head_before


def test_caller_head_changed_target_file_refuses(source: Path, tmp_path: Path):
    job = _job_clone(source, tmp_path)
    (source / "app.py").write_text("def add(a, b):\n    return 0\n")
    git(source, "add", ".")
    git(source, "commit", "-qm", "change add")
    hashes_before = _file_hashes(source)
    head_before = git(source, "rev-parse", "HEAD")

    result = apply_job_diff(job, source)

    assert result.applied is False
    assert "caller's branch has moved on" in result.reason
    assert "app.py" in result.reason
    assert _file_hashes(source) == hashes_before
    assert git(source, "rev-parse", "HEAD") == head_before


def test_caller_head_changed_unrelated_files_still_applies(source: Path, tmp_path: Path):
    job = _job_clone(source, tmp_path)
    (source / "uv.lock").write_text("changed lock\n")
    git(source, "add", ".")
    git(source, "commit", "-qm", "change lock")
    head_before = git(source, "rev-parse", "HEAD")

    result = apply_job_diff(job, source)

    assert result.applied is True
    assert result.reason is None
    assert set(result.files) == {"app.py", "tests/test_app.py"}
    assert "def sub(a, b):" in (source / "app.py").read_text()
    assert (source / "uv.lock").read_text() == "changed lock\n"
    assert git(source, "rev-parse", "HEAD") == head_before
    assert _exit_code(source, "diff", "--cached", "--quiet") == 0
    assert _exit_code(source, "diff", "--quiet") != 0


def test_empty_diff_reports_nothing_to_apply(source: Path, tmp_path: Path):
    job = _job_clone(source, tmp_path, change=False)

    result = apply_job_diff(job, source)

    assert result.applied is False
    assert result.reason == "the job changed no files"
    assert result.files == []
    assert git(source, "status", "--porcelain", "--untracked-files=all") == ""


def test_file_names_are_paths_not_pathspec_patterns(source: Path, tmp_path: Path):
    # A job file named "*.py" must not be read as a glob that matches the
    # caller's own untracked notes.py.
    job = _job_clone(source, tmp_path, change=False, extra_change={"*.py": "x = 1\n"})
    (source / "notes.py").write_text("mine\n")
    result = apply_job_diff(job, source)
    assert result.applied, result.reason
    assert (source / "*.py").read_text() == "x = 1\n"
    assert (source / "notes.py").read_text() == "mine\n"
