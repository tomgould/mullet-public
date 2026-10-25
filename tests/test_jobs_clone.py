"""Tests for mullet.jobs.clone, using real git repositories in tmp_path."""

import subprocess
from pathlib import Path

import pytest

from mullet.jobs.clone import GitError, JobClone, resolve_commit


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


@pytest.fixture
def jobs_home(tmp_path, monkeypatch):
    home = tmp_path / "jobs"
    home.mkdir()
    monkeypatch.setenv("MULLET_JOBS_DIR", str(home))
    return home


@pytest.fixture
def source(tmp_path):
    repo = tmp_path / "src"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    (repo / "app.py").write_text("print('hi')\n")
    (repo / "README.md").write_text("readme\n")
    (repo / ".gitignore").write_text("node_modules/\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "init")
    return repo


def _user_repo_state(repo: Path) -> tuple[str, str, str]:
    return (
        git(repo, "for-each-ref", "--format=%(refname) %(objectname)"),
        git(repo, "status", "--porcelain", "--untracked-files=all"),
        git(repo, "rev-parse", "HEAD"),
    )


@pytest.fixture
def clone(jobs_home, source):
    base = resolve_commit(source, "HEAD")
    return JobClone.create(source, base, jobs_home / "job1", "mullet/job1", identity=("Dev", "dev@example.com"))


def test_create_checks_out_a_private_branch(clone, source):
    assert clone.branch() == "mullet/job1"
    assert clone.head() == resolve_commit(source, "HEAD")
    assert (clone.work / "app.py").read_text() == "print('hi')\n"
    assert clone.git("remote").stdout.strip() == ""
    assert clone.git("config", "core.hooksPath").stdout.strip() == "/dev/null"
    assert clone.git_dir.is_dir() and not (clone.work / ".git").is_dir()


def test_create_leaves_the_user_repo_untouched(jobs_home, source):
    before = _user_repo_state(source)
    JobClone.create(source, resolve_commit(source, "HEAD"), jobs_home / "j", "mullet/j")
    assert _user_repo_state(source) == before


def test_create_refuses_paths_outside_jobs_home(jobs_home, source, tmp_path):
    with pytest.raises(ValueError):
        JobClone.create(source, resolve_commit(source, "HEAD"), tmp_path / "elsewhere", "b")


def test_create_refuses_an_existing_root(jobs_home, source):
    (jobs_home / "taken").mkdir()
    with pytest.raises(FileExistsError):
        JobClone.create(source, resolve_commit(source, "HEAD"), jobs_home / "taken", "b")


def test_snapshot_sees_adds_edits_deletes_and_untracked(clone):
    (clone.work / "app.py").write_text("print('hello')\n")
    (clone.work / "README.md").unlink()
    (clone.work / "new.py").write_text("x = 1\ny = 2\n")
    snap = clone.snapshot()
    by_path = {f.path: f.status for f in snap.files}
    assert by_path == {"app.py": "M", "README.md": "D", "new.py": "A"}
    assert snap.added_lines == 3 and snap.removed_lines == 2
    assert "+print('hello')" in snap.diff


def test_snapshot_of_untouched_clone_is_empty(clone):
    snap = clone.snapshot()
    assert snap.empty and snap.diff == ""


def test_snapshot_leaves_the_real_index_alone(clone):
    (clone.work / "new.py").write_text("x = 1\n")
    clone.snapshot()
    assert clone.git("diff", "--cached", "--name-only").stdout.strip() == ""


def test_materialize_drops_files_hidden_by_gitignore(clone):
    (clone.work / "conftest.py").write_text("raise SystemExit(0)\n")
    with open(clone.work / ".gitignore", "a") as fh:
        fh.write("conftest.py\n")
    (clone.work / "app.py").write_text("print('hello')\n")
    snap = clone.snapshot()
    assert {f.path for f in snap.files} == {".gitignore", "app.py"}
    clone.materialize(snap.tree)
    assert not (clone.work / "conftest.py").exists()
    assert (clone.work / "app.py").read_text() == "print('hello')\n"


def test_materialize_keeps_dependency_dirs(clone):
    (clone.work / "node_modules").mkdir()
    (clone.work / "node_modules" / "dep.js").write_text("dep")
    snap = clone.snapshot()
    (clone.work / "junk.tmp").write_text("junk")  # appears after the snapshot
    clone.materialize(snap.tree, keep=["node_modules"])
    assert (clone.work / "node_modules" / "dep.js").exists()
    assert not (clone.work / "junk.tmp").exists()


def test_commit_snapshot_commits_exactly_the_snapshot(clone):
    (clone.work / "app.py").write_text("print('hello')\n")
    snap = clone.snapshot()
    clone.materialize(snap.tree)
    # A gate that scribbles on a tracked file or leaves artifacts behind.
    (clone.work / "README.md").write_text("changed by a gate\n")
    (clone.work / "report.xml").write_text("<x/>")
    sha = clone.commit_snapshot(snap, "node n1")
    assert clone.head() == sha
    assert clone.git("rev-parse", f"{sha}^{{tree}}").stdout.strip() == snap.tree
    assert (clone.work / "README.md").read_text() == "readme\n"
    assert not (clone.work / "report.xml").exists()
    assert clone.git("log", "-1", "--format=%an <%ae>|%s").stdout.strip() == "Dev <dev@example.com>|node n1"


def test_commit_snapshot_refuses_a_moved_branch(clone):
    (clone.work / "a.txt").write_text("a")
    first = clone.snapshot()
    clone.commit_snapshot(first, "first")
    with pytest.raises(GitError, match="moved"):
        clone.commit_snapshot(first, "again")


def test_rewritten_gitfile_cannot_redirect_mullet(clone, source):
    before = _user_repo_state(source)
    (clone.work / ".git").write_text(f"gitdir: {source / '.git'}\n")
    (clone.work / "app.py").write_text("print('hello')\n")
    snap = clone.snapshot()
    sha = clone.commit_snapshot(snap, "node")
    assert clone.git("rev-parse", "HEAD").stdout.strip() == sha
    assert _user_repo_state(source) == before


def test_reset_to_discards_everything_since(clone):
    start = clone.head()
    (clone.work / "app.py").write_text("broken\n")
    (clone.work / "extra").mkdir()
    (clone.work / "extra" / "x.py").write_text("x")
    clone.reset_to(start)
    assert (clone.work / "app.py").read_text() == "print('hi')\n"
    assert not (clone.work / "extra").exists()


def test_symlinks_and_nested_repos_show_their_modes(clone):
    (clone.work / "link").symlink_to("/etc/passwd")
    nested = clone.work / "vendor_repo"
    nested.mkdir()
    git(nested, "init", "-q")
    (nested / "f").write_text("f")
    git(nested, "add", ".")
    git(nested, "commit", "-qm", "n")
    modes = {f.path: f.new_mode for f in clone.snapshot().files}
    assert modes["link"] == "120000"
    assert modes["vendor_repo"] == "160000"


def test_blob_reads_from_a_tree(clone):
    (clone.work / "app.py").write_text("print('hello')\n")
    snap = clone.snapshot()
    assert clone.blob(snap.base, "app.py") == b"print('hi')\n"
    assert clone.blob(snap.tree, "app.py") == b"print('hello')\n"
    assert clone.blob(snap.tree, "missing.py") is None


def test_fetch_into_adds_only_the_target_branch(clone, source):
    (clone.work / "app.py").write_text("print('hello')\n")
    sha = clone.commit_snapshot(clone.snapshot(), "node")
    head_before = git(source, "rev-parse", "HEAD")
    status_before = git(source, "status", "--porcelain")
    assert clone.fetch_into(source, "mullet/job1") == sha
    assert git(source, "rev-parse", "refs/heads/mullet/job1") == sha
    assert git(source, "rev-parse", "HEAD") == head_before
    assert git(source, "status", "--porcelain") == status_before


def test_fetch_into_refuses_the_checked_out_branch(clone, source):
    (clone.work / "app.py").write_text("print('hello')\n")
    clone.commit_snapshot(clone.snapshot(), "node")
    with pytest.raises(GitError):
        clone.fetch_into(source, "main")


def test_fetch_into_refuses_a_non_fast_forward(clone, source):
    (clone.work / "app.py").write_text("print('hello')\n")
    clone.commit_snapshot(clone.snapshot(), "node")
    git(source, "branch", "mullet/other", "HEAD")
    git(source, "checkout", "-q", "mullet/other")
    (source / "other.txt").write_text("o")
    git(source, "add", ".")
    git(source, "commit", "-qm", "diverge")
    git(source, "checkout", "-q", "main")
    with pytest.raises(GitError):
        clone.fetch_into(source, "mullet/other")


def test_remove_deletes_the_job_root(clone):
    root = clone.root
    clone.remove()
    assert not root.exists()


def _objects(git_dir: Path) -> set[str]:
    out = subprocess.run(["git", f"--git-dir={git_dir}", "cat-file", "--batch-all-objects", "--batch-check"],
                         capture_output=True, text=True, check=True).stdout
    return {line.split()[0] for line in out.splitlines()}


def test_the_clone_holds_only_the_base_commits_history(jobs_home, source):
    base = resolve_commit(source, "HEAD")
    git(source, "tag", "-a", "v-local", "-m", "a local annotated tag")
    tag = git(source, "rev-parse", "v-local")
    git(source, "checkout", "-qb", "unpushed")
    (source / "secret.txt").write_text("an unpushed secret\n")
    git(source, "add", "secret.txt")
    git(source, "commit", "-qm", "unpushed work")
    branch, blob = git(source, "rev-parse", "HEAD"), git(source, "rev-parse", "HEAD:secret.txt")
    git(source, "checkout", "-q", "main")
    (source / "app.py").write_text("print('stashed')\n")
    git(source, "stash", "-q")
    stash = git(source, "rev-parse", "stash@{0}")
    c = JobClone.create(source, base, jobs_home / "iso", "mullet/iso", identity=("Dev", "dev@example.com"))
    objects = _objects(c.git_dir)
    assert base in objects and git(source, "rev-parse", "HEAD^{tree}") in objects
    assert not {tag, branch, blob, stash} & objects
    assert c.git("for-each-ref", "--format=%(refname)").stdout.split() == ["refs/heads/mullet/iso"]
    assert c.git("remote").stdout.strip() == "" and not (c.git_dir / "FETCH_HEAD").exists()
    assert (c.work / "app.py").read_text() == "print('hi')\n"  # and the job still works
    (c.work / "new.py").write_text("x = 1\n")
    assert c.commit_snapshot(c.snapshot(), "add new.py") != base


def test_the_objects_are_copied_not_hard_linked(jobs_home, source):
    c = JobClone.create(source, resolve_commit(source, "HEAD"), jobs_home / "copy", "mullet/copy")
    links = {p.stat().st_nlink for p in (c.git_dir / "objects").rglob("*") if p.is_file()}
    assert links == {1}


def test_a_base_that_isnt_a_full_hash_is_refused(jobs_home, source):
    with pytest.raises(GitError, match="full commit hash"):
        JobClone.create(source, "HEAD", jobs_home / "bad", "mullet/bad")
