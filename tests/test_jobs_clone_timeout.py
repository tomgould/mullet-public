"""A hung git command is killed after jobs.git_timeout_seconds instead of hanging a job forever."""

import os
import stat
import time
from pathlib import Path

import pytest

from mullet.jobs.clone import GitError, JobClone, _git_env, _run, resolve_commit
from tests.test_jobs_clone import git, jobs_home, source  # noqa: F401  (fixtures)


def _fake_git(tmp_path: Path, monkeypatch, pid_file: Path) -> None:
    """A `git` that records its child's pid and then sleeps far longer than any test."""
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    script = bin_dir / "git"
    script.write_text(f"#!/bin/sh\nsleep 60 &\necho $! > {pid_file}\nwait\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


def _set_timeout(tmp_path: Path, monkeypatch, seconds: str) -> None:
    config = tmp_path / "config"
    config.mkdir(exist_ok=True)
    (config / "config.yaml").write_text(f"jobs:\n  git_timeout_seconds: {seconds}\n")
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(config))


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_a_hung_git_is_killed_with_its_children_and_named_in_the_error(tmp_path, monkeypatch):
    pid_file = tmp_path / "pid"
    _fake_git(tmp_path, monkeypatch, pid_file)
    _set_timeout(tmp_path, monkeypatch, "1")
    started = time.monotonic()
    with pytest.raises(GitError) as err:
        _run(["git", "-C", str(tmp_path), "fetch", "--quiet"], cwd=tmp_path, env=_git_env())
    assert time.monotonic() - started < 15
    message = str(err.value)
    assert "fetch" in message and "jobs.git_timeout_seconds" in message and "1" in message
    time.sleep(0.3)
    assert not _alive(int(pid_file.read_text()))


def test_create_fails_with_a_clear_error_when_git_hangs(jobs_home, source, tmp_path, monkeypatch):  # noqa: F811
    base = resolve_commit(source, "HEAD")
    pid_file = tmp_path / "pid"
    _fake_git(tmp_path, monkeypatch, pid_file)
    _set_timeout(tmp_path, monkeypatch, "1")
    with pytest.raises(GitError, match="jobs.git_timeout_seconds"):
        JobClone.create(source, base, jobs_home / "job1", "mullet/job1", identity=("D", "d@example.com"))


def test_a_normal_git_is_unaffected_by_a_tiny_timeout(jobs_home, source, tmp_path, monkeypatch):  # noqa: F811
    _set_timeout(tmp_path, monkeypatch, "30")
    base = resolve_commit(source, "HEAD")
    clone = JobClone.create(source, base, jobs_home / "job1", "mullet/job1", identity=("D", "d@example.com"))
    assert clone.head() == base


def test_git_runs_with_the_c_locale_so_parsed_output_is_stable(monkeypatch):
    monkeypatch.setenv("LC_ALL", "de_DE.UTF-8")
    assert _git_env()["LC_ALL"] == "C"
    assert _git_env(isolated=False)["LC_ALL"] == "C"


def _script_git(tmp_path: Path, monkeypatch, body: str) -> None:
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    script = bin_dir / "git"
    script.write_text("#!/bin/sh\n" + body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


def test_a_timed_out_git_is_asked_to_stop_first_so_it_can_remove_its_lock(tmp_path, monkeypatch):
    import mullet.jobs.clone as clone_mod

    lock, marker = tmp_path / "index.lock", tmp_path / "terminated"
    _script_git(tmp_path, monkeypatch, f"touch {lock}\ntrap 'rm -f {lock}; touch {marker}; exit 143' TERM\n"
                                       "sleep 60 &\nwait\n")
    _set_timeout(tmp_path, monkeypatch, "1")
    monkeypatch.setattr(clone_mod, "GIT_TERM_GRACE_SECONDS", 5.0)
    started = time.monotonic()
    with pytest.raises(GitError, match="jobs.git_timeout_seconds"):
        _run(["git", "-C", str(tmp_path), "commit", "-q"], cwd=tmp_path, env=_git_env())
    assert marker.exists() and not lock.exists()
    assert time.monotonic() - started < 5  # it stopped on SIGTERM; nothing waited out the grace


def test_a_git_that_ignores_sigterm_is_killed_after_the_grace(tmp_path, monkeypatch):
    import mullet.jobs.clone as clone_mod

    pid_file = tmp_path / "pid"
    _script_git(tmp_path, monkeypatch, f"trap '' TERM\necho $$ > {pid_file}\nwhile :; do sleep 0.1; done\n")
    _set_timeout(tmp_path, monkeypatch, "1")
    monkeypatch.setattr(clone_mod, "GIT_TERM_GRACE_SECONDS", 1.5)
    started = time.monotonic()
    with pytest.raises(GitError, match="jobs.git_timeout_seconds"):
        _run(["git", "-C", str(tmp_path), "fetch"], cwd=tmp_path, env=_git_env())
    elapsed = time.monotonic() - started
    assert 2.4 <= elapsed < 10  # the timeout, then the whole grace, then SIGKILL
    time.sleep(0.3)
    assert not _alive(int(pid_file.read_text()))


def test_the_term_grace_is_a_few_seconds():
    from mullet.jobs.clone import GIT_TERM_GRACE_SECONDS
    assert 1 <= GIT_TERM_GRACE_SECONDS <= 10
