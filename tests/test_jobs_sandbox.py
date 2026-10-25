"""Tests for mullet.jobs.sandbox. They run real bubblewrap and skip without it."""

import os
import subprocess
import time
from pathlib import Path

import pytest

from mullet.jobs import children, sandbox
from mullet.jobs.sandbox import SandboxCheck, SandboxSpec, SandboxUnavailable

needs_bwrap = pytest.mark.skipif(not sandbox.check().ok, reason=sandbox.check().reason)


@pytest.fixture
def work(tmp_path):
    w = tmp_path / "work"
    w.mkdir()
    return w


def _run(script, spec, tmp_path, timeout=30):
    return sandbox.run_shell(script, spec, timeout=timeout, log_path=tmp_path / "logs" / "run.log")


@needs_bwrap
def test_writes_inside_writable_succeed(work, tmp_path):
    result = _run("echo ok > inside.txt", SandboxSpec(cwd=work, writable=[work]), tmp_path)
    assert result.exit_code == 0
    assert (work / "inside.txt").read_text() == "ok\n"


@needs_bwrap
def test_writes_outside_writable_fail(work, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    script = f"echo x > {outside}/escape.txt; echo x > {Path.home()}/mullet-escape-test.txt"
    result = _run(script, SandboxSpec(cwd=work, writable=[work]), tmp_path)
    assert not (outside / "escape.txt").exists()
    # $HOME inside the sandbox is an empty tmpfs: the write lands there and vanishes.
    assert not (Path.home() / "mullet-escape-test.txt").exists()
    # tmp_path is under the hidden /tmp, so the outside directory doesn't exist in there.
    assert "Directory nonexistent" in result.tail


@needs_bwrap
def test_visible_paths_outside_writable_are_read_only(work, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    spec = SandboxSpec(cwd=work, writable=[work], hide=[Path.home()])
    result = _run(f"echo x > {outside}/escape.txt", spec, tmp_path)
    assert not (outside / "escape.txt").exists()
    assert "Read-only file system" in result.tail


@needs_bwrap
def test_home_is_hidden(work, tmp_path):
    entries = [p for p in Path.home().iterdir() if not str(work).startswith(str(p))]
    if not entries:
        pytest.skip("home directory is empty")
    probe = entries[0]
    result = _run(f"test -e '{probe}' && echo visible || echo hidden", SandboxSpec(cwd=work, writable=[work]), tmp_path)
    assert "hidden" in result.tail


@needs_bwrap
def test_hide_and_expose(work, tmp_path):
    private = tmp_path / "private"
    (private / "tool").mkdir(parents=True)
    (private / "secret.txt").write_text("TOPSECRET-4711")
    (private / "tool" / "bin.sh").write_text("echo tool-ran")
    spec = SandboxSpec(cwd=work, writable=[work], hide=[private], expose_ro=[private / "tool"])
    result = _run(f"cat {private}/secret.txt; sh {private}/tool/bin.sh", spec, tmp_path)
    assert "TOPSECRET-4711" not in result.tail
    assert "tool-ran" in result.tail


@needs_bwrap
def test_environment_is_an_allowlist(work, tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_TEST_SECRET", "s3cret-value")
    spec = SandboxSpec(cwd=work, writable=[work], env={"MULLET_JUNIT": "/x/junit.xml"})
    result = _run("env", spec, tmp_path)
    assert "s3cret-value" not in result.tail
    assert "MULLET_JUNIT=/x/junit.xml" in result.tail
    assert "TERM=dumb" in result.tail


@needs_bwrap
def test_no_network_by_default(work, tmp_path):
    script = "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 53), 2)\" && echo connected || echo blocked"
    result = _run(script, SandboxSpec(cwd=work, writable=[work]), tmp_path)
    assert "blocked" in result.tail


@needs_bwrap
def test_readonly_overlay_inside_writable(work, tmp_path):
    deps = work / "node_modules"
    deps.mkdir()
    spec = SandboxSpec(cwd=work, writable=[work], readonly_overlays=[deps])
    result = _run("echo x > node_modules/bin.js; echo y > src.js", spec, tmp_path)
    assert not (deps / "bin.js").exists()
    assert (work / "src.js").exists()
    assert "Read-only file system" in result.tail


@needs_bwrap
def test_run_dir_is_hidden(work, tmp_path):
    if not Path("/run/user").exists():
        pytest.skip("no /run/user on this host")
    result = _run("test -e /run/user && echo visible || echo hidden", SandboxSpec(cwd=work, writable=[work]), tmp_path)
    assert "hidden" in result.tail


@needs_bwrap
def test_timeout_kills_the_process_tree(work, tmp_path):
    started = time.monotonic()
    result = _run("sleep 60 & sleep 60", SandboxSpec(cwd=work, writable=[work]), tmp_path, timeout=1)
    assert result.timed_out
    assert result.exit_code is None
    assert time.monotonic() - started < 20


@needs_bwrap
def test_full_output_goes_to_the_log(work, tmp_path):
    result = _run("seq 1 20000", SandboxSpec(cwd=work, writable=[work]), tmp_path)
    log = result.log_path.read_text()
    assert log.splitlines()[0] == "1"
    assert log.splitlines()[-1] == "20000"
    assert len(result.tail) <= 4000


@needs_bwrap
def test_mount_order(work):
    deps = work / "deps"
    deps.mkdir()
    argv = sandbox.build_argv(["true"], SandboxSpec(cwd=work, writable=[work], readonly_overlays=[deps]))
    tmpfs_home = argv.index(str(Path.home()))
    bind_work = argv.index("--bind")
    overlay = len(argv) - 1 - argv[::-1].index("--ro-bind")
    assert tmpfs_home < bind_work < overlay
    assert "--unshare-net" in argv
    assert "--clearenv" in argv


@needs_bwrap
def test_writable_must_be_a_directory(tmp_path):
    with pytest.raises(ValueError):
        sandbox.build_argv(["true"], SandboxSpec(cwd=tmp_path, writable=[tmp_path / "missing"]))


def test_no_new_commands_start_after_a_stop(tmp_path):
    children.configure(None)
    children.request_stop()
    try:
        with pytest.raises(children.Stopped):
            sandbox.run_shell("true", SandboxSpec(cwd=tmp_path, writable=[tmp_path]), timeout=5,
                              log_path=tmp_path / "run.log")
    finally:
        children.configure(None)  # a new runner starts with the stop cleared


def test_kill_in_dir_leaves_older_processes_alone(tmp_path):
    old = subprocess.Popen(["sleep", "30"], cwd=tmp_path)
    try:
        time.sleep(3)  # clear of kill_in_dir's 1.5s tolerance for psutil's coarse start times
        cutoff = time.time()
        new = subprocess.Popen(["sleep", "30"], cwd=tmp_path)
        time.sleep(0.2)
        assert children.kill_in_dir(tmp_path, since=cutoff) == 1
        assert new.wait(timeout=10) is not None
        assert old.poll() is None
    finally:
        old.kill()


def test_unavailable_sandbox_refuses(monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox, "check", lambda: SandboxCheck(False, "bwrap", "no userns"))
    with pytest.raises(SandboxUnavailable, match="no userns"):
        sandbox.build_argv(["true"], SandboxSpec(cwd=tmp_path, writable=[tmp_path]))


def test_build_env_never_inherits_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-should-not-leak")
    env = sandbox.build_env(SandboxSpec(cwd=tmp_path, writable=[tmp_path]))
    assert "OPENROUTER_API_KEY" not in env
    assert env["HOME"] == str(Path.home())
    assert set(env) <= {"PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "HOME", "TMPDIR", "TERM"}


def test_mullets_own_variables_win_over_a_specs(tmp_path):
    spec = SandboxSpec(cwd=tmp_path, writable=[tmp_path],
                       env={"HOME": "/evil", "TMPDIR": "/evil", "TERM": "xterm", "FOO": "bar"})
    env = sandbox.build_env(spec)
    assert env["HOME"] == str(Path.home()) and env["TMPDIR"] == "/tmp" and env["TERM"] == "dumb"
    assert env["FOO"] == "bar"


def test_read_tail_handles_missing_file(tmp_path):
    assert sandbox.read_tail(tmp_path / "nope.log") == ""


def test_check_reports_platform(monkeypatch):
    sandbox.check.cache_clear()
    monkeypatch.setattr(sandbox.sys, "platform", "darwin")
    try:
        status = sandbox.check()
        assert not status.ok and "darwin" in status.reason
    finally:
        sandbox.check.cache_clear()
    assert os.environ.get("PATH")
