"""Tests for mullet.jobs.children: each owner stops only its own child processes."""

import fcntl
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import psutil
import pytest

from mullet.jobs import children


@pytest.fixture
def spawn():
    procs: list[subprocess.Popen] = []

    def start() -> subprocess.Popen:
        proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
        procs.append(proc)
        return proc

    yield start
    for proc in procs:
        proc.kill()
        proc.wait()


def test_a_stop_on_one_tracker_leaves_anothers_processes_alive(spawn):
    first, second = children.Tracker(), children.Tracker()
    a, b = spawn(), spawn()
    with first.track(a), second.track(b):
        first.request_stop()
        assert a.wait(timeout=10) is not None
        assert b.poll() is None
        with pytest.raises(children.Stopped):
            first.check_not_stopping()
        second.check_not_stopping()


def test_configure_clears_only_its_own_trackers_stop():
    first, second = children.Tracker(), children.Tracker()
    first.request_stop()
    second.configure(None)
    with pytest.raises(children.Stopped):
        first.check_not_stopping()
    first.configure(None)
    first.check_not_stopping()


def test_module_functions_act_on_the_tracker_bound_in_this_thread(spawn):
    bound = children.Tracker()
    outside, inside = spawn(), spawn()
    with children.track(outside):
        with children.use(bound):
            assert children.current() is bound
            with children.track(inside):
                children.request_stop()
                assert inside.wait(timeout=10) is not None
        assert children.current() is not bound
        assert outside.poll() is None
        children.check_not_stopping()


def test_a_new_thread_starts_on_the_default_tracker():
    seen = []
    with children.use(children.Tracker()):
        thread = threading.Thread(target=lambda: seen.append(children.current()))
        thread.start()
        thread.join()
    assert seen == [children.current()]


def test_concurrent_owners_in_threads_stop_independently(spawn):
    procs = {"one": spawn(), "two": spawn()}
    trackers = {"one": children.Tracker(), "two": children.Tracker()}
    tracking = threading.Barrier(3)
    done = threading.Event()

    def owner(name: str) -> None:
        with children.use(trackers[name]), children.track(procs[name]):
            tracking.wait()
            done.wait(30)

    threads = [threading.Thread(target=owner, args=(name,)) for name in procs]
    for thread in threads:
        thread.start()
    try:
        tracking.wait(10)
        with children.use(trackers["one"]):
            assert children.kill_all() == 1
        assert procs["one"].wait(timeout=10) is not None
        assert procs["two"].poll() is None
    finally:
        done.set()
        for thread in threads:
            thread.join(30)


def test_each_tracker_persists_only_its_own_groups(tmp_path, spawn):
    first, second = children.Tracker(), children.Tracker()
    first.configure(tmp_path / "first.json")
    second.configure(tmp_path / "second.json")
    a, b = spawn(), spawn()
    with first.track(a), second.track(b):
        assert list(json.loads((tmp_path / "first.json").read_text())) == [str(a.pid)]
        assert list(json.loads((tmp_path / "second.json").read_text())) == [str(b.pid)]
    assert json.loads((tmp_path / "first.json").read_text()) == {}


# A process is known by its start in clock ticks since boot, which a step of the wall clock doesn't move


JOB = "20261026-101500-a1b2c3"


def _step_the_clock(monkeypatch, seconds: float = 3600.0) -> None:
    """psutil's create_time adds the boot time the kernel derives from the wall clock, so NTP or a resume moves it."""
    real = psutil.Process.create_time
    monkeypatch.setattr(psutil.Process, "create_time", lambda self: real(self) + seconds)


def _stat_line(comm: bytes, starttime: int) -> bytes:
    # Fields 3 to 21 after the comm, then field 22 (starttime) and field 23.
    return b"4242 (" + comm + b") S " + b" ".join(str(n).encode() for n in range(4, 22)) + b" %d 1000\n" % starttime


@pytest.mark.parametrize("comm", [b"sleep", b"a b", b"a) R 9 9 (b", b")", b"(("])
def test_the_start_is_read_after_the_last_paren(comm):
    assert children.parse_start_ticks(_stat_line(comm, 987654)) == 987654


@pytest.mark.parametrize("stat", [b"", b"4242 (sleep S 1 2", b"4242 (sleep) S 1 2 3", b"4242 (x) S " + b"y " * 30])
def test_a_malformed_stat_line_gives_no_start(stat):
    assert children.parse_start_ticks(stat) is None


def test_a_live_process_whose_name_holds_parens_and_spaces_is_read_right():
    name = b"a) R 9 9 (b"
    script = (f"import ctypes, sys; ctypes.CDLL(None).prctl(15, {name!r}, 0, 0, 0); print('ready', flush=True); "
              "sys.stdin.read()")
    proc = subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        assert proc.stdout.readline() == b"ready\n"
        assert Path(f"/proc/{proc.pid}/comm").read_bytes() == name + b"\n"
        expected = round((psutil.Process(proc.pid).create_time() - psutil.boot_time()) * os.sysconf("SC_CLK_TCK"))
        assert children.start_ticks(proc.pid) == expected
    finally:
        proc.kill()
        proc.wait()
    assert children.start_ticks(proc.pid) is None


def test_the_identity_survives_a_clock_step(monkeypatch):
    me = os.getpid()
    identity = children.process_identity(me)
    assert identity == f"{children.start_ticks(me)}@{children.boot_id()}"
    _step_the_clock(monkeypatch)
    assert children.same_process(me, identity)


def test_a_recycled_pid_or_another_boot_is_not_the_same_process():
    me = os.getpid()
    ticks = children.start_ticks(me)
    assert not children.same_process(me, f"{ticks + 1}@{children.boot_id()}")
    assert not children.same_process(me, f"{ticks}@another-boot")
    assert not children.same_process(_dead_pid(), children.process_identity(me))
    assert not children.same_process(me, "not a start time")


def test_a_start_time_from_an_older_mullet_is_still_understood():
    me = os.getpid()
    assert children.same_process(me, repr(psutil.Process(me).create_time()))
    assert children.same_process(me, psutil.Process(me).create_time())
    assert not children.same_process(me, 12345.0)


def test_a_stop_after_a_clock_step_still_kills_the_tracked_group(spawn, monkeypatch):
    tracker = children.Tracker()
    proc = spawn()
    with tracker.track(proc):
        _step_the_clock(monkeypatch)
        assert tracker.kill_all() == 1
        assert proc.wait(timeout=10) is not None


def test_kill_stale_after_a_clock_step_kills_the_recorded_group(spawn, tmp_path, monkeypatch):
    record = tmp_path / "children.json"
    tracker = children.Tracker()
    tracker.configure(record)
    proc = spawn()
    with tracker.track(proc):
        saved = record.read_text()
    record.write_text(saved)  # what a runner killed with -9 leaves behind
    _step_the_clock(monkeypatch)
    monkeypatch.setattr("mullet.jobs.sandbox_backends.container.reap_orphans", lambda engines=None: 0)
    assert children.kill_stale(record) == 1
    assert proc.wait(timeout=10) is not None and not record.exists()


def test_the_runner_pid_file_survives_a_clock_step(tmp_path, monkeypatch):
    from mullet.cli import job_cli
    from mullet.jobs import reaper

    monkeypatch.setenv("MULLET_JOBS_DIR", str(tmp_path / "jobs"))
    job = tmp_path / "jobs" / JOB
    job.mkdir(parents=True)
    record = children.pid_record()
    assert record == f"{os.getpid()} {children.process_identity(os.getpid())}"
    (job / "runner.pid").write_text(record)
    fd = os.open(job / "runner.lock", os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        _step_the_clock(monkeypatch)
        assert reaper.runner_state(JOB) == "live"
        assert job_cli._runner_pid(JOB) == os.getpid()
        (job / "runner.pid").write_text(f"{os.getpid()} {children.start_ticks(os.getpid()) + 1}@{children.boot_id()}")
        assert reaper.runner_state(JOB) == "dead" and job_cli._runner_pid(JOB) is None
        (job / "runner.pid").write_text(f"{_dead_pid()} {children.process_identity(os.getpid())}")
        assert reaper.runner_state(JOB) == "dead" and job_cli._runner_pid(JOB) is None
    finally:
        os.close(fd)


def _dead_pid() -> int:
    done = subprocess.Popen(["true"])
    done.wait()
    return done.pid
