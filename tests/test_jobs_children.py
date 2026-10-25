"""Tests for mullet.jobs.children: each owner stops only its own child processes."""

import json
import subprocess
import threading

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
