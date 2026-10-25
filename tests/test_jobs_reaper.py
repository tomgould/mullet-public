"""Tests for the reaper's safety core. Fake processes only: nothing real is signalled."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest

from mullet.jobs import reaper
from mullet.jobs.reaper import (
    Candidate,
    ProcInfo,
    SightingsStore,
    classify_parent,
    idle_limit,
    idle_limits,
    index_by_pid,
    snapshot,
    stop_process,
)

UID = 4242


class Clock:
    """A fake clock whose sleep only moves time forward."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def _deny():
    raise psutil.AccessDenied(1)


class SnapProc:
    """A psutil-like object for `snapshot()`."""

    def __init__(self, pid, *, ppid=1, cmdline=("x",), create_time=10.0, uid=UID):
        self.pid = pid
        self._ppid, self._cmd, self._ct, self._uid = ppid, cmdline, create_time, uid

    def _get(self, v):
        if isinstance(v, BaseException):
            raise v
        return v

    def uids(self):
        return SimpleNamespace(real=self._get(self._uid))

    def create_time(self):
        return self._get(self._ct)

    def ppid(self):
        return self._get(self._ppid)

    def cmdline(self):
        return list(self._get(self._cmd))


class TestSnapshot:
    def test_reads_fields(self):
        procs = snapshot(lambda: [SnapProc(5, ppid=3, cmdline=("a", "b"), create_time=7.0)], uid=UID)
        assert procs == [ProcInfo(pid=5, ppid=3, cmdline=("a", "b"), create_time=7.0, uid=UID)]

    def test_unreadable_fields_are_none_not_values(self):
        procs = snapshot(lambda: [
            SnapProc(5, ppid=psutil.AccessDenied(5), cmdline=psutil.ZombieProcess(5)),
        ], uid=UID)
        assert procs[0].ppid is None and procs[0].cmdline is None

    def test_other_users_and_unreadable_start_left_out(self):
        procs = snapshot(lambda: [
            SnapProc(5, uid=0),
            SnapProc(6, create_time=psutil.AccessDenied(6)),
            SnapProc(7, uid=psutil.AccessDenied(7)),
            SnapProc(8),
        ], uid=UID)
        assert [p.pid for p in procs] == [8]


def _p(pid, ppid, cmdline):
    return ProcInfo(pid=pid, ppid=ppid, cmdline=cmdline, create_time=1.0, uid=UID)


class TestClassifyParent:
    def _classify(self, proc, others=(), exists=False):
        by_pid = index_by_pid([proc, *others])
        return classify_parent(proc, by_pid, pid_exists=lambda pid: exists)

    def test_ppid_1_is_orphan(self):
        assert self._classify(_p(10, 1, ("w",))) == "orphan"

    def test_missing_parent_is_orphan(self):
        assert self._classify(_p(10, 99, ("w",)), exists=False) == "orphan"

    def test_parent_missing_from_snapshot_but_alive_is_unknown(self):
        # Another user's process, or one started after the snapshot.
        assert self._classify(_p(10, 99, ("w",)), exists=True) == "unknown"

    def test_systemd_user_parent_is_orphan(self):
        parent = _p(50, 1, ("/lib/systemd/systemd", "--user"))
        assert self._classify(_p(10, 50, ("w",)), [parent]) == "orphan"

    def test_init_parent_is_orphan(self):
        parent = _p(50, 0, ("/sbin/init",))
        assert self._classify(_p(10, 50, ("w",)), [parent]) == "orphan"

    def test_bash_parent_is_parented(self):
        parent = _p(50, 1, ("/bin/bash",))
        assert self._classify(_p(10, 50, ("w",)), [parent]) == "parented"

    def test_systemd_without_user_flag_elsewhere_is_parented(self):
        parent = _p(50, 1, ("/opt/systemd", "--system"))
        assert self._classify(_p(10, 50, ("w",)), [parent]) == "parented"

    def test_unknown_ppid_is_ignored(self):
        assert self._classify(_p(10, None, ("w",))) == "unknown"

    def test_unreadable_parent_cmdline_is_unknown(self):
        parent = _p(50, 1, None)
        assert self._classify(_p(10, 50, ("w",)), [parent]) == "unknown"

    def test_empty_parent_cmdline_is_unknown(self):
        parent = _p(50, 1, ())
        assert self._classify(_p(10, 50, ("w",)), [parent]) == "unknown"


def _cand(kind="xdist", pid=10, ct=100.0, limit=120.0):
    return Candidate(kind=kind, reason="test", idle_limit=limit, pid=pid, create_time=ct)


class TestSightings:
    def test_first_seen_is_read_only(self, tmp_path):
        store = SightingsStore(tmp_path / "s.json")
        assert store.first_seen("k", 5.0) == 5.0
        assert store.first_seen("k", 9.0) == 9.0
        assert not (tmp_path / "s.json").exists()

    def test_other_kinds_survive_a_commit(self, tmp_path):
        path = tmp_path / "s.json"
        a = _cand("xdist", 10)
        b = _cand("mcp", 20)
        store = SightingsStore(path)
        store.commit([a.key, b.key], 100.0)
        # The next pass: both finders report, then one commit.
        store = SightingsStore(path)
        store.commit([a.key, b.key, _cand("opencode", 30).key], 200.0)
        store = SightingsStore(path)
        assert store.first_seen(a.key, 300.0) == 100.0
        assert store.first_seen(b.key, 300.0) == 100.0

    def test_unseen_keys_are_dropped(self, tmp_path):
        path = tmp_path / "s.json"
        a, b = _cand("xdist", 10), _cand("mcp", 20)
        store = SightingsStore(path)
        store.commit([a.key, b.key], 100.0)
        store.commit([a.key], 200.0)
        assert SightingsStore(path).first_seen(b.key, 300.0) == 300.0

    def test_recycled_pid_has_its_own_key(self):
        assert _cand(ct=100.0).key != _cand(ct=101.0).key
        assert _cand("xdist").key != _cand("mcp").key

    def test_candidate_reaches_its_limit_across_two_passes(self, tmp_path):
        path = tmp_path / "s.json"
        clock = Clock(1000.0)
        c = _cand(limit=120.0)
        store = SightingsStore(path)
        assert not store.is_due(c, clock.now())
        store.commit([c.key], clock.now())
        clock.t += 60
        store = SightingsStore(path)
        assert not store.is_due(c, clock.now())
        store.commit([c.key], clock.now())
        clock.t += 61
        store = SightingsStore(path)
        assert store.is_due(c, clock.now())

    def test_corrupt_file_starts_empty(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text("{not json")
        assert SightingsStore(path).first_seen("k", 1.0) == 1.0
        path.write_text(json.dumps({"k": "soon"}))
        assert SightingsStore(path).first_seen("k", 1.0) == 1.0

    def test_default_path_is_in_state_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(reaper, "state_dir", lambda: tmp_path)
        assert SightingsStore().path == tmp_path / "reap-sightings.json"


class TestLimits:
    def test_defaults(self):
        assert idle_limits({}) == {"xdist": 120.0, "opencode": 120.0, "mcp": 120.0,
                                   "lmstudio": 600.0, "tmp": 3600.0}

    def test_env_overrides(self):
        assert idle_limit("xdist", {"MULLET_REAP_XDIST_S": "30"}) == 30.0
        assert idle_limit("lmstudio", {"MULLET_REAP_LMSTUDIO_S": " 5.5 "}) == 5.5

    @pytest.mark.parametrize("bad", ["", "abc", "-1", "nan", "inf"])
    def test_bad_values_ignored(self, bad):
        assert idle_limit("tmp", {"MULLET_REAP_TMP_S": bad}) == 3600.0


class World:
    """Fake process table. `factory` returns a fresh view each call, like psutil.Process."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.procs: dict[int, dict] = {}
        self.signals: list[tuple[int, int]] = []
        self.made = 0

    def add(self, pid, ct, *, children=(), exits_on=(signal.SIGTERM, signal.SIGKILL),
            running=True, exit_after=0.0):
        self.procs[pid] = {"ct": ct, "children": list(children), "exits_on": set(exits_on),
                           "running": running, "exit_after": exit_after, "dies_at": None}

    def _alive(self, pid):
        info = self.procs.get(pid)
        if info is None or not info["running"]:
            return False
        if info["dies_at"] is not None and self.clock.t >= info["dies_at"]:
            info["running"] = False
            return False
        return True

    def factory(self, pid):
        self.made += 1
        if pid not in self.procs:
            raise psutil.NoSuchProcess(pid)
        world = self

        class View:
            def __init__(self):
                self.pid = pid

            def is_running(self):
                return world._alive(pid)

            def create_time(self):
                if not world._alive(pid):
                    raise psutil.NoSuchProcess(pid)
                return world.procs[pid]["ct"]

            def status(self):
                return psutil.STATUS_SLEEPING

            def children(self, recursive=False):
                out = []
                for c in world.procs[pid]["children"]:
                    if world._alive(c):
                        v = world.factory(c)
                        out.append(v)
                return out

            def send_signal(self, sig):
                world.signals.append((pid, sig))
                info = world.procs[pid]
                if sig in info["exits_on"]:
                    info["dies_at"] = world.clock.t + info["exit_after"]

        return View()


def _stop(world, cand, clock, **kw):
    return stop_process(cand, sleep=clock.sleep, now=clock.now,
                        process_factory=world.factory, **kw)


class TestStopProcess:
    def test_recycled_pid_is_skipped(self):
        clock = Clock()
        world = World(clock)
        world.add(10, ct=200.0)  # pid 10 now belongs to a newer process
        res = _stop(world, _cand(pid=10, ct=100.0), clock)
        assert res.action == "skipped" and not res.ok
        assert world.signals == []
        assert res.as_record()["action"] == "skipped"

    def test_not_running_is_skipped(self):
        clock = Clock()
        world = World(clock)
        world.add(10, ct=100.0, running=False)
        res = _stop(world, _cand(pid=10, ct=100.0), clock)
        assert res.action == "skipped" and world.signals == []

    def test_vanished_is_skipped(self):
        clock = Clock()
        world = World(clock)
        res = _stop(world, _cand(pid=10, ct=100.0), clock)
        assert res.action == "skipped" and world.signals == []

    def test_uses_a_fresh_process_object(self):
        clock = Clock()
        world = World(clock)
        world.add(10, ct=100.0)
        _stop(world, _cand(pid=10, ct=100.0), clock)
        # One for the check, one before SIGTERM, more while waiting.
        assert world.made >= 3

    def test_term_stops_process_and_children(self):
        clock = Clock()
        world = World(clock)
        world.add(10, ct=100.0, children=[11, 12])
        world.add(11, ct=101.0)
        world.add(12, ct=102.0)
        res = _stop(world, _cand(pid=10, ct=100.0), clock)
        assert res.action == "stopped" and res.ok
        assert sorted(res.terminated) == [10, 11, 12] and res.killed == []
        assert all(sig == signal.SIGTERM for _, sig in world.signals)

    def test_stubborn_process_is_killed_after_grace_on_fake_clock(self):
        clock = Clock(1000.0)
        world = World(clock)
        world.add(10, ct=100.0, exits_on={signal.SIGKILL})
        res = _stop(world, _cand(pid=10, ct=100.0), clock)
        assert res.killed == [10]
        assert world.signals == [(10, signal.SIGTERM), (10, signal.SIGKILL)]
        assert clock.t == pytest.approx(1010.0)
        assert all(s <= 0.25 + 1e-9 for s in clock.sleeps)

    def test_slow_exit_within_grace_needs_no_kill(self):
        clock = Clock(1000.0)
        world = World(clock)
        world.add(10, ct=100.0, exit_after=3.0)
        res = _stop(world, _cand(pid=10, ct=100.0), clock)
        assert res.killed == [] and clock.t < 1004.0

    def test_recycled_child_is_not_signalled(self):
        clock = Clock()
        world = World(clock)
        world.add(10, ct=100.0, children=[11])
        world.add(11, ct=101.0)
        real_factory = world.factory
        def factory(pid):
            # Children are listed through the parent's view, so the first time
            # pid 11 is asked for here is the fresh check before signalling:
            # by then it has been recycled.
            if pid == 11:
                world.procs[11]["ct"] = 555.0
            return real_factory(pid)

        res = stop_process(_cand(pid=10, ct=100.0), sleep=clock.sleep, now=clock.now,
                           process_factory=factory)
        assert res.terminated == [10]
        assert (11, signal.SIGTERM) not in world.signals

    def test_dry_run_lists_children_and_signals_nothing(self):
        clock = Clock()
        world = World(clock)
        world.add(10, ct=100.0, children=[11])
        world.add(11, ct=101.0)
        res = _stop(world, _cand(pid=10, ct=100.0), clock, dry_run=True)
        assert res.action == "would_stop" and res.children == [11]
        assert "11" in res.detail and world.signals == []

    def test_candidate_without_pid_is_skipped(self):
        clock = Clock()
        world = World(clock)
        c = Candidate(kind="tmp", reason="old", idle_limit=1.0, path="/tmp/mullet-x")
        assert _stop(world, c, clock).action == "skipped"


# ---------------------------------------------------------------- finders

NOW = 100_000.0
OLD = NOW - 50_000.0  # a process start time well past any grace period
BOOT = "import sys;exec(eval(sys.stdin.readline()))"
JOB = "20261025-032134-2160f6"


def _proc(pid, cmdline, *, ppid=1, ct=OLD):
    return ProcInfo(pid=pid, ppid=ppid, cmdline=tuple(cmdline) if cmdline is not None else None,
                    create_time=ct, uid=UID)


def _seeded(tmp_path, procs, kinds=("xdist", "mcp", "opencode"), at=NOW - 10_000.0):
    """A store in which every process in `procs` was first seen long ago, under every kind."""
    path = tmp_path / "sightings.json"
    keys = [reaper.sighting_key(k, pid=p.pid, create_time=p.create_time)
            for p in procs for k in kinds]
    SightingsStore(path).commit(keys, at)
    return SightingsStore(path)


class FakeLedger:
    def __init__(self, jobs):
        self.jobs = jobs

    def get_job(self, job_id):
        if job_id not in self.jobs:
            raise KeyError(job_id)
        return SimpleNamespace(status=self.jobs[job_id])


def _no_factory(pid):
    raise AssertionError("no runner should be looked up")


NEVER = [
    ("code", "/home/u/src/pytest-xdist"),
    ("/usr/bin/vim", "/home/u/src/pytest-xdist/src/xdist/remote.py"),
    ("less", "execnet.log"),
    ("/usr/bin/python3", "-m", "pytest", "-n", "8"),
    ("python3", "-m", "xdist"),
    ("vim", "opencode", "run"),
    ("/usr/local/bin/opencode", "serve"),
    ("opencode",),
    ("python3",),
    ("/usr/bin/python3", "/home/u/myscript.py"),
    ("/usr/lib/firefox/firefox", "-contentproc", "--dir", f"/tmp/{JOB}"),
    ("/opt/google/chrome/chrome", "--type=renderer"),
    ("bash", "-c", BOOT),
    ("python3", "-c", BOOT, "extra"),
    ("python3", "-c", "import sys;exec(eval(sys.stdin.readline()));print(1)"),
    ("python3", "-x", "-c", BOOT),
    ("python3", "-u", "-u", "-c", BOOT),
    ("vim", "mullet", "mcp"),
    ("mullet", "mcpx"),
    ("mullet", "job", "mcp"),
    ("python3", "-m", "mullet", "paths"),
    ("python3", "/home/u/.local/bin/mullet-hud", "mcp"),
    (),
    None,
]


class TestNeverQualify:
    @pytest.mark.parametrize("cmdline", NEVER)
    def test_other_processes_are_never_candidates(self, cmdline, tmp_path, monkeypatch):
        monkeypatch.setenv("MULLET_JOBS_DIR", str(tmp_path / "jobs"))
        p = _proc(10, cmdline, ppid=1)
        store = _seeded(tmp_path, [p])
        ledger = FakeLedger({JOB: reaper.JobStatus.DONE})
        assert reaper.find_xdist_orphans([p], store, NOW, pid_exists=lambda pid: False) == []
        assert reaper.find_mcp_orphans([p], store, NOW, pid_exists=lambda pid: False) == []
        assert reaper.find_opencode_strays([p], store, NOW, ledger=ledger,
                                           process_factory=_no_factory) == []


class TestXdist:
    def _find(self, procs, store, now, **kw):
        kw.setdefault("pid_exists", lambda pid: False)
        return reaper.find_xdist_orphans(procs, store, now, env={}, **kw)

    @pytest.mark.parametrize("cmdline", [
        ("/usr/bin/python3", "-c", BOOT),
        ("/home/u/.venv/bin/python", "-u", "-c", BOOT),
        ("python3.10", "-u", "-B", "-c", BOOT),
    ])
    def test_bootstrap_command_lines_match(self, cmdline):
        assert reaper.is_xdist_worker(cmdline)

    def test_matches_what_execnet_really_runs(self):
        gateway_io = pytest.importorskip("execnet.gateway_io")
        xspec = pytest.importorskip("execnet").XSpec("popen")
        argv = tuple(gateway_io.popen_args(xspec))
        assert reaper.is_xdist_worker(argv), argv

    def test_orphan_qualifies_after_120_s_not_before(self, tmp_path):
        path = tmp_path / "s.json"
        worker = _proc(10, ("/usr/bin/python3", "-u", "-c", BOOT), ppid=1)
        t0 = NOW
        for t, due in ((t0, False), (t0 + 60, False), (t0 + 119.5, False), (t0 + 120, True)):
            store = SightingsStore(path)
            found = self._find([worker], store, t)
            assert len(found) == 1, t
            assert found[0].due is due, t
            assert reaper.due_only(found) == (found if due else [])
            store.commit(reaper.seen_keys(found), t)
        assert found[0].idle_s == 120.0
        assert found[0].pid == 10 and found[0].create_time == OLD

    def test_first_sighting_is_returned_so_its_key_is_committed(self, tmp_path):
        path = tmp_path / "s.json"
        worker = _proc(10, ("python3", "-c", BOOT))
        found = self._find([worker], SightingsStore(path), NOW)
        assert [c.due for c in found] == [False]
        SightingsStore(path).commit(reaper.seen_keys(found), NOW)
        assert SightingsStore(path).first_seen(found[0].key, NOW + 500) == NOW

    def test_worker_with_live_pytest_parent_is_spared(self, tmp_path):
        parent = _proc(50, ("/usr/bin/python3", "-m", "pytest", "-n", "8"), ppid=40)
        worker = _proc(10, ("python3", "-c", BOOT), ppid=50)
        store = _seeded(tmp_path, [worker])
        assert self._find([parent, worker], store, NOW) == []

    def test_unknown_parentage_is_spared(self, tmp_path):
        worker = _proc(10, ("python3", "-c", BOOT), ppid=None)
        hidden = _proc(11, ("python3", "-c", BOOT), ppid=77)  # parent alive, not ours
        store = _seeded(tmp_path, [worker, hidden])
        assert self._find([worker, hidden], store, NOW, pid_exists=lambda pid: True) == []

    def test_parent_gone_from_snapshot_is_orphan(self, tmp_path):
        worker = _proc(10, ("python3", "-c", BOOT), ppid=77)
        store = _seeded(tmp_path, [worker])
        assert [c.due for c in self._find([worker], store, NOW)] == [True]

    def test_young_process_waits_for_its_grace_period(self, tmp_path):
        worker = _proc(10, ("python3", "-c", BOOT), ct=NOW - 200.0)
        store = _seeded(tmp_path, [worker], at=NOW - 150.0)
        found = self._find([worker], store, NOW, limit=120.0, grace=300.0)
        assert found[0].idle_s == 150.0 and not found[0].due
        assert self._find([worker], store, NOW + 100.0, limit=120.0, grace=300.0)[0].due

    def test_env_limit_is_used(self, tmp_path):
        worker = _proc(10, ("python3", "-c", BOOT))
        store = _seeded(tmp_path, [worker], at=NOW - 40.0)
        found = reaper.find_xdist_orphans([worker], store, NOW, pid_exists=lambda pid: False,
                                          env={"MULLET_REAP_XDIST_S": "30"})
        assert found[0].idle_limit == 30.0 and found[0].due


class TestMcp:
    def _find(self, procs, store, now=NOW, **kw):
        kw.setdefault("pid_exists", lambda pid: False)
        return reaper.find_mcp_orphans(procs, store, now, env={}, **kw)

    @pytest.mark.parametrize("cmdline", [
        ("/usr/bin/python3", "-m", "mullet", "mcp"),
        ("python3", "-m", "mullet", "mcp", "--stdio"),
        ("/usr/local/bin/mullet", "mcp"),
        # A console script with a #! line: interpreter first, then the script.
        ("/usr/bin/python3", "/home/u/.local/bin/mullet", "mcp"),
        ("/home/u/.venv/bin/python3.12", "/home/u/.venv/bin/mullet", "mcp", "--foo"),
    ])
    def test_orphaned_server_qualifies(self, cmdline, tmp_path):
        server = _proc(20, cmdline, ppid=1)
        found = self._find([server], _seeded(tmp_path, [server]))
        assert [(c.kind, c.pid, c.due) for c in found] == [("mcp", 20, True)]

    def test_server_with_live_client_is_spared(self, tmp_path):
        client = _proc(50, ("/usr/bin/claude",), ppid=40)
        server = _proc(20, ("/usr/bin/python3", "/home/u/.local/bin/mullet", "mcp"), ppid=50)
        assert self._find([client, server], _seeded(tmp_path, [server])) == []

    def test_server_under_systemd_user_is_orphan(self, tmp_path):
        manager = _proc(5, ("/lib/systemd/systemd", "--user"), ppid=1)
        server = _proc(20, ("mullet", "mcp"), ppid=5)
        assert len(self._find([manager, server], _seeded(tmp_path, [server]))) == 1

    def test_not_due_on_first_sighting(self, tmp_path):
        server = _proc(20, ("mullet", "mcp"))
        found = self._find([server], SightingsStore(tmp_path / "s.json"))
        assert [c.due for c in found] == [False]


class TestSharedSightings:
    def test_one_commit_keeps_every_finders_sightings(self, tmp_path):
        path = tmp_path / "s.json"
        worker = _proc(10, ("python3", "-c", BOOT))
        server = _proc(20, ("mullet", "mcp"))
        procs = [worker, server]
        nope = {"pid_exists": lambda pid: False, "env": {}}
        for t in (NOW, NOW + 120.0):
            store = SightingsStore(path)
            found = (reaper.find_xdist_orphans(procs, store, t, **nope)
                     + reaper.find_mcp_orphans(procs, store, t, **nope))
            store.commit(reaper.seen_keys(found), t)
        assert sorted((c.kind, c.due) for c in found) == [("mcp", True), ("xdist", True)]


class TestOpencodeStrays:
    @pytest.fixture
    def home(self, monkeypatch, tmp_path):
        # The env var moves jobs_home() and job_dir() together.
        home = tmp_path / "jobs"
        monkeypatch.setenv("MULLET_JOBS_DIR", str(home))
        work = home / JOB / "clone" / "work"
        work.mkdir(parents=True)
        return home

    def _oc(self, work, pid=30, ct=OLD):
        return _proc(pid, ("/home/u/.opencode/bin/opencode", "run", "do the thing", "--pure",
                           "--dir", str(work), "--model", "m", "--format", "json"),
                     ppid=999, ct=ct)

    def _runner(self, home, pid, started):
        (home / JOB / "runner.pid").write_text(f"{pid} {started}")

    def _find(self, procs, store, ledger, factory=_no_factory, now=NOW):
        return reaper.find_opencode_strays(procs, store, now, ledger=ledger, env={},
                                           process_factory=factory)

    def test_job_dir_and_jobs_home_agree(self, home):
        from mullet.jobs.paths import job_dir
        assert job_dir(JOB) == (home / JOB).resolve()

    def test_done_job_qualifies(self, home, tmp_path):
        oc = self._oc(home / JOB / "clone" / "work")
        found = self._find([oc], _seeded(tmp_path, [oc]), FakeLedger({JOB: reaper.JobStatus.DONE}))
        assert [(c.kind, c.pid, c.due) for c in found] == [("opencode", 30, True)]
        assert found[0].extra["job_id"] == JOB
        assert "done" in found[0].reason

    def test_done_job_not_due_on_first_sighting(self, home, tmp_path):
        oc = self._oc(home / JOB / "clone" / "work")
        found = self._find([oc], SightingsStore(tmp_path / "s.json"),
                           FakeLedger({JOB: reaper.JobStatus.DONE}))
        assert [c.due for c in found] == [False]

    def test_dir_equals_form_is_read(self, home, tmp_path):
        work = home / JOB / "clone" / "work"
        oc = _proc(30, ("opencode", "run", "msg", f"--dir={work}"))
        found = self._find([oc], _seeded(tmp_path, [oc]), FakeLedger({JOB: reaper.JobStatus.FAILED}))
        assert len(found) == 1

    def test_running_job_with_live_runner_is_spared(self, home, tmp_path):
        clock = Clock()
        world = World(clock)
        world.add(777, ct=555.5)
        self._runner(home, 777, 555.5)
        assert reaper.runner_state(JOB, world.factory) == "live"
        oc = self._oc(home / JOB / "clone" / "work")
        store = _seeded(tmp_path, [oc])  # long past its limit, so only the runner spares it
        assert self._find([oc], store, FakeLedger({JOB: reaper.JobStatus.RUNNING}),
                          world.factory) == []
        # The same process would qualify were the job not running.
        assert len(self._find([oc], store, FakeLedger({JOB: reaper.JobStatus.PAUSED}))) == 1

    def test_running_job_with_recycled_runner_pid_qualifies(self, home, tmp_path):
        clock = Clock()
        world = World(clock)
        world.add(777, ct=9999.0)  # pid 777 now belongs to a newer process
        self._runner(home, 777, 555.5)
        assert reaper.runner_state(JOB, world.factory) == "dead"
        oc = self._oc(home / JOB / "clone" / "work")
        found = self._find([oc], _seeded(tmp_path, [oc]),
                           FakeLedger({JOB: reaper.JobStatus.RUNNING}), world.factory)
        assert [c.due for c in found] == [True]
        assert "runner has exited" in found[0].reason

    def test_running_job_with_exited_runner_qualifies(self, home, tmp_path):
        world = World(Clock())
        self._runner(home, 777, 555.5)  # no such pid in the world
        oc = self._oc(home / JOB / "clone" / "work")
        found = self._find([oc], _seeded(tmp_path, [oc]),
                           FakeLedger({JOB: reaper.JobStatus.RUNNING}), world.factory)
        assert len(found) == 1

    def test_running_job_without_pid_file_qualifies(self, home, tmp_path):
        oc = self._oc(home / JOB / "clone" / "work")
        assert reaper.runner_state(JOB, _no_factory) == "dead"
        found = self._find([oc], _seeded(tmp_path, [oc]), FakeLedger({JOB: reaper.JobStatus.RUNNING}))
        assert len(found) == 1

    @pytest.mark.parametrize("text", ["", "777", "abc 1.0", "777 1.0 extra"])
    def test_unreadable_pid_file_is_spared(self, home, tmp_path, text):
        (home / JOB / "runner.pid").write_text(text)
        oc = self._oc(home / JOB / "clone" / "work")
        assert reaper.runner_state(JOB, _no_factory) == "unknown"
        assert self._find([oc], _seeded(tmp_path, [oc]),
                          FakeLedger({JOB: reaper.JobStatus.RUNNING})) == []

    def test_unreadable_runner_process_is_spared(self, home, tmp_path):
        self._runner(home, 777, 555.5)

        def factory(pid):
            raise psutil.AccessDenied(pid)

        oc = self._oc(home / JOB / "clone" / "work")
        assert reaper.runner_state(JOB, factory) == "unknown"
        assert self._find([oc], _seeded(tmp_path, [oc]),
                          FakeLedger({JOB: reaper.JobStatus.RUNNING}), factory) == []

    def test_job_missing_from_ledger_is_ignored(self, home, tmp_path):
        oc = self._oc(home / JOB / "clone" / "work")
        assert self._find([oc], _seeded(tmp_path, [oc]), FakeLedger({})) == []

    def test_ledger_error_is_ignored(self, home, tmp_path):
        class Broken:
            def get_job(self, job_id):
                raise RuntimeError("database is locked")

        oc = self._oc(home / JOB / "clone" / "work")
        assert self._find([oc], _seeded(tmp_path, [oc]), Broken()) == []

    @pytest.mark.parametrize("where", ["outside", "invalid_id", "relative", "home_itself", "none"])
    def test_unattributable_dirs_are_ignored(self, home, tmp_path, where):
        outside = tmp_path / "elsewhere" / JOB
        outside.mkdir(parents=True)
        bad = home / "not-a-job" / "work"
        bad.mkdir(parents=True)
        dirs = {"outside": str(outside), "invalid_id": str(bad), "relative": f"{JOB}/clone/work",
                "home_itself": str(home), "none": None}
        argv = ["opencode", "run", "msg"] + (["--dir", dirs[where]] if dirs[where] else [])
        oc = _proc(30, argv)
        ledger = FakeLedger({JOB: reaper.JobStatus.DONE, "not-a-job": reaper.JobStatus.DONE})
        assert self._find([oc], _seeded(tmp_path, [oc]), ledger) == []

    def test_symlink_out_of_jobs_home_is_ignored(self, home, tmp_path):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        link = home / JOB / "clone" / "escape"
        link.symlink_to(elsewhere)
        oc = _proc(30, ("opencode", "run", "msg", "--dir", str(link)))
        assert self._find([oc], _seeded(tmp_path, [oc]), FakeLedger({JOB: reaper.JobStatus.DONE})) == []

    def test_two_different_dirs_are_ignored(self, home, tmp_path):
        work = home / JOB / "clone" / "work"
        oc = _proc(30, ("opencode", "run", "msg", "--dir", str(work), "--dir", str(tmp_path)))
        assert self._find([oc], _seeded(tmp_path, [oc]), FakeLedger({JOB: reaper.JobStatus.DONE})) == []

    def test_works_with_a_real_ledger(self, home, tmp_path):
        from mullet.jobs.ledger import Ledger
        with Ledger(home / "ledger.db") as ledger:
            job_id = ledger.create_job(tmp_path, "goal", "main", "0" * 40, {})
            ledger.set_job_status(job_id, reaper.JobStatus.FAILED)
            work = home / job_id / "clone" / "work"
            work.mkdir(parents=True)
            oc = self._oc(work)
            stray = self._oc(home / JOB / "clone" / "work", pid=31)  # unknown to this ledger
            found = self._find([oc, stray], _seeded(tmp_path, [oc, stray]), ledger)
        assert [(c.pid, c.extra["job_id"], c.due) for c in found] == [(30, job_id, True)]


class TestTmpDirs:
    def test_old_unused_dir_qualifies(self, tmp_path):
        # Create a tmp directory that looks like a mullet tmp dir
        tmp_root = tmp_path / "tmp"
        tmp_root.mkdir()
        mullet_dir = tmp_root / "mullet-test123"
        mullet_dir.mkdir()
        
        # Set mtime to be old enough to qualify
        import time
        old_time = time.time() - 4000  # 4000 seconds ago
        os.utime(mullet_dir, (old_time, old_time))
        
        # Mock open_paths to return no open files
        def open_paths():
            return []
            
        # Find should return our directory
        found = reaper.find_stale_tmp_dirs(tmp_root, now=time.time(), open_paths=open_paths)
        assert len(found) == 1
        assert found[0].kind == "tmp"
        assert found[0].path == str(mullet_dir)
        assert found[0].due is True

    def test_young_dir_does_not_qualify(self, tmp_path):
        # Create a tmp directory that looks like a mullet tmp dir
        tmp_root = tmp_path / "tmp"
        tmp_root.mkdir()
        mullet_dir = tmp_root / "mullet-test123"
        mullet_dir.mkdir()
        
        # Set mtime to be recent (young)
        import time
        now = time.time()
        os.utime(mullet_dir, (now, now))
        
        # Mock open_paths to return no open files
        def open_paths():
            return []
            
        # Find should not return our directory (too young)
        found = reaper.find_stale_tmp_dirs(tmp_root, now=now, open_paths=open_paths)
        assert len(found) == 0

    def test_dir_with_open_file_inside_does_not_qualify(self, tmp_path):
        # Create a tmp directory that looks like a mullet tmp dir
        tmp_root = tmp_path / "tmp"
        tmp_root.mkdir()
        mullet_dir = tmp_root / "mullet-test123"
        mullet_dir.mkdir()
        
        # Create a file inside the directory
        test_file = mullet_dir / "test.txt"
        test_file.write_text("test")
        
        # Set mtime to be old enough to qualify
        import time
        old_time = time.time() - 4000  # 4000 seconds ago
        os.utime(mullet_dir, (old_time, old_time))
        
        # Mock open_paths to return the file as open
        def open_paths():
            return [("", [str(test_file)])]
            
        # Find should not return our directory (has open file)
        found = reaper.find_stale_tmp_dirs(tmp_root, now=time.time(), open_paths=open_paths)
        assert len(found) == 0

    def test_symlink_does_not_qualify(self, tmp_path):
        # Create a tmp directory that looks like a mullet tmp dir
        tmp_root = tmp_path / "tmp"
        tmp_root.mkdir()
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        symlink_dir = tmp_root / "mullet-test123"
        symlink_dir.symlink_to(real_dir)
        
        # Set mtime to be old enough to qualify
        import time
        old_time = time.time() - 4000  # 4000 seconds ago
        os.utime(real_dir, (old_time, old_time))
        
        # Mock open_paths to return no open files
        def open_paths():
            return []
            
        # Find should not return our directory (it's a symlink)
        found = reaper.find_stale_tmp_dirs(tmp_root, now=time.time(), open_paths=open_paths)
        assert len(found) == 0

    def test_notmullet_dir_does_not_qualify(self, tmp_path):
        # Create a tmp directory that doesn't start with mullet-
        tmp_root = tmp_path / "tmp"
        tmp_root.mkdir()
        not_mullet_dir = tmp_root / "notmullet-test123"
        not_mullet_dir.mkdir()
        
        # Set mtime to be old enough to qualify
        import time
        old_time = time.time() - 4000  # 4000 seconds ago
        os.utime(not_mullet_dir, (old_time, old_time))
        
        # Mock open_paths to return no open files
        def open_paths():
            return []
            
        # Find should not return our directory (doesn't start with mullet-)
        found = reaper.find_stale_tmp_dirs(tmp_root, now=time.time(), open_paths=open_paths)
        assert len(found) == 0

    def test_px_dir_qualifies(self, tmp_path):
        # Create a tmp directory that looks like a mullet-px- dir
        tmp_root = tmp_path / "tmp"
        tmp_root.mkdir()
        mullet_dir = tmp_root / "mullet-px-test123"
        mullet_dir.mkdir()
        
        # Set mtime to be old enough to qualify
        import time
        old_time = time.time() - 4000  # 4000 seconds ago
        os.utime(mullet_dir, (old_time, old_time))
        
        # Mock open_paths to return no open files
        def open_paths():
            return []
            
        # Find should return our directory (starts with mullet- which covers mullet-px-)
        found = reaper.find_stale_tmp_dirs(tmp_root, now=time.time(), open_paths=open_paths)
        assert len(found) == 1
        assert found[0].kind == "tmp"
        assert found[0].path == str(mullet_dir)
        assert found[0].due is True

    def test_remove_dir_removes_valid_dir(self, tmp_path):
        # Create a tmp directory that looks like a mullet tmp dir
        tmp_root = tmp_path / "tmp"
        tmp_root.mkdir()
        mullet_dir = tmp_root / "mullet-test123"
        mullet_dir.mkdir()
        
        # Create a file inside the directory
        test_file = mullet_dir / "test.txt"
        test_file.write_text("test")
        
        # Set mtime to be old enough to qualify
        import time
        old_time = time.time() - 4000  # 4000 seconds ago
        os.utime(mullet_dir, (old_time, old_time))
        
        # Mock open_paths to return no open files
        def open_paths():
            return []
            
        # Remove should succeed
        result = reaper.remove_dir(str(mullet_dir), tmp_root=tmp_root, now=time.time(), open_paths=open_paths)
        assert result is True
        assert not mullet_dir.exists()

    def test_remove_dir_skips_non_mullet_dir(self, tmp_path):
        # Create a tmp directory that doesn't start with mullet-
        tmp_root = tmp_path / "tmp"
        tmp_root.mkdir()
        not_mullet_dir = tmp_root / "notmullet-test123"
        not_mullet_dir.mkdir()
        
        # Set mtime to be old enough
        import time
        old_time = time.time() - 4000  # 4000 seconds ago
        os.utime(not_mullet_dir, (old_time, old_time))
        
        # Mock open_paths to return no open files
        def open_paths():
            return []
            
        # Remove should skip (return False)
        result = reaper.remove_dir(str(not_mullet_dir), tmp_root=tmp_root, now=time.time(), open_paths=open_paths)
        assert result is False
        # Directory should still exist
        assert not_mullet_dir.exists()

    def test_remove_dir_skips_if_open(self, tmp_path):
        # Create a tmp directory that looks like a mullet tmp dir
        tmp_root = tmp_path / "tmp"
        tmp_root.mkdir()
        mullet_dir = tmp_root / "mullet-test123"
        mullet_dir.mkdir()
        
        # Create a file inside the directory
        test_file = mullet_dir / "test.txt"
        test_file.write_text("test")
        
        # Set mtime to be old enough to qualify
        import time
        old_time = time.time() - 4000  # 4000 seconds ago
        os.utime(mullet_dir, (old_time, old_time))
        
        # Mock open_paths to return the file as open
        def open_paths():
            return [("", [str(test_file)])]
            
        # Remove should skip (return False) because directory has open files
        result = reaper.remove_dir(str(mullet_dir), tmp_root=tmp_root, now=time.time(), open_paths=open_paths)
        assert result is False
        # Directory should still exist
        assert mullet_dir.exists()

    def test_remove_dir_skips_young_dir(self, tmp_path):
        # Create a tmp directory that looks like a mullet tmp dir
        tmp_root = tmp_path / "tmp"
        tmp_root.mkdir()
        mullet_dir = tmp_root / "mullet-test123"
        mullet_dir.mkdir()
        
        # Set mtime to be recent (young)
        import time
        now = time.time()
        os.utime(mullet_dir, (now, now))
        
        # Mock open_paths to return no open files
        def open_paths():
            return []
            
        # Remove should skip (return False) because directory is too young
        result = reaper.remove_dir(str(mullet_dir), tmp_root=tmp_root, now=now, open_paths=open_paths)
        assert result is False
        # Directory should still exist
        assert mullet_dir.exists()


class TestLMStudio:
    def _model(self, *, last_used=None, state="loaded", type="llm", model_id="model-1"):
        model = {"id": model_id, "type": type, "state": state}
        if last_used is not None:
            model["last_used"] = last_used
        return {"data": [model]}

    def test_unreachable_server_gives_no_candidates_and_no_error(self, tmp_path):
        store = SightingsStore(tmp_path / "s.json")

        def http_get(url):
            raise ConnectionError("refused")

        found = reaper.find_idle_lmstudio_models(store, NOW, http_get=http_get)
        assert found == []

    def test_non_loaded_models_are_ignored(self, tmp_path):
        store = SightingsStore(tmp_path / "s.json")
        models = {"data": [
            {"id": "m1", "type": "llm", "state": "loading"},
            {"id": "m2", "type": "embedding", "state": "loaded"},
        ]}
        found = reaper.find_idle_lmstudio_models(store, NOW, http_get=lambda url: models)
        assert found == []

    def test_loaded_model_with_no_request_signal_is_reported_not_due(self, tmp_path):
        store = SightingsStore(tmp_path / "s.json")
        found = reaper.find_idle_lmstudio_models(store, NOW, http_get=lambda url: self._model())
        assert len(found) == 1
        assert found[0].kind == "lmstudio"
        assert found[0].identifier == "model-1"
        assert not found[0].due
        assert found[0].extra["unload_url"] is None

    def test_base_url_read_from_env(self, tmp_path):
        store = SightingsStore(tmp_path / "s.json")
        requested = []

        def http_get(url):
            requested.append(url)
            return {"data": []}

        env = {"MULLET_REAP_LMSTUDIO_URL": "http://custom:5678"}
        reaper.find_idle_lmstudio_models(store, NOW, http_get=http_get, env=env)
        assert requested == ["http://custom:5678/api/v0/models"]

    def test_idle_model_without_unload_url_is_reported_not_posted(self, tmp_path):
        store = SightingsStore(tmp_path / "s.json")
        models = self._model(last_used=NOW - 700)
        posts = []

        def http_post(url, data):
            posts.append((url, data))

        # First sighting: not due yet.
        found = reaper.find_idle_lmstudio_models(store, NOW, http_get=lambda url: models, env={})
        assert len(found) == 1 and not found[0].due
        store.commit(reaper.seen_keys(found), NOW)
        # After the limit: due, but no unload URL is configured.
        store = SightingsStore(tmp_path / "s.json")
        found = reaper.find_idle_lmstudio_models(store, NOW + 600, http_get=lambda url: models, env={})
        assert len(found) == 1 and found[0].due
        result = reaper.unload_lmstudio_model(found[0], http_post=http_post)
        assert result.action == "reported"
        assert posts == []

    def test_idle_model_with_unload_url_is_posted_once_after_limit(self, tmp_path):
        store = SightingsStore(tmp_path / "s.json")
        models = self._model(last_used=NOW - 700)
        posts = []

        def http_post(url, data):
            posts.append((url, data))
            return {"ok": True}

        env = {"MULLET_REAP_LMSTUDIO_UNLOAD_URL": "http://localhost:1234/api/v0/models/unload"}
        # Not due on first sighting.
        found = reaper.find_idle_lmstudio_models(store, NOW, http_get=lambda url: models, env=env)
        assert len(found) == 1 and not found[0].due
        store.commit(reaper.seen_keys(found), NOW)
        # Just before the limit: still not due.
        store = SightingsStore(tmp_path / "s.json")
        found = reaper.find_idle_lmstudio_models(store, NOW + 599, http_get=lambda url: models, env=env)
        assert len(found) == 1 and not found[0].due
        # At the limit: due and unload posts exactly once.
        store = SightingsStore(tmp_path / "s.json")
        found = reaper.find_idle_lmstudio_models(store, NOW + 600, http_get=lambda url: models, env=env)
        assert len(found) == 1 and found[0].due
        result = reaper.unload_lmstudio_model(found[0], http_post=http_post)
        assert result.action == "unloaded"
        assert result.ok
        assert len(posts) == 1
        assert posts[0] == ("http://localhost:1234/api/v0/models/unload", {"model_id": "model-1"})

    def test_dry_run_does_not_post(self, tmp_path):
        store = SightingsStore(tmp_path / "s.json")
        models = self._model(last_used=NOW - 700)
        posts = []

        def http_post(url, data):
            posts.append((url, data))

        env = {"MULLET_REAP_LMSTUDIO_UNLOAD_URL": "http://localhost:1234/api/v0/models/unload"}
        found = reaper.find_idle_lmstudio_models(store, NOW, http_get=lambda url: models, env=env)
        store.commit(reaper.seen_keys(found), NOW)
        store = SightingsStore(tmp_path / "s.json")
        found = reaper.find_idle_lmstudio_models(store, NOW + 600, http_get=lambda url: models, env=env)
        result = reaper.unload_lmstudio_model(found[0], http_post=http_post, dry_run=True)
        assert result.action == "would_unload"
        assert result.ok
        assert posts == []

    def test_failed_unload_is_skipped(self, tmp_path):
        store = SightingsStore(tmp_path / "s.json")
        models = self._model(last_used=NOW - 700)

        def http_post(url, data):
            raise RuntimeError("server error")

        env = {"MULLET_REAP_LMSTUDIO_UNLOAD_URL": "http://localhost:1234/api/v0/models/unload"}
        found = reaper.find_idle_lmstudio_models(store, NOW, http_get=lambda url: models, env=env)
        store.commit(reaper.seen_keys(found), NOW)
        store = SightingsStore(tmp_path / "s.json")
        found = reaper.find_idle_lmstudio_models(store, NOW + 600, http_get=lambda url: models, env=env)
        result = reaper.unload_lmstudio_model(found[0], http_post=http_post)
        assert result.action == "skipped"
        assert not result.ok


# ---------------------------------------------------------------- a pass


@pytest.fixture
def state(monkeypatch, tmp_path):
    """Point state_dir() and the jobs home at tmp_path so no real user state is touched."""
    state = tmp_path / "state"
    monkeypatch.setattr(reaper, "state_dir", lambda: state)
    home = tmp_path / "jobs"
    monkeypatch.setenv("MULLET_JOBS_DIR", str(home))
    for name in ("MULLET_REAP_XDIST_S", "MULLET_REAP_OPENCODE_S", "MULLET_REAP_MCP_S",
                 "MULLET_REAP_LMSTUDIO_S", "MULLET_REAP_TMP_S",
                 "MULLET_REAP_LMSTUDIO_UNLOAD_URL"):
        monkeypatch.delenv(name, raising=False)
    (home / JOB / "clone" / "work").mkdir(parents=True)
    return SimpleNamespace(dir=state, home=home, log=state / "reap.log",
                           sightings=state / "reap-sightings.json")


class Stopper:
    def __init__(self):
        self.calls: list[Candidate] = []

    def __call__(self, cand):
        self.calls.append(cand)
        return reaper.StopResult(cand, "stopped", "fake stop", terminated=[cand.pid])


class FakeHttp:
    def __init__(self, models):
        self.models = models
        self.posts: list[tuple[str, dict]] = []

    def get(self, url):
        return self.models

    def post(self, url, data):
        self.posts.append((url, dict(data)))
        return {"ok": True}


def _no_pid_exists(pid):
    return False


def _pass(procs, now, *, dry_run=False, stopper=None, ledger=None, **kw):
    kw.setdefault("open_paths", None)
    return reaper.run_pass(dry_run=dry_run, now=now, snapshot_fn=lambda: list(procs),
                           ledger=ledger, http=kw.pop("http", None),
                           stopper=stopper if stopper is not None else Stopper(),
                           pid_exists=_no_pid_exists, process_factory=_no_factory, **kw)


def _log_lines(state):
    if not state.log.exists():
        return []
    return [json.loads(line) for line in state.log.read_text().splitlines()]


WORKER = _proc(10, ("/usr/bin/python3", "-u", "-c", BOOT), ppid=1)
SERVER = _proc(20, ("/usr/bin/python3", "-m", "mullet", "mcp"), ppid=1)


def _opencode(home, pid=30):
    return _proc(pid, ("/home/u/.opencode/bin/opencode", "run", "go", "--dir",
                       str(home / JOB / "clone" / "work")), ppid=999)


class TestRunPass:
    def test_run_pass_with_xdist_candidates_only_does_not_wipe_opencode_sightings(self, state):
        oc = _opencode(state.home)
        ledger = FakeLedger({JOB: reaper.JobStatus.DONE})
        procs = [WORKER, oc]
        oc_key = reaper.sighting_key("opencode", pid=oc.pid, create_time=oc.create_time)
        x_key = reaper.sighting_key("xdist", pid=WORKER.pid, create_time=WORKER.create_time)

        assert _pass(procs, NOW, ledger=ledger, dry_run=True) == []
        saved = json.loads(state.sightings.read_text())
        assert saved == {oc_key: NOW, x_key: NOW}

        assert _pass(procs, NOW + 60, ledger=ledger, dry_run=True) == []
        assert json.loads(state.sightings.read_text())[oc_key] == NOW

        records = _pass(procs, NOW + 120, ledger=ledger, dry_run=True)
        assert sorted((r["kind"], r["pid"]) for r in records) == [("opencode", 30), ("xdist", 10)]
        assert json.loads(state.sightings.read_text()) == {oc_key: NOW, x_key: NOW}

    def test_mcp_first_seen_survives_alongside_xdist_and_both_become_due(self, state):
        procs = [WORKER, SERVER]
        _pass(procs, NOW, dry_run=True)
        records = _pass(procs, NOW + 120, dry_run=True)
        assert sorted(r["kind"] for r in records) == ["mcp", "xdist"]

    def test_uses_the_default_sightings_file_in_state_dir(self, state):
        _pass([WORKER], NOW, dry_run=True)
        assert state.sightings.exists()

    def test_dry_run_never_calls_stopper_and_logs_would_stop(self, state):
        stopper = Stopper()
        _pass([WORKER, SERVER], NOW, dry_run=True, stopper=stopper)
        records = _pass([WORKER, SERVER], NOW + 120, dry_run=True, stopper=stopper)
        assert stopper.calls == []
        assert sorted((r["kind"], r["action"]) for r in records) == [
            ("mcp", "would_stop"), ("xdist", "would_stop")]
        lines = _log_lines(state)
        assert lines == records
        by_kind = {r["kind"]: r for r in lines}
        assert by_kind["xdist"]["pid"] == 10
        assert by_kind["xdist"]["create_time"] == OLD
        assert by_kind["xdist"]["ts"] == NOW + 120
        for line in lines:
            assert set(line) >= {"ts", "kind", "pid", "create_time", "reason", "action", "result"}

    def test_real_pass_calls_stopper_once_per_due_candidate(self, state):
        young = _proc(11, ("python3", "-c", BOOT), ppid=1)
        stopper = Stopper()
        _pass([WORKER, SERVER], NOW, stopper=stopper)
        assert stopper.calls == []
        records = _pass([WORKER, SERVER, young], NOW + 120, stopper=stopper)
        assert sorted((c.kind, c.pid, c.create_time) for c in stopper.calls) == [
            ("mcp", 20, OLD), ("xdist", 10, OLD)]
        assert all(c.pid != 11 for c in stopper.calls)
        assert sorted((r["kind"], r["action"]) for r in records) == [
            ("mcp", "stopped"), ("xdist", "stopped")]
        assert len(_log_lines(state)) == 2

    def test_nothing_due_writes_no_log_lines(self, state):
        assert _pass([WORKER], NOW) == []
        assert _log_lines(state) == []

    def test_recycled_pid_is_skipped_by_the_real_stopper(self, state):
        clock = Clock()
        world = World(clock)
        world.add(10, ct=OLD + 5.0)  # pid 10 now names a newer process

        def stopper(cand):
            return stop_process(cand, sleep=clock.sleep, now=clock.now,
                                process_factory=world.factory)

        _pass([WORKER], NOW, stopper=stopper)
        records = _pass([WORKER], NOW + 120, stopper=stopper)
        assert [r["action"] for r in records] == ["skipped"]
        assert world.signals == []

    def test_stopper_error_is_logged_as_skipped(self, state):
        def stopper(cand):
            raise RuntimeError("boom")

        _pass([WORKER], NOW, stopper=stopper)
        records = _pass([WORKER], NOW + 120, stopper=stopper)
        assert [(r["action"], r["result"]) for r in records] == [("skipped", "stopping failed: boom")]

    def test_lmstudio_unload_goes_through_the_helper(self, state, monkeypatch):
        url = "http://localhost:1234/api/v0/models/unload"
        monkeypatch.setenv("MULLET_REAP_LMSTUDIO_UNLOAD_URL", url)
        http = FakeHttp({"data": [{"id": "model-1", "type": "llm", "state": "loaded",
                                   "last_used": NOW - 700}]})
        stopper = Stopper()
        _pass([], NOW, http=http, stopper=stopper)
        dry = _pass([], NOW + 600, http=http, stopper=stopper, dry_run=True)
        assert [(r["model"], r["action"]) for r in dry] == [("model-1", "would_stop")]
        assert http.posts == []
        real = _pass([], NOW + 600, http=http, stopper=stopper)
        assert [(r["model"], r["action"]) for r in real] == [("model-1", "stopped")]
        assert http.posts == [(url, {"model_id": "model-1"})]
        assert stopper.calls == []

    def test_lmstudio_without_unload_url_is_reported(self, state):
        http = FakeHttp({"data": [{"id": "model-1", "type": "llm", "state": "loaded",
                                   "last_used": NOW - 700}]})
        _pass([], NOW, http=http)
        records = _pass([], NOW + 600, http=http)
        assert [r["action"] for r in records] == ["reported"]
        assert http.posts == []

    def test_tmp_dir_dry_run_keeps_it_and_real_pass_removes_it(self, state, tmp_path):
        root = tmp_path / "tmp"
        old = root / "mullet-px-abc"
        old.mkdir(parents=True)
        os.utime(old, (NOW - 5000, NOW - 5000))
        stopper = Stopper()
        kw = {"tmp_root": root, "open_paths": lambda: [], "stopper": stopper}
        dry = _pass([], NOW, dry_run=True, **kw)
        assert [(r["path"], r["action"]) for r in dry] == [(str(old), "would_stop")]
        assert old.exists()
        real = _pass([], NOW, **kw)
        assert [(r["path"], r["action"]) for r in real] == [(str(old), "stopped")]
        assert not old.exists()
        assert stopper.calls == []

    def test_actions_are_only_the_documented_ones(self, state, monkeypatch):
        monkeypatch.setenv("MULLET_REAP_LMSTUDIO_UNLOAD_URL", "http://x/unload")
        http = FakeHttp({"data": [{"id": "m", "type": "llm", "state": "loaded",
                                   "last_used": NOW - 700}]})
        _pass([WORKER], NOW, http=http)
        _pass([WORKER], NOW + 600, http=http, dry_run=True)
        _pass([WORKER], NOW + 600, http=http)
        actions = {line["action"] for line in _log_lines(state)}
        assert actions and actions <= {"stopped", "would_stop", "skipped", "reported"}


class TestTimerWindow:
    def test_no_install_file_means_not_in_dry_run(self, state):
        assert reaper.timer_dry_run_active(NOW) is False

    def test_window_flips_after_24_hours(self, state):
        path = reaper.record_timer_install(NOW)
        assert path == state.dir / "reap-timer-installed"
        assert reaper.timer_dry_run_active(NOW) is True
        assert reaper.timer_dry_run_active(NOW + 86_399.0) is True
        assert reaper.timer_dry_run_active(NOW + 86_400.0) is False
        assert reaper.timer_dry_run_active(NOW + 200_000.0) is False

    def test_corrupt_install_file_stays_in_dry_run(self, state):
        state.dir.mkdir(parents=True)
        (state.dir / "reap-timer-installed").write_text("not a time")
        assert reaper.timer_dry_run_active(NOW + 1_000_000.0) is True


# ---------------------------------------------------------------- CLI


class TestReapCmd:
    @pytest.fixture
    def cfg(self, tmp_path):
        return tmp_path / "config"

    @pytest.fixture
    def systemctl(self):
        calls: list[list[str]] = []

        def runner(args: list[str]):
            calls.append(list(args))
            return subprocess.CompletedProcess(args, returncode=0, stdout="", stderr="")

        return runner, calls

    def test_install_timer_writes_units_and_runs_systemctl(self, cfg, systemctl, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        runner, calls = systemctl
        from mullet.cli import reap_cmd
        assert reap_cmd.cmd_reap(["--install-timer"], config_home=cfg, systemctl=runner) == 0
        service = cfg / "systemd" / "user" / "mullet-reap.service"
        timer = cfg / "systemd" / "user" / "mullet-reap.timer"
        assert service.exists() and timer.exists()
        service_text = service.read_text()
        assert "Type=oneshot" in service_text
        assert "ExecStart=" in service_text and "reap --from-timer" in service_text
        timer_text = timer.read_text()
        assert "OnBootSec=3min" in timer_text
        assert "OnUnitActiveSec=3min" in timer_text
        assert calls == [["daemon-reload"], ["enable", "--now", "mullet-reap.timer"]]

    def test_install_timer_dry_run_writes_nothing_and_runs_nothing(self, cfg, systemctl, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        runner, calls = systemctl
        from mullet.cli import reap_cmd
        assert reap_cmd.cmd_reap(["--install-timer", "--dry-run"], config_home=cfg, systemctl=runner) == 0
        assert not (cfg / "systemd" / "user" / "mullet-reap.service").exists()
        assert not (cfg / "systemd" / "user" / "mullet-reap.timer").exists()
        assert calls == []

    def test_remove_timer_deletes_files_and_disables(self, cfg, systemctl, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        runner, calls = systemctl
        from mullet.cli import reap_cmd
        # Install first so the files exist.
        assert reap_cmd.cmd_reap(["--install-timer"], config_home=cfg, systemctl=runner) == 0
        calls.clear()
        assert reap_cmd.cmd_reap(["--remove-timer"], config_home=cfg, systemctl=runner) == 0
        assert not (cfg / "systemd" / "user" / "mullet-reap.service").exists()
        assert not (cfg / "systemd" / "user" / "mullet-reap.timer").exists()
        assert calls[0] == ["disable", "mullet-reap.timer"]
        assert ["daemon-reload"] in calls

    def test_reap_dry_run_exits_zero(self, monkeypatch):
        from mullet.cli import reap_cmd
        monkeypatch.setattr(reap_cmd, "_run_pass", lambda dry_run: [])
        assert reap_cmd.cmd_reap(["--dry-run"]) == 0

    def test_main_dispatches_to_cmd_reap(self, monkeypatch, tmp_path):
        from mullet.cli import reap_cmd
        captured: list[list[str]] = []
        monkeypatch.setattr(reap_cmd, "cmd_reap", lambda argv, **kw: (captured.append(argv), 0)[1])
        monkeypatch.setattr(sys, "argv", ["mullet", "reap", "--dry-run"])
        with pytest.raises(SystemExit) as exc_info:
            import mullet.cli.main as main_module
            main_module._main_inner()
        assert exc_info.value.code == 0
        assert captured == [["--dry-run"]]
