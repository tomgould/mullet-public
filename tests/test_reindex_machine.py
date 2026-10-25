"""The Trawl reindexer gets the machine ready before indexing a pool.

With a GPU it waits for headroom, takes the GPU lease and re-checks, all
within MULLET_GPU_WAIT_MAX_S, and falls back to the CPU when the GPU stays
busy. Without one it takes no lease. The indexing itself
(`_reindex_one_locked`) is replaced with a recorder, the machine's readings
come from fixed snapshots, and waiting runs on a fake clock, so nothing loads
a model or touches the GPU.
"""

from __future__ import annotations

import os
import sys
import threading
import types
from argparse import Namespace

import pytest

from mullet.core import env
from mullet.jobs import machine
from tests.test_reindex_process_pool import _load_reindex

ROOMY = machine.Headroom(vram_free_mb=8000, vram_total_mb=12282, gpu_temp_c=50.0,
                         ram_free_mb=40000, load_1m=0.5)
SHORT_OF_VRAM = machine.Headroom(vram_free_mb=100, vram_total_mb=12282, gpu_temp_c=50.0,
                                 ram_free_mb=40000, load_1m=0.5)
NO_GPU = machine.Headroom(ram_free_mb=40000, load_1m=0.5)


class FakeTime:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []
        self.on_sleep = None

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        if self.on_sleep:
            self.on_sleep()


@pytest.fixture
def reindex(tmp_path, monkeypatch):
    module = _load_reindex()
    monkeypatch.setattr(module, "TRAWL_POOLS_DIR", tmp_path)
    for pool in ("news", "docs"):
        (tmp_path / pool).mkdir()
    for name in ("MULLET_GPU_MIN_FREE_MB", "MULLET_RAM_MIN_FREE_MB", "MULLET_GPU_MAX_TEMP_C",
                 "MULLET_MAX_LOAD", "MULLET_GPU_WAIT_MAX_S"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(env, "dotenv_files", lambda: [])
    monkeypatch.setattr(machine, "headroom", lambda: ROOMY)
    fake = FakeTime()
    monkeypatch.setattr(machine, "_clock", fake.clock)
    monkeypatch.setattr(machine, "_sleep", fake.sleep)
    module.time_ = fake
    # The real one would empty CUDA's cache if another test loaded the model cache.
    released: list[machine.GpuHolder | None] = []
    monkeypatch.setattr(module, "real_release_gpu_memory", module._release_gpu_memory, raising=False)
    monkeypatch.setattr(module, "_release_gpu_memory", lambda: released.append(machine.current_holder()))
    module.released = released
    on_cpu: list[str] = []
    monkeypatch.setattr(module, "real_cpu_only", module._cpu_only, raising=False)

    class _Cpu:
        def __enter__(self):
            on_cpu.append("cpu")

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(module, "_cpu_only", _Cpu)
    module.on_cpu = on_cpu
    return module


@pytest.fixture
def indexed(reindex, monkeypatch):
    """Record, for each pool indexed, who held the GPU lease at the time."""
    seen: list[tuple[str, machine.GpuHolder | None]] = []

    def fake_locked(name, args, quiet=False):
        seen.append((name, machine.current_holder()))
        return 1.5

    monkeypatch.setattr(reindex, "_reindex_one_locked", fake_locked)
    return seen


@pytest.fixture
def other_holder():
    """Another thread holding the GPU lease; ``release()`` frees it."""
    taken, done = threading.Event(), threading.Event()

    def hold():
        with machine.gpu_lease("reindex of pool trag"):
            taken.set()
            done.wait(60)

    worker = threading.Thread(target=hold)
    worker.start()
    assert taken.wait(30)

    def release():
        done.set()
        worker.join(30)

    yield types.SimpleNamespace(release=release)
    release()


def _line(text: str) -> str:
    return f"{'news':<{22}} {text}"


def _lines(capsys) -> list[str]:
    return [ln.strip() for ln in capsys.readouterr().out.splitlines() if ln.strip()]


def test_a_pool_is_indexed_holding_the_gpu_lease(reindex, indexed, capsys):
    assert reindex._reindex_one("news", Namespace(no_wait=False), quiet=True) == 1.5
    ((name, holder),) = indexed
    assert name == "news" and holder.purpose == "reindex of pool news" and holder.pid == os.getpid()
    assert reindex.released == [holder], "models are released before the lease is"
    assert reindex.on_cpu == []
    assert machine.current_holder() is None
    assert _lines(capsys) == []


def test_headroom_is_waited_for_before_the_lease_is_taken(reindex, indexed, monkeypatch, capsys):
    calls = []

    def sample():
        calls.append(machine.current_holder())
        return SHORT_OF_VRAM if len(calls) <= 3 else ROOMY

    monkeypatch.setattr(machine, "headroom", sample)
    reindex._reindex_one("news", Namespace(no_wait=False), quiet=True)
    assert calls[:4] == [None] * 4, "no lease while waiting for headroom"
    assert calls[4].purpose == "reindex of pool news", "re-checked holding the lease"
    lines = _lines(capsys)
    assert lines[0].startswith(_line("waiting up to "))
    assert lines[0].endswith("for headroom, then the CPU (--no-wait skips): GPU has 100 MiB free, "
                             "needs 2048 MiB [MULLET_GPU_MIN_FREE_MB]")
    assert lines[1].startswith(_line("starting after waiting "))
    assert len(lines) == 2
    assert reindex.time_.sleeps == [5.0, 5.0, 5.0]
    assert indexed[0][1].purpose == "reindex of pool news" and reindex.on_cpu == []


def test_waiting_for_the_gpu_names_the_holder(reindex, indexed, other_holder, capsys):
    reindex.time_.on_sleep = other_holder.release
    reindex._reindex_one("news", Namespace(no_wait=False), quiet=True)
    lines = _lines(capsys)
    assert lines[0].startswith(_line("waiting up to "))
    assert (f"for the GPU, then the CPU (--no-wait skips): held by reindex of pool trag "
            f"(pid {os.getpid()}, since ") in lines[0]
    assert lines[1].startswith(_line("starting after waiting "))
    assert indexed[0][1].purpose == "reindex of pool news" and reindex.on_cpu == []


def test_no_headroom_in_time_indexes_on_the_cpu(reindex, indexed, monkeypatch, capsys):
    monkeypatch.setenv("MULLET_GPU_WAIT_MAX_S", "12")
    monkeypatch.setattr(machine, "headroom", lambda: SHORT_OF_VRAM)
    assert reindex._reindex_one("news", Namespace(no_wait=False), quiet=True) == 1.5
    assert reindex.time_.sleeps == [5.0, 5.0, 2.0]
    assert reindex.on_cpu == ["cpu"] and indexed == [("news", None)], "on the CPU, without the lease"
    assert _lines(capsys)[-1] == _line("gave up after 12s waiting for headroom: GPU has 100 MiB free, "
                                       "needs 2048 MiB [MULLET_GPU_MIN_FREE_MB]; indexing on the CPU instead")


def test_a_gpu_that_stays_held_indexes_on_the_cpu(reindex, indexed, other_holder, monkeypatch, capsys):
    monkeypatch.setenv("MULLET_GPU_WAIT_MAX_S", "12")
    reindex._reindex_one("news", Namespace(no_wait=False), quiet=True)
    assert reindex.on_cpu == ["cpu"]
    assert indexed[0][1].purpose == "reindex of pool trag", "the other holder still has it"
    last = _lines(capsys)[-1]
    assert last.startswith(_line("gave up after 12s waiting for the GPU: reindex of pool trag (pid "))
    assert last.endswith("; indexing on the CPU instead")


def test_a_failed_recheck_lets_go_of_the_lease_and_uses_the_cpu(reindex, indexed, monkeypatch):
    calls = []

    def sample():
        calls.append(1)
        return ROOMY if len(calls) == 1 else SHORT_OF_VRAM

    monkeypatch.setattr(machine, "headroom", sample)
    reindex._reindex_one("news", Namespace(no_wait=False), quiet=True)
    assert sum(reindex.time_.sleeps) == reindex._RECHECK_S
    assert reindex.on_cpu == ["cpu"] and indexed == [("news", None)]


def test_without_a_gpu_there_is_no_lease(reindex, indexed, monkeypatch, capsys):
    monkeypatch.setattr(machine, "headroom", lambda: NO_GPU)
    monkeypatch.setattr(machine, "gpu_lease", lambda *a, **k: pytest.fail("took the lease without a GPU"))
    reindex._reindex_one("news", Namespace(no_wait=False), quiet=True)
    assert indexed == [("news", None)] and reindex.on_cpu == []
    assert _lines(capsys) == []


def test_without_a_gpu_workers_run_side_by_side(reindex, monkeypatch, capsys):
    monkeypatch.setattr(machine, "headroom", lambda: NO_GPU)
    both, inside = threading.Barrier(2, timeout=30), []

    def fake_locked(name, args, quiet=False):
        inside.append(name)
        both.wait()  # only passes if both pools are indexing at once
        return 1.0

    monkeypatch.setattr(reindex, "_reindex_one_locked", fake_locked)
    monkeypatch.setattr(sys, "argv", ["reindex", "--all", "--workers", "2"])
    monkeypatch.setattr(reindex, "discover_clients", lambda clients_dir=None: ["news", "docs"])
    _no_model_cache(monkeypatch)
    reindex.main()
    assert sorted(inside) == ["docs", "news"]


def test_no_wait_skips_a_pool_while_the_gpu_is_busy(reindex, indexed):
    with machine.gpu_lease("reindex of pool trag"):
        with pytest.raises(machine.GpuBusy) as caught:
            reindex._reindex_one("news", Namespace(no_wait=True), quiet=True)
    assert caught.value.purpose == "reindex of pool trag"
    assert indexed == [] and reindex.time_.sleeps == [] and reindex.on_cpu == []


def test_no_wait_skips_a_pool_without_headroom(reindex, indexed, monkeypatch):
    monkeypatch.setattr(machine, "headroom", lambda: SHORT_OF_VRAM)
    with pytest.raises(machine.NoHeadroom) as caught:
        reindex._reindex_one("news", Namespace(no_wait=True), quiet=True)
    assert "[MULLET_GPU_MIN_FREE_MB]" in str(caught.value)
    assert indexed == [] and reindex.time_.sleeps == []
    assert machine.current_holder() is None


def test_no_wait_from_the_command_line_exits_75_naming_the_holder(reindex, indexed, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["reindex", "--pool", "news", "--no-wait"])
    with machine.gpu_lease("reindex of pool trag"):
        with pytest.raises(SystemExit) as caught:
            reindex.main()
    assert caught.value.code == reindex.EXIT_MACHINE_BUSY == 75
    assert "FAILED: skipped: the GPU is in use: reindex of pool trag (pid " in capsys.readouterr().out
    assert indexed == []


def _no_model_cache(monkeypatch):
    """The --all loops clear the model cache between pools; keep CUDA out of it."""
    from mullet.knowledge.engine import model_cache

    monkeypatch.setattr(model_cache, "get_model_cache", lambda: types.SimpleNamespace(clear=lambda: None))


@pytest.mark.parametrize("mode", [[], ["--workers", "2"]])
def test_all_with_no_wait_exits_75_when_a_pool_was_skipped(reindex, indexed, monkeypatch, capsys, mode):
    _no_model_cache(monkeypatch)
    monkeypatch.setattr(reindex, "discover_clients", lambda clients_dir=None: ["news", "docs"])
    monkeypatch.setattr(sys, "argv", ["reindex", "--all", "--no-wait", *mode])
    with machine.gpu_lease("reindex of pool trag"):
        with pytest.raises(SystemExit) as caught:
            reindex.main()
    assert caught.value.code == 75
    out = capsys.readouterr().out
    assert out.count("FAILED: skipped: the GPU is in use: reindex of pool trag") == 2
    assert indexed == []


def test_all_with_no_wait_exits_normally_when_nothing_was_busy(reindex, indexed, monkeypatch):
    _no_model_cache(monkeypatch)
    monkeypatch.setattr(reindex, "discover_clients", lambda clients_dir=None: ["news", "docs"])
    monkeypatch.setattr(sys, "argv", ["reindex", "--all", "--no-wait"])
    reindex.main()
    assert [name for name, _ in indexed] == ["news", "docs"]


def test_wait_lines_share_the_print_lock(reindex, monkeypatch, capsys):
    taken = []

    class RecordingLock:
        def __enter__(self):
            taken.append(True)

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(reindex, "_print_lock", RecordingLock())
    reindex._say("news", "hello")
    assert taken == [True] and _lines(capsys) == [_line("hello")]


def test_a_locked_pool_never_asks_for_the_gpu(reindex, indexed, monkeypatch):
    asked = []
    monkeypatch.setattr(machine, "gpu_lease", lambda *a, **k: asked.append(a))
    monkeypatch.setattr(machine, "headroom", lambda: asked.append("headroom") or ROOMY)
    with reindex._pool_lock("news"):
        reindex._reindex_one("news", Namespace(no_wait=False), quiet=True)
    assert asked == [] and indexed == []


def test_cpu_only_forces_the_embedding_device_in_this_thread(reindex):
    from mullet.knowledge.engine.model_cache import embedding_device

    normal = embedding_device()
    seen_elsewhere = []
    with reindex.real_cpu_only():
        assert embedding_device() == "cpu"
        other = threading.Thread(target=lambda: seen_elsewhere.append(embedding_device()))
        other.start()
        other.join(30)
    assert embedding_device() == normal
    assert seen_elsewhere == [normal], "another thread keeps its own device"


def test_releasing_gpu_memory_only_touches_a_loaded_model_cache(reindex, monkeypatch):
    release = reindex.real_release_gpu_memory
    monkeypatch.delitem(sys.modules, "mullet.knowledge.engine.model_cache", raising=False)
    release()  # nothing loaded: nothing imported
    assert "mullet.knowledge.engine.model_cache" not in sys.modules
    cleared = []
    fake = types.SimpleNamespace(get_model_cache=lambda: types.SimpleNamespace(clear=lambda: cleared.append(1)))
    monkeypatch.setitem(sys.modules, "mullet.knowledge.engine.model_cache", fake)
    release()
    assert cleared == [1]
