"""The machine guard: the GPU lease, nvidia-smi parsing, and waiting for headroom.

No GPU is needed. nvidia-smi is replaced by fixed sample output (or a fake
script on PATH; conftest's stand-in otherwise says there is no GPU), the lease lives in the throwaway HOME's state directory
(tests/conftest.py), and waiting runs on a fake clock.
"""

from __future__ import annotations

import os
import pickle
import select
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from mullet.core import env
from mullet.jobs import machine

REPO = Path(machine.__file__).resolve().parents[2]

# What nvidia-smi prints on a 12 GB laptop GPU with LM Studio loaded.
GPU_CSV = "NVIDIA GeForce RTX 4070 Laptop GPU, 514, 12282, 63, 23, [N/A], 150.00, 150.00, 5.00, 175.00\n"
GPU_CSV_BASIC = "NVIDIA GeForce RTX 4070 Laptop GPU, 514, 12282, 63, 23\n"
APPS_CSV = (
    "21145, /usr/bin/sunshine, 186\n"
    "1954453, /home/u/.cache/lm-studio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda-avx2-2.64.0/"
    "llama-server, 6752\n"
    "1972421, /usr/bin/python3, 2846\n"
)

_HOLD = r"""
import sys
from mullet.jobs import machine
with machine.gpu_lease(sys.argv[1]):
    print("held", flush=True)
    sys.stdin.readline()
"""


class Holder:
    """Another process holding the GPU lease until ``release()``."""

    def __init__(self, purpose: str) -> None:
        child_env = {**os.environ, "PYTHONPATH": str(REPO)}
        self.proc = subprocess.Popen([sys.executable, "-c", _HOLD, purpose], cwd=REPO, env=child_env,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        ready, _, _ = select.select([self.proc.stdout], [], [], 60)
        line = self.proc.stdout.readline() if ready else ""
        if line.strip() != "held":
            self.proc.kill()
            raise AssertionError(f"the holder process didn't take the lease (got {line!r})")
        self.pid = self.proc.pid

    def release(self) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.write("\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=30)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=30)
        for stream in (self.proc.stdin, self.proc.stdout):
            stream.close()


@pytest.fixture
def holder():
    made: list[Holder] = []

    def start(purpose: str = "reindex of pool docs") -> Holder:
        made.append(Holder(purpose))
        return made[-1]

    yield start
    for h in made:
        h.close()


class FakeTime:
    """A clock that only moves when something sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _dead_pid() -> int:
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


@pytest.fixture
def no_settings(monkeypatch):
    """No MULLET_* machine settings from the environment or any .env file."""
    for name in ("MULLET_GPU_MIN_FREE_MB", "MULLET_RAM_MIN_FREE_MB", "MULLET_GPU_MAX_TEMP_C",
                 "MULLET_MAX_LOAD", "MULLET_GPU_WAIT_MAX_S"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(env, "dotenv_files", lambda: [])


# --- The lease ----------------------------------------------------------------------


def test_the_lease_lives_in_the_state_dir_and_records_its_holder():
    assert machine.lock_path() == Path.home() / ".local" / "state" / "mullet" / "gpu.lock"
    assert machine.current_holder() is None
    with machine.gpu_lease("reindex of pool docs") as me:
        assert me.pid == os.getpid() and me.purpose == "reindex of pool docs"
        assert machine.current_holder() == me
        assert machine.holder_path().stat().st_mode & 0o777 == 0o600
    assert machine.current_holder() is None
    assert not machine.holder_path().exists()
    assert machine.lock_path().exists()


def test_a_second_process_is_refused_with_the_holders_purpose(holder):
    other = holder("reindex of pool trag")
    with pytest.raises(machine.GpuBusy) as caught:
        with machine.gpu_lease("my work", wait=False):
            pytest.fail("took a lease another process holds")
    busy = caught.value
    assert busy.holder.pid == other.pid and busy.purpose == "reindex of pool trag"
    assert busy.waited is None
    assert str(busy).startswith(f"the GPU is in use: reindex of pool trag (pid {other.pid}, since ")
    other.release()
    with machine.gpu_lease("my work", wait=False) as me:
        assert machine.current_holder() == me


def test_waiting_reports_the_holder_once_and_takes_the_lease_when_freed(holder):
    other = holder()
    seen, sleeps = [], []

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            other.release()

    with machine.gpu_lease("next", poll=7.0, on_wait=seen.append, sleep=sleep) as me:
        assert machine.current_holder() == me
    assert [h.purpose for h in seen] == ["reindex of pool docs"]
    assert sleeps == [7.0, 7.0]


def test_running_out_of_time_raises_with_the_holder(holder):
    other = holder()
    fake, seen = FakeTime(), []
    with pytest.raises(machine.GpuBusy) as caught:
        with machine.gpu_lease("next", timeout=10, poll=3, on_wait=seen.append,
                               clock=fake.clock, sleep=fake.sleep):
            pytest.fail("took a lease another process holds")
    assert fake.sleeps == [3, 3, 3, 1]
    assert caught.value.waited == 10 and caught.value.holder.pid == other.pid
    assert str(caught.value).startswith("gave up after 10s waiting for the GPU: reindex of pool docs")
    assert len(seen) == 1


def test_a_zero_timeout_tries_once_without_calling_on_wait(holder):
    holder()
    fake, seen = FakeTime(), []
    with pytest.raises(machine.GpuBusy):
        with machine.gpu_lease("next", timeout=0, on_wait=seen.append, clock=fake.clock, sleep=fake.sleep):
            pass
    assert fake.sleeps == [] and seen == []


def test_the_module_clock_and_sleep_are_used_by_default(holder, monkeypatch):
    holder()
    fake = FakeTime()
    monkeypatch.setattr(machine, "_clock", fake.clock)
    monkeypatch.setattr(machine, "_sleep", fake.sleep)
    with pytest.raises(machine.GpuBusy):
        with machine.gpu_lease("next", timeout=6, poll=5):
            pass
    assert fake.sleeps == [5, 1]


# 12 s less a gap the fake clock at 1000 cannot hold: the last pause would not move it.
OFF_GRID = 12.0 - ((3.0 + 2.5e-5) - 3.0)


def _bounded(fake: FakeTime, limit: int = 50):
    def sleep(seconds: float) -> None:
        fake.sleep(seconds)
        if len(fake.sleeps) > limit:
            pytest.fail(f"still waiting after {limit} pauses: {fake.sleeps[-3:]}")
    return sleep


def test_a_wait_whose_gap_the_clock_cannot_hold_gives_up_in_time(holder):
    holder()
    fake = FakeTime()
    with pytest.raises(machine.GpuBusy) as caught:
        with machine.gpu_lease("next", timeout=OFF_GRID, poll=5, clock=fake.clock, sleep=_bounded(fake)):
            pass
    assert fake.sleeps[:3] == [5, 5, pytest.approx(2.0, abs=1e-3)] and len(fake.sleeps) <= 4
    assert str(caught.value).startswith("gave up after 12s waiting for the GPU")


def test_a_clock_that_stops_moving_ends_a_finite_wait(holder):
    holder()
    fake = FakeTime()
    with pytest.raises(machine.GpuBusy):
        with machine.gpu_lease("next", timeout=30, poll=5, clock=fake.clock,
                               sleep=lambda s: fake.sleeps.append(s)):
            pass
    assert fake.sleeps == [5]


def test_another_thread_in_the_same_process_waits_too():
    taken, done = threading.Event(), threading.Event()

    def hold():
        with machine.gpu_lease("thread"):
            taken.set()
            done.wait(30)

    worker = threading.Thread(target=hold)
    worker.start()
    try:
        assert taken.wait(30)
        with pytest.raises(machine.GpuBusy) as caught:
            with machine.gpu_lease("main", wait=False):
                pass
        assert caught.value.purpose == "thread"
    finally:
        done.set()
        worker.join(30)


def test_a_record_left_by_a_dead_process_is_ignored():
    path = machine.holder_path()
    path.parent.mkdir(parents=True)
    path.write_text(f'{{"pid": {_dead_pid()}, "purpose": "crashed", "started": {time.time()}}}')
    assert machine.current_holder() is None
    with machine.gpu_lease("mine", wait=False) as me:
        assert machine.current_holder() == me
    assert not path.exists()


def test_a_record_whose_pid_was_reused_is_ignored():
    path = machine.holder_path()
    path.parent.mkdir(parents=True)
    path.write_text(f'{{"pid": {os.getpid()}, "purpose": "old", "started": 1.0, "pid_started": 2.0}}')
    assert machine.current_holder() is None


@pytest.mark.parametrize("text", ["", "{nope", "[]", '{"pid": "1", "purpose": "x", "started": 1}',
                                  '{"pid": true, "purpose": "x", "started": 1}',
                                  '{"pid": 1, "purpose": 2, "started": 1}'])
def test_an_unreadable_record_means_no_holder(text):
    path = machine.holder_path()
    path.parent.mkdir(parents=True)
    path.write_text(text)
    assert machine.current_holder() is None


def test_a_held_lock_with_a_stale_record_says_the_holder_is_unknown(holder):
    holder()
    machine.holder_path().write_text(f'{{"pid": {_dead_pid()}, "purpose": "x", "started": 1.0}}')
    with pytest.raises(machine.GpuBusy) as caught:
        with machine.gpu_lease("mine", wait=False):
            pass
    assert caught.value.holder is None and caught.value.purpose is None
    assert "left no record" in str(caught.value)


def test_describe_adds_the_date_when_the_lease_is_from_another_day():
    started = time.mktime((2026, 10, 23, 22, 41, 0, 0, 0, -1))
    holder = machine.GpuHolder(42, "reindex of pool x", started)
    assert holder.describe(now=started + 60) == "reindex of pool x (pid 42, since 22:41)"
    assert holder.describe(now=started + 86400) == "reindex of pool x (pid 42, since 2026-10-23 22:41)"


def test_the_exceptions_survive_a_process_pool():
    busy = machine.GpuBusy(machine.GpuHolder(7, "reindex of pool a", 1.0), 12.0)
    again = pickle.loads(pickle.dumps(busy))
    assert (again.holder, again.waited, str(again)) == (busy.holder, 12.0, str(busy))
    short = [machine.Shortfall("ram", "100 MiB RAM available, needs 8192 MiB")]
    room = machine.NoHeadroom(machine.Headroom(ram_free_mb=100), short)
    again = pickle.loads(pickle.dumps(room))
    assert (again.headroom, again.shortfalls, str(again)) == (room.headroom, short, str(room))
    assert isinstance(again, machine.MachineBusy)


# --- Own processes --------------------------------------------------------------------


@pytest.fixture
def children():
    """Start sleeping child processes; they're killed afterwards."""
    made: list[subprocess.Popen] = []

    def start(**kwargs) -> subprocess.Popen:
        made.append(subprocess.Popen(["sleep", "60"], **kwargs))
        return made[-1]

    yield start
    for proc in made:
        proc.kill()
        proc.wait()


def test_related_pids_cover_children_and_siblings_in_the_process_group(children):
    first, second = children(), children()
    elsewhere = children(start_new_session=True)
    mine = machine.related_pids()
    assert {os.getpid(), first.pid, second.pid, elsewhere.pid} <= mine, "children always count"
    sibling_view = machine.related_pids(first.pid)
    assert {first.pid, os.getpid(), second.pid} <= sibling_view
    assert elsewhere.pid not in sibling_view, "a sibling in another process group isn't ours"


def test_related_pids_of_a_dead_pid_is_just_that_pid():
    pid = _dead_pid()
    assert machine.related_pids(pid) == frozenset({pid})


def test_vram_held_by_a_child_counts_as_free_by_default(no_settings, children):
    child = children()
    room = machine.Headroom(vram_free_mb=500, gpu_processes=(machine.GpuProcess(child.pid, "python3", 3000),))
    assert machine.wait_for_headroom(need_ram_mb=0, max_load=10**6, timeout=0, sample=lambda: room) is room
    with pytest.raises(machine.NoHeadroom):
        machine.wait_for_headroom(need_ram_mb=0, max_load=10**6, timeout=0, own_pids=(), sample=lambda: room)


# --- nvidia-smi --------------------------------------------------------------------------


def test_gpu_csv_is_parsed_with_its_power_limits():
    room = machine.parse_gpu_csv(GPU_CSV)
    assert room == machine.Headroom(
        gpu_name="NVIDIA GeForce RTX 4070 Laptop GPU", vram_free_mb=514, vram_total_mb=12282,
        gpu_temp_c=63.0, gpu_util_pct=23.0,
        power=machine.PowerLimits(current_w=150.0, default_w=150.0, min_w=5.0, max_w=175.0))
    assert room.has_gpu and room.power.at_default and room.power.suggestion() == 100.0


def test_the_basic_query_has_no_power_limits():
    room = machine.parse_gpu_csv(GPU_CSV_BASIC, with_power=False)
    assert room.vram_free_mb == 514 and room.power is None


def test_the_first_gpu_is_read_and_unreported_values_are_none():
    room = machine.parse_gpu_csv("GPU A, 100, [N/A], [Not Supported], [N/A], [N/A], [N/A], [N/A], [N/A], [N/A]\n"
                                 "GPU B, 9000, 9000, 40, 0, 1, 1, 1, 1, 1\n")
    assert (room.gpu_name, room.vram_free_mb, room.vram_total_mb) == ("GPU A", 100, None)
    assert room.gpu_temp_c is None and room.gpu_util_pct is None and room.power is None


@pytest.mark.parametrize("text", [
    None, "", "\n",
    'Field "bogus.field" is not a valid field to query.\n',
    "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n",
    "a, b, c, d, e, f, g, h, i, j\n", "1, 2, 3\n",
])
def test_garbage_gpu_output_means_no_gpu(text):
    room = machine.parse_gpu_csv(text)
    assert room == machine.Headroom() and not room.has_gpu


def test_compute_apps_are_parsed_and_lm_studio_flagged():
    apps = machine.parse_compute_apps(APPS_CSV + "77, /opt/lmstudio/llmster, [N/A]\nNo running processes found\n")
    assert [(p.pid, p.used_mb) for p in apps] == [(21145, 186), (1954453, 6752), (1972421, 2846), (77, None)]
    assert [p.is_lm_studio for p in apps] == [False, True, False, True]
    room = machine.Headroom(gpu_processes=apps)
    assert [p.pid for p in room.lm_studio_processes] == [1954453, 77]


def test_a_process_name_with_a_comma_is_kept_whole():
    (app,) = machine.parse_compute_apps("5, /opt/a, b/llama-server, 10\n")
    assert (app.pid, app.name, app.used_mb, app.is_lm_studio) == (5, "/opt/a, b/llama-server", 10, True)


def test_power_suggestions():
    lowered = machine.parse_gpu_csv("G, 1, 2, 3, 4, 100.00, 100.00, 150.00, 5.00, 175.00").power
    assert not lowered.at_default and lowered.suggestion() is None
    assert machine.PowerLimits(150.0, 150.0, 120.0, 175.0).suggestion() == 120.0
    assert machine.PowerLimits(None, 150.0, 5.0, 175.0).suggestion() is None
    assert not machine.PowerLimits(None, 150.0, 5.0, 175.0).at_default


def test_headroom_without_nvidia_smi_has_no_gpu_but_still_ram_and_load(monkeypatch):
    calls = []
    monkeypatch.setattr(machine, "_nvidia_smi", lambda args, timeout=None: calls.append(args[0]))
    room = machine.headroom()
    assert not room.has_gpu and room.gpu_processes == () and room.power is None
    assert room.ram_free_mb > 0 and room.load_1m >= 0
    assert calls == [f"--query-gpu={machine.GPU_QUERY}", f"--query-gpu={machine.GPU_QUERY_BASIC}"], \
        "no GPU, so the processes aren't queried"


def test_headroom_reads_the_gpu_and_its_processes_in_two_calls(monkeypatch):
    answers = {f"--query-gpu={machine.GPU_QUERY}": GPU_CSV,
               f"--query-compute-apps={machine.APPS_QUERY}": APPS_CSV}
    calls = []
    monkeypatch.setattr(machine, "_nvidia_smi",
                        lambda args, timeout=None: calls.append(timeout) or answers[args[0]])
    room = machine.headroom(timeout=2.0)
    assert room.vram_free_mb == 514 and len(room.gpu_processes) == 3 and room.power.at_default
    assert calls == [2.0, 2.0]


def test_a_driver_that_rejects_the_power_fields_still_gives_the_gpu(monkeypatch):
    answers = {f"--query-gpu={machine.GPU_QUERY}": None,
               f"--query-gpu={machine.GPU_QUERY_BASIC}": GPU_CSV_BASIC,
               f"--query-compute-apps={machine.APPS_QUERY}": ""}
    monkeypatch.setattr(machine, "_nvidia_smi", lambda args, timeout=None: answers[args[0]])
    room = machine.headroom()
    assert room.vram_free_mb == 514 and room.power is None


def test_a_hung_nvidia_smi_is_asked_once(monkeypatch, caplog):
    calls = []

    def hung(args, timeout=None):
        calls.append(args)
        raise machine._Hung("timed out")

    monkeypatch.setattr(machine, "_nvidia_smi", hung)
    assert not machine.headroom().has_gpu
    assert len(calls) == 1 and "didn't answer" in caplog.text


def _fake_nvidia_smi(tmp_path, monkeypatch, script: str) -> Path:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "nvidia-smi"
    exe.write_text("#!/bin/sh\n" + f'echo "$1" >> "{bindir}/calls"\n' + script)
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return bindir / "calls"


def test_the_tests_see_no_gpu():
    assert machine._nvidia_smi(["-L"]) is None
    assert not machine.headroom().has_gpu


def test_a_missing_nvidia_smi_gives_none(monkeypatch):
    monkeypatch.setattr(machine.shutil, "which", lambda name: None)
    assert machine._nvidia_smi(["-L"]) is None
    assert not machine.headroom().has_gpu


def test_a_failing_nvidia_smi_gives_none(tmp_path, monkeypatch):
    _fake_nvidia_smi(tmp_path, monkeypatch, 'echo "NVIDIA-SMI has failed"\nexit 9\n')
    assert machine._nvidia_smi(["-L"]) is None
    assert not machine.headroom().has_gpu


def test_a_slow_nvidia_smi_times_out_and_is_not_asked_again(tmp_path, monkeypatch):
    calls = _fake_nvidia_smi(tmp_path, monkeypatch, "exec sleep 30\n")
    started = time.monotonic()
    assert not machine.headroom(timeout=0.5).has_gpu
    assert time.monotonic() - started < 10
    assert len(calls.read_text().splitlines()) == 1


def test_garbage_from_nvidia_smi_means_no_gpu(tmp_path, monkeypatch):
    _fake_nvidia_smi(tmp_path, monkeypatch, 'printf "\\377\\376 not csv at all\\n"\n')
    assert not machine.headroom().has_gpu


def test_a_working_nvidia_smi_is_read_end_to_end(tmp_path, monkeypatch):
    _fake_nvidia_smi(tmp_path, monkeypatch, f"""case "$1" in
  --query-gpu=*) printf '{GPU_CSV}';;
  --query-compute-apps=*) printf '{APPS_CSV}';;
  *) exit 2;;
esac
""")
    room = machine.headroom()
    assert room.gpu_name == "NVIDIA GeForce RTX 4070 Laptop GPU" and room.vram_free_mb == 514
    assert [p.pid for p in room.lm_studio_processes] == [1954453]
    assert room.power.at_default


# --- Settings ---------------------------------------------------------------------------


def test_limits_default(no_settings):
    assert machine.limits() == machine.Limits(2048.0, 8192.0, 83.0, float(os.cpu_count()))
    assert machine.max_wait_s() == 900.0


def test_limits_come_from_the_environment_then_the_env_file(no_settings, tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("MULLET_GPU_MIN_FREE_MB=4096\nMULLET_MAX_LOAD=12\nMULLET_GPU_MAX_TEMP_C=80\n"
                      "MULLET_GPU_WAIT_MAX_S=60\n")
    monkeypatch.setattr(env, "dotenv_files", lambda: [dotenv])
    monkeypatch.setenv("MULLET_GPU_MAX_TEMP_C", "75")
    monkeypatch.setenv("MULLET_RAM_MIN_FREE_MB", "1024")
    assert machine.limits() == machine.Limits(4096.0, 1024.0, 75.0, 12.0)
    assert machine.max_wait_s() == 60.0


@pytest.mark.parametrize("value", ["lots", "-1", "nan", "inf"])
def test_a_bad_setting_keeps_the_default(no_settings, monkeypatch, caplog, value):
    monkeypatch.setenv("MULLET_GPU_MIN_FREE_MB", value)
    assert machine.limits().gpu_min_free_mb == 2048.0
    assert "MULLET_GPU_MIN_FREE_MB" in caplog.text and value not in caplog.text


def test_the_lease_waits_for_the_max_wait_setting_by_default(no_settings, holder, monkeypatch):
    holder()
    monkeypatch.setenv("MULLET_GPU_WAIT_MAX_S", "10")
    fake = FakeTime()
    with pytest.raises(machine.GpuBusy) as caught:
        with machine.gpu_lease("next", poll=4, clock=fake.clock, sleep=fake.sleep):
            pass
    assert fake.sleeps == [4, 4, 2] and caught.value.waited == 10


# --- Waiting for headroom ----------------------------------------------------------------

ROOMY = machine.Headroom(vram_free_mb=8000, vram_total_mb=12282, gpu_temp_c=50.0,
                         ram_free_mb=40000, load_1m=2.0)
LIMITS = machine.Limits(2048, 8192, 83, 16)


def test_shortfalls_name_each_failing_check_and_its_setting():
    assert ROOMY.shortfalls(LIMITS) == []
    lms = (machine.GpuProcess(9, "/x/llama-server", 6000),)
    busy = machine.Headroom(vram_free_mb=500, gpu_temp_c=88.0, gpu_processes=lms, ram_free_mb=100,
                            load_1m=40.0)
    short = busy.shortfalls(LIMITS)
    assert [str(s) for s in short] == [
        "GPU has 500 MiB free, needs 2048 MiB (LM Studio holds 6000 MiB)",
        "GPU at 88°C, over 83°C",
        "100 MiB RAM available, needs 8192 MiB",
        "load average 40.0, over 16",
    ]
    assert [s.setting for s in short] == ["MULLET_GPU_MIN_FREE_MB", "MULLET_GPU_MAX_TEMP_C",
                                          "MULLET_RAM_MIN_FREE_MB", "MULLET_MAX_LOAD"]
    assert short[0].explain() == ("GPU has 500 MiB free, needs 2048 MiB (LM Studio holds 6000 MiB) "
                                  "[MULLET_GPU_MIN_FREE_MB]")


def test_missing_readings_pass():
    assert machine.Headroom().shortfalls(LIMITS) == []


def test_vram_our_own_processes_hold_counts_as_free():
    room = machine.Headroom(vram_free_mb=500, gpu_processes=(machine.GpuProcess(os.getpid(), "python3", 3000),))
    assert room.vram_free_for({os.getpid()}) == 3500
    assert room.shortfalls(LIMITS, own_pids={os.getpid()}) == []
    assert [s.check for s in room.shortfalls(LIMITS)] == ["vram"]


def test_returns_at_once_when_there_is_room(no_settings):
    fake = FakeTime()
    assert machine.wait_for_headroom(sample=lambda: ROOMY, clock=fake.clock, sleep=fake.sleep) is ROOMY
    assert fake.sleeps == []


def test_waits_until_there_is_room_and_reports_changes(no_settings):
    fake = FakeTime()
    samples = iter([
        machine.Headroom(vram_free_mb=500),
        machine.Headroom(vram_free_mb=900),  # still short of VRAM: not reported again
        machine.Headroom(vram_free_mb=9000, gpu_temp_c=90.0),
        ROOMY,
    ])
    seen = []
    room = machine.wait_for_headroom(poll=4, sample=lambda: next(samples), clock=fake.clock,
                                     sleep=fake.sleep, on_wait=lambda r, s: seen.append([str(x) for x in s]))
    assert room is ROOMY
    assert fake.sleeps == [4, 4, 4]
    assert seen == [["GPU has 500 MiB free, needs 2048 MiB"], ["GPU at 90°C, over 83°C"]]


def test_gives_up_after_the_timeout(no_settings):
    fake = FakeTime()
    busy = machine.Headroom(vram_free_mb=9000, ram_free_mb=10, load_1m=1.0)
    with pytest.raises(machine.NoHeadroom) as caught:
        machine.wait_for_headroom(timeout=12, poll=5, sample=lambda: busy, clock=fake.clock, sleep=fake.sleep)
    assert fake.sleeps == [5, 5, 2]
    assert caught.value.waited == 12 and caught.value.headroom is busy
    assert str(caught.value) == ("gave up after 12s waiting for headroom: 10 MiB RAM available, "
                                 "needs 8192 MiB [MULLET_RAM_MIN_FREE_MB]")


def test_a_headroom_wait_whose_gap_the_clock_cannot_hold_gives_up_in_time(no_settings):
    fake = FakeTime()
    busy = machine.Headroom(vram_free_mb=9000, ram_free_mb=10, load_1m=1.0)
    with pytest.raises(machine.NoHeadroom):
        machine.wait_for_headroom(timeout=OFF_GRID, poll=5, sample=lambda: busy, clock=fake.clock,
                                  sleep=_bounded(fake))
    assert len(fake.sleeps) <= 4


def test_a_clock_that_stops_moving_ends_a_finite_headroom_wait(no_settings):
    fake = FakeTime()
    busy = machine.Headroom(vram_free_mb=9000, ram_free_mb=10, load_1m=1.0)
    with pytest.raises(machine.NoHeadroom):
        machine.wait_for_headroom(timeout=30, poll=5, sample=lambda: busy, clock=fake.clock,
                                  sleep=lambda s: fake.sleeps.append(s))
    assert fake.sleeps == [5]


def test_the_default_timeout_is_the_max_wait_setting(no_settings, monkeypatch):
    monkeypatch.setenv("MULLET_GPU_WAIT_MAX_S", "10")
    fake = FakeTime()
    with pytest.raises(machine.NoHeadroom):
        machine.wait_for_headroom(poll=4, sample=lambda: machine.Headroom(vram_free_mb=1),
                                  clock=fake.clock, sleep=fake.sleep)
    assert fake.sleeps == [4, 4, 2]


def test_a_zero_timeout_checks_once(no_settings):
    fake, seen = FakeTime(), []
    with pytest.raises(machine.NoHeadroom) as caught:
        machine.wait_for_headroom(timeout=0, sample=lambda: machine.Headroom(vram_free_mb=9000, load_1m=999.0),
                                  on_wait=lambda *a: seen.append(a), clock=fake.clock, sleep=fake.sleep)
    assert fake.sleeps == [] and seen == []
    assert str(caught.value).startswith("not enough headroom: load average 999.0")


def test_thresholds_from_the_caller_beat_the_settings(no_settings, monkeypatch):
    monkeypatch.setenv("MULLET_GPU_MIN_FREE_MB", "100000")
    monkeypatch.setenv("MULLET_MAX_LOAD", "0.5")
    room = machine.Headroom(vram_free_mb=3000, load_1m=4.0)
    with pytest.raises(machine.NoHeadroom) as caught:
        machine.wait_for_headroom(timeout=0, sample=lambda: room)
    assert [s.check for s in caught.value.shortfalls] == ["vram", "load"]
    assert machine.wait_for_headroom(need_vram_mb=2048, max_load=8, timeout=0, sample=lambda: room) is room


def test_without_a_gpu_a_busy_machine_waits_briefly_then_goes_ahead(no_settings):
    fake, seen = FakeTime(), []
    busy = machine.Headroom(ram_free_mb=10, load_1m=999.0)
    room = machine.wait_for_headroom(poll=25, sample=lambda: busy, clock=fake.clock, sleep=fake.sleep,
                                     on_wait=lambda r, s: seen.append([x.check for x in s]))
    assert room is busy
    assert fake.sleeps == [25, 25, 10] and sum(fake.sleeps) == machine.NO_GPU_MAX_WAIT_S
    assert seen == [["ram", "load"]]


def test_without_a_gpu_no_wait_goes_ahead_at_once(no_settings):
    fake = FakeTime()
    busy = machine.Headroom(ram_free_mb=10, load_1m=999.0)
    assert machine.wait_for_headroom(timeout=0, sample=lambda: busy, clock=fake.clock, sleep=fake.sleep) is busy
    assert fake.sleeps == []


def test_a_real_machine_without_a_gpu_never_raises(no_settings):
    room = machine.wait_for_headroom(need_ram_mb=10**9, max_load=0, timeout=0)
    assert not room.has_gpu


def test_one_whole_test_suite_at_a_time():
    import fcntl

    path = machine.paths.state_dir() / machine.SUITE_LOCK_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)  # another job's suite is running
    entered = threading.Event()

    def suite():
        with machine.suite_lease(poll=0.01):
            entered.set()

    thread = threading.Thread(target=suite)
    thread.start()
    assert not entered.wait(0.3)
    os.close(fd)  # that suite finishes
    assert entered.wait(5)
    thread.join(5)
