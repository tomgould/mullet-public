"""The machine guard: one GPU lease at a time, and waiting until the machine has room.

Mullet's heavy local work (a Trawl reindex embedding chunks on the GPU, for
one) shares a machine with LM Studio, test suites and whatever else is
running. Two things keep it from overloading the machine:

- ``gpu_lease(purpose)`` holds the GPU lease: an ``flock`` on
  ``state_dir()/gpu.lock``, so one lease across every Mullet process of this
  user. The holder's pid, purpose and start time sit beside it in
  ``gpu.lock.json``, so a waiter can say what it is waiting for. The kernel
  drops the lock when its holder dies, so a record left by a dead process is
  ignored.
- ``suite_lease()`` lets one whole test suite run at a time, so two jobs'
  gates never slow each other into false failures.
- ``wait_for_headroom()`` waits until there is enough free VRAM and RAM, the
  GPU is cool enough and the load average is low enough. ``headroom()`` takes
  the snapshot from ``nvidia-smi``, ``psutil`` and ``os.getloadavg()``.

Nothing waits for long. Both waits give up after ``MULLET_GPU_WAIT_MAX_S``
unless the caller says otherwise. Without an NVIDIA GPU (no ``nvidia-smi``,
or it fails) there is nothing to lease or check on the GPU, and RAM and load
are waited on for ``NO_GPU_MAX_WAIT_S`` at most before the work goes ahead,
so machines without one (a Mac, a small box) never sit idle.

Settings (the environment first, then Mullet's ``.env`` files):

- ``MULLET_GPU_MIN_FREE_MB``: free VRAM to wait for, default 2048 MiB;
- ``MULLET_RAM_MIN_FREE_MB``: available RAM to wait for, default 8192 MiB;
- ``MULLET_GPU_MAX_TEMP_C``: the hottest GPU to start on, default 83 °C;
- ``MULLET_MAX_LOAD``: the highest 1-minute load average, default the CPU count;
- ``MULLET_GPU_WAIT_MAX_S``: the longest wait, default 900 s (LM Studio
  unloads an idle model after 600 s).
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Collection, Iterator

import psutil

from mullet.core import env, paths

logger = logging.getLogger(__name__)

LOCK_NAME = "gpu.lock"
HOLDER_NAME = "gpu.lock.json"
# LM Studio's model server and its headless daemon, as nvidia-smi names them.
LM_STUDIO_PROCESS_NAMES = ("llama-server", "llmster")
NVIDIA_SMI_TIMEOUT_S = 3.0
# The power limit `mullet doctor` suggests for all-day local work, in watts.
SUGGESTED_POWER_LIMIT_W = 100.0
# Without a GPU, the longest wait for RAM and load before the work goes ahead.
NO_GPU_MAX_WAIT_S = 60.0

DEFAULT_GPU_MIN_FREE_MB = 2048.0
DEFAULT_RAM_MIN_FREE_MB = 8192.0
DEFAULT_GPU_MAX_TEMP_C = 83.0
DEFAULT_GPU_WAIT_MAX_S = 900.0

# One query for the GPU's state and its power limits; the basic one is the
# fallback for a driver that rejects one of the power fields.
GPU_QUERY_BASIC = "name,memory.free,memory.total,temperature.gpu,utilization.gpu"
GPU_QUERY = (GPU_QUERY_BASIC + ",power.limit,enforced.power.limit,power.default_limit,"
             "power.min_limit,power.max_limit")
APPS_QUERY = "pid,process_name,used_memory"
_CSV = "--format=csv,noheader,nounits"

# The setting behind each check, so messages can name it.
SETTING_FOR_CHECK = {
    "vram": "MULLET_GPU_MIN_FREE_MB",
    "temperature": "MULLET_GPU_MAX_TEMP_C",
    "ram": "MULLET_RAM_MIN_FREE_MB",
    "load": "MULLET_MAX_LOAD",
}

_LOCK_BUSY = frozenset({errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK})
_NOT_REPORTED = object()
# What waiting uses unless the caller passes its own; looked up at call time,
# so tests can replace them here.
_clock: Callable[[], float] = time.monotonic
_sleep: Callable[[float], None] = time.sleep


# --- Settings ------------------------------------------------------------------------


@dataclass(frozen=True)
class Limits:
    """What work needs from the machine before it starts."""

    gpu_min_free_mb: float
    ram_min_free_mb: float
    gpu_max_temp_c: float
    max_load: float


def _number(text: str) -> float | None:
    """A finite number, or None for "[N/A]", "[Not Supported]" and other text."""
    try:
        value = float(text.strip())
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _setting(name: str, default: float) -> float:
    """A number setting from the environment, else Mullet's .env files, else ``default``."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        raw = env.dotenv_value(name)
    if raw is None or not raw.strip():
        return default
    value = _number(raw)
    if value is None or value < 0:
        logger.warning("machine: ignoring %s, which isn't a number of 0 or more; using %g", name, default)
        return default
    return value


def limits() -> Limits:
    """The thresholds from settings (see the module docstring for names and defaults)."""
    return Limits(
        gpu_min_free_mb=_setting("MULLET_GPU_MIN_FREE_MB", DEFAULT_GPU_MIN_FREE_MB),
        ram_min_free_mb=_setting("MULLET_RAM_MIN_FREE_MB", DEFAULT_RAM_MIN_FREE_MB),
        gpu_max_temp_c=_setting("MULLET_GPU_MAX_TEMP_C", DEFAULT_GPU_MAX_TEMP_C),
        max_load=_setting("MULLET_MAX_LOAD", float(os.cpu_count() or 1)),
    )


def max_wait_s() -> float:
    """``MULLET_GPU_WAIT_MAX_S``: how long a wait lasts unless the caller says otherwise."""
    return _setting("MULLET_GPU_WAIT_MAX_S", DEFAULT_GPU_WAIT_MAX_S)


# --- The GPU lease -------------------------------------------------------------------


@dataclass(frozen=True)
class GpuHolder:
    """Who holds the GPU lease: a process, what it is doing and since when.

    ``started`` is the wall-clock time the lease was taken. ``pid_started`` is
    the process's own start time, which tells a reused pid apart.
    """

    pid: int
    purpose: str
    started: float
    pid_started: float | None = None

    def describe(self, now: float | None = None) -> str:
        """Like ``reindex of pool docs (pid 4242, since 22:41)``; the date shows when it isn't today."""
        when = time.localtime(self.started)
        today = time.localtime(time.time() if now is None else now)
        same_day = when[:3] == today[:3]
        since = time.strftime("%H:%M" if same_day else "%Y-%m-%d %H:%M", when)
        return f"{self.purpose} (pid {self.pid}, since {since})"


class MachineBusy(RuntimeError):
    """The machine can't take the work now: the base of ``GpuBusy`` and ``NoHeadroom``."""


class GpuBusy(MachineBusy):
    """Another process holds the GPU lease.

    ``holder`` is who, or None when the holder left no live record. ``waited``
    is how long the caller waited before giving up, None when it didn't wait.
    """

    def __init__(self, holder: GpuHolder | None, waited: float | None = None) -> None:
        self.holder = holder
        self.waited = waited
        who = holder.describe() if holder else "another Mullet process (it left no record)"
        if waited:
            message = f"gave up after {waited:.0f}s waiting for the GPU: {who}"
        else:
            message = f"the GPU is in use: {who}"
        super().__init__(message)

    @property
    def purpose(self) -> str | None:
        """What the holder is doing, when known."""
        return self.holder.purpose if self.holder else None

    def __reduce__(self):  # keep it picklable across a process pool
        return (type(self), (self.holder, self.waited))


def lock_path() -> Path:
    """The lock file behind the GPU lease."""
    return paths.state_dir() / LOCK_NAME


def holder_path() -> Path:
    """The record of who holds the GPU lease, beside the lock file."""
    return paths.state_dir() / HOLDER_NAME


@contextlib.contextmanager
def gpu_lease(
    purpose: str,
    *,
    wait: bool = True,
    timeout: float | None = None,
    poll: float = 5.0,
    on_wait: Callable[[GpuHolder | None], None] | None = None,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> Iterator[GpuHolder]:
    """Hold the GPU lease for the ``with`` block; yields this process's record.

    While another process holds it this waits, trying again every ``poll``
    seconds, for ``timeout`` seconds at most: ``max_wait_s()`` when None,
    and ``math.inf`` waits as long as it takes. ``on_wait(holder)`` is called
    before the first pause and again whenever the holder changes (``holder``
    is None when the record is missing or stale). ``wait=False`` raises
    ``GpuBusy`` at once and running out of time raises it too; both carry
    the holder. ``clock`` and ``sleep`` replace ``time.monotonic`` and
    ``time.sleep``.

    Not re-entrant: taking it again in the same process, from another thread
    say, waits like any other process would.
    """
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        _acquire(fd, wait=wait, timeout=max_wait_s() if timeout is None else timeout, poll=poll,
                 on_wait=on_wait, clock=clock or _clock, sleep=sleep or _sleep)
        try:
            me = GpuHolder(os.getpid(), purpose, time.time(), _process_started(os.getpid()))
            _write_record(me)
            try:
                yield me
            finally:
                _clear_record(me)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


SUITE_LOCK_NAME = "suite.lock"


@contextlib.contextmanager
def suite_lease(*, poll: float = 5.0, sleep: Callable[[float], None] | None = None) -> Iterator[None]:
    """Run one whole test suite at a time on this machine, across every job.

    Two full suites side by side (two jobs' gates, say) each run slower and
    fail timing-sensitive tests that pass alone, which a gate then reads as
    regressions. This waits, as long as it takes, for the other suite: that
    one is bounded by its own timeout. An ``flock`` on
    ``state_dir()/suite.lock``, dropped by the kernel if the holder dies.
    """
    path = paths.state_dir() / SUITE_LOCK_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        _acquire(fd, wait=True, timeout=math.inf, poll=poll, on_wait=None,
                 clock=_clock, sleep=sleep or _sleep)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _acquire(fd: int, *, wait: bool, timeout: float, poll: float,
             on_wait: Callable[[GpuHolder | None], None] | None,
             clock: Callable[[], float], sleep: Callable[[float], None]) -> None:
    started = clock()
    reported: object = _NOT_REPORTED
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError as exc:
            if exc.errno not in _LOCK_BUSY:
                raise
        holder = current_holder()
        if not wait:
            raise GpuBusy(holder)
        waited = clock() - started
        if waited >= timeout:
            raise GpuBusy(holder, waited)
        if on_wait is not None and holder != reported:
            reported = holder
            on_wait(holder)
        sleep(max(0.0, min(poll, timeout - waited)))


def current_holder() -> GpuHolder | None:
    """The holder of the GPU lease, from its record, or None.

    A record left by a process that has exited, or whose pid now belongs to a
    newer process, is stale and ignored. This reads the record only and never
    touches the lock.
    """
    holder = _read_record()
    if holder is None or not _is_running(holder):
        return None
    return holder


def _read_record() -> GpuHolder | None:
    try:
        data = json.loads(holder_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    pid, purpose, started = data.get("pid"), data.get("purpose"), data.get("started")
    pid_started = data.get("pid_started")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    if not isinstance(purpose, str) or not _is_number(started):
        return None
    return GpuHolder(pid, purpose, float(started),
                     float(pid_started) if _is_number(pid_started) else None)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_running(holder: GpuHolder) -> bool:
    try:
        proc = psutil.Process(holder.pid)
        if proc.status() == psutil.STATUS_ZOMBIE:
            return False
        if holder.pid_started is not None and abs(proc.create_time() - holder.pid_started) > 1.0:
            return False  # the pid was reused
        return True
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True  # it exists; trust the record
    except psutil.Error:
        return False


def _process_started(pid: int) -> float | None:
    try:
        return psutil.Process(pid).create_time()
    except psutil.Error:
        return None


def _write_record(holder: GpuHolder) -> None:
    """Write the holder's record atomically. Failing to is logged, not fatal."""
    target = holder_path()
    try:
        fd, tmp = tempfile.mkstemp(prefix=f".{HOLDER_NAME}.", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(asdict(holder), fh)
            os.replace(tmp, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
    except OSError as exc:
        logger.warning("machine: couldn't record the GPU lease holder in %s: %s", target, exc)


def _clear_record(holder: GpuHolder) -> None:
    record = _read_record()
    if record is not None and (record.pid, record.started) == (holder.pid, holder.started):
        with contextlib.suppress(OSError):
            holder_path().unlink()


def related_pids(pid: int | None = None) -> frozenset[int]:
    """``pid`` (default: this process), its descendants, and its parent's family in its process group.

    The VRAM these hold counts as the caller's own: a reindex's idle
    ``--processes`` siblings, for one, share its parent and process group.
    A process outside the group (another job started from the same shell)
    doesn't count.
    """
    me = os.getpid() if pid is None else pid
    pids = {me}
    try:
        proc = psutil.Process(me)
        pids.update(child.pid for child in proc.children(recursive=True))
        group = os.getpgid(me)
        parent = proc.parent()
        if parent is not None:
            for relative in (parent, *parent.children(recursive=True)):
                with contextlib.suppress(OSError):
                    if os.getpgid(relative.pid) == group:
                        pids.add(relative.pid)
    except (psutil.Error, OSError):
        pass
    return frozenset(pids)


# --- Headroom ----------------------------------------------------------------------


@dataclass(frozen=True)
class GpuProcess:
    """A process using the GPU, from ``nvidia-smi --query-compute-apps``."""

    pid: int
    name: str
    used_mb: int | None = None

    @property
    def is_lm_studio(self) -> bool:
        """True for LM Studio's model server (``llama-server``) or daemon (``llmster``)."""
        base = os.path.basename(self.name).lower()
        return any(name in base for name in LM_STUDIO_PROCESS_NAMES)


@dataclass(frozen=True)
class PowerLimits:
    """The GPU's power limits in watts; None where nvidia-smi doesn't report one."""

    current_w: float | None
    default_w: float | None
    min_w: float | None
    max_w: float | None

    @property
    def at_default(self) -> bool:
        """True when the limit in force is the default."""
        if self.current_w is None or self.default_w is None:
            return False
        return abs(self.current_w - self.default_w) < 0.5

    def suggestion(self, target: float = SUGGESTED_POWER_LIMIT_W) -> float | None:
        """``target`` within the GPU's range, or None when the limit is already that low."""
        if self.current_w is None:
            return None
        value = target
        if self.max_w is not None:
            value = min(value, self.max_w)
        if self.min_w is not None:
            value = max(value, self.min_w)
        return value if value < self.current_w - 0.5 else None


@dataclass(frozen=True)
class Shortfall:
    """One check the machine fails: ``check`` is "vram", "temperature", "ram" or "load"."""

    check: str
    message: str

    @property
    def setting(self) -> str:
        """The setting that sets this check's threshold."""
        return SETTING_FOR_CHECK.get(self.check, "")

    def explain(self) -> str:
        """The message with its setting, like ``... needs 2048 MiB [MULLET_GPU_MIN_FREE_MB]``."""
        return f"{self.message} [{self.setting}]" if self.setting else self.message

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class Headroom:
    """A snapshot of what the machine has to spare.

    VRAM figures are in MiB, as nvidia-smi reports them, for the first GPU.
    A GPU field is None without a working nvidia-smi, and any field is None
    when it couldn't be read.
    """

    gpu_name: str | None = None
    vram_free_mb: int | None = None
    vram_total_mb: int | None = None
    gpu_temp_c: float | None = None
    gpu_util_pct: float | None = None
    gpu_processes: tuple[GpuProcess, ...] = ()
    power: PowerLimits | None = None
    ram_free_mb: int | None = None
    load_1m: float | None = None

    @property
    def has_gpu(self) -> bool:
        """True when nvidia-smi reported a GPU's memory."""
        return self.vram_free_mb is not None or self.vram_total_mb is not None

    @property
    def lm_studio_processes(self) -> tuple[GpuProcess, ...]:
        """LM Studio's processes on the GPU."""
        return tuple(p for p in self.gpu_processes if p.is_lm_studio)

    def vram_free_for(self, own_pids: Collection[int]) -> int | None:
        """Free VRAM plus what ``own_pids`` already hold: what those processes can count on."""
        if self.vram_free_mb is None:
            return None
        return self.vram_free_mb + sum(p.used_mb or 0 for p in self.gpu_processes if p.pid in own_pids)

    def shortfalls(self, limits: Limits, *, own_pids: Collection[int] = ()) -> list[Shortfall]:
        """The checks this snapshot fails; empty when the work can start.

        VRAM held by ``own_pids`` counts as free, so a process never waits on
        its own memory. A field that is None passes.
        """
        out: list[Shortfall] = []
        free = self.vram_free_for(own_pids)
        if free is not None and free < limits.gpu_min_free_mb:
            message = f"GPU has {free} MiB free, needs {limits.gpu_min_free_mb:.0f} MiB"
            lms = sum(p.used_mb or 0 for p in self.lm_studio_processes)
            if lms:
                message += f" (LM Studio holds {lms} MiB)"
            out.append(Shortfall("vram", message))
        if self.gpu_temp_c is not None and self.gpu_temp_c > limits.gpu_max_temp_c:
            out.append(Shortfall("temperature", f"GPU at {self.gpu_temp_c:.0f}°C, over "
                                 f"{limits.gpu_max_temp_c:.0f}°C"))
        if self.ram_free_mb is not None and self.ram_free_mb < limits.ram_min_free_mb:
            out.append(Shortfall("ram", f"{self.ram_free_mb} MiB RAM available, needs "
                                 f"{limits.ram_min_free_mb:.0f} MiB"))
        if self.load_1m is not None and self.load_1m > limits.max_load:
            out.append(Shortfall("load", f"load average {self.load_1m:.1f}, over {limits.max_load:g}"))
        return out


class NoHeadroom(MachineBusy):
    """The machine didn't have room for the work in time.

    ``shortfalls`` are the checks the last snapshot (``headroom``) failed,
    each named with its setting in the message. ``waited`` is how long the
    caller waited, None when it didn't wait.
    """

    def __init__(self, headroom: Headroom, shortfalls: list[Shortfall],
                 waited: float | None = None) -> None:
        self.headroom = headroom
        self.shortfalls = list(shortfalls)
        self.waited = waited
        reasons = "; ".join(s.explain() for s in self.shortfalls) or "no reason given"
        if waited:
            message = f"gave up after {waited:.0f}s waiting for headroom: {reasons}"
        else:
            message = f"not enough headroom: {reasons}"
        super().__init__(message)

    def __reduce__(self):  # keep it picklable across a process pool
        return (type(self), (self.headroom, self.shortfalls, self.waited))


class _Hung(Exception):
    """nvidia-smi didn't answer in time."""


def _nvidia_smi(args: list[str], timeout: float = NVIDIA_SMI_TIMEOUT_S) -> str | None:
    """stdout of ``nvidia-smi <args>``, or None when it is missing or fails.

    Raises ``_Hung`` when it doesn't answer within ``timeout`` seconds, so the
    caller can skip asking it again.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        proc = subprocess.run([exe, *args], capture_output=True, text=True, errors="replace",
                              check=False, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as exc:
        raise _Hung(str(exc)) from None
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def _whole(value: float | None) -> int | None:
    return None if value is None else int(value)


def parse_gpu_csv(text: str | None, *, with_power: bool = True) -> Headroom:
    """The GPU fields of a ``Headroom`` from ``nvidia-smi --query-gpu=`` (csv, no header or units).

    ``text`` answers ``GPU_QUERY``, or ``GPU_QUERY_BASIC`` with
    ``with_power=False``. Reads the first GPU, the one CUDA uses unless told
    otherwise. A line without a readable memory figure is skipped; with
    none, the GPU fields stay None. The current power limit is
    ``power.limit``, or ``enforced.power.limit`` where the first reads
    "[N/A]" (as on some laptops).
    """
    count = 10 if with_power else 5
    for line in (text or "").splitlines():
        parts = line.rsplit(",", count - 1)
        if len(parts) != count:
            continue
        numbers = [_number(p) for p in parts[1:]]
        free, total, temp, util = numbers[:4]
        if free is None and total is None:
            continue
        power = None
        if with_power:
            limit, enforced, default, low, high = numbers[4:]
            current = limit if limit is not None else enforced
            if current is not None or default is not None:
                power = PowerLimits(current_w=current, default_w=default, min_w=low, max_w=high)
        return Headroom(gpu_name=parts[0].strip() or None, vram_free_mb=_whole(free),
                        vram_total_mb=_whole(total), gpu_temp_c=temp, gpu_util_pct=util, power=power)
    return Headroom()


def parse_compute_apps(text: str | None) -> tuple[GpuProcess, ...]:
    """Processes from ``nvidia-smi --query-compute-apps=pid,process_name,used_memory`` (csv, no header or units).

    Lines that don't start with a pid are skipped.
    """
    out: list[GpuProcess] = []
    for line in (text or "").splitlines():
        pid_text, sep, rest = line.partition(",")
        name, sep2, used = rest.rpartition(",")
        if not (sep and sep2):
            continue
        try:
            pid = int(pid_text.strip())
        except ValueError:
            continue
        out.append(GpuProcess(pid, name.strip(), _whole(_number(used))))
    return tuple(out)


def _ram_free_mb() -> int | None:
    try:
        return int(psutil.virtual_memory().available // (1024 * 1024))
    except (OSError, psutil.Error, RuntimeError):
        return None


def _load_1m() -> float | None:
    try:
        return os.getloadavg()[0]
    except (OSError, AttributeError):
        return None


def _gpu(timeout: float) -> Headroom:
    """The GPU part of a snapshot: two nvidia-smi calls at most when it works."""
    try:
        text = _nvidia_smi([f"--query-gpu={GPU_QUERY}", _CSV], timeout)
        if text is not None:
            room = parse_gpu_csv(text)
        else:  # missing, or a driver that rejects a power field
            room = parse_gpu_csv(_nvidia_smi([f"--query-gpu={GPU_QUERY_BASIC}", _CSV], timeout),
                                 with_power=False)
        if not room.has_gpu:
            return room
        apps = parse_compute_apps(_nvidia_smi([f"--query-compute-apps={APPS_QUERY}", _CSV], timeout))
        return replace(room, gpu_processes=apps)
    except _Hung as exc:
        logger.warning("machine: nvidia-smi didn't answer (%s); treating the GPU as unknown", exc)
        return Headroom()


def headroom(*, timeout: float = NVIDIA_SMI_TIMEOUT_S) -> Headroom:
    """Measure the machine now. Never raises: whatever can't be read is None.

    Each nvidia-smi call gets ``timeout`` seconds; one that hangs ends the
    GPU readings there.
    """
    return replace(_gpu(timeout), ram_free_mb=_ram_free_mb(), load_1m=_load_1m())


def wait_for_headroom(
    need_vram_mb: float | None = None,
    need_ram_mb: float | None = None,
    max_temp_c: float | None = None,
    max_load: float | None = None,
    *,
    timeout: float | None = None,
    poll: float = 5.0,
    on_wait: Callable[[Headroom, list[Shortfall]], None] | None = None,
    own_pids: Collection[int] | None = None,
    no_gpu_wait: float = NO_GPU_MAX_WAIT_S,
    sample: Callable[[], Headroom] | None = None,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> Headroom:
    """Wait until the machine can take the work; return the last snapshot.

    A threshold left as None comes from ``limits()``. VRAM held by
    ``own_pids`` (default ``related_pids()``) counts as free.
    ``on_wait(headroom, shortfalls)`` is called before the first pause and
    again whenever the set of failing checks changes.

    With a GPU this waits ``timeout`` seconds at most (``max_wait_s()`` when
    None, ``math.inf`` for as long as it takes; 0 checks once), then raises
    ``NoHeadroom``. Without one there are no GPU checks, and RAM and load
    are waited on for ``no_gpu_wait`` seconds at most (or ``timeout``, if
    shorter) before it returns anyway: such a machine never sits idle.
    ``sample``, ``clock`` and ``sleep`` replace ``headroom()``,
    ``time.monotonic`` and ``time.sleep``.
    """
    defaults = limits()
    wanted = Limits(
        gpu_min_free_mb=defaults.gpu_min_free_mb if need_vram_mb is None else need_vram_mb,
        ram_min_free_mb=defaults.ram_min_free_mb if need_ram_mb is None else need_ram_mb,
        gpu_max_temp_c=defaults.gpu_max_temp_c if max_temp_c is None else max_temp_c,
        max_load=defaults.max_load if max_load is None else max_load,
    )
    limit = max_wait_s() if timeout is None else timeout
    mine = related_pids() if own_pids is None else own_pids
    measure = sample or headroom
    clock = clock or _clock
    sleep = sleep or _sleep
    started = clock()
    reported: tuple[str, ...] | None = None
    while True:
        room = measure()
        short = room.shortfalls(wanted, own_pids=mine)
        if not short:
            return room
        waited = clock() - started
        deadline = limit if room.has_gpu else min(limit, no_gpu_wait)
        if waited >= deadline:
            if room.has_gpu:
                raise NoHeadroom(room, short, waited)
            logger.info("machine: no GPU, going ahead anyway: %s", "; ".join(map(str, short)))
            return room
        checks = tuple(s.check for s in short)
        if on_wait is not None and checks != reported:
            reported = checks
            on_wait(room, short)
        sleep(max(0.0, min(poll, deadline - waited)))
