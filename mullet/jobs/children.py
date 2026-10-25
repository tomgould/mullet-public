"""Track the process groups a runner starts, so none outlive it.

Executors and sandboxed checks run in their own process groups so a timeout
can kill a whole tree. That also means killing the runner doesn't kill them.
Each group is recorded here, in memory and in a file in the job directory,
so `kill_all()` can stop them on a stop request and `kill_stale()` can clean
up after a runner that crashed. A recorded process is only killed if its
start time still matches, so a recycled PID is never hit.
"""

from __future__ import annotations

import json
import os
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from subprocess import Popen

import psutil

_lock = threading.Lock()
_live: dict[int, float] = {}  # pgid -> leader start time
_record_path: Path | None = None
_stopping = threading.Event()


class Stopped(RuntimeError):
    """A stop was requested, so no new child process may start."""


def configure(record_path: Path | None) -> None:
    """Persist tracked groups to `record_path` (None stops persisting) and clear any stop."""
    global _record_path
    with _lock:
        _record_path = record_path
        _stopping.clear()
        _save()


def request_stop() -> None:
    """Refuse new children from now on and kill the running ones."""
    _stopping.set()
    kill_all()


def check_not_stopping() -> None:
    """Raise `Stopped` if a stop was requested; call before starting a child."""
    if _stopping.is_set():
        raise Stopped("a stop was requested")


def _save() -> None:
    if _record_path is None:
        return
    tmp = _record_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({str(k): v for k, v in _live.items()}))
    os.replace(tmp, _record_path)


def _start_time(pid: int) -> float | None:
    try:
        return psutil.Process(pid).create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


@contextmanager
def track(proc: Popen) -> Iterator[None]:
    """Record `proc`'s process group for as long as the block runs."""
    started = _start_time(proc.pid)
    with _lock:
        if started is not None:
            _live[proc.pid] = started
            _save()
    try:
        yield
    finally:
        with _lock:
            _live.pop(proc.pid, None)
            _save()


def _kill_group(pgid: int, started: float) -> bool:
    if _start_time(pgid) != started:
        return False
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            return True
        try:
            psutil.Process(pgid).wait(timeout=5)
            return True
        except psutil.TimeoutExpired:
            continue
        except psutil.NoSuchProcess:
            return True
    return True


def kill_all() -> int:
    """Kill every group this process is tracking. Returns how many were killed."""
    with _lock:
        groups = dict(_live)
    return sum(_kill_group(pgid, started) for pgid, started in groups.items())


def kill_in_dir(directory: Path, since: float) -> int:
    """Kill this user's processes started after `since` whose working directory is in `directory`.

    An executor can leave background processes behind (a dev server, a
    watcher) that would keep changing the work tree while the gates read it.
    `since` keeps it from touching anything older, such as a shell the user
    opened in the clone to look around.
    """
    root = str(Path(directory).resolve())
    me = os.getpid()
    uid = os.getuid()
    # psutil derives start times from a boot time recorded in whole seconds,
    # so they can be up to a second off wall-clock time.
    cutoff = since - 1.5
    killed = 0
    for proc in psutil.process_iter(["pid", "uids", "create_time"]):
        try:
            if (proc.pid == me or proc.info["uids"] is None or proc.info["uids"].real != uid
                    or (proc.info["create_time"] or 0) < cutoff):
                continue
            cwd = proc.cwd()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        if cwd == root or cwd.startswith(root + os.sep):
            try:
                proc.kill()
                killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    return killed


def kill_stale(record_path: Path) -> int:
    """Kill groups left behind by a previous runner for this job."""
    try:
        groups = json.loads(record_path.read_text())
    except (OSError, ValueError):
        return 0
    killed = sum(_kill_group(int(pgid), float(started)) for pgid, started in groups.items())
    record_path.unlink(missing_ok=True)
    return killed
