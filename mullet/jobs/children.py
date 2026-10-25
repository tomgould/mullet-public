"""Track the process groups a runner starts, so none outlive it.

Executors and sandboxed checks run in their own process groups so a timeout
can kill a whole tree. That also means killing the runner doesn't kill them.
Each group is recorded here, in memory and in a file in the job directory,
so `kill_all()` can stop them on a stop request and `kill_stale()` can clean
up after a runner that crashed. A recorded process is only killed if its
start time still matches, so a recycled PID is never hit.

Each owner has its own `Tracker`. The module-level functions act on the
tracker bound to the calling thread with `use()`, or on the process default
when none is bound, so a runner that never binds one keeps working as before.
"""

from __future__ import annotations

import json
import os
import signal
import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from pathlib import Path
from subprocess import Popen

import psutil

class Stopped(RuntimeError):
    """A stop was requested, so no new child process may start."""


def _start_time(pid: int) -> float | None:
    try:
        return psutil.Process(pid).create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


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


class Tracker:
    """The process groups one owner (a runner, a delegation) started, and its stop flag.

    A stop on one tracker kills only the groups recorded in it, so two runners
    in the same process never stop each other's children.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._live: dict[int, float] = {}  # pgid -> leader start time
        self._record_path: Path | None = None
        self._stopping = threading.Event()

    def configure(self, record_path: Path | None) -> None:
        """Persist tracked groups to `record_path` (None stops persisting) and clear any stop."""
        with self._lock:
            self._record_path = record_path
            self._stopping.clear()
            self._save()

    def request_stop(self) -> None:
        """Refuse new children from now on and kill the running ones."""
        self._stopping.set()
        self.kill_all()

    def check_not_stopping(self) -> None:
        """Raise `Stopped` if a stop was requested; call before starting a child."""
        if self._stopping.is_set():
            raise Stopped("a stop was requested")

    def _save(self) -> None:
        if self._record_path is None:
            return
        tmp = self._record_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({str(k): v for k, v in self._live.items()}))
        os.replace(tmp, self._record_path)

    @contextmanager
    def track(self, proc: Popen) -> Iterator[None]:
        """Record `proc`'s process group for as long as the block runs."""
        started = _start_time(proc.pid)
        with self._lock:
            if started is not None:
                self._live[proc.pid] = started
                self._save()
        try:
            yield
        finally:
            with self._lock:
                self._live.pop(proc.pid, None)
                self._save()

    def kill_all(self) -> int:
        """Kill every group this tracker holds. Returns how many were killed."""
        with self._lock:
            groups = dict(self._live)
        return sum(_kill_group(pgid, started) for pgid, started in groups.items())


_default = Tracker()
_current: ContextVar[Tracker | None] = ContextVar("mullet_children_tracker", default=None)


def current() -> Tracker:
    """The tracker the module-level functions use here: the bound one, else the process default."""
    return _current.get() or _default


@contextmanager
def use(tracker: Tracker) -> Iterator[Tracker]:
    """Make `tracker` the one the module-level functions use in this thread for the block.

    A new thread starts with the process default, so an owner that runs its
    work in a thread binds its tracker inside that thread.
    """
    token = _current.set(tracker)
    try:
        yield tracker
    finally:
        _current.reset(token)


def configure(record_path: Path | None) -> None:
    """Persist the current tracker's groups to `record_path` (None stops) and clear its stop."""
    current().configure(record_path)


def request_stop() -> None:
    """Refuse new children of the current tracker and kill its running ones."""
    current().request_stop()


def check_not_stopping() -> None:
    """Raise `Stopped` if the current tracker was asked to stop; call before starting a child."""
    current().check_not_stopping()


def track(proc: Popen) -> AbstractContextManager[None]:
    """Record `proc`'s process group in the current tracker for as long as the block runs."""
    return current().track(proc)


def kill_all() -> int:
    """Kill every group the current tracker holds. Returns how many were killed."""
    return current().kill_all()


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
