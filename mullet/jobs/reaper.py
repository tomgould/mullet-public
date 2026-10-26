"""Find and stop idle or orphaned Mullet work that is hogging the machine.

This module holds the safety core of `mullet reap`. The reaper may only stop
work it can attribute positively to Mullet, by command line and parentage,
and only once that work has been idle for longer than its kind's limit.

- `snapshot()` reads the current user's processes once. A field psutil won't
  let us read is recorded as None ("unknown"), never guessed.
- `classify_parent()` says whether a process is orphaned. Unknown is never
  treated as orphaned.
- `SightingsStore` remembers when each candidate was first seen, keyed by
  kind, pid and start time, so a recycled pid starts its idle time afresh.
  It is read during a pass and written once, after every finder has run.
- `find_xdist_orphans()`, `find_mcp_orphans()` and `find_opencode_strays()`
  match exact command lines only and return every match, with `due` set on
  those past their idle limit and grace period.
- `stop_process()` checks the pid still names the same process just before
  signalling, then sends SIGTERM to it and its own children, waits, and
  sends SIGKILL to whatever is left.

The clock and sleep are injected everywhere so tests never wait or kill.
"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import psutil

from mullet.core.env import local_llm_host
from mullet.core.paths import state_dir
from mullet.jobs import children
from mullet.jobs.ledger import Ledger, valid_job_id
from mullet.jobs.paths import job_dir, jobs_home
from mullet.jobs.types import JobStatus

KIND_XDIST = "xdist"
KIND_OPENCODE = "opencode"
KIND_MCP = "mcp"
KIND_LMSTUDIO = "lmstudio"
KIND_TMP = "tmp"

# kind -> (env var, default idle limit in seconds)
LIMIT_ENV: dict[str, tuple[str, float]] = {
    KIND_XDIST: ("MULLET_REAP_XDIST_S", 120.0),
    KIND_OPENCODE: ("MULLET_REAP_OPENCODE_S", 120.0),
    KIND_MCP: ("MULLET_REAP_MCP_S", 120.0),
    KIND_LMSTUDIO: ("MULLET_REAP_LMSTUDIO_S", 600.0),
    KIND_TMP: ("MULLET_REAP_TMP_S", 3600.0),
}

# How long a stopped process gets to exit after SIGTERM before SIGKILL.
TERM_GRACE_S = 10.0
_POLL_S = 0.25

# Command lines that positively name init or a user's service manager.
_INIT_ARGV0 = frozenset({"/sbin/init", "/lib/systemd/systemd", "/usr/lib/systemd/systemd"})

_PSUTIL_ERRORS = (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError)

ParentState = Literal["orphan", "parented", "unknown"]


# ---------------------------------------------------------------- snapshot


@dataclass(frozen=True)
class ProcInfo:
    """One process as read at snapshot time.

    `ppid` and `cmdline` are None when they could not be read; callers must
    treat None as unknown, not as a value.
    """

    pid: int
    ppid: int | None
    cmdline: tuple[str, ...] | None
    create_time: float
    uid: int


def _read(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except _PSUTIL_ERRORS:
        return None


def snapshot(
    process_iter: Callable[[], Iterable[Any]] = psutil.process_iter,
    *,
    uid: int | None = None,
) -> list[ProcInfo]:
    """Read the current user's processes into frozen `ProcInfo` records.

    A process whose owner or start time can't be read is left out, since it
    could never be attributed or checked against a recycled pid. A process
    whose ppid or command line can't be read is kept, with that field None.
    """
    me = os.getuid() if uid is None else uid
    out: list[ProcInfo] = []
    for proc in process_iter():
        uids = _read(proc.uids)
        if uids is None or getattr(uids, "real", None) != me:
            continue
        created = _read(proc.create_time)
        if not isinstance(created, (int, float)):
            continue
        ppid = _read(proc.ppid)
        cmdline = _read(proc.cmdline)
        out.append(ProcInfo(
            pid=int(proc.pid),
            ppid=int(ppid) if isinstance(ppid, int) else None,
            cmdline=tuple(cmdline) if cmdline is not None else None,
            create_time=float(created),
            uid=me,
        ))
    return out


def index_by_pid(procs: Iterable[ProcInfo]) -> dict[int, ProcInfo]:
    """Map pid to `ProcInfo` for `classify_parent()`."""
    return {p.pid: p for p in procs}


# ---------------------------------------------------------------- parentage


def is_init_cmdline(cmdline: tuple[str, ...], pid: int) -> bool:
    """True when `cmdline` positively names init or `systemd --user`."""
    if not cmdline:
        return False
    argv0 = cmdline[0]
    base = os.path.basename(argv0)
    if base == "systemd" and "--user" in cmdline[1:]:
        return True
    if argv0 in _INIT_ARGV0 and "--user" not in cmdline[1:]:
        return True
    return pid == 1 and base in ("systemd", "init")


def classify_parent(
    proc: ProcInfo,
    by_pid: Mapping[int, ProcInfo],
    *,
    pid_exists: Callable[[int], bool] = psutil.pid_exists,
) -> ParentState:
    """Say whether `proc` is orphaned, has a live parent, or can't be told.

    Orphan only when the ppid is 1, the parent has exited, or the parent is
    positively init or `systemd --user`. The snapshot holds only this user's
    processes, so a parent missing from it is checked with `pid_exists`: one
    that still exists belongs to someone else (or is new) and gives unknown.
    An unreadable ppid or parent command line also gives unknown.
    """
    ppid = proc.ppid
    if ppid is None or ppid < 0:
        return "unknown"
    if ppid == 1:
        return "orphan"
    if ppid == 0:
        return "unknown"
    parent = by_pid.get(ppid)
    if parent is None:
        try:
            exists = pid_exists(ppid)
        except Exception:
            return "unknown"
        return "unknown" if exists else "orphan"
    if not parent.cmdline:
        return "unknown"
    return "orphan" if is_init_cmdline(parent.cmdline, parent.pid) else "parented"


# ---------------------------------------------------------------- candidates


@dataclass(frozen=True)
class Candidate:
    """Something the reaper may stop once it has been idle for `idle_limit` seconds.

    Processes carry `pid` and the `create_time` recorded when they were
    found; directories carry `path` instead.
    """

    kind: str
    reason: str
    idle_limit: float
    pid: int | None = None
    create_time: float | None = None
    path: str | None = None
    identifier: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)
    idle_s: float = field(default=0.0, compare=False)
    due: bool = field(default=False, compare=False)

    @property
    def key(self) -> str:
        """The sightings key: kind plus pid plus start time, path, or identifier."""
        return sighting_key(self.kind, pid=self.pid, create_time=self.create_time,
                            path=self.path, identifier=self.identifier)

    def as_record(self) -> dict[str, Any]:
        """A JSON-ready description for reap.log and `--json`."""
        rec: dict[str, Any] = {"kind": self.kind, "reason": self.reason,
                               "idle_limit_s": self.idle_limit, "idle_s": self.idle_s}
        if self.pid is not None:
            rec["pid"] = self.pid
            rec["create_time"] = self.create_time
        if self.path is not None:
            rec["path"] = self.path
        if self.identifier is not None:
            rec["identifier"] = self.identifier
        rec.update(self.extra)
        return rec


def sighting_key(kind: str, *, pid: int | None = None, create_time: float | None = None,
                 path: str | None = None, identifier: str | None = None) -> str:
    """Build a sightings key for a process, a path, or a non-process identifier."""
    if path is not None:
        return json.dumps([kind, "path", path])
    if identifier is not None:
        return json.dumps([kind, "id", identifier])
    if pid is None or create_time is None:
        raise ValueError("a process key needs both pid and create_time")
    return json.dumps([kind, "pid", int(pid), repr(float(create_time))])


# ---------------------------------------------------------------- sightings


def sightings_path() -> Path:
    """Where first-seen times are kept between passes."""
    return state_dir() / "reap-sightings.json"


class SightingsStore:
    """First-seen times for candidates, shared by every finder.

    Read with `first_seen()` during a pass; nothing is written until
    `commit()`, which is called once per pass after every finder has run.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else sightings_path()
        self._seen: dict[str, float] = self._load()

    def _load(self) -> dict[str, float]:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        out: dict[str, float] = {}
        for k, v in data.items():
            if isinstance(k, str) and isinstance(v, (int, float)) and math.isfinite(v):
                out[k] = float(v)
        return out

    def first_seen(self, key: str, now: float) -> float:
        """When `key` was first seen, or `now` if this is its first sighting. Read-only."""
        return self._seen.get(key, now)

    def idle_for(self, key: str, now: float) -> float:
        """Seconds since `key` was first seen (0 on a first sighting)."""
        return max(0.0, now - self.first_seen(key, now))

    def is_due(self, candidate: Candidate, now: float) -> bool:
        """True when `candidate` has been seen for at least its idle limit."""
        return self.idle_for(candidate.key, now) >= candidate.idle_limit

    def commit(self, seen_keys: Iterable[str], now: float) -> None:
        """Keep first-seen times for keys seen this pass, add new ones at `now`, drop the rest."""
        self._seen = {k: self._seen.get(k, now) for k in seen_keys}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._seen, sort_keys=True))
        os.replace(tmp, self.path)


# ---------------------------------------------------------------- limits


def idle_limit(kind: str, env: Mapping[str, str] | None = None) -> float:
    """The idle limit in seconds for `kind`, from its env var; a bad value gives the default."""
    name, default = LIMIT_ENV[kind]
    raw = (os.environ if env is None else env).get(name)
    if raw is None:
        return default
    try:
        value = float(raw.strip())
    except (ValueError, AttributeError):
        return default
    if not math.isfinite(value) or value < 0:
        return default
    return value


def idle_limits(env: Mapping[str, str] | None = None) -> dict[str, float]:
    """Idle limits for every kind."""
    return {kind: idle_limit(kind, env) for kind in LIMIT_ENV}


# ---------------------------------------------------------------- finders
#
# Each finder looks over one snapshot and returns every process it can
# attribute positively, due or not, so the pass can commit all their keys to
# the shared sightings store once, after every finder has run. Only those
# with `due` set may be stopped: seen for at least the idle limit, and older
# than the grace period (which defaults to the idle limit).


def seen_keys(candidates: Iterable[Candidate]) -> list[str]:
    """The sightings keys of every candidate found this pass, for `SightingsStore.commit()`."""
    return [c.key for c in candidates]


def due_only(candidates: Iterable[Candidate]) -> list[Candidate]:
    """The candidates that have reached their idle limit and grace period."""
    return [c for c in candidates if c.due]


def _sighted(kind: str, proc: ProcInfo, reason: str, sightings: SightingsStore, now: float,
             limit: float, grace: float, extra: Mapping[str, Any] | None = None) -> Candidate:
    key = sighting_key(kind, pid=proc.pid, create_time=proc.create_time)
    idle = sightings.idle_for(key, now)
    old_enough = now - proc.create_time >= grace
    return Candidate(kind=kind, reason=reason, idle_limit=limit, pid=proc.pid,
                     create_time=proc.create_time, extra=dict(extra or {}),
                     idle_s=idle, due=idle >= limit and old_enough)


_PYTHON_RE = re.compile(r"python(\d+(\.\d+)*)?")

# execnet's popen gateway starts each pytest-xdist worker as
# `<python> [-u] [-B] -c "import sys;exec(eval(sys.stdin.readline()))"`
# (execnet.gateway_io.popen_args). Nothing else is accepted.
XDIST_BOOTSTRAP = "import sys;exec(eval(sys.stdin.readline()))"
_XDIST_FLAGS = ("-u", "-B")


def _is_python(argv0: str) -> bool:
    return bool(_PYTHON_RE.fullmatch(os.path.basename(argv0)))


def is_xdist_worker(cmdline: tuple[str, ...] | None) -> bool:
    """True only for the exact execnet popen bootstrap command line pytest-xdist workers use."""
    if not cmdline or len(cmdline) < 3 or not _is_python(cmdline[0]):
        return False
    if tuple(cmdline[-2:]) != ("-c", XDIST_BOOTSTRAP):
        return False
    flags = cmdline[1:-2]
    return len(set(flags)) == len(flags) and all(f in _XDIST_FLAGS for f in flags)


def is_mcp_server(cmdline: tuple[str, ...] | None) -> bool:
    """True for `python -m mullet mcp ...`, `mullet mcp ...` or `python .../mullet mcp ...`.

    A console script with a #! line shows the interpreter first and the
    script second, so that form is matched too. The word after `mullet`
    must be exactly `mcp`.
    """
    if not cmdline:
        return False
    argv = tuple(cmdline)
    if os.path.basename(argv[0]) == "mullet":
        return len(argv) >= 2 and argv[1] == "mcp"
    if not _is_python(argv[0]):
        return False
    if argv[1:4] == ("-m", "mullet", "mcp"):
        return True
    return len(argv) >= 3 and os.path.basename(argv[1]) == "mullet" and argv[2] == "mcp"


def opencode_run_dir(cmdline: tuple[str, ...] | None) -> str | None:
    """The `--dir` of an `opencode run` command line, or None if it isn't one or names no single dir."""
    if not cmdline or len(cmdline) < 2:
        return None
    if os.path.basename(cmdline[0]) != "opencode" or cmdline[1] != "run":
        return None
    dirs: list[str] = []
    rest = list(cmdline[2:])
    for i, arg in enumerate(rest):
        if arg == "--dir" and i + 1 < len(rest):
            dirs.append(rest[i + 1])
        elif arg.startswith("--dir="):
            dirs.append(arg[len("--dir="):])
    if len(set(dirs)) != 1:
        return None
    return dirs[0]


def job_id_for_dir(work: str) -> str | None:
    """The job id whose directory under `jobs_home()` holds `work`, or None if it lies elsewhere."""
    if not os.path.isabs(work):
        return None
    try:
        rel = Path(work).resolve().relative_to(jobs_home())
    except (ValueError, OSError, RuntimeError):
        return None
    if not rel.parts:
        return None
    job_id = rel.parts[0]
    return job_id if valid_job_id(job_id) else None


RunnerState = Literal["live", "dead", "unknown"]


def runner_state(job_id: str, process_factory: Callable[[int], Any] = psutil.Process) -> RunnerState:
    """Whether the runner named in the job's runner.pid is still the live process that wrote it.

    No pid file, a vanished pid or a different start time give `dead`; an
    unreadable file or process gives `unknown`. The start is in clock ticks
    since boot; a bare number is a create_time an older runner wrote.
    """
    try:
        text = (job_dir(job_id) / "runner.pid").read_text()
    except FileNotFoundError:
        return "dead"
    except (OSError, ValueError):
        return "unknown"
    parts = text.split()
    if len(parts) != 2:
        return "unknown"
    if "@" in parts[1]:  # <start ticks>@<boot id>, which a clock step doesn't move
        try:
            pid = int(parts[0])
        except ValueError:
            return "unknown"
        if children.same_process(pid, parts[1]):
            return "live"
        unreadable = children.process_identity(pid) is None and os.path.exists(f"/proc/{pid}")
        return "unknown" if unreadable else "dead"
    try:
        pid, started = int(parts[0]), float(parts[1])
    except ValueError:
        return "unknown"
    try:
        proc = process_factory(pid)
        if not proc.is_running():
            return "dead"
        created = float(proc.create_time())
    except psutil.NoSuchProcess:
        return "dead"
    except (psutil.Error, OSError):
        return "unknown"
    return "live" if abs(created - started) <= 0.01 else "dead"


def _limits(kind: str, limit: float | None, grace: float | None,
            env: Mapping[str, str] | None) -> tuple[float, float]:
    lim = idle_limit(kind, env) if limit is None else limit
    return lim, lim if grace is None else grace


def find_xdist_orphans(
    procs: Iterable[ProcInfo],
    sightings: SightingsStore,
    now: float,
    *,
    limit: float | None = None,
    grace: float | None = None,
    env: Mapping[str, str] | None = None,
    pid_exists: Callable[[int], bool] = psutil.pid_exists,
) -> list[Candidate]:
    """Every pytest-xdist worker in `procs` whose parent has exited, due or not.

    Matched only by the exact execnet bootstrap command line, and orphaned
    only as `classify_parent()` says; unknown parentage is left alone.
    """
    procs = list(procs)
    by_pid = index_by_pid(procs)
    lim, gr = _limits(KIND_XDIST, limit, grace, env)
    out: list[Candidate] = []
    for proc in procs:
        if not is_xdist_worker(proc.cmdline):
            continue
        if classify_parent(proc, by_pid, pid_exists=pid_exists) != "orphan":
            continue
        out.append(_sighted(KIND_XDIST, proc, "pytest-xdist worker whose parent has exited",
                            sightings, now, lim, gr, {"ppid": proc.ppid}))
    return out


def find_mcp_orphans(
    procs: Iterable[ProcInfo],
    sightings: SightingsStore,
    now: float,
    *,
    limit: float | None = None,
    grace: float | None = None,
    env: Mapping[str, str] | None = None,
    pid_exists: Callable[[int], bool] = psutil.pid_exists,
) -> list[Candidate]:
    """Every `mullet mcp` server in `procs` whose parent has exited, due or not."""
    procs = list(procs)
    by_pid = index_by_pid(procs)
    lim, gr = _limits(KIND_MCP, limit, grace, env)
    out: list[Candidate] = []
    for proc in procs:
        if not is_mcp_server(proc.cmdline):
            continue
        if classify_parent(proc, by_pid, pid_exists=pid_exists) != "orphan":
            continue
        out.append(_sighted(KIND_MCP, proc, "mullet mcp server whose parent has exited",
                            sightings, now, lim, gr, {"ppid": proc.ppid}))
    return out


def find_opencode_strays(
    procs: Iterable[ProcInfo],
    sightings: SightingsStore,
    now: float,
    *,
    ledger: Ledger,
    limit: float | None = None,
    grace: float | None = None,
    env: Mapping[str, str] | None = None,
    process_factory: Callable[[int], Any] = psutil.Process,
) -> list[Candidate]:
    """Every `opencode run` in `procs` started for a Mullet job that is no longer running, due or not.

    The job comes from the `--dir` argument, which must lie inside
    `jobs_home()/<job_id>` for a valid job id the ledger knows. The job is
    running when the ledger says RUNNING and its runner.pid names a live
    process with the recorded start time. Anything that can't be attributed
    or read (no `--dir`, a dir elsewhere, an unknown job, an unreadable
    runner) is left alone.
    """
    lim, gr = _limits(KIND_OPENCODE, limit, grace, env)
    out: list[Candidate] = []
    for proc in procs:
        work = opencode_run_dir(proc.cmdline)
        if work is None:
            continue
        job_id = job_id_for_dir(work)
        if job_id is None:
            continue
        try:
            status = JobStatus(ledger.get_job(job_id).status)
        except Exception:
            continue
        if status is JobStatus.RUNNING:
            if runner_state(job_id, process_factory) != "dead":
                continue
            reason = f"opencode run for job {job_id}, whose runner has exited"
        else:
            reason = f"opencode run for job {job_id}, which is {status.value}"
        out.append(_sighted(KIND_OPENCODE, proc, reason, sightings, now, lim, gr,
                            {"job_id": job_id, "dir": work}))
    return out


# ---------------------------------------------------------------- lm studio


def lmstudio_base_url(env: Mapping[str, str] | None = None) -> str:
    """The LM Studio server root, from env or `local_llm_host()`."""
    e = os.environ if env is None else env
    override = (e.get("MULLET_REAP_LMSTUDIO_URL") or "").strip().rstrip("/")
    if override:
        return override
    return local_llm_host()


def lmstudio_unload_url(env: Mapping[str, str] | None = None,
                        base_url: str | None = None) -> str | None:
    """The configured unload endpoint, or None when no endpoint is confirmed safe."""
    e = os.environ if env is None else env
    explicit = (e.get("MULLET_REAP_LMSTUDIO_UNLOAD_URL") or "").strip()
    if explicit:
        return explicit
    # LM Studio documents a POST unload endpoint in newer REST versions, but
    # the exact path has not been confirmed here, so the default is unset.
    return None


def _model_last_request_signal(model: Mapping[str, Any]) -> float | None:
    """A positive timestamp for the model's last request, or None if none exists."""
    for key in ("last_used", "last_inference_time", "last_request_at",
                "lastRequestAt", "last_request_time", "lastInferenceTime"):
        value = model.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return None


def find_idle_lmstudio_models(
    sightings: SightingsStore,
    now: float,
    *,
    base_url: str | None = None,
    http_get: Callable[[str], Any] | None = None,
    limit: float | None = None,
    env: Mapping[str, str] | None = None,
) -> list[Candidate]:
    """Every loaded LM Studio model that is idle, plus loaded models reported when no signal exists.

    Models are fetched from GET {base_url}/api/v0/models. A model is only a
    candidate when the response positively shows it is loaded (type == llm,
    state == loaded). It is marked due only when it has been tracked for the
    idle limit and a last-request signal in the response shows no request in
    that window. With no such signal the model is reported but never acted on.
    The LM Studio process itself is never examined or signalled.
    """
    lim = idle_limit(KIND_LMSTUDIO, env) if limit is None else limit
    base = (base_url or lmstudio_base_url(env)).rstrip("/")
    unload = lmstudio_unload_url(env, base)

    if http_get is None:
        return []

    try:
        data = http_get(f"{base}/api/v0/models")
    except Exception:
        return []

    if not isinstance(data, dict):
        return []

    models = data.get("data", [])
    if not isinstance(models, list):
        return []

    out: list[Candidate] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        if model.get("type") != "llm" or model.get("state") != "loaded":
            continue
        model_id = model.get("id") or model.get("modelKey")
        if not model_id:
            continue

        last_request = _model_last_request_signal(model)
        key = sighting_key(KIND_LMSTUDIO, identifier=str(model_id))
        idle_s = sightings.idle_for(key, now)

        if last_request is None:
            reason = "LM Studio model loaded (no request signal)"
            due = False
        else:
            no_request_for = now - last_request
            reason = "LM Studio model with no recent request"
            due = idle_s >= lim and no_request_for >= lim

        out.append(Candidate(
            kind=KIND_LMSTUDIO,
            reason=reason,
            idle_limit=lim,
            identifier=str(model_id),
            extra={
                "model_id": str(model_id),
                "base_url": base,
                "unload_url": unload,
                "last_request_at": last_request,
            },
            idle_s=idle_s,
            due=due,
        ))
    return out


def find_stale_tmp_dirs(
    tmp_root: Path = Path("/tmp"),
    *,
    now: float,
    open_paths: Callable[[], Iterable[tuple[str, list[str]]]],
    limit: float | None = None,
    env: Mapping[str, str] | None = None,
) -> list[Candidate]:
    """Find directories directly under tmp_root whose name starts with `mullet-` that are:
    - real directories (not symlinks)
    - owned by the current user
    - older than MULLET_REAP_TMP_S (default 3600 seconds)
    - not open by any running process
    
    Args:
        tmp_root: Root directory to search (default /tmp)
        now: Current timestamp
        open_paths: Function that returns (cwd, [open_file_paths]) for each process
        limit: Override idle limit in seconds (default from env or 3600)
        env: Environment variables for limit lookup
        
    Returns:
        List of Candidate objects for stale directories
    """
    lim = idle_limit(KIND_TMP, env) if limit is None else limit
    out: list[Candidate] = []
    
    try:
        entries = tmp_root.iterdir()
    except (OSError, PermissionError):
        return []
        
    # Get current user ID for ownership check
    uid = os.getuid()
    
    # Get all open paths once
    open_paths_map: dict[str, set[str]] = {}
    for cwd, paths in open_paths():
        for path in paths:
            open_paths_map[path] = open_paths_map.get(path, set())
            open_paths_map[path].add(cwd)
        if cwd:
            open_paths_map[cwd] = open_paths_map.get(cwd, set())
            open_paths_map[cwd].add(cwd)
    
    for entry in entries:
        # Skip if not a directory or is a symlink
        try:
            if not entry.is_dir() or entry.is_symlink():
                continue
            # Check if name starts with 'mullet-'
            if not entry.name.startswith("mullet-"):
                continue
            # Check ownership
            if entry.stat().st_uid != uid:
                continue
            # Check age
            mtime = entry.stat().st_mtime
            if now - mtime < lim:
                continue
                
            # Check if any process has this directory or files inside it open
            is_open = False
            entry_str = str(entry)
            for open_path, cwds in open_paths_map.items():
                # Check if the directory itself is a cwd
                if entry_str in cwds:
                    is_open = True
                    break
                # Check if any file inside the directory is open
                try:
                    if Path(open_path).resolve().is_relative_to(entry.resolve()):
                        is_open = True
                        break
                except (OSError, ValueError):
                    # Can't resolve path, skip this check
                    continue
                    
            if is_open:
                continue
                
            # Directory qualifies as stale
            out.append(Candidate(
                kind=KIND_TMP,
                reason="stale tmp directory",
                idle_limit=lim,
                path=str(entry),
                idle_s=now - mtime,
                due=True
            ))
            
        except (OSError, PermissionError):
            # Skip entries we can't access
            continue
            
    return out


def remove_dir(
    path: str | Path,
    *,
    tmp_root: Path = Path("/tmp"),
    now: float,
    open_paths: Callable[[], Iterable[tuple[str, list[str]]]],
    limit: float | None = None,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Remove a directory after re-checking all conditions.
    
    Args:
        path: Path to directory to remove
        tmp_root: Root directory that paths must be under (default /tmp)
        now: Current timestamp
        open_paths: Function that returns (cwd, [open_file_paths]) for each process
        limit: Override idle limit in seconds (default from env or 3600)
        env: Environment variables for limit lookup
        
    Returns:
        True if directory was removed, False otherwise
    """
    path_obj = Path(path)
    lim = idle_limit(KIND_TMP, env) if limit is None else limit
    
    # Re-check all conditions before removal
    
    # Check if path exists
    if not path_obj.exists():
        return False
    
    # Check if it's a directory
    if not path_obj.is_dir():
        return False
        
    # Check if it's a symlink
    if path_obj.is_symlink():
        return False
        
    # Check if it's under tmp_root
    try:
        if path_obj.parent != tmp_root:
            return False
    except (OSError, ValueError):
        return False
    
    # Check name condition
    if not path_obj.name.startswith("mullet-"):
        return False
    
    # Check ownership
    uid = os.getuid()
    try:
        if path_obj.stat().st_uid != uid:
            return False
    except (OSError, PermissionError):
        return False
    
    # Check age
    try:
        mtime = path_obj.stat().st_mtime
        if now - mtime < lim:
            return False
    except (OSError, PermissionError):
        return False
    
    # Check if any process has this directory or files inside it open
    try:
        open_paths_list = list(open_paths())
        path_str = str(path_obj)
        
        # Build a map of open paths
        open_paths_map: dict[str, set[str]] = {}
        for cwd, paths in open_paths_list:
            for p in paths:
                open_paths_map[p] = open_paths_map.get(p, set())
                open_paths_map[p].add(cwd)
            if cwd:
                open_paths_map[cwd] = open_paths_map.get(cwd, set())
                open_paths_map[cwd].add(cwd)
                
        # Check if directory is in use
        is_open = False
        for open_path, cwds in open_paths_map.items():
            # Check if the directory itself is a cwd
            if path_str in cwds:
                is_open = True
                break
            # Check if any file inside the directory is open
            try:
                if Path(open_path).resolve().is_relative_to(path_obj.resolve()):
                    is_open = True
                    break
            except (OSError, ValueError):
                # Can't resolve path, skip this check
                continue
                
        if is_open:
            return False
            
    except (OSError, PermissionError):
        return False
    
    # All checks passed, remove the directory
    try:
        import shutil
        shutil.rmtree(path_obj)
        return True
    except (OSError, PermissionError):
        return False


# ---------------------------------------------------------------- stopping


@dataclass
class StopResult:
    """What `stop_process()` did, faithfully: `action` is `stopped`,
    `would_stop` or `skipped`, and a skip never claims a signal was sent."""

    candidate: Candidate
    action: str
    detail: str
    terminated: list[int] = field(default_factory=list)
    killed: list[int] = field(default_factory=list)
    children: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True only when a signal was really sent, would be in a dry run, or a model was unloaded."""
        return self.action in ("stopped", "would_stop", "unloaded", "would_unload")

    def as_record(self) -> dict[str, Any]:
        """A JSON-ready line for reap.log."""
        rec = self.candidate.as_record()
        rec.update({"action": self.action, "detail": self.detail,
                    "children": list(self.children),
                    "terminated": list(self.terminated), "killed": list(self.killed)})
        return rec


def _same_process(pid: int, created: float,
                  process_factory: Callable[[int], Any]) -> Any | None:
    """A fresh process object for `pid` if it is still the one that started at `created`.

    psutil caches create_time on a Process object, so a new object is made
    each time. Start times are compared exactly, as mullet.jobs.children does.
    """
    try:
        fresh = process_factory(pid)
        if not fresh.is_running():
            return None
        if fresh.create_time() != created:
            return None
        return fresh
    except _PSUTIL_ERRORS:
        return None


def _gone(pid: int, created: float, process_factory: Callable[[int], Any]) -> bool:
    fresh = _same_process(pid, created, process_factory)
    if fresh is None:
        return True
    try:
        return fresh.status() == psutil.STATUS_ZOMBIE
    except _PSUTIL_ERRORS:
        return True
    except AttributeError:
        return False


def _signal(pid: int, created: float, sig: int,
            process_factory: Callable[[int], Any]) -> bool:
    fresh = _same_process(pid, created, process_factory)
    if fresh is None:
        return False
    try:
        fresh.send_signal(sig)
        return True
    except _PSUTIL_ERRORS:
        return False


def stop_process(
    candidate: Candidate,
    *,
    sleep: Callable[[float], None],
    now: Callable[[], float],
    process_factory: Callable[[int], Any] = psutil.Process,
    dry_run: bool = False,
    grace: float = TERM_GRACE_S,
) -> StopResult:
    """Stop `candidate`'s process and its own children, checking for a recycled pid first.

    Just before signalling, a fresh process object must say the pid is
    running and has the start time recorded at discovery; on any mismatch
    or error nothing is sent and the result is `skipped`. Otherwise SIGTERM
    goes to the process and its children (each re-checked the same way),
    the wait polls `now()` with `sleep()` for up to `grace` seconds, and
    survivors get SIGKILL, again re-checked. A dry run reports the same
    targets, children included, without signalling.
    """
    pid, created = candidate.pid, candidate.create_time
    if pid is None or created is None:
        return StopResult(candidate, "skipped", "not a process with a recorded start time")
    target = _same_process(pid, created, process_factory)
    if target is None:
        return StopResult(candidate, "skipped",
                          f"pid {pid} has exited or now belongs to another process")
    kids: list[tuple[int, float]] = []
    try:
        for child in target.children(recursive=True):
            try:
                kids.append((int(child.pid), float(child.create_time())))
            except _PSUTIL_ERRORS:
                continue
    except _PSUTIL_ERRORS:
        kids = []
    child_pids = [c for c, _ in kids]
    if dry_run:
        detail = f"would send SIGTERM to pid {pid}"
        if child_pids:
            detail += " and its children " + ", ".join(str(c) for c in child_pids)
        return StopResult(candidate, "would_stop", detail, children=child_pids)

    if not _signal(pid, created, signal.SIGTERM, process_factory):
        return StopResult(candidate, "skipped",
                          f"pid {pid} changed or exited before it could be signalled",
                          children=child_pids)
    terminated = [pid]
    for cpid, ccreated in kids:
        if _signal(cpid, ccreated, signal.SIGTERM, process_factory):
            terminated.append(cpid)

    tracked = [(pid, created)] + [k for k in kids if k[0] in terminated]
    deadline = now() + grace
    while True:
        alive = [t for t in tracked if not _gone(t[0], t[1], process_factory)]
        if not alive:
            break
        remaining = deadline - now()
        if remaining <= 0:
            break
        sleep(min(_POLL_S, remaining))

    killed = [p for p, c in alive if _signal(p, c, signal.SIGKILL, process_factory)]
    detail = f"sent SIGTERM to {len(terminated)} process(es)"
    if killed:
        detail += f", SIGKILL to {len(killed)} that did not exit within {grace:g} s"
    return StopResult(candidate, "stopped", detail, terminated=terminated,
                      killed=killed, children=child_pids)


def unload_lmstudio_model(
    candidate: Candidate,
    *,
    http_post: Callable[[str, Mapping[str, Any]], Any],
    dry_run: bool = False,
) -> StopResult:
    """Unload an idle LM Studio model through its REST API.

    The unload endpoint is taken from `candidate.extra["unload_url"]`. When it
    is unset the candidate is reported only and the LM Studio process is never
    touched. On success the model is considered unloaded; on failure it is
    skipped.
    """
    model_id = candidate.extra.get("model_id")
    unload_url = candidate.extra.get("unload_url")

    if not unload_url:
        return StopResult(
            candidate,
            "reported",
            "no LM Studio unload URL configured; model reported only, process not touched",
        )

    if dry_run:
        return StopResult(
            candidate,
            "would_unload",
            f"would POST to unload LM Studio model {model_id}",
        )

    try:
        http_post(unload_url, {"model_id": model_id})
    except Exception as exc:
        return StopResult(
            candidate,
            "skipped",
            f"LM Studio unload request failed: {exc}",
        )

    return StopResult(
        candidate,
        "unloaded",
        f"POSTed to unload LM Studio model {model_id}",
    )


# ---------------------------------------------------------------- a pass

# The actions reap.log records; the unload helper's own names map onto these.
_LOG_ACTIONS = {"stopped": "stopped", "would_stop": "would_stop", "skipped": "skipped",
                "reported": "reported", "unloaded": "stopped", "would_unload": "would_stop"}

_PROCESS_KINDS = (KIND_XDIST, KIND_OPENCODE, KIND_MCP)


def reap_log_path() -> Path:
    """Where every action and would-act is appended, one JSON line each."""
    return state_dir() / "reap.log"


def process_open_paths(
    process_iter: Callable[..., Iterable[Any]] = psutil.process_iter,
) -> list[tuple[str, list[str]]]:
    """(cwd, open file paths) for each of the current user's processes, for `find_stale_tmp_dirs()`."""
    me = os.getuid()
    out: list[tuple[str, list[str]]] = []
    for proc in process_iter():
        uids = _read(proc.uids)
        if uids is None or getattr(uids, "real", None) != me:
            continue
        cwd = _read(proc.cwd) or ""
        files = _read(proc.open_files) or []
        out.append((str(cwd), [str(getattr(f, "path", "")) for f in files if getattr(f, "path", "")]))
    return out


def _default_stopper(candidate: Candidate) -> StopResult:
    return stop_process(candidate, sleep=time.sleep, now=time.time)


def _log_record(candidate: Candidate, action: str, result: str, now: float,
                dry_run: bool) -> dict[str, Any]:
    rec: dict[str, Any] = {"ts": now, "kind": candidate.kind}
    if candidate.pid is not None:
        rec["pid"] = candidate.pid
        rec["create_time"] = candidate.create_time
    if candidate.path is not None:
        rec["path"] = candidate.path
    if candidate.kind == KIND_LMSTUDIO and candidate.identifier is not None:
        rec["model"] = candidate.identifier
    rec.update({"reason": candidate.reason, "action": _LOG_ACTIONS.get(action, "skipped"),
                "result": result, "idle_s": candidate.idle_s,
                "idle_limit_s": candidate.idle_limit, "dry_run": dry_run})
    return rec


def _append_log(records: Iterable[Mapping[str, Any]], path: Path) -> None:
    lines = [json.dumps(r, sort_keys=True) + "\n" for r in records]
    if not lines:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.writelines(lines)


def run_pass(
    *,
    dry_run: bool,
    now: float,
    snapshot_fn: Callable[[], Iterable[ProcInfo]] = snapshot,
    ledger: Ledger | None = None,
    http: Any | None = None,
    stopper: Callable[[Candidate], StopResult] = _default_stopper,
    sightings: SightingsStore | None = None,
    env: Mapping[str, str] | None = None,
    pid_exists: Callable[[int], bool] = psutil.pid_exists,
    process_factory: Callable[[int], Any] = psutil.Process,
    tmp_root: Path = Path("/tmp"),
    open_paths: Callable[[], Iterable[tuple[str, list[str]]]] | None = process_open_paths,
    log_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Run one reaper pass and return the reap.log records it wrote.

    One snapshot is taken and every finder runs against the same sightings
    store; the store is committed exactly once, after all of them, so no
    finder wipes another's first-seen times. Then each due candidate is
    acted on: processes through `stopper` (which must re-check the pid's
    start time), LM Studio models through `unload_lmstudio_model()` with
    `http.post`, and tmp directories through `remove_dir()`. In a dry run
    nothing is stopped, posted or removed, and `stopper` is never called.

    `ledger` None skips `opencode run` processes, `http` None (an object with
    `get(url)` and `post(url, data)`) skips LM Studio, and `open_paths` None
    skips tmp directories. Each action or would-act is appended as one JSON
    line to `log_path` (default state_dir()/reap.log).
    """
    store = sightings if sightings is not None else SightingsStore()
    procs = list(snapshot_fn())

    found: list[Candidate] = []
    found += find_xdist_orphans(procs, store, now, env=env, pid_exists=pid_exists)
    if ledger is not None:
        found += find_opencode_strays(procs, store, now, ledger=ledger, env=env,
                                      process_factory=process_factory)
    found += find_mcp_orphans(procs, store, now, env=env, pid_exists=pid_exists)
    if http is not None:
        found += find_idle_lmstudio_models(store, now, http_get=http.get, env=env)
    if open_paths is not None:
        found += find_stale_tmp_dirs(tmp_root, now=now, open_paths=open_paths, env=env)
    store.commit(seen_keys(found), now)

    records: list[dict[str, Any]] = []
    for cand in due_only(found):
        if cand.kind in _PROCESS_KINDS:
            if dry_run:
                action, result = "would_stop", f"would send SIGTERM to pid {cand.pid}"
            else:
                try:
                    res = stopper(cand)
                    action, result = res.action, res.detail
                except Exception as exc:
                    action, result = "skipped", f"stopping failed: {exc}"
        elif cand.kind == KIND_LMSTUDIO:
            if http is None:
                continue
            res = unload_lmstudio_model(cand, http_post=http.post, dry_run=dry_run)
            action, result = res.action, res.detail
        elif cand.kind == KIND_TMP and cand.path is not None and open_paths is not None:
            if dry_run:
                action, result = "would_stop", f"would remove {cand.path}"
            elif remove_dir(cand.path, tmp_root=tmp_root, now=now, open_paths=open_paths, env=env):
                action, result = "stopped", f"removed {cand.path}"
            else:
                action, result = "skipped", f"{cand.path} no longer qualified when re-checked"
        else:
            continue
        records.append(_log_record(cand, action, result, now, dry_run))

    _append_log(records, log_path if log_path is not None else reap_log_path())
    return records


# ---------------------------------------------------------------- timer window

# The timer only reports, never acts, for this long after it is installed.
TIMER_DRY_RUN_S = 24 * 3600.0


def timer_install_path() -> Path:
    """The file holding the time the reap timer was installed."""
    return state_dir() / "reap-timer-installed"


def record_timer_install(now: float) -> Path:
    """Write `now` as the timer's install time and return the file written."""
    path = timer_install_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{now!r}\n")
    return path


def timer_dry_run_active(now: float) -> bool:
    """True for the first 24 hours after the timer was installed.

    No install file means the window has never started, so False. A file
    that can't be read or holds no sensible time gives True, so the timer
    stays on the safe side and only reports.
    """
    try:
        text = timer_install_path().read_text()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    try:
        installed = float(text.strip())
    except ValueError:
        return True
    if not math.isfinite(installed):
        return True
    return now - installed < TIMER_DRY_RUN_S
