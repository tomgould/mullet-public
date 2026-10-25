"""Run commands in a sandbox: bubblewrap on Linux.

Mullet runs code a model has just written: the repo's setup, its tests and
its build. Every such process gets:

- the whole filesystem read-only, with $HOME, /tmp and /run hidden, which
  also hides sockets such as Docker's and the SSH agent's;
- write access only to the paths the caller names, normally the clone;
- read-only overlays for dependency directories inside those paths;
- no network unless the caller asks for it (only setup does);
- a fresh environment built from an allowlist, never Mullet's own, which
  holds API keys loaded from .env.

There is no unsandboxed fallback. If the sandbox can't start,
`SandboxUnavailable` is raised and the caller must refuse the work.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from mullet.jobs import children

_PASSTHROUGH_ENV = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ")
_TAIL_CHARS = 4000


class SandboxUnavailable(RuntimeError):
    """The sandbox can't run on this machine, so nothing may run."""


@dataclass(frozen=True)
class SandboxCheck:
    """Whether the sandbox works here, and why not if it doesn't."""

    ok: bool
    backend: str
    reason: str = ""


@dataclass
class SandboxSpec:
    """What a sandboxed process may see and touch."""

    cwd: Path
    writable: list[Path]
    readonly_overlays: list[Path] = field(default_factory=list)
    expose_ro: list[Path] = field(default_factory=list)
    # Hidden: home, temp dirs, and every usual home of a privileged socket
    # (Docker, LXD/Incus, Podman, systemd). /var/run is usually a symlink to
    # /run; build_argv skips symlinks and directories that don't exist.
    hide: list[Path] = field(
        default_factory=lambda: [
            Path.home(), Path("/tmp"), Path("/run"), Path("/var/run"), Path("/var/snap/lxd"),
            Path("/var/lib/lxd"), Path("/var/lib/incus"),
        ]
    )
    network: bool = False
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class RunResult:
    """How a sandboxed command ended. The full output is in `log_path`."""

    exit_code: int | None
    timed_out: bool
    duration_s: float
    log_path: Path
    tail: str


@lru_cache(maxsize=1)
def check() -> SandboxCheck:
    """Probe once per process whether bubblewrap can create a sandbox."""
    if sys.platform != "linux":
        return SandboxCheck(False, "none", f"no sandbox backend for {sys.platform} yet")
    bwrap = shutil.which("bwrap")
    if not bwrap:
        return SandboxCheck(False, "bwrap", "bubblewrap (bwrap) is not installed")
    probe = [
        bwrap, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
        "--unshare-net", "--unshare-pid", "--die-with-parent", "true",
    ]
    try:
        proc = subprocess.run(probe, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return SandboxCheck(False, "bwrap", f"bubblewrap probe failed: {exc}")
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip() or f"exit {proc.returncode}"
        return SandboxCheck(False, "bwrap", f"bubblewrap can't create a sandbox: {detail}")
    return SandboxCheck(True, "bwrap")


def _require() -> str:
    status = check()
    if not status.ok:
        raise SandboxUnavailable(status.reason)
    return shutil.which("bwrap") or "bwrap"


def build_env(spec: SandboxSpec) -> dict[str, str]:
    """The complete environment a sandboxed process starts with. Mullet's own values are set last."""
    env = {k: os.environ[k] for k in _PASSTHROUGH_ENV if k in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env.update(spec.env)
    env.update({"HOME": str(Path.home()), "TMPDIR": "/tmp", "TERM": "dumb"})
    return env


def build_argv(command: list[str], spec: SandboxSpec) -> list[str]:
    """The bwrap command line that runs `command` under `spec`.

    Mount order matters: hidden directories become empty tmpfs first, then
    toolchains are exposed read-only, then writable paths are bound, and
    last the read-only overlays go on top of the writable ones.
    """
    bwrap = _require()
    argv = [
        bwrap, "--die-with-parent", "--new-session",
        "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-cgroup-try",
    ]
    if not spec.network:
        argv.append("--unshare-net")
    argv += ["--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc"]
    for hidden in spec.hide:
        if hidden.is_dir() and not hidden.is_symlink():
            argv += ["--tmpfs", str(hidden)]
    if spec.network:
        # /etc/resolv.conf often points into /run, which is hidden above.
        resolver = Path("/run/systemd/resolve")
        if resolver.is_dir():
            argv += ["--ro-bind", str(resolver), str(resolver)]
    # A symlinked bind source would mount its target instead, which could be
    # anywhere, including a directory hidden above. Writable paths and
    # overlays can come from the work tree, so a symlink there is refused.
    # Toolchain paths come from the user's own profile and are followed.
    for path in [*spec.writable, *spec.readonly_overlays]:
        if path.is_symlink():
            raise ValueError(f"refusing to mount {path}: it is a symlink")
    for path in spec.expose_ro:
        if path.exists():
            argv += ["--ro-bind", str(path.resolve()), str(path)]
    for path in spec.writable:
        if not path.is_dir():
            raise ValueError(f"writable path {path} is not a directory")
        argv += ["--bind", str(path), str(path)]
    for path in spec.readonly_overlays:
        if path.exists():
            argv += ["--ro-bind", str(path), str(path)]
    argv.append("--clearenv")
    for key, value in build_env(spec).items():
        argv += ["--setenv", key, value]
    argv += ["--chdir", str(spec.cwd), "--"]
    return argv + command


def run(command: list[str], spec: SandboxSpec, *, timeout: float, log_path: Path) -> RunResult:
    """Run `command` in the sandbox, writing all output to `log_path`.

    On timeout the whole process group is killed: TERM, then KILL after a
    grace period. bwrap's own PID namespace takes the children with it.
    """
    children.check_not_stopping()
    argv = build_argv(command, spec)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        )
        with children.track(proc):
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_group(proc)
    duration = time.monotonic() - started
    return RunResult(
        exit_code=None if timed_out else proc.returncode,
        timed_out=timed_out,
        duration_s=duration,
        log_path=log_path,
        tail=read_tail(log_path),
    )


def run_shell(script: str, spec: SandboxSpec, *, timeout: float, log_path: Path) -> RunResult:
    """Run a shell command string (a profile command) in the sandbox."""
    return run(["/bin/sh", "-c", script], spec, timeout=timeout, log_path=log_path)


def _kill_group(proc: subprocess.Popen, grace: float = 5.0) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


def read_tail(path: Path, chars: int = _TAIL_CHARS) -> str:
    """The last `chars` characters of a log, for summaries and retry feedback."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - chars * 4))
            data = fh.read()
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")[-chars:]
