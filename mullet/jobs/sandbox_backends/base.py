"""What every sandbox backend shares: the spec, the results, the environment and the interface.

A backend turns a `SandboxSpec` into a confined process. The guarantees it
must keep are in docs/adr/0007-sandbox-backends.md; the shared contract tests
are `backend_contract` in tests/test_sandbox_backends.py.
"""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Protocol, runtime_checkable

PASSTHROUGH_ENV = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ")
TAIL_CHARS = 4000


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


@runtime_checkable
class SandboxBackend(Protocol):
    """One way of confining a process. Callers reach it through `mullet.jobs.sandbox`.

    `mullet.jobs.sandbox` runs `check()` once per process and refuses all work
    when it fails, before any other method is called. A backend never runs a
    command unconfined: when it can't confine one it raises.
    """

    name: str

    def check(self) -> SandboxCheck:
        """Probe whether this backend can confine a process here. Not cached."""
        ...

    def build_argv(self, command: list[str], spec: SandboxSpec) -> list[str]:
        """The host command line that runs `command` confined by `spec`."""
        ...

    def spawn(self, command: list[str], spec: SandboxSpec, *, stdout: int | IO[bytes] | None,
              stderr: int | IO[bytes] | None, cwd: Path | None = None,
              text: bool = False) -> subprocess.Popen:
        """Start `command` confined by `spec` in its own process group, for a long-lived worker."""
        ...

    def run(self, command: list[str], spec: SandboxSpec, *, timeout: float, log_path: Path) -> RunResult:
        """Run `command` to the end or the timeout, all output to `log_path`."""
        ...

    def run_shell(self, script: str, spec: SandboxSpec, *, timeout: float, log_path: Path) -> RunResult:
        """Run a shell command string (a profile command) like `run`."""
        ...

    def nested_argv(self, readonly: list[Path], hidden: list[Path]) -> list[str]:
        """An argv prefix that, run inside this backend's sandbox, starts a second one.

        The inner sandbox has no network at all, its own process namespaces,
        `readonly` paths read-only and `hidden` paths covered. opencode's
        worker runs each shell command through it, away from the proxy port.
        """
        ...

    def nested_check(self) -> bool:
        """Whether `nested_argv` works inside this backend's sandbox here. Not cached."""
        ...


def build_env(spec: SandboxSpec) -> dict[str, str]:
    """The complete environment a sandboxed process starts with. Mullet's own values are set last."""
    env = {k: os.environ[k] for k in PASSTHROUGH_ENV if k in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env.update(spec.env)
    env.update({"HOME": str(Path.home()), "TMPDIR": "/tmp", "TERM": "dumb"})
    return env


def launcher_env() -> dict[str, str]:
    """The environment the host-side launcher (bwrap, docker) starts with: PATH only."""
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}


def kill_group(proc: subprocess.Popen, grace: float = 5.0) -> None:
    """TERM the process group, then KILL it if it is still there after `grace` seconds."""
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


def read_tail(path: Path, chars: int = TAIL_CHARS) -> str:
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
