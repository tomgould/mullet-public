"""The bubblewrap backend: Linux user namespaces, no daemon and no root.

The filesystem is the host's, mounted read-only, with home, /tmp and /run
covered by empty tmpfs; only the spec's writable paths are bound read-write.
The network namespace is unshared unless the spec asks for network.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import IO

from mullet.jobs import children
from mullet.jobs.sandbox_backends.base import (
    RunResult,
    SandboxCheck,
    SandboxSpec,
    build_env,
    kill_group,
    launcher_env,
    read_tail,
)


class BubblewrapBackend:
    """Confine processes with bubblewrap (`bwrap`)."""

    name = "bubblewrap"

    def _bwrap(self) -> str:
        return shutil.which("bwrap") or "bwrap"

    def check(self) -> SandboxCheck:
        """Probe whether bubblewrap can create a sandbox."""
        if sys.platform != "linux":
            return SandboxCheck(False, self.name, f"no sandbox backend for {sys.platform} yet")
        bwrap = shutil.which("bwrap")
        if not bwrap:
            return SandboxCheck(False, self.name, "bubblewrap (bwrap) is not installed")
        probe = [
            bwrap, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
            "--unshare-net", "--unshare-pid", "--die-with-parent", "true",
        ]
        try:
            proc = subprocess.run(probe, capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return SandboxCheck(False, self.name, f"bubblewrap probe failed: {exc}")
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip() or f"exit {proc.returncode}"
            return SandboxCheck(False, self.name, f"bubblewrap can't create a sandbox: {detail}")
        return SandboxCheck(True, self.name)

    def build_argv(self, command: list[str], spec: SandboxSpec) -> list[str]:
        """The bwrap command line that runs `command` under `spec`.

        Mount order matters: hidden directories become empty tmpfs first, then
        toolchains are exposed read-only, then writable paths are bound, and
        last the read-only overlays go on top of the writable ones.
        """
        argv = [
            self._bwrap(), "--die-with-parent", "--new-session",
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

    def spawn(self, command: list[str], spec: SandboxSpec, *, stdout: int | IO[bytes] | None,
              stderr: int | IO[bytes] | None, cwd: Path | None = None,
              text: bool = False) -> subprocess.Popen:
        """Start bwrap in a new session with stdin closed and only PATH in its own environment."""
        return self._start(self.build_argv(command, spec), stdout=stdout, stderr=stderr, cwd=cwd, text=text)

    def _start(self, argv: list[str], *, stdout: int | IO[bytes] | None, stderr: int | IO[bytes] | None,
               cwd: Path | None = None, text: bool = False) -> subprocess.Popen:
        extra: dict = {"text": True, "encoding": "utf-8", "errors": "replace"} if text else {}
        return subprocess.Popen(
            argv,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            env=launcher_env(),
            **extra,
        )

    def run(self, command: list[str], spec: SandboxSpec, *, timeout: float, log_path: Path) -> RunResult:
        """Run `command` in the sandbox, writing all output to `log_path`.

        The spec is checked before the log is created, so a refused spec leaves
        no log. On timeout the whole process group is killed: TERM, then KILL
        after a grace period. bwrap's own PID namespace takes the children with it.
        `mullet.jobs.sandbox.run` has already refused work after a stop.
        """
        argv = self.build_argv(command, spec)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        timed_out = False
        with open(log_path, "wb") as log:
            proc = self._start(argv, stdout=log, stderr=subprocess.STDOUT)
            with children.track(proc):
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    kill_group(proc)
        duration = time.monotonic() - started
        return RunResult(
            exit_code=None if timed_out else proc.returncode,
            timed_out=timed_out,
            duration_s=duration,
            log_path=log_path,
            tail=read_tail(log_path),
        )

    def run_shell(self, script: str, spec: SandboxSpec, *, timeout: float, log_path: Path) -> RunResult:
        """Run a shell command string (a profile command) in the sandbox."""
        return self.run(["/bin/sh", "-c", script], spec, timeout=timeout, log_path=log_path)

    def nested_argv(self, readonly: list[Path], hidden: list[Path]) -> list[str]:
        """A second bwrap, started inside the first one, ending in `--`.

        bwrap must start from a cleared environment (the caller runs it under
        `env -i`): its own init process keeps whatever it was started with, and
        would show it in /proc. Its own session, pid, IPC and UTS namespaces
        and /proc as well, so the outer sandbox's processes aren't even listed.
        --bind / / keeps the outer view, writable paths included; `readonly`
        goes read-only on top, and `hidden` is covered by an empty tmpfs.
        """
        argv = [self._bwrap(), "--die-with-parent", "--new-session", "--unshare-net", "--unshare-pid",
                "--unshare-ipc", "--unshare-uts", "--bind", "/", "/", "--dev", "/dev", "--proc", "/proc"]
        for path in readonly:
            argv += ["--ro-bind", str(path), str(path)]
        for path in hidden:
            argv += ["--tmpfs", str(path)]
        return argv + ["--"]

    def nested_check(self) -> bool:
        """Whether a bwrap sandbox can start another, network-less one inside it."""
        bwrap = shutil.which("bwrap")
        if not bwrap:
            return False
        probe = [bwrap, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--unshare-pid",
                 "--", *self.nested_argv([], []), "true"]
        try:
            return subprocess.run(probe, capture_output=True, timeout=15).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False
