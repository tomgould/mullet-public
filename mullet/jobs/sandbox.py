"""Run commands in a sandbox, through the backend the `sandbox.backend` setting picks.

Mullet runs code a model has just written: the repo's setup, its tests and
its build. Every such process gets:

- the whole filesystem read-only, with $HOME, /tmp and /run hidden, which
  also hides sockets such as Docker's and the SSH agent's;
- write access only to the paths the caller names, normally the clone;
- read-only overlays for dependency directories inside those paths;
- no network unless the caller asks for it (only setup does, see below);
- a fresh environment built from an allowlist, never Mullet's own, which
  holds API keys loaded from .env.

Setup commands share the host's network. `GateRunner.setup` passes
`network=True`, so those commands keep the host's network and can reach
anything the host can, loopback services and the local network included: the
package managers in them (`pip`, `npm`, `composer`, `cargo`) need the
internet, and there is no per-host filter here. Setup is still sandboxed in
every other way (filesystem, environment, no Mullet keys), and it is the only
step with network. Workers, checks and gates run with the network unshared.
Setup commands come from the profile: the user's stored profile, and the
repo's `.mullet/project.yaml` as committed at the job's base commit, never
from files the worker edits. Treat a repo you don't trust as able to run
arbitrary network-capable commands during setup.

There is no unsandboxed fallback. If the sandbox can't start,
`SandboxUnavailable` is raised and the caller must refuse the work.

The backends live in `mullet.jobs.sandbox_backends`: bubblewrap is the default
and the only one so far. This module keeps the names every caller uses,
selects one backend per process and refuses all work when it can't run, so
no caller needs to know which backend that is.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from mullet.jobs import children
from mullet.jobs.sandbox_backends import get_backend
from mullet.jobs.sandbox_backends.base import (
    RunResult,
    SandboxBackend,
    SandboxCheck,
    SandboxSpec,
    SandboxUnavailable,
    build_env,
    read_tail,
)
from mullet.jobs.sandbox_backends.settings import AUTO, SandboxSettings, load_settings

# Which backends `auto` tries on each platform, best first.
AUTO_ORDER: dict[str, tuple[str, ...]] = {"linux": ("bubblewrap",)}

__all__ = [
    "AUTO_ORDER", "RunResult", "SandboxBackend", "SandboxCheck", "SandboxSpec", "SandboxUnavailable",
    "Selection", "backend", "build_env", "check", "clear_cache", "nested_argv", "nested_check",
    "read_tail", "run", "run_shell", "select", "selection", "spawn",
]


@dataclass(frozen=True)
class Selection:
    """The backend the setting picked and why, or why none can be used."""

    name: str
    reason: str
    backend: SandboxBackend | None
    error: str = ""
    probe: SandboxCheck | None = None
    problems: tuple[str, ...] = ()


def select(settings: SandboxSettings, platform: str | None = None) -> Selection:
    """Pick the backend `settings` asks for. Never an unsandboxed one: on failure `backend` is None."""
    problems = tuple(settings.problems)
    requested = settings.backend
    if requested != AUTO:
        why = "set in the config file" if settings.from_config else "the default"
        try:
            return Selection(requested, why, get_backend(requested), problems=problems)
        except SandboxUnavailable as exc:
            return Selection(requested, why, None, error=str(exc), problems=problems)
    platform = platform or sys.platform
    tried = []
    for name in AUTO_ORDER.get(platform, ()):
        candidate = get_backend(name)
        probe = candidate.check()
        if probe.ok:
            return Selection(name, "sandbox.backend is auto, and it works here", candidate, probe=probe,
                             problems=problems)
        tried.append(f"{name}: {probe.reason}")
    detail = "; ".join(tried) if tried else f"no sandbox backend for {platform} yet"
    return Selection(AUTO, "sandbox.backend is auto", None,
                     error=f"sandbox.backend is auto, and no backend works here ({detail})", problems=problems)


@dataclass(frozen=True)
class _Resolved:
    """The selection and its probe, cached together so they can never disagree."""

    selection: Selection
    probe: SandboxCheck


_lock = threading.Lock()
_resolved: _Resolved | None = None


def _probe(chosen: Selection) -> SandboxCheck:
    if chosen.backend is None:
        return SandboxCheck(False, chosen.name, chosen.error)
    return chosen.probe or chosen.backend.check()


def _resolve() -> _Resolved:
    global _resolved
    with _lock:
        if _resolved is None:
            chosen = select(load_settings())
            _resolved = _Resolved(chosen, _probe(chosen))
        return _resolved


def clear_cache() -> None:
    """Forget the selection and its probe together; the next call reads the config file again."""
    global _resolved
    with _lock:
        _resolved = None


def selection() -> Selection:
    """The backend for this process, from the user's config file. Read once."""
    return _resolve().selection


def backend() -> SandboxBackend:
    """The selected backend, without probing it; `SandboxUnavailable` when none can be used."""
    chosen = selection()
    if chosen.backend is None:
        raise SandboxUnavailable(chosen.error)
    return chosen.backend


def check() -> SandboxCheck:
    """Probe once per process whether the selected backend can create a sandbox."""
    chosen = selection()
    resolved = _resolve()
    if resolved.selection is chosen:
        return resolved.probe
    return _probe(chosen)


selection.cache_clear = clear_cache  # type: ignore[attr-defined]
check.cache_clear = clear_cache  # type: ignore[attr-defined]


def _require() -> SandboxBackend:
    status = check()
    if not status.ok:
        raise SandboxUnavailable(status.reason)
    return backend()


def spawn(command: list[str], spec: SandboxSpec, *, stdout: int | IO[bytes] | None,
          stderr: int | IO[bytes] | None, cwd: Path | None = None, text: bool = False) -> subprocess.Popen:
    """Start `command` under `spec` in its own process group, for a long-lived worker."""
    return _require().spawn(command, spec, stdout=stdout, stderr=stderr, cwd=cwd, text=text)


def run(command: list[str], spec: SandboxSpec, *, timeout: float, log_path: Path) -> RunResult:
    """Run `command` in the sandbox, writing all output to `log_path`.

    On timeout the whole process group is killed: TERM, then KILL after a
    grace period.
    """
    children.check_not_stopping()
    return _require().run(command, spec, timeout=timeout, log_path=log_path)


def run_shell(script: str, spec: SandboxSpec, *, timeout: float, log_path: Path) -> RunResult:
    """Run a shell command string (a profile command) in the sandbox."""
    children.check_not_stopping()
    return _require().run_shell(script, spec, timeout=timeout, log_path=log_path)


def nested_argv(readonly: list[Path], hidden: list[Path]) -> list[str]:
    """The selected backend's prefix for a network-less sandbox inside its own.

    Raises `SandboxUnavailable` when no backend can be used, which `shell_wrapper`
    (opencode executor) passes on. That is safe there: the executor's
    `check_runnable` runs `check()` before any wrapper is built, and refuses the
    node when the sandbox is unavailable, so the exception is never the first sign.
    """
    return backend().nested_argv(readonly, hidden)


def nested_check() -> bool:
    """Whether the selected backend works and can start a nested sandbox inside its own."""
    return check().ok and backend().nested_check()
