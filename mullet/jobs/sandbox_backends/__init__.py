"""Sandbox backends: the ways Mullet can confine a process.

Callers use `mullet.jobs.sandbox`, which picks one backend from the
``sandbox.backend`` setting and refuses all work when it can't run.
"""

from __future__ import annotations

from collections.abc import Callable

from mullet.jobs.sandbox_backends.base import (
    RunResult,
    SandboxBackend,
    SandboxCheck,
    SandboxSpec,
    SandboxUnavailable,
)
from mullet.jobs.sandbox_backends.bubblewrap import BubblewrapBackend
from mullet.jobs.sandbox_backends.settings import AUTO, RESERVED

# The order is `auto`'s order of preference.
BACKENDS: dict[str, Callable[[], SandboxBackend]] = {
    "bubblewrap": BubblewrapBackend,
}


def get_backend(name: str) -> SandboxBackend:
    """A new backend called `name`; `SandboxUnavailable` when this Mullet has none by that name."""
    factory = BACKENDS.get(name)
    if factory is not None:
        return factory()
    if name in RESERVED:
        raise SandboxUnavailable(
            f"sandbox.backend is {name}, which this version of Mullet does not have yet; "
            f"use {' or '.join([*BACKENDS, AUTO])}")
    raise SandboxUnavailable(f"unknown sandbox backend {name!r}; use {' or '.join([*BACKENDS, AUTO])}")


__all__ = [
    "BACKENDS", "BubblewrapBackend", "RunResult", "SandboxBackend", "SandboxCheck", "SandboxSpec",
    "SandboxUnavailable", "get_backend",
]
