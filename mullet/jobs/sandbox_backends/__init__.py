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
from mullet.jobs.sandbox_backends.container import DockerBackend, PodmanBackend
from mullet.jobs.sandbox_backends.settings import AUTO, RESERVED, SandboxSettings

# `auto`'s order of preference is AUTO_ORDER in mullet.jobs.sandbox.
BACKENDS: dict[str, Callable[..., SandboxBackend]] = {
    "bubblewrap": BubblewrapBackend,
    "docker": DockerBackend,
    "podman": PodmanBackend,
}
# Backends that read the image and limits from the ``sandbox:`` settings.
_TAKES_SETTINGS = {"docker", "podman"}


def get_backend(name: str, settings: SandboxSettings | None = None) -> SandboxBackend:
    """A new backend called `name`; `SandboxUnavailable` when this Mullet has none by that name.

    The container backends take `settings` (the user's config file when None).
    """
    factory = BACKENDS.get(name)
    if factory is not None:
        return factory(settings) if name in _TAKES_SETTINGS else factory()
    if name in RESERVED:
        raise SandboxUnavailable(
            f"sandbox.backend is {name}, which this version of Mullet does not have yet; "
            f"use {' or '.join([*BACKENDS, AUTO])}")
    raise SandboxUnavailable(f"unknown sandbox backend {name!r}; use {' or '.join([*BACKENDS, AUTO])}")


__all__ = [
    "BACKENDS", "BubblewrapBackend", "DockerBackend", "PodmanBackend", "RunResult", "SandboxBackend", "SandboxCheck", "SandboxSpec",
    "SandboxUnavailable", "get_backend",
]
