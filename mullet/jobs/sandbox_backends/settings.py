"""The ``sandbox:`` block of the Mullet config file::

    sandbox:
      backend: bubblewrap   # bubblewrap (default), docker, podman or auto
      image: mullet-sandbox:<version>   # docker and podman only
      memory: 8g            # docker and podman only: the container's memory limit
      pids_limit: 4096      # docker and podman only: the most processes and threads

``auto`` takes the first backend that works on this machine (bubblewrap, then
docker, then podman on Linux), and refuses all work when none does.
``seatbelt`` is reserved for a backend still to come: choosing it refuses all
work with a message that says so. A bad value is reported and its default
kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mullet import __version__

DEFAULT_BACKEND = "bubblewrap"
AUTO = "auto"
CONTAINER_BACKENDS = ("docker", "podman")
AVAILABLE = ("bubblewrap", *CONTAINER_BACKENDS)
RESERVED = ("seatbelt",)
CHOICES = (*AVAILABLE, AUTO, *RESERVED)
KEYS = ("backend", "image", "memory", "pids_limit")

DEFAULT_IMAGE = f"mullet-sandbox:{__version__}"
DEFAULT_MEMORY = "8g"
DEFAULT_PIDS_LIMIT = 4096
MIN_MEMORY_BYTES = 64 * 1024 ** 2
PIDS_RANGE = (64, 1_000_000)
_MEMORY = re.compile(r"([1-9][0-9]*)([kmg])")
_UNITS = {"k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}
# A local image reference: name[:tag] or name@digest, no spaces or options.
_IMAGE = re.compile(r"[a-z0-9][a-z0-9._/:@-]*", re.IGNORECASE)


@dataclass
class SandboxSettings:
    """The ``sandbox:`` settings, with any problems found while reading them.

    `from_config` says whether `backend` came from the config file.
    """

    backend: str = DEFAULT_BACKEND
    image: str = DEFAULT_IMAGE
    memory: str = DEFAULT_MEMORY
    pids_limit: int = DEFAULT_PIDS_LIMIT
    from_config: bool = False
    problems: list[str] = field(default_factory=list)


def _parse_backend(value: Any, source: str, settings: SandboxSettings) -> None:
    name = value.strip().lower() if isinstance(value, str) else None
    if name not in CHOICES:
        settings.problems.append(f"{source}.backend: expected one of {', '.join(CHOICES)} "
                                 f"(got {value!r}); {DEFAULT_BACKEND} is used")
        return
    settings.backend = name
    settings.from_config = True


def _parse_image(value: Any, source: str, settings: SandboxSettings) -> None:
    if isinstance(value, str) and _IMAGE.fullmatch(value.strip()):
        settings.image = value.strip()
        return
    settings.problems.append(f"{source}.image: expected an image name such as mullet-sandbox:{__version__} "
                             f"(got {value!r}); {DEFAULT_IMAGE} is used")


def memory_bytes(value: str) -> int:
    """Bytes in a docker-style size such as ``8g`` or ``512m``; ValueError when it isn't one."""
    match = _MEMORY.fullmatch(value.strip().lower())
    if not match:
        raise ValueError(value)
    return int(match.group(1)) * _UNITS[match.group(2)]


def _parse_memory(value: Any, source: str, settings: SandboxSettings) -> None:
    try:
        if not isinstance(value, str) or memory_bytes(value) < MIN_MEMORY_BYTES:
            raise ValueError(value)
    except ValueError:
        settings.problems.append(f"{source}.memory: expected a size of at least 64m, such as 8g or 4096m "
                                 f"(got {value!r}); {DEFAULT_MEMORY} is used")
        return
    settings.memory = value.strip().lower()


def _parse_pids(value: Any, source: str, settings: SandboxSettings) -> None:
    low, high = PIDS_RANGE
    if isinstance(value, int) and not isinstance(value, bool) and low <= value <= high:
        settings.pids_limit = value
        return
    settings.problems.append(f"{source}.pids_limit: expected a whole number from {low} to {high} "
                             f"(got {value!r}); {DEFAULT_PIDS_LIMIT} is used")


_PARSERS = {"backend": _parse_backend, "image": _parse_image, "memory": _parse_memory,
            "pids_limit": _parse_pids}


def parse_settings(data: Any, source: str = "sandbox") -> SandboxSettings:
    """Read the raw ``sandbox:`` mapping; a bad value is reported and its default kept."""
    settings = SandboxSettings()
    if data is None:
        return settings
    if not isinstance(data, dict):
        settings.problems.append(f"{source}: expected a mapping; ignored")
        return settings
    for key in data:
        if key not in KEYS:
            settings.problems.append(f"{source}.{key}: unknown setting; ignored")
    for key, parse in _PARSERS.items():
        if data.get(key) is not None:
            parse(data[key], source, settings)
    return settings


def settings_path() -> Path:
    """The user's Mullet config file, which holds the ``sandbox:`` block."""
    from mullet.core.paths import config_file

    return config_file("config.yaml")


def load_settings(path: Path | None = None) -> SandboxSettings:
    """The ``sandbox:`` settings from the user's config file; defaults when there are none."""
    target = path or settings_path()
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return SandboxSettings()
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        settings = SandboxSettings()
        settings.problems.append(f"{target}: unreadable ({exc}); {DEFAULT_BACKEND} is used")
        return settings
    return parse_settings(data.get("sandbox") if isinstance(data, dict) else None, f"{target} sandbox")
