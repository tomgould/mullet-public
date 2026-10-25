"""The ``sandbox:`` block of the Mullet config file::

    sandbox:
      backend: bubblewrap   # bubblewrap (default) or auto

``auto`` takes the first backend that works on this machine, and refuses all
work when none does. ``docker``, ``podman`` and ``seatbelt`` are reserved for
backends still to come: choosing one now refuses all work with a message that
says so. A bad value is reported and the default kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_BACKEND = "bubblewrap"
AUTO = "auto"
AVAILABLE = ("bubblewrap",)
RESERVED = ("docker", "podman", "seatbelt")
CHOICES = (*AVAILABLE, AUTO, *RESERVED)
KEYS = ("backend",)


@dataclass
class SandboxSettings:
    """The ``sandbox:`` settings, with any problems found while reading them."""

    backend: str = DEFAULT_BACKEND
    from_config: bool = False
    problems: list[str] = field(default_factory=list)


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
    value = data.get("backend")
    if value is None:
        return settings
    name = value.strip().lower() if isinstance(value, str) else None
    if name not in CHOICES:
        settings.problems.append(f"{source}.backend: expected one of {', '.join(CHOICES)} "
                                 f"(got {value!r}); {DEFAULT_BACKEND} is used")
        return settings
    settings.backend = name
    settings.from_config = True
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
