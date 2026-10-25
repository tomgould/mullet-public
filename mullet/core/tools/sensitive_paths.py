"""Locations the file and docker tools refuse, because they hold the user's credentials.

The default list is the user's SSH, cloud and token files, Mullet's own config and
keys, and ``.env`` files. ``delegation.file_deny`` in the Mullet config file replaces
it, with patterns like these::

    delegation:
      file_deny: ["~/.ssh", "~/.aws", "*.pem", ".env", ".env.*"]

A pattern with no ``/`` matches a file or directory name at any depth. One with a
``/`` is a path (``~`` is the home directory) and covers everything under it, and
may use ``*``.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from collections.abc import Iterable
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_DENY: tuple[str, ...] = (
    "~/.ssh", "~/.gnupg", "~/.aws", "~/.azure", "~/.kube", "~/.config/gcloud", "~/.config/gh",
    "~/.docker/config.json", "~/.netrc", "~/.git-credentials", "~/.npmrc", "~/.pypirc",
    "~/.mullet", "~/.config/mullet", "~/.local/share/mullet", "~/.local/share/opencode/auth.json",
    "~/.claude/.credentials.json", "~/.claude.json",
    ".env", ".env.*", "*.pem", "id_rsa", "id_ed25519", "id_ecdsa",
    "/var/run/docker.sock", "/run/docker.sock", "/etc/shadow",
)


def load_deny(config: Path | None = None) -> tuple[str, ...]:
    """The patterns to refuse: ``delegation.file_deny`` from the config file, else the default list."""
    if config is None:
        from mullet.core.paths import config_file

        config = config_file("config.yaml")
    try:
        data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return DEFAULT_DENY
    block = data.get("delegation") if isinstance(data, dict) else None
    given = block.get("file_deny") if isinstance(block, dict) else None
    if isinstance(given, list) and all(isinstance(p, str) for p in given):
        return tuple(given)
    if given is not None:
        logger.warning("delegation.file_deny should be a list of patterns; using the default list")
    return DEFAULT_DENY


def denied_by(path: str | os.PathLike[str], patterns: Iterable[str]) -> str | None:
    """The pattern that refuses `path`, or None. Symlinks are resolved first, so a link cannot hide a target."""
    resolved = Path(os.path.abspath(path)).resolve()
    here = str(resolved)
    for pattern in patterns:
        if "/" not in pattern:
            if any(fnmatch.fnmatch(part, pattern) for part in resolved.parts):
                return pattern
            continue
        base = os.path.expanduser(pattern)
        if "*" in base:
            if fnmatch.fnmatch(here, base) or fnmatch.fnmatch(here, base.rstrip("/") + "/*"):
                return pattern
            continue
        target = str(Path(base).resolve())
        if here == target or here.startswith(target.rstrip("/") + "/"):
            return pattern
    return None


def covers(path: str | os.PathLike[str], patterns: Iterable[str]) -> str | None:
    """The pattern for a location *inside* `path`, so mounting or reading a parent directory is refused too."""
    resolved = str(Path(os.path.abspath(path)).resolve())
    prefix = resolved.rstrip("/") + "/"
    for pattern in patterns:
        if "/" not in pattern or "*" in pattern:
            continue
        target = str(Path(os.path.expanduser(pattern)).resolve())
        if target == resolved or target.startswith(prefix):
            return pattern
    return None
