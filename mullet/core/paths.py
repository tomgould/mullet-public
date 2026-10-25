"""Where Mullet keeps its configuration, state, data and caches.

The rule, in one place:

- Configuration lives in ``config_dir()``: ``$MULLET_CONFIG_DIR``, else
  ``$XDG_CONFIG_HOME/mullet``, else ``~/.config/mullet``. A config file that an
  older install kept in ``~/.mullet`` (``config.yaml``, ``license.key``) is
  still read from there while ``config_dir()`` has no copy of it; see
  ``config_file()``. Nothing new is written to ``~/.mullet``.
- State, data and caches live in ``~/.mullet`` while it holds an older install
  (``legacy_active()``), so that install keeps working untouched. Otherwise
  they follow the XDG base directory spec: ``$XDG_STATE_HOME/mullet``
  (default ``~/.local/state/mullet``), ``$XDG_DATA_HOME/mullet``
  (``~/.local/share/mullet``) and ``$XDG_CACHE_HOME/mullet``
  (``~/.cache/mullet``).
- An XDG variable that is unset, empty or relative is ignored, as the spec
  requires.
- Trawl pools live in ``$TRAWL_POOLS_DIR``, else ``data_dir() / "pools"``.
- Nothing moves on its own. ``mullet migrate-state`` moves an older install
  into the XDG directories and leaves ``~/.mullet/MOVED.txt`` behind, after
  which ``~/.mullet`` no longer counts as an install.

Every function reads the environment when it is called, never at import, so
a changed ``HOME`` or ``XDG_*`` (the test suite gives every test its own HOME)
takes effect at once. None of them creates a directory: whatever writes a
file creates its parent.

A project's own ``.mullet/`` directory (session overrides, observations,
decisions, the in-repo job profile) is a different thing and stays where it
is.
"""

from __future__ import annotations

import os
from pathlib import Path

# The note `mullet migrate-state` leaves in ~/.mullet. While it is there,
# ~/.mullet no longer counts as an install.
MOVED_NOTE = "MOVED.txt"

# What an older install keeps in ~/.mullet, and which kind of directory each
# entry belongs to. `mullet migrate-state` moves every entry to
# `<that kind's XDG directory>/<same name>`; an entry not listed here goes to
# the data directory.
LEGACY_LAYOUT: dict[str, str] = {
    # configuration
    "config.yaml": "config",
    ".env": "config",
    "license.key": "config",
    "credentials.json": "config",
    # state: logs, history and what Mullet records about its own use
    "routing.log": "state",
    "routing_history": "state",
    "logs": "state",
    "interactions.db": "state",
    "cost_history.jsonl": "state",
    "delegation_spend.db": "state",
    "analytics": "state",
    "sessions": "state",
    "sessions.db": "state",
    "history.json": "state",
    "skill_stats.json": "state",
    "learned_constraints.json": "state",
    "unavailable_models.json": "state",
    "opencode_known_bad_models.json": "state",
    "state": "state",
    "shared-memory": "state",
    ".trial_start": "state",
    # data: what the user or Mullet built up and would miss
    "pools": "data",
    "skills": "data",
    "learned_skills": "data",
    "learned_skills.json": "data",
    "agents": "data",
    "plugins": "data",
    "prompts": "data",
    "library_pools.json": "data",
    "memory.json": "data",
    "notepad": "data",
    "mullet.db": "data",
    "mcp-tokens": "data",
    # caches: rebuilt when missing
    "model_cache.json": "cache",
    "moonshot_pricing_cache.json": "cache",
    "preflight.jsonl": "cache",
    "usage-cache.json": "cache",
    "hud-transcript": "cache",
    "cache": "cache",
}

# Names a project's own .mullet/ directory can hold too. When Mullet runs with
# HOME as its working directory, the project directory is ~/.mullet, so these
# alone don't show that an older install lives there.
_ALSO_PROJECT_NAMES = frozenset({"state", "skills", "plugins", "prompts", "license.key"})

_INSTALL_MARKERS = frozenset(LEGACY_LAYOUT) - _ALSO_PROJECT_NAMES


def _xdg(var: str, default: str) -> Path:
    """The XDG base directory in ``var``, or ``~/<default>`` when unset or relative."""
    value = os.environ.get(var, "")
    return Path(value) if os.path.isabs(value) else Path.home() / default


def legacy_dir() -> Path:
    """``~/.mullet``, where installs before XDG support kept everything."""
    return Path.home() / ".mullet"


def legacy_active() -> bool:
    """True while ``~/.mullet`` holds an older install that hasn't been migrated.

    It counts when it has no ``MOVED.txt`` and holds at least one entry that
    only a global install writes (see ``LEGACY_LAYOUT``).
    """
    try:
        names = set(os.listdir(legacy_dir()))
    except OSError:
        return False
    return MOVED_NOTE not in names and bool(names & _INSTALL_MARKERS)


def config_dir() -> Path:
    """Directory of Mullet's configuration: ``$MULLET_CONFIG_DIR``, else XDG."""
    override = os.environ.get("MULLET_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    return _xdg("XDG_CONFIG_HOME", ".config") / "mullet"


def xdg_state_dir() -> Path:
    """Where state goes on an XDG install, whether or not ``~/.mullet`` is in use."""
    return _xdg("XDG_STATE_HOME", ".local/state") / "mullet"


def xdg_data_dir() -> Path:
    """Where data goes on an XDG install, whether or not ``~/.mullet`` is in use."""
    return _xdg("XDG_DATA_HOME", ".local/share") / "mullet"


def xdg_cache_dir() -> Path:
    """Where caches go on an XDG install, whether or not ``~/.mullet`` is in use."""
    return _xdg("XDG_CACHE_HOME", ".cache") / "mullet"


def state_dir() -> Path:
    """Directory of logs, history and other state: ``~/.mullet`` on an older install."""
    return legacy_dir() if legacy_active() else xdg_state_dir()


def data_dir() -> Path:
    """Directory of pools, skills, memory and other data: ``~/.mullet`` on an older install."""
    return legacy_dir() if legacy_active() else xdg_data_dir()


def cache_dir() -> Path:
    """Directory of caches Mullet can rebuild: ``~/.mullet`` on an older install."""
    return legacy_dir() if legacy_active() else xdg_cache_dir()


def pools_dir() -> Path:
    """Directory of Trawl pools: ``$TRAWL_POOLS_DIR`` (verbatim), else ``data_dir()/pools``."""
    override = os.environ.get("TRAWL_POOLS_DIR")
    if override:
        return Path(override)
    return data_dir() / "pools"


def config_file(name: str) -> Path:
    """Path of the config file ``name`` to read.

    ``config_dir()/name``, unless only an older install's ``~/.mullet/name``
    exists. Write config to ``config_dir()`` directly.
    """
    path = config_dir() / name
    if not path.exists():
        legacy = legacy_dir() / name
        if legacy.is_file():
            return legacy
    return path
