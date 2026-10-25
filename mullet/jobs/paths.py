"""Where Mullet jobs keep their clones, ledger, logs and project profiles."""

from __future__ import annotations

import os
from pathlib import Path

from mullet.core.paths import config_dir, xdg_data_dir


def jobs_home() -> Path:
    """Root directory for all job data. `MULLET_JOBS_DIR` overrides it.

    Always under the XDG data directory, even while an older install still
    uses ~/.mullet, since jobs never lived there.
    """
    override = os.environ.get("MULLET_JOBS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (xdg_data_dir() / "jobs").resolve()


def profiles_dir() -> Path:
    """Directory of per-repo profiles. `MULLET_PROFILES_DIR` overrides it."""
    override = os.environ.get("MULLET_PROFILES_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (config_dir() / "projects").resolve()


def job_dir(job_id: str) -> Path:
    """Directory holding one job's clone, git dir and logs."""
    return ensure_inside(jobs_home() / job_id, jobs_home())


def ensure_inside(path: Path, root: Path) -> Path:
    """Return `path` resolved, or raise if it is not strictly inside `root`.

    Every destructive operation on job data goes through this, so a bad job
    id or a planted symlink can never point a reset or delete at the user's
    own files.
    """
    resolved = Path(path).resolve()
    root_resolved = Path(root).resolve()
    if resolved == root_resolved or root_resolved not in resolved.parents:
        raise ValueError(f"{resolved} is not inside {root_resolved}")
    return resolved
