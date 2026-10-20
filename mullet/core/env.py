"""Environment-driven settings — single source of truth."""
import os
from pathlib import Path


def get_pools_dir() -> Path:
    """Return the canonical Trawl pools directory.

    Precedence:
    1. ``TRAWL_POOLS_DIR`` env var (if set and non-empty, used verbatim)
    2. Default: ``~/.mullet/pools``
    """
    override = os.environ.get("TRAWL_POOLS_DIR")
    if override:
        return Path(override)
    return Path.home() / ".mullet" / "pools"
