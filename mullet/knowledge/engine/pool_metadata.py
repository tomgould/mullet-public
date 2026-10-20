"""Per-pool usage metadata: created_at, last_indexed_at, last_searched_at, search_count.

Stored as `.pool_metadata.json` inside each pool directory. Atomic writes via
temp-file + os.replace. All operations are best-effort — metadata failures must
NEVER block search or indexing.
"""

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

METADATA_FILENAME = ".pool_metadata.json"

_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _metadata_path(pool_dir: Path) -> Path:
    return pool_dir / METADATA_FILENAME


def read_metadata(pool_dir: Path) -> dict:
    """Read a pool's metadata file. Returns empty dict if missing or unreadable."""
    path = _metadata_path(pool_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.debug(f"Failed to read pool metadata at {path}: {e}")
        return {}


def _write_metadata(pool_dir: Path, data: dict) -> None:
    """Atomic write of metadata JSON. Best-effort — logs and swallows errors."""
    path = _metadata_path(pool_dir)
    try:
        pool_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".pool_metadata.", suffix=".tmp", dir=str(pool_dir))
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
    except Exception as e:
        logger.debug(f"Failed to write pool metadata at {path}: {e}")


def ensure_created(pool_dir: Path) -> None:
    """Stamp created_at if not already set. Idempotent."""
    with _lock:
        data = read_metadata(pool_dir)
        if "created_at" not in data:
            data["created_at"] = _now_iso()
            _write_metadata(pool_dir, data)


def mark_searched(pool_dir: Path) -> None:
    """Bump last_searched_at + search_count.

    No-op if ``pool_dir`` does not already exist: usage tracking must never
    conjure an empty pool directory as a side effect (callers may pass an
    unresolved display name).
    """
    if not pool_dir.is_dir():
        logger.debug("mark_searched: skipping, pool dir does not exist: %s", pool_dir)
        return
    with _lock:
        data = read_metadata(pool_dir)
        data["last_searched_at"] = _now_iso()
        data["search_count"] = int(data.get("search_count", 0)) + 1
        if "created_at" not in data:
            data["created_at"] = data["last_searched_at"]
        _write_metadata(pool_dir, data)


def mark_indexed(pool_dir: Path) -> None:
    """Bump last_indexed_at.

    No-op if ``pool_dir`` does not already exist (see ``mark_searched``).
    """
    if not pool_dir.is_dir():
        logger.debug("mark_indexed: skipping, pool dir does not exist: %s", pool_dir)
        return
    with _lock:
        data = read_metadata(pool_dir)
        data["last_indexed_at"] = _now_iso()
        if "created_at" not in data:
            data["created_at"] = data["last_indexed_at"]
        _write_metadata(pool_dir, data)


def delete_pool(pool_dir: Path) -> None:
    """Recursively delete a pool directory. Raises on error."""
    import shutil
    shutil.rmtree(pool_dir)
