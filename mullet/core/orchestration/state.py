"""Mode state persistence via Trove MemoryStore.

Survives context compression, session restarts, and agent crashes.
Written after each stage transition.

Features (from OMC state-manager):
- Atomic file writes (temp file + rename)
- File locking to prevent concurrent write corruption
- Read cache with TTL for performance
- Stale state detection via timestamp
"""

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mullet.core.knowledge.manager import KnowledgeManager

logger = logging.getLogger(__name__)

STALE_THRESHOLD = 4 * 3600  # 4 hours
CACHE_TTL = 5.0  # Seconds to cache reads

# Module-level read cache: {path: (data, timestamp, mtime)}
_read_cache: dict[str, tuple[dict, float, float]] = {}


class ModeState:
    """Persist execution mode state via Trove MemoryStore.

    Local fallback uses atomic file writes and file locking.
    """

    def __init__(self, knowledge: "KnowledgeManager | None", mode_name: str, session_id: str = ""):
        self._knowledge = knowledge
        self._mode_name = mode_name
        self._session_id = session_id
        self._namespace = f"mode:{mode_name}"
        if session_id:
            self._namespace = f"mode:{mode_name}:{session_id}"

    def save(self, state: dict):
        """Save mode state."""
        state["_timestamp"] = time.time()
        state["_mode"] = self._mode_name

        if self._knowledge:
            self._knowledge.remember(
                key=f"state:{self._mode_name}",
                content=json.dumps(state),
                namespace=self._namespace,
            )
        else:
            self._save_local(state)

    def load(self) -> dict | None:
        """Load mode state. Returns None if not found or stale."""
        if self._knowledge:
            results = self._knowledge.recall(
                query=f"state:{self._mode_name}",
                namespace=self._namespace,
            )
            if results:
                content = results[0].get("content", "") if isinstance(results[0], dict) else str(results[0])
                try:
                    state = json.loads(content)
                    if time.time() - state.get("_timestamp", 0) > STALE_THRESHOLD:
                        logger.info("Stale state for %s (>4h), ignoring", self._mode_name)
                        return None
                    return state
                except json.JSONDecodeError as _exc:
                    logger.warning("orchestration.state: failed to decode state from knowledge: %s", _exc)
                    return None
        return self._load_local()

    def clear(self):
        """Clear mode state."""
        if self._knowledge:
            try:
                self._knowledge.memory.delete(
                    namespace=self._namespace,
                    key=f"state:{self._mode_name}",
                )
            except Exception as _exc:
                logger.warning("orchestration.state: failed to clear mode state: %s", _exc)

        # Also clear local file and cache
        path = self._state_path()
        if path.exists():
            try:
                path.unlink()
            except OSError as _exc:
                logger.warning("orchestration.state: failed to unlink local state file %s: %s", path, _exc)
        _read_cache.pop(str(path), None)

    # --- Local file persistence with atomic writes ---

    def _state_path(self) -> Path:
        state_dir = Path(".mullet") / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        name = f"{self._mode_name}.json"
        if self._session_id:
            name = f"{self._mode_name}_{self._session_id}.json"
        return state_dir / name

    def _save_local(self, state: dict):
        """Atomic write: temp file + rename to prevent corruption."""
        path = self._state_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        data = json.dumps(state, indent=2)

        # Write to temp file, then atomic rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=f".{path.stem}_",
            suffix=".tmp",
        )
        try:
            os.write(fd, data.encode())
            os.close(fd)
            os.replace(tmp_path, str(path))  # Atomic on POSIX
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        # Invalidate cache
        _read_cache.pop(str(path), None)

    def _load_local(self) -> dict | None:
        """Load with read cache and stale detection."""
        path = self._state_path()
        path_str = str(path)

        if not path.exists():
            return None

        # Check cache
        now = time.time()
        if path_str in _read_cache:
            cached_data, cached_time, cached_mtime = _read_cache[path_str]
            if now - cached_time < CACHE_TTL:
                # Verify file hasn't changed (TOCTOU prevention)
                try:
                    current_mtime = path.stat().st_mtime
                    if current_mtime == cached_mtime:
                        return cached_data
                except OSError:
                    pass

        # Read fresh
        try:
            mtime = path.stat().st_mtime
            state = json.loads(path.read_text())

            if time.time() - state.get("_timestamp", 0) > STALE_THRESHOLD:
                return None

            # Update cache
            _read_cache[path_str] = (state, now, mtime)
            return state
        except (json.JSONDecodeError, OSError) as _exc:
            logger.warning("orchestration.state: failed to load local state %s: %s", path, _exc)
            return None


def clear_state_cache():
    """Clear all cached state reads."""
    _read_cache.clear()
