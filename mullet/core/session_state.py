"""Session-scoped provider override state, persisted to .mullet/session.json."""

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

_DEFAULT_TTL_HOURS = 5
_STATE_FILE = "session.json"


def _state_path(state_dir: Optional[Path]) -> Path:
    """Resolve the session.json path, creating the directory if needed."""
    d = state_dir if state_dir is not None else Path.cwd() / ".mullet"
    d.mkdir(parents=True, exist_ok=True)
    return d / _STATE_FILE


def _load_raw(state_dir: Optional[Path]) -> dict:
    """Load the raw JSON dict from disk, returning {} on missing or malformed file."""
    path = _state_path(state_dir)
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_raw(data: dict, state_dir: Optional[Path]) -> None:
    """Atomically write *data* to session.json via a sibling tmp file."""
    path = _state_path(state_dir)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".session_tmp_")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def set_override(
    provider: str,
    model: Optional[str] = None,
    expires_at_iso: Optional[str] = None,
    state_dir: Optional[Path] = None,
) -> None:
    """Write (or replace) the active provider override in session.json."""
    now = datetime.now(timezone.utc)
    expires = (
        datetime.fromisoformat(expires_at_iso)
        if expires_at_iso
        else now + timedelta(hours=_DEFAULT_TTL_HOURS)
    )
    existing = _load_raw(state_dir)
    existing["override"] = {
        "provider": provider,
        "model": model,
        "activated_at": now.isoformat(),
        "expires_at": expires.isoformat(),
    }
    _save_raw(existing, state_dir)


def get_override(state_dir: Optional[Path] = None) -> Optional[dict]:
    """Return the active override dict, or None if absent, malformed, or expired."""
    raw = _load_raw(state_dir)
    override = raw.get("override")
    if not isinstance(override, dict):
        return None
    expires_raw = override.get("expires_at")
    if expires_raw:
        try:
            expires = datetime.fromisoformat(expires_raw)
            if expires < datetime.now(timezone.utc):
                clear_override(state_dir)
                return None
        except ValueError:
            clear_override(state_dir)
            return None
    return override


def clear_override(state_dir: Optional[Path] = None) -> None:
    """Remove the override key from session.json, preserving other keys."""
    raw = _load_raw(state_dir)
    if "override" in raw:
        del raw["override"]
        _save_raw(raw, state_dir)


def is_expired(state_dir: Optional[Path] = None) -> bool:
    """Return True if the stored override exists but has passed its expiry time."""
    raw = _load_raw(state_dir)
    override = raw.get("override")
    if not isinstance(override, dict):
        return False
    expires_raw = override.get("expires_at")
    if not expires_raw:
        return False
    try:
        return datetime.fromisoformat(expires_raw) < datetime.now(timezone.utc)
    except ValueError:
        return False
