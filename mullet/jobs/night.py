"""Night mode: lean on the local GPU while the user is away.

State lives in one JSON file in Mullet's state directory.  The file is written
atomically so a crash never leaves it half-written.  Every outside call is
injectable: the clock, the idle probe and the state path, so tests run with
fakes only.

The machine guard's temperature, VRAM and RAM limits are never relaxed by night
mode.  Only how long the runner is willing to wait for headroom changes.
"""

from __future__ import annotations

import contextlib
import getpass
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from mullet.core.paths import state_dir

logger = logging.getLogger(__name__)

STATE_NAME = "night.json"
# A user is back if there has been input in the last two minutes.
BACK_IDLE_MS = 120_000
# Night mode must have been on for ten minutes before the user can be back,
# so a quick trip to the kettle does not stop it.
MIN_ON_S = 600


def _default_path() -> Path:
    return state_dir() / STATE_NAME


def _default_who() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "cli"


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


# --- Time helpers --------------------------------------------------------------------


def default_until(now: float) -> float:
    """The next 07:30 local time, strictly after ``now``."""
    local = datetime.fromtimestamp(now)
    target = local.replace(hour=7, minute=30, second=0, microsecond=0)
    if target <= local:
        target += timedelta(days=1)
    return target.timestamp()


def parse_until(text: str, now: float) -> float:
    """The next occurrence of ``HH:MM`` local time, strictly after ``now``.

    Raises ``ValueError`` when ``text`` is not a valid ``HH:MM`` value.
    """
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"until must be HH:MM, got {text!r}")
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"until must be HH:MM, got {text!r}") from exc
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ValueError(f"until must be HH:MM, got {text!r}")
    local = datetime.fromtimestamp(now)
    target = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= local:
        target += timedelta(days=1)
    return target.timestamp()


# --- State ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NightState:
    """The public view of night mode state."""

    on: bool = False
    until: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class NightSettings:
    """Adjustments to the ladder while night mode is on.

    The machine guard's temperature, VRAM and RAM limits are never relaxed.
    """

    cheap_minutes_factor: float = 2.0
    extra_below_claude: int = 2
    local_headroom_wait_s: float = 1800.0


SETTINGS = NightSettings()


def read_state(path: Path | None = None) -> NightState:
    """Read the state file.  A missing or corrupt file reads as off."""
    target = path or _default_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return NightState()
    if not isinstance(data, dict):
        return NightState()
    return NightState(
        on=bool(data.get("on")),
        until=_as_float(data.get("until")),
        reason=data.get("off_reason"),
    )


def _write_state(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` to ``path`` atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def switch_on(
    until: float | None = None,
    who: str | None = None,
    now: float | None = None,
    path: Path | None = None,
) -> None:
    """Switch night mode on, writing a new state file."""
    if now is None:
        now = time.time()
    if until is None:
        until = default_until(now)
    if who is None:
        who = _default_who()
    data = {
        "on": True,
        "since": now,
        "until": until,
        "who": who,
        "off_reason": None,
    }
    _write_state(path or _default_path(), data)


def switch_off(reason: str, now: float | None = None, path: Path | None = None) -> None:
    """Switch night mode off and record why."""
    target = path or _default_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    data["on"] = False
    data["off_reason"] = reason
    _write_state(target, data)


def is_on(now: float | None = None, path: Path | None = None) -> bool:
    """True only while night mode is on and before its ``until`` time."""
    if now is None:
        now = time.time()
    state = read_state(path)
    if not state.on:
        return False
    until = state.until
    if until is None or now >= until:
        ended = datetime.fromtimestamp(until if until is not None else now)
        switch_off(ended.strftime("ended at %H:%M"), now=now, path=path)
        return False
    return True


# --- User presence -------------------------------------------------------------------


def _probe_idle_ms(
    which: Callable[[str], str | None],
    run: Callable[..., Any],
) -> int | None:
    """The user's idle time in milliseconds, or None when it cannot be found."""
    xprintidle = which("xprintidle")
    if xprintidle:
        try:
            result = run(
                [xprintidle],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception:
            pass

    if not which("loginctl"):
        return None

    session = os.environ.get("XDG_SESSION_ID")
    if not session:
        try:
            user = getpass.getuser()
            result = run(
                ["loginctl", "list-sessions", "--no-legend"],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    parts = line.split()
                    if len(parts) >= 4 and parts[2] == user:
                        session = parts[0]
                        break
        except Exception:
            pass

    if not session:
        return None

    try:
        result = run(
            [
                "loginctl",
                "show-session",
                session,
                "-p",
                "IdleHint",
                "-p",
                "Type",
                "-p",
                "LockedHint",
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if result.returncode != 0:
            return None
        props: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                props[key.strip()] = value.strip()
        if props.get("IdleHint") != "no":
            return None
        if props.get("Type") not in ("x11", "wayland"):
            return None
        if props.get("LockedHint") != "no":
            return None
        # The session is active, graphical and unlocked: report no idle time.
        return 0
    except Exception:
        return None


def user_is_back(
    now: float | None = None,
    path: Path | None = None,
    idle_ms: int | None = None,
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., Any] = subprocess.run,
) -> bool:
    """True when the user has been active recently and night mode is old enough.

    This function never raises.  If the idle time cannot be discovered, the
    user is treated as not back.
    """
    try:
        if now is None:
            now = time.time()
        target = path or _default_path()
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(data, dict) or not data.get("on"):
            return False
        since = _as_float(data.get("since"))
        if since is None or now - since < MIN_ON_S:
            return False

        idle = idle_ms if idle_ms is not None else _probe_idle_ms(which, run)
        if idle is None:
            return False
        return idle < BACK_IDLE_MS
    except Exception:
        return False


# --- Check ---------------------------------------------------------------------------


def check(
    now: float | None = None,
    path: Path | None = None,
    back: bool | None = None,
) -> NightState:
    """Update and return the current night state.

    If night mode is on and the user is back, switch it off with the reason
    "you're back".
    """
    if now is None:
        now = time.time()
    if not is_on(now, path):
        return read_state(path)
    is_back = back if back is not None else user_is_back(now, path)
    if is_back:
        switch_off("you're back", now=now, path=path)
    return read_state(path)
