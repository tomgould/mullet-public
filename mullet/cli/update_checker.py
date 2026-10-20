"""Version update checker — queries PyPI for latest mullet release."""

from __future__ import annotations

import json
import sys
import urllib.request


def get_installed_version() -> str:
    """Return the installed mullet version string."""
    from mullet import __version__
    return __version__


def _version_gt(a: str, b: str) -> bool:
    """Return True if version a > b."""
    try:
        from packaging.version import Version
        return Version(a) > Version(b)
    except ImportError:
        def _parts(v: str) -> tuple[int, ...]:
            return tuple(int(x) for x in v.split(".") if x.isdigit())
        return _parts(a) > _parts(b)


def get_latest_pypi_version(timeout: float = 5.0) -> str | None:
    """Query PyPI JSON API for latest mullet release version.

    Returns version string like "1.7.1" or None on any error.
    Uses urllib.request — no extra deps.
    """
    url = "https://pypi.org/pypi/mullet/json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data["info"]["version"]
    except Exception:
        return None


def check_for_update(timeout: float = 5.0) -> dict:
    """Check if a newer version is available.

    Returns dict with keys:
      - installed: str — current installed version
      - latest: str | None — latest PyPI version (None if check failed)
      - update_available: bool — True when latest > installed (semver)
      - message: str — human-readable status line
    """
    installed = get_installed_version()
    latest = get_latest_pypi_version(timeout=timeout)

    if latest is None:
        return {
            "installed": installed,
            "latest": None,
            "update_available": False,
            "message": f"mullet {installed} installed (PyPI check failed)",
        }

    update_available = _version_gt(latest, installed)

    if update_available:
        message = (
            f"mullet {latest} available (installed: {installed})."
            f" Update: pip install -U mullet"
        )
    else:
        message = f"mullet {installed} is up to date"

    return {
        "installed": installed,
        "latest": latest,
        "update_available": update_available,
        "message": message,
    }


def maybe_print_update_notice(timeout: float = 3.0) -> None:
    """Print a one-line update notice if a newer version is available.

    Prints nothing if up to date, if PyPI is unreachable, or if latest <= installed.
    Handles all exceptions silently — never crashes the calling command.
    """
    try:
        result = check_for_update(timeout=timeout)
        if result["update_available"]:
            print(f"  mullet {result['latest']} available (installed: {result['installed']})."
                  f" Update: pip install -U mullet", file=sys.stderr)
    except Exception:
        pass
