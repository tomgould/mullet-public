"""Version update checker: compares the installed mullet with the latest on GitHub.

Mullet is installed from https://github.com/tomgould/mullet-public, not from
PyPI (the "mullet" package on PyPI is an unrelated project), so the latest
version is read from ``mullet/__init__.py`` on the repository's main branch.
That is the same place the installed version comes from, so the two always
compare like with like.
"""

from __future__ import annotations

import re
import sys
import urllib.request

REPO_URL = "https://github.com/tomgould/mullet-public"
VERSION_URL = (
    "https://raw.githubusercontent.com/tomgould/mullet-public/main/mullet/__init__.py"
)
UPDATE_HINT = "git pull in your Mullet clone, then pip install -e . (or re-run scripts/install.sh)"

_VERSION_RE = re.compile(r"""^__version__\s*=\s*["']([^"']+)["']""", re.MULTILINE)


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


def get_latest_version(timeout: float = 5.0) -> str | None:
    """Return the version on the public repository's main branch.

    Returns a version string like "1.9.0", or None on any error (offline,
    repository unreachable, or no version line found). Uses urllib.request,
    so it needs no extra dependencies.
    """
    try:
        with urllib.request.urlopen(VERSION_URL, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    match = _VERSION_RE.search(text)
    return match.group(1) if match else None


def check_for_update(timeout: float = 5.0) -> dict:
    """Check if a newer version is available.

    Returns dict with keys:
      - installed: str — current installed version
      - latest: str | None — latest version on GitHub (None if check failed)
      - update_available: bool — True when latest > installed (semver)
      - message: str — human-readable status line
    """
    installed = get_installed_version()
    latest = get_latest_version(timeout=timeout)

    if latest is None:
        return {
            "installed": installed,
            "latest": None,
            "update_available": False,
            "message": f"mullet {installed} installed (GitHub version check failed)",
        }

    update_available = _version_gt(latest, installed)

    if update_available:
        message = (
            f"mullet {latest} available (installed: {installed})."
            f" Update: {UPDATE_HINT}"
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

    Prints nothing if up to date, if GitHub is unreachable, or if latest <= installed.
    Handles all exceptions silently — never crashes the calling command.
    """
    try:
        result = check_for_update(timeout=timeout)
        if result["update_available"]:
            print(f"  mullet {result['latest']} available (installed: {result['installed']})."
                  f" Update: {UPDATE_HINT}", file=sys.stderr)
    except Exception:
        pass
