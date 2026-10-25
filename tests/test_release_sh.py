"""Tests for scripts/release.sh argument handling (nothing is modified)."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RELEASE_SH = ROOT / "scripts" / "release.sh"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(RELEASE_SH), *args], capture_output=True, text=True, cwd=ROOT
    )


def test_release_sh_uses_strict_mode() -> None:
    assert "set -euo pipefail" in RELEASE_SH.read_text()


def test_release_sh_dry_run_accepts_good_versions() -> None:
    for good in ("2.0.1", "10.20.30", "2.1.0-rc1", "2.1.0-beta.2"):
        result = _run("--dry-run", good)
        assert result.returncode == 0, (good, result.stderr)
        assert f"v{good}" in result.stdout


def test_release_sh_rejects_bad_versions() -> None:
    bad_versions = ("", "2.0", "v2.0.0", "2.0.0;ls", '2.0.0"/x/', "2.0.0/../x", "2.0.0-", "a.b.c", "2.0.0 -rc")
    for bad in bad_versions:
        result = _run("--dry-run", bad)
        assert result.returncode != 0, bad
        assert result.stdout == ""


def test_release_sh_needs_a_version_and_one_only() -> None:
    assert _run().returncode != 0
    assert _run("--dry-run").returncode != 0
    assert _run("--dry-run", "1.0.0", "2.0.0").returncode != 0


def test_release_sh_rejects_bad_version_without_dry_run() -> None:
    before = (ROOT / "pyproject.toml").read_text()
    result = _run("1.0.0;touch /nonexistent-marker")
    assert result.returncode != 0
    assert (ROOT / "pyproject.toml").read_text() == before
