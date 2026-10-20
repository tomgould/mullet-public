"""Tests for the release script."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_release_sh_exists_executable() -> None:
    path = Path("scripts/release.sh")
    assert path.is_file()
    assert os.access(path, os.X_OK)


def test_release_sh_handles_missing_arg() -> None:
    path = Path("scripts/release.sh")
    result = subprocess.run(["bash", str(path)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "usage:" in (result.stdout + result.stderr).lower()
