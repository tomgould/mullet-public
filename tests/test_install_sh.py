"""Tests for the install script and Python version cap."""
from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = ROOT / "scripts" / "install.sh"
PYPROJECT = ROOT / "pyproject.toml"


def test_install_sh_passes_bash_syntax_check() -> None:
    result = subprocess.run(["bash", "-n", str(INSTALL_SH)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_install_sh_only_prints_after_finding_installer() -> None:
    script = INSTALL_SH.read_text()
    # The "Installing ..." line must not appear before the installer check.
    before_check = script.split("if command -v uv >/dev/null 2>&1; then", 1)[0]
    assert 'echo "Installing Mullet from $SPEC ..."' not in before_check


def test_install_sh_pins_uv_python_to_3_12() -> None:
    script = INSTALL_SH.read_text()
    assert "--python 3.12" in script


def test_pyproject_requires_python_below_3_13() -> None:
    for line in PYPROJECT.read_text().splitlines():
        if line.startswith("requires-python"):
            assert line == 'requires-python = ">=3.10,<3.13"'
            return
    raise AssertionError("requires-python not found in pyproject.toml")
