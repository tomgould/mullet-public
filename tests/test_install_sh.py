"""Tests for the install script and Python version cap."""
from __future__ import annotations

import os
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


def _run(*args: str, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "MULLET_REF"}
    env.update(env_extra or {})
    return subprocess.run(
        ["bash", str(INSTALL_SH), *args], capture_output=True, text=True, env=env
    )


def test_install_sh_help_exits_zero() -> None:
    result = _run("--help")
    assert result.returncode == 0
    assert "--version" in result.stdout and "--ref" in result.stdout


def test_install_sh_dry_run_version_pins_tag() -> None:
    result = _run("--version", "2.0.0", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "mullet-public.git@v2.0.0" in result.stdout
    assert "Ref: v2.0.0" in result.stdout


def test_install_sh_dry_run_default_names_default_branch() -> None:
    result = _run("--dry-run")
    assert result.returncode == 0, result.stderr
    assert ".git@" not in result.stdout
    assert "default branch" in result.stdout


def test_install_sh_dry_run_ref_and_env() -> None:
    assert ".git@feature/x-1" in _run("--ref", "feature/x-1", "--dry-run").stdout
    assert ".git@mybranch" in _run("--dry-run", env_extra={"MULLET_REF": "mybranch"}).stdout


def test_install_sh_rejects_bad_refs() -> None:
    for bad in ("-x", "a b", "a;rm", "$(id)", "a`id`", ""):
        result = _run("--ref", bad, "--dry-run")
        assert result.returncode != 0, bad
        assert "Error" in result.stderr


def test_install_sh_rejects_bad_version_and_unknown_args() -> None:
    assert _run("--version", "2.0", "--dry-run").returncode != 0
    assert _run("--version", "1.0.0;ls", "--dry-run").returncode != 0
    assert _run("--version").returncode != 0
    assert _run("--bogus").returncode != 0
