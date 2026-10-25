"""Tests for the doctor command."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _doctor(timeout: int = 30) -> subprocess.CompletedProcess:
    """Run `mullet doctor` from this checkout; use with the fake_claude fixture."""
    return subprocess.run(
        [sys.executable, "-m", "mullet.cli.main", "doctor"],
        capture_output=True, text=True, timeout=timeout,
        cwd="/tmp", env={**os.environ, "PYTHONPATH": str(ROOT)},
    )


def test_doctor_runs(fake_claude):
    """mullet doctor should run without errors."""
    result = _doctor()
    assert result.returncode == 0
    assert "System Health Check" in result.stdout
    assert "Python" in result.stdout
    assert "Skills" in result.stdout
    assert "Tools" in result.stdout


def test_doctor_shows_checks(fake_claude):
    """Doctor should show pass/fail counts, and check the Claude Code CLI."""
    result = _doctor()
    assert "checks passed" in result.stdout
    assert "9.9.9 (Claude Code)" in result.stdout
    assert "Version drift" not in result.stdout


def test_doctor_shows_pool_coverage_row(fake_claude):
    """mullet doctor should show a pool coverage summary row with healthy/total counts."""
    result = _doctor()
    assert result.returncode == 0
    # Must have a row that mentions pool counts, e.g. "Pool coverage  3/5 healthy"
    assert "Pool coverage" in result.stdout or "pools healthy" in result.stdout or "pools configured" in result.stdout


def test_version_command():
    """mullet --version should print version."""
    result = subprocess.run(
        [sys.executable, "-m", "mullet.cli.main", "--version"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    import mullet
    assert mullet.__version__ in result.stdout


def test_tools_command():
    """mullet --tools should list tools."""
    result = subprocess.run(
        [sys.executable, "-m", "mullet.cli.main", "--tools"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "terminal" in result.stdout
    assert "web_search" in result.stdout


def test_skills_command():
    """mullet --skills should list skills."""
    result = subprocess.run(
        [sys.executable, "-m", "mullet.cli.main", "--skills"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "brainstorm" in result.stdout
    assert "mullet-job" in result.stdout
