"""Tests for the doctor command."""

import subprocess
import sys


def test_doctor_runs():
    """mullet doctor should run without errors."""
    result = subprocess.run(
        [sys.executable, "-m", "mullet.cli.main", "doctor"],
        capture_output=True, text=True, timeout=15,
        cwd="/tmp",
    )
    assert result.returncode == 0
    assert "System Health Check" in result.stdout
    assert "Python" in result.stdout
    assert "Skills" in result.stdout
    assert "Tools" in result.stdout


def test_doctor_shows_checks():
    """Doctor should show pass/fail counts."""
    result = subprocess.run(
        [sys.executable, "-m", "mullet.cli.main", "doctor"],
        capture_output=True, text=True, timeout=15,
        cwd="/tmp",
    )
    assert "checks passed" in result.stdout


def test_doctor_shows_version_drift_row():
    """mullet doctor should include a version drift / git-sync check row."""
    result = subprocess.run(
        [sys.executable, "-m", "mullet.cli.main", "doctor"],
        capture_output=True, text=True, timeout=15,
        cwd="/tmp",
    )
    assert result.returncode == 0
    # The row label must appear; the status (ok/warn) varies by network availability
    assert "Version" in result.stdout or "version" in result.stdout


def test_doctor_shows_pool_coverage_row():
    """mullet doctor should show a pool coverage summary row with healthy/total counts."""
    result = subprocess.run(
        [sys.executable, "-m", "mullet.cli.main", "doctor"],
        capture_output=True, text=True, timeout=20,
        cwd="/tmp",
    )
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
    assert "autopilot" in result.stdout
    assert "ralph" in result.stdout
