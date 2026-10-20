"""Tests for the explain command."""

import subprocess
import sys


def test_explain_simple():
    result = subprocess.run(
        [sys.executable, "-m", "mullet.cli.main", "explain", "list files"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "LOW" in result.stdout
    assert "Simple message: Yes" in result.stdout


def test_explain_complex():
    result = subprocess.run(
        [sys.executable, "-m", "mullet.cli.main", "explain",
         "refactor the authentication system for OAuth2 with JWT in production"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "ADVANCED" in result.stdout
    assert "architecture" in result.stdout.lower() or "risk" in result.stdout.lower()


def test_explain_medium():
    result = subprocess.run(
        [sys.executable, "-m", "mullet.cli.main", "explain",
         "implement input validation for the user registration form and add tests"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "score:" in result.stdout.lower() or "Complexity" in result.stdout


def test_explain_no_task():
    result = subprocess.run(
        [sys.executable, "-m", "mullet.cli.main", "explain"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0
    assert "Usage" in result.stderr
