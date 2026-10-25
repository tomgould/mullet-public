"""End-to-end CLI tests via subprocess.

These test the actual mullet command as users would invoke it.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run_mullet(*args, timeout=15):
    """Run mullet as subprocess, from this checkout."""
    return subprocess.run(
        [sys.executable, "-m", "mullet.cli.main"] + list(args),
        capture_output=True, text=True, timeout=timeout,
        cwd="/tmp", env={**os.environ, "PYTHONPATH": str(ROOT)},
    )


def test_help():
    result = _run_mullet("--help")
    assert result.returncode == 0
    assert "Cloud in the front" in result.stdout
    assert "mullet job --help" in result.stdout


def test_version():
    result = _run_mullet("--version")
    assert result.returncode == 0
    assert "mullet" in result.stdout


def test_tools_list():
    result = _run_mullet("--tools")
    assert result.returncode == 0
    assert "terminal" in result.stdout
    assert "browse" in result.stdout
    assert "delegate" in result.stdout
    lines = [ln for ln in result.stdout.split("\n") if ln.strip().startswith("- ")]
    assert len(lines) >= 13


def test_skills_list():
    result = _run_mullet("--skills")
    assert result.returncode == 0
    assert "/brainstorm" in result.stdout
    assert "/mullet-job" in result.stdout
    assert "/code-review" in result.stdout


def test_doctor(fake_claude):
    result = _run_mullet("doctor")
    assert result.returncode == 0
    assert "System Health Check" in result.stdout
    assert "checks passed" in result.stdout


def _init_env(tmp_path):
    """Env for `mullet init` runs that isolates side effects to tmp_path.

    Without TRAWL_POOLS_DIR, the init wizard creates a pool under the user's
    real ~/.mullet/pools (slug derived from the test's tmp dir name), which
    leaks artifacts named test_init_creates_config0, test_init_no_overwrite0
    after every test run.
    """
    import os
    env = os.environ.copy()
    env["TRAWL_POOLS_DIR"] = str(tmp_path / "pools")
    return env


def test_init_creates_config(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "mullet.cli.main", "init"],
        capture_output=True, text=True, timeout=15,
        cwd=str(tmp_path),
        env=_init_env(tmp_path),
    )
    assert result.returncode == 0
    assert (tmp_path / "mullet.yaml").exists()


def test_init_no_overwrite(tmp_path):
    (tmp_path / "mullet.yaml").write_text("existing: true")
    result = subprocess.run(
        [sys.executable, "-m", "mullet.cli.main", "init"],
        capture_output=True, text=True, timeout=15,
        cwd=str(tmp_path),
        env=_init_env(tmp_path),
    )
    # The init wizard prints "mullet.yaml exists, skipping" when refusing to
    # overwrite; the key behavioural assertion is that the existing content
    # is preserved verbatim.
    assert "exists" in result.stdout and "skipping" in result.stdout
    assert (tmp_path / "mullet.yaml").read_text() == "existing: true"


def test_explain_simple():
    result = _run_mullet("explain", "list files")
    assert result.returncode == 0
    assert "LOW" in result.stdout
    assert "score" in result.stdout.lower()


def test_explain_complex():
    result = _run_mullet("explain", "refactor auth for OAuth2 with JWT in production")
    assert result.returncode == 0
    assert "ADVANCED" in result.stdout


def test_explain_no_task():
    result = _run_mullet("explain")
    assert result.returncode != 0


def test_completion_bash():
    result = _run_mullet("completion", "bash")
    assert result.returncode == 0
    assert "complete" in result.stdout


def test_completion_zsh():
    result = _run_mullet("completion", "zsh")
    assert result.returncode == 0
    assert "compdef" in result.stdout


def test_history():
    result = _run_mullet("history")
    assert result.returncode == 0


def test_config():
    result = _run_mullet("config")
    assert result.returncode == 0


def test_stats():
    result = _run_mullet("stats")
    assert result.returncode == 0


def test_tools_shows_git():
    """New git tool should appear in --tools output."""
    result = _run_mullet("--tools")
    assert "git" in result.stdout
    assert "run_tests" in result.stdout


def test_doctor_shows_git():
    """Doctor should check git version."""
    result = _run_mullet("doctor")
    assert "Git" in result.stdout


def test_cost():
    """Cost command should show pricing tiers."""
    result = _run_mullet("cost")
    assert result.returncode == 0
    assert "LOCAL" in result.stdout
    # Tier names: pre-refresh it was "MEDIUM (cheap)" / "HIGH (premium)";
    # post-refresh (eef19fe) it's "CHEAP (sub-$0.01/1K)" / "PREMIUM".
    # Accept either form so the test survives future tier-label tweaks.
    assert ("MEDIUM" in result.stdout) or ("CHEAP" in result.stdout)
    assert ("HIGH" in result.stdout) or ("PREMIUM" in result.stdout)
    assert "$0.0000" in result.stdout  # Local is free


def test_review_no_git(tmp_path):
    """Review with no git repo should handle gracefully."""
    result = subprocess.run(
        [sys.executable, "-m", "mullet.cli.main", "review"],
        capture_output=True, text=True, timeout=15, cwd=str(tmp_path),
    )
    # No git repo — should exit with "No changes" or error
    assert result.returncode in (0, 1)
