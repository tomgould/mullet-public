"""Tests for stdin pipe support."""

import subprocess
import sys
import pytest

try:
    from mullet.cli.main import detect_local_provider, get_local_loaded_context
    # Require 512+ context as basic sanity check
    HAS_LLM = detect_local_provider() is not None and get_local_loaded_context() >= 512
except Exception:
    HAS_LLM = False


@pytest.mark.live
@pytest.mark.skipif(not HAS_LLM, reason="No local LLM")
def test_pipe_stdin_as_task():
    """echo 'task' | mullet should use stdin as task."""
    result = subprocess.run(
        [sys.executable, "-m", "mullet.cli.main"],
        input="what is 2+2? just the number",
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0
    assert "4" in result.stdout


@pytest.mark.live
@pytest.mark.skipif(not HAS_LLM, reason="No local LLM")
def test_pipe_content_with_task():
    """echo 'content' | mullet 'instruction' should combine."""
    result = subprocess.run(
        [sys.executable, "-m", "mullet.cli.main", "what number does this return?"],
        input="def answer(): return 42",
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0
    assert "42" in result.stdout
