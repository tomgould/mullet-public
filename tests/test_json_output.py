"""Tests for JSON output mode."""

import json
import subprocess
import sys
import pytest

try:
    from mullet.cli.main import detect_local_provider, get_local_loaded_context
    # Require 512+ token context as basic sanity check
    HAS_LLM = detect_local_provider() is not None and get_local_loaded_context() >= 512
except Exception:
    HAS_LLM = False


@pytest.mark.live
@pytest.mark.skipif(not HAS_LLM, reason="No local LLM")
def test_json_output_fields():
    """--json should include all expected fields."""
    result = subprocess.run(
        [sys.executable, "-m", "mullet.cli.main", "--json", "what is 1+1?"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0

    data = json.loads(result.stdout)
    assert "completed" in data
    assert "response" in data
    assert "tool_calls" in data
    assert "tokens" in data
    assert "provider" in data
    assert "tier" in data
    assert "cost" in data
    assert "savings" in data
    assert "duration" in data
    assert data["duration"] > 0
    assert data["tokens"]["total"] > 0


@pytest.mark.live
@pytest.mark.skipif(not HAS_LLM, reason="No local LLM")
def test_json_output_pipeable():
    """JSON output should be valid JSON that can be piped."""
    result = subprocess.run(
        [sys.executable, "-m", "mullet.cli.main", "--json", "say hello"],
        capture_output=True, text=True, timeout=60,
    )
    # Should be valid JSON
    data = json.loads(result.stdout)
    assert isinstance(data, dict)
    # Stderr should be empty (no provider hint in JSON mode)
    # (verbose/router output goes to stderr, but provider hint should be suppressed)
