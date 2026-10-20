"""End-to-end smoke test for the HarnessEngine (Phase 1)."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mullet.cli.harness_cli import cmd_shell
from mullet.harness.engine import HarnessEngine

pytestmark = [pytest.mark.harness]


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``live`` marker if it is not already in pytest config."""
    markers = config.getini("markers")
    if isinstance(markers, str):
        markers = markers.splitlines()
    if not any(m.startswith("live:") for m in markers):
        config.addinivalue_line("markers", "live: tests against real APIs")


@pytest.mark.harness
@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("KIMI_API_KEY") and not os.environ.get("MOONSHOT_API_KEY"),
    reason="Kimi API key not set",
)
def test_harness_engine_live_smoke() -> None:
    """Spawn ``mullet shell`` against real Kimi and verify deterministic response."""
    # Live test invocation:
    #   pytest tests/harness/test_engine_smoke.py -m live -q
    # Default invocation skips it:
    #   pytest tests/harness/ -q
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mullet",
            "shell",
            "--provider",
            "kimi",
            "Reply with exactly: HARNESS_OK",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "harness_ok" in result.stdout.lower()
    assert "Traceback" not in result.stderr


def test_one_shot_text_output_smoke_with_mock(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """Monkey-patch ``HarnessEngine.run`` and verify ``cmd_shell`` prints response."""
    fixed_result = {
        "session_id": "sess-smoke-123",
        "response": "HARNESS_OK",
        "stop_reason": "stop",
        "input_tokens": 1,
        "output_tokens": 1,
    }
    monkeypatch.setattr(HarnessEngine, "run", lambda self, prompt, **kwargs: fixed_result)

    args = SimpleNamespace(
        prompt="Reply with exactly: HARNESS_OK",
        session_id=None,
        provider="kimi",
        model=None,
        output_format="text",
        list_sessions=False,
    )
    rc = cmd_shell(args)
    captured = capsys.readouterr()
    assert rc == 0
    assert "HARNESS_OK" in captured.out
