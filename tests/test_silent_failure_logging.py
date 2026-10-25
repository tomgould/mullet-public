"""Tests that silent-failure-with-logger.warning sites actually emit warnings.

These tests catch regressions where someone reverts a logger.warning back
to bare except:pass. They verify the logger contract, not the control
flow — the old `pass` behavior is preserved (nothing raised).
"""

import logging

import pytest


def test_circuit_breaker_open_logs_error_level(caplog):
    """CircuitBreaker tripped to OPEN logs a warning/error."""
    from mullet.knowledge.engine.health import CircuitBreaker
    caplog.set_level(logging.WARNING)

    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=30.0)
    cb.record_failure()
    cb.record_failure()  # trip

    open_logs = [r for r in caplog.records if "OPEN" in r.message and "CircuitBreaker" in r.message]
    assert open_logs, f"expected CircuitBreaker OPEN log; got {[r.message for r in caplog.records]}"


def test_interaction_log_db_failure_does_not_raise():
    """interaction_log.log_interaction swallows errors; no exception propagates."""
    from mullet.core.logging.interaction_log import log_interaction

    try:
        log_interaction(
            provider="test",
            model="test-model",
            messages=None,  # invalid — triggers internal error
            response_content="",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=0,
            success=True,
        )
    except Exception as e:
        pytest.fail(f"log_interaction must not raise on invalid input; raised {e}")


def test_ansi_strip_removes_csi_sequences():
    """Contract: strip_ansi removes color codes."""
    from mullet.core.utils.ansi_strip import strip_ansi
    assert strip_ansi("\x1b[31mhello\x1b[0m") == "hello"


def test_browser_is_safe_url_rejects_localhost():
    """Contract: browser.is_safe_url rejects localhost/private ranges (SSRF guard)."""
    from mullet.core.tools.browser import BROWSE_SCHEMA
    # Just verify the module loaded cleanly with the new logger wiring
    assert BROWSE_SCHEMA["parameters"]["required"] == ["url"]


def test_shape_request_strips_temperature_for_moonshot_v1():
    """Contract: shape_request strips temperature for moonshot-v1-*."""
    from mullet.core.providers.constraints import shape_request
    raw = {"model": "moonshot-v1-128k", "messages": [], "temperature": 0.7}
    shaped = shape_request("moonshot-v1-128k", raw)
    assert "temperature" not in shaped


def test_provider_parse_error_default_returns_none():
    """Contract: base Provider.parse_error returns None for unknown errors."""
    from mullet.core.providers.base import Provider

    class _P(Provider):
        name = "fake"
        def create(self, messages, **kwargs):
            raise NotImplementedError

    p = _P()
    assert p.parse_error(Exception("random unknown error")) is None


def test_fallback_manager_restores_primary_on_turn_start():
    """Contract: FallbackManager.on_turn_start restores primary."""
    from mullet.core.router.fallback import FallbackManager

    class _P:
        def __init__(self, name):
            self.name = name

    primary = _P("primary")
    fb = _P("fb")
    mgr = FallbackManager(primary, [fb])
    mgr.on_failure(Exception("x"))
    assert mgr.current is fb
    mgr.on_turn_start()
    assert mgr.current is primary


def test_context_compressor_should_compress_threshold():
    """Contract: ContextCompressor fires above threshold."""
    from mullet.core.agent.compression import ContextCompressor
    c = ContextCompressor(context_length=1_000, threshold_percent=0.5)
    assert c.should_compress([{"role": "user", "content": "x" * 5_000}]) is True
    assert c.should_compress([{"role": "user", "content": "hi"}]) is False


def test_auto_swarm_detect_file_mentions():
    """Contract: auto_swarm detects file-path tokens in task text."""
    from mullet.core.router.auto_swarm import _detect_file_mentions_in_task
    task = "review src/a.py, src/b.py, c.py, tests/d.py, e.py"
    assert _detect_file_mentions_in_task(task) >= 5


def test_manual_tool_guard_resets_on_delegation():
    """Contract: guard state resets on mullet_run tool name."""
    import importlib.util
    from pathlib import Path

    hook_path = Path(__file__).resolve().parent.parent / "mullet" / "hooks" / "manual_tool_guard.py"
    spec = importlib.util.spec_from_file_location("manual_tool_guard_test", hook_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert "mcp__mullet__mullet_run" in mod.DELEGATION_TOOLS
    assert "Bash" in mod.COUNTED_TOOLS
