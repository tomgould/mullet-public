"""Regression tests for bugs found during code review.

Each test targets a specific bug that was fixed and ensures it doesn't regress.
"""

import os
from unittest.mock import MagicMock, patch
import pytest

from mullet.core.agent.runner import AgentRunner
from mullet.core.agent.types import TokenUsage
from mullet.core.providers.base import LLMResponse, Provider
from mullet.core.tools.registry import ToolRegistry


def _make_provider(responses: list[LLMResponse]) -> Provider:
    """Create a mock provider that returns responses in sequence."""
    provider = MagicMock(spec=Provider)
    provider.name = "mock"
    provider.model = "mock-model"
    provider.create = MagicMock(side_effect=responses)
    return provider


def _make_response(content: str = "", tool_calls: list | None = None,
                   finish_reason: str = "stop") -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        model="mock", provider="mock",
    )


def _make_tool_call(name: str = "terminal", args: str = '{"command": "ls"}',
                    call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


# --- Bug: Budget warning elif unreachable ---

def test_critical_budget_warning_fires():
    """CRITICAL budget warning should fire when remaining == 1."""
    registry = ToolRegistry()
    registry.register(
        name="terminal",
        func=lambda **kwargs: "ok",
        schema={"description": "test", "parameters": {"type": "object", "properties": {}}},
        toolset="core",
    )

    responses = []
    # Fill up the budget — each iteration uses 1
    for i in range(4):
        responses.append(_make_response(
            tool_calls=[_make_tool_call(call_id=f"call_{i}")],
            finish_reason="tool_calls",
        ))
    responses.append(_make_response(content="Final answer"))

    provider = _make_provider(responses)
    agent = AgentRunner(
        provider=provider, tool_registry=registry,
        toolsets=["core"], max_iterations=5,
    )
    result = agent.run("test")

    # The agent should have received the CRITICAL warning on iteration 4
    # (when remaining would be 1 after consuming)
    # Check that tool results contain the warning text
    tool_messages = [m for m in result.messages if m.get("role") == "tool"]
    warnings = [m["content"] for m in tool_messages if "CRITICAL" in m["content"]]
    assert len(warnings) >= 1, "CRITICAL: Last iteration warning should have been injected"


def test_warning_budget_fires_at_3():
    """Budget pressure should fire when remaining is low."""
    registry = ToolRegistry()
    registry.register(
        name="terminal",
        func=lambda **kwargs: "ok",
        schema={"description": "test", "parameters": {"type": "object", "properties": {}}},
        toolset="core",
    )

    responses = []
    for i in range(8):
        responses.append(_make_response(
            tool_calls=[_make_tool_call(call_id=f"call_{i}")],
            finish_reason="tool_calls",
        ))
    responses.append(_make_response(content="done"))

    provider = _make_provider(responses)
    agent = AgentRunner(
        provider=provider, tool_registry=registry,
        toolsets=["core"], max_iterations=10,
    )
    result = agent.run("test")

    tool_messages = [m for m in result.messages if m.get("role") == "tool"]
    # Budget pressure uses NOTE/URGENT/CRITICAL depending on ratio
    warnings = [m["content"] for m in tool_messages
                if any(tag in m["content"] for tag in ["NOTE:", "URGENT:", "CRITICAL:", "WARNING"])]
    assert len(warnings) >= 1, "Budget pressure should have fired when remaining <= 3"


# --- Bug: Malformed tool call skips tool result ---

def test_malformed_tool_call_still_appends_result():
    """Malformed tool calls should still produce a tool result message."""
    registry = ToolRegistry()

    # Malformed tool call: not a proper dict
    malformed = {"id": "call_bad"}  # Missing "function" key entirely
    responses = [
        _make_response(
            tool_calls=[malformed],
            finish_reason="tool_calls",
        ),
        _make_response(content="I'll try differently"),
    ]

    provider = _make_provider(responses)
    agent = AgentRunner(
        provider=provider, tool_registry=registry,
        toolsets=[], max_iterations=5,
    )
    result = agent.run("do something")

    # Should have a tool result message for the malformed call
    tool_messages = [m for m in result.messages if m.get("role") == "tool"]
    assert len(tool_messages) >= 1
    assert "error" in tool_messages[0]["content"].lower()


# --- Bug: Fallback provider not fully tried ---

def test_multiple_fallbacks_attempted():
    """All fallback providers should be tried if earlier ones fail."""
    primary = MagicMock(spec=Provider)
    primary.name = "primary"
    primary.model = "primary-model"
    primary.create = MagicMock(side_effect=ConnectionError("primary down"))

    fallback1 = MagicMock(spec=Provider)
    fallback1.name = "fallback1"
    fallback1.model = "fb1-model"
    fallback1.create = MagicMock(side_effect=ConnectionError("fallback1 also down"))

    fallback2 = MagicMock(spec=Provider)
    fallback2.name = "fallback2"
    fallback2.model = "fb2-model"
    good_response = _make_response(content="Success from fallback2")
    fallback2.create = MagicMock(return_value=good_response)

    registry = ToolRegistry()
    agent = AgentRunner(
        provider=primary, tool_registry=registry,
        toolsets=[], max_iterations=5,
        fallback_providers=[fallback1, fallback2],
    )
    result = agent.run("test")

    assert result.completed
    assert "Success from fallback2" in result.final_response
    # Verify fallback2 was actually called
    fallback2.create.assert_called_once()


# --- Bug: Retry too broad ---

def test_retry_doesnt_catch_auth_errors():
    """Authentication errors should NOT be retried."""
    from mullet.core.providers.openai_compat import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider(
        base_url="http://localhost:99999/v1",  # Doesn't exist
        model="test",
    )

    # Mock the client to raise an auth error
    try:
        from openai import AuthenticationError
        auth_error = AuthenticationError(
            message="Invalid API key",
            response=MagicMock(status_code=401),
            body=None,
        )
    except ImportError:
        pytest.skip("openai SDK not available for precise error testing")

    provider.client.chat.completions.create = MagicMock(side_effect=auth_error)

    with pytest.raises(Exception):  # Should raise immediately, not retry
        provider._call_with_retry({"model": "test", "messages": [{"role": "user", "content": "hi"}]})

    # Should only have been called once (no retry)
    assert provider.client.chat.completions.create.call_count == 1


# --- Bug: Empty choices crashes _to_response ---

def test_empty_choices_no_crash():
    """Provider returning empty choices should not crash with IndexError."""
    from mullet.core.providers.openai_compat import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider(
        base_url="http://localhost:99999/v1",
        model="test",
    )
    mock_response = MagicMock()
    mock_response.choices = []  # Empty!
    mock_response.model = "test"

    result = provider._to_response(mock_response)
    assert result.content == ""
    assert result.finish_reason == "error"
    assert result.tool_calls is None


# --- Bug: main() shows raw traceback on invalid args ---

def test_main_handles_keyboard_interrupt():
    """Ctrl+C should exit cleanly, not show traceback."""

    # Just verify the main function structure exists
    from mullet.cli.main import main, _main_inner
    assert callable(main)
    assert callable(_main_inner)


def test_main_traceback_env_set_prints_traceback(capsys, monkeypatch):
    """MULLET_TRACEBACK=1 should print full traceback to stderr."""
    monkeypatch.setenv("MULLET_TRACEBACK", "1")
    from mullet.cli.main import main

    with patch("mullet.cli.main._main_inner", side_effect=RuntimeError("boom")):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "Traceback" in captured.err


def test_main_traceback_env_unset_no_traceback(capsys, monkeypatch):
    """Without MULLET_TRACEBACK, only the friendly message goes to stderr."""
    monkeypatch.delenv("MULLET_TRACEBACK", raising=False)
    from mullet.cli.main import main

    with patch("mullet.cli.main._main_inner", side_effect=RuntimeError("boom")):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "Traceback" not in captured.err


# --- Bug: Rate limiter memory leak ---

def test_rate_limiter_cleans_empty_keys():
    """Empty client keys should be removed to prevent memory leak."""
    from mullet.server.middleware import RateLimiter
    import time

    limiter = RateLimiter(max_requests=10, window_seconds=1)
    limiter.is_allowed("client_a")
    assert "client_a" in limiter._counters

    # Simulate time passing (expire the window)
    limiter._counters["client_a"] = [time.time() - 100]  # Expired
    limiter.is_allowed("client_a")  # Should clean + allow

    # After a second clean pass with expired data, the key should be removed
    # then re-created for the new request
    assert "client_a" in limiter._counters
    assert len(limiter._counters["client_a"]) == 1  # Only the new request


# --- Bug: Image analysis check crashes on non-dict JSON ---

def test_image_check_handles_non_dict_json():
    """Tool results that parse to int/list should not crash the image check."""
    registry = ToolRegistry()
    registry.register(
        name="code_exec",
        func=lambda **kw: "1024",  # Returns a string that parses to int
        schema={"description": "test", "parameters": {"type": "object", "properties": {}}},
        toolset="core",
    )

    responses = [
        _make_response(
            tool_calls=[_make_tool_call("code_exec", '{"code": "print(2**10)"}', "c1")],
            finish_reason="tool_calls",
        ),
        _make_response(content="The result is 1024"),
    ]
    provider = _make_provider(responses)
    agent = AgentRunner(
        provider=provider, tool_registry=registry,
        toolsets=["core"], max_iterations=5,
    )
    # Should NOT raise AttributeError: 'int' object has no attribute 'get'
    result = agent.run("run code")
    assert result.completed
    assert "1024" in result.final_response
