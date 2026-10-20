"""Scenario tests for the agent loop — multi-tool interaction patterns.

Tests realistic agent execution scenarios with mock providers to ensure
the agent loop handles complex tool call patterns correctly.
"""

import json
from unittest.mock import MagicMock
import pytest

from mullet.core.agent.runner import AgentRunner
from mullet.core.agent.types import TokenUsage
from mullet.core.providers.base import LLMResponse, Provider
from mullet.core.tools.registry import ToolRegistry


def _mock_provider(responses: list[LLMResponse]) -> Provider:
    p = MagicMock(spec=Provider)
    p.name = "mock"
    p.model = "test"
    p.create = MagicMock(side_effect=responses)
    return p


def _response(content="", tool_calls=None, finish="stop") -> LLMResponse:
    return LLMResponse(
        content=content, tool_calls=tool_calls, finish_reason=finish,
        usage=TokenUsage(10, 5, 15), model="mock", provider="mock",
    )


def _tool_call(name: str, args: dict, call_id: str = "call_1") -> dict:
    return {
        "id": call_id, "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(name="read_file", func=lambda args: "file contents here",
                 schema={"description": "test", "parameters": {"type": "object", "properties": {}}},
                 toolset="test")
    reg.register(name="write_file", func=lambda args: "written",
                 schema={"description": "test", "parameters": {"type": "object", "properties": {}}},
                 toolset="test")
    reg.register(name="terminal", func=lambda args: "command output",
                 schema={"description": "test", "parameters": {"type": "object", "properties": {}}},
                 toolset="test")
    return reg


def test_read_then_write(registry):
    """Agent reads a file then writes a modified version."""
    responses = [
        _response(tool_calls=[_tool_call("read_file", {"path": "main.py"}, "c1")],
                   finish="tool_calls"),
        _response(tool_calls=[_tool_call("write_file", {"path": "main.py", "content": "new"}, "c2")],
                   finish="tool_calls"),
        _response(content="Done! Updated main.py"),
    ]
    agent = AgentRunner(
        provider=_mock_provider(responses),
        tool_registry=registry, toolsets=["test"], max_iterations=10,
    )
    result = agent.run("update main.py")
    assert result.completed
    assert result.tool_calls_made == 2
    assert "Updated" in result.final_response


def test_parallel_tool_calls(registry):
    """Agent makes multiple tool calls in one response."""
    responses = [
        _response(tool_calls=[
            _tool_call("read_file", {"path": "a.py"}, "c1"),
            _tool_call("read_file", {"path": "b.py"}, "c2"),
        ], finish="tool_calls"),
        _response(content="Both files read"),
    ]
    agent = AgentRunner(
        provider=_mock_provider(responses),
        tool_registry=registry, toolsets=["test"], max_iterations=10,
    )
    result = agent.run("compare a.py and b.py")
    assert result.completed
    assert result.tool_calls_made == 2


def test_tool_error_recovery(registry):
    """Agent recovers from tool error and tries different approach."""
    reg = ToolRegistry()
    call_count = [0]
    def failing_then_ok(args):
        call_count[0] += 1
        if call_count[0] == 1:
            raise FileNotFoundError("no such file")
        return "success"
    reg.register(name="read_file", func=failing_then_ok,
                 schema={"description": "test", "parameters": {"type": "object", "properties": {}}},
                 toolset="test")

    responses = [
        _response(tool_calls=[_tool_call("read_file", {"path": "bad.py"}, "c1")],
                   finish="tool_calls"),
        _response(tool_calls=[_tool_call("read_file", {"path": "good.py"}, "c2")],
                   finish="tool_calls"),
        _response(content="Found it"),
    ]
    agent = AgentRunner(
        provider=_mock_provider(responses),
        tool_registry=reg, toolsets=["test"], max_iterations=10,
    )
    result = agent.run("read the file")
    assert result.completed
    assert result.tool_calls_made == 2


def test_think_blocks_stripped(registry):
    """Agent response with thinking blocks should be stripped."""
    responses = [
        _response(content="<think>reasoning here</think>The answer is 42"),
    ]
    agent = AgentRunner(
        provider=_mock_provider(responses),
        tool_registry=registry, toolsets=["test"], max_iterations=5,
    )
    result = agent.run("what is the answer?")
    assert result.completed
    assert "42" in result.final_response
    assert "reasoning here" not in result.final_response


def test_empty_content_response(registry):
    """Provider returning empty content should be handled."""
    responses = [
        _response(content=""),
    ]
    agent = AgentRunner(
        provider=_mock_provider(responses),
        tool_registry=registry, toolsets=["test"], max_iterations=5,
    )
    result = agent.run("hello")
    assert result.completed


def test_budget_exhaustion(registry):
    """Agent with tight budget should stop gracefully."""
    # Make 4 tool calls but budget is only 3
    responses = []
    for i in range(5):
        responses.append(_response(
            tool_calls=[_tool_call("terminal", {"command": "ls"}, f"c{i}")],
            finish="tool_calls",
        ))
    responses.append(_response(content="done"))

    agent = AgentRunner(
        provider=_mock_provider(responses),
        tool_registry=registry, toolsets=["test"], max_iterations=3,
    )
    result = agent.run("list everything")
    assert not result.completed  # Should hit budget limit
    assert "budget" in result.final_response.lower() or result.iterations_used >= 3


def test_streaming_callbacks(registry):
    """Streaming callbacks should fire for tokens and tool calls."""
    tokens = []
    tools = []

    responses = [
        _response(tool_calls=[_tool_call("terminal", {"command": "ls"}, "c1")],
                   content="Let me check", finish="tool_calls"),
        _response(content="Found the files"),
    ]
    agent = AgentRunner(
        provider=_mock_provider(responses),
        tool_registry=registry, toolsets=["test"], max_iterations=10,
        on_token=lambda t: tokens.append(t),
        on_tool_call=lambda n, p: tools.append(n),
    )
    result = agent.run("list files")
    assert result.completed
    assert "terminal" in tools
    assert len(tokens) >= 1  # At least the final response


def test_conversation_history_preserved(registry):
    """Agent should work with conversation history."""
    responses = [
        _response(content="Hello! I remember you asked about auth."),
    ]
    history = [
        {"role": "user", "content": "tell me about the auth module"},
        {"role": "assistant", "content": "The auth module handles authentication."},
    ]
    agent = AgentRunner(
        provider=_mock_provider(responses),
        tool_registry=registry, toolsets=["test"], max_iterations=5,
    )
    result = agent.run("what did I just ask about?", conversation_history=history)
    assert result.completed
    # Messages should include history + new user message + response
    assert len(result.messages) >= 4  # system + 2 history + user + assistant
