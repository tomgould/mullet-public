"""Tests for handling malformed tool calls from LLMs."""

from mullet.core.agent.runner import AgentRunner
from mullet.core.agent.types import TokenUsage
from mullet.core.providers.base import LLMResponse, Provider
from mullet.core.tools.registry import ToolRegistry


class MalformedProvider(Provider):
    name = "test"
    def __init__(self, response):
        self._resp = response
        self._call = 0
    def create(self, messages, **kw):
        self._call += 1
        if self._call == 1:
            return self._resp
        return LLMResponse(content="Recovered.", usage=TokenUsage(total_tokens=10))


def _registry():
    r = ToolRegistry()
    r.register(name="echo", func=lambda text="", **_: text,
               schema={"description": "Echo", "parameters": {"type": "object", "properties": {}}})
    return r


def test_missing_function_key():
    """Tool call missing 'function' key."""
    provider = MalformedProvider(LLMResponse(
        content="",
        tool_calls=[{"id": "c1", "type": "function"}],  # No "function" key
        usage=TokenUsage(total_tokens=20),
    ))
    agent = AgentRunner(provider=provider, tool_registry=_registry(), max_iterations=3)
    result = agent.run("test")
    # Should not crash — either skip or handle gracefully
    assert result is not None


def test_missing_tool_id():
    """Tool call missing 'id' field."""
    provider = MalformedProvider(LLMResponse(
        content="",
        tool_calls=[{
            "type": "function",
            "function": {"name": "echo", "arguments": '{"text": "hi"}'},
            # No "id" field
        }],
        usage=TokenUsage(total_tokens=20),
    ))
    agent = AgentRunner(provider=provider, tool_registry=_registry(), max_iterations=3)
    result = agent.run("test")
    assert result is not None
    assert result.tool_calls_made >= 0


def test_empty_tool_calls_list():
    """Empty tool_calls list treated as no tools."""
    provider = MalformedProvider(LLMResponse(
        content="No tools needed.",
        tool_calls=[],
        usage=TokenUsage(total_tokens=10),
    ))
    agent = AgentRunner(provider=provider, tool_registry=_registry(), max_iterations=3)
    result = agent.run("test")
    assert result.completed is True
    assert result.final_response == "No tools needed."


def test_none_tool_calls():
    """None tool_calls treated as no tools."""
    provider = MalformedProvider(LLMResponse(
        content="Direct response.",
        tool_calls=None,
        usage=TokenUsage(total_tokens=10),
    ))
    agent = AgentRunner(provider=provider, tool_registry=_registry(), max_iterations=3)
    result = agent.run("test")
    assert result.completed is True
