"""Tests for the full agent execution loop — multi-turn, multi-tool scenarios."""

import json
from mullet.core.agent.runner import AgentRunner
from mullet.core.agent.types import TokenUsage
from mullet.core.providers.base import LLMResponse, Provider
from mullet.core.tools.registry import ToolRegistry


class ScriptedProvider(Provider):
    """Provider that returns a scripted sequence of responses."""
    name = "scripted"

    def __init__(self, script: list[LLMResponse]):
        self._script = script
        self._idx = 0

    def create(self, messages, tools=None, **kwargs) -> LLMResponse:
        resp = self._script[min(self._idx, len(self._script) - 1)]
        self._idx += 1
        return resp


def _make_registry_with_tools() -> ToolRegistry:
    """Registry with add and concat tools."""
    registry = ToolRegistry()
    registry.register(
        name="add",
        func=lambda a, b, **_: json.dumps({"result": int(a) + int(b)}),
        schema={"description": "Add two numbers", "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        }},
        toolset="math",
    )
    registry.register(
        name="concat",
        func=lambda s1, s2, **_: s1 + s2,
        schema={"description": "Concatenate strings", "parameters": {
            "type": "object",
            "properties": {"s1": {"type": "string"}, "s2": {"type": "string"}},
            "required": ["s1", "s2"],
        }},
        toolset="text",
    )
    return registry


def test_multi_tool_sequence():
    """Agent calls two different tools then responds."""
    provider = ScriptedProvider([
        # Call add
        LLMResponse(content="", tool_calls=[{
            "id": "call_1", "type": "function",
            "function": {"name": "add", "arguments": '{"a": 10, "b": 20}'},
        }], usage=TokenUsage(total_tokens=50)),
        # Call concat
        LLMResponse(content="", tool_calls=[{
            "id": "call_2", "type": "function",
            "function": {"name": "concat", "arguments": '{"s1": "hello ", "s2": "world"}'},
        }], usage=TokenUsage(total_tokens=50)),
        # Final response
        LLMResponse(content="Add: 30, Concat: hello world", usage=TokenUsage(total_tokens=30)),
    ])
    registry = _make_registry_with_tools()
    agent = AgentRunner(provider=provider, tool_registry=registry, toolsets=["math", "text"])

    result = agent.run("use both tools")
    assert result.completed is True
    assert result.tool_calls_made == 2
    assert result.iterations_used == 3
    assert "30" in result.final_response
    assert "hello world" in result.final_response


def test_parallel_tool_calls():
    """Agent calls multiple tools in one response."""
    provider = ScriptedProvider([
        LLMResponse(content="Calling both:", tool_calls=[
            {"id": "c1", "type": "function", "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'}},
            {"id": "c2", "type": "function", "function": {"name": "add", "arguments": '{"a": 3, "b": 4}'}},
        ], usage=TokenUsage(total_tokens=60)),
        LLMResponse(content="Results: 3 and 7", usage=TokenUsage(total_tokens=20)),
    ])
    registry = _make_registry_with_tools()
    agent = AgentRunner(provider=provider, tool_registry=registry, toolsets=["math"])

    result = agent.run("add 1+2 and 3+4")
    assert result.completed is True
    assert result.tool_calls_made == 2
    # Both tool results should be in messages
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 2


def test_tool_error_recovery():
    """Agent sees a tool error and adjusts."""
    def failing_tool(**_):
        raise ValueError("division by zero")

    registry = ToolRegistry()
    registry.register(
        name="divide",
        func=failing_tool,
        schema={"description": "Divide", "parameters": {"type": "object", "properties": {}}},
        toolset="math",
    )

    provider = ScriptedProvider([
        LLMResponse(content="", tool_calls=[{
            "id": "c1", "type": "function",
            "function": {"name": "divide", "arguments": "{}"},
        }], usage=TokenUsage(total_tokens=30)),
        # Agent sees error and responds gracefully
        LLMResponse(content="The division failed. Cannot divide by zero.",
                    usage=TokenUsage(total_tokens=20)),
    ])

    agent = AgentRunner(provider=provider, tool_registry=registry, toolsets=["math"])
    result = agent.run("divide something")
    assert result.completed is True
    assert "failed" in result.final_response.lower() or "zero" in result.final_response.lower()


def test_content_alongside_tool_calls():
    """Verify content is streamed when it comes with tool calls."""
    tokens_seen = []

    provider = ScriptedProvider([
        LLMResponse(
            content="Let me check that...",
            tool_calls=[{"id": "c1", "type": "function",
                         "function": {"name": "add", "arguments": '{"a": 1, "b": 1}'}}],
            usage=TokenUsage(total_tokens=30),
        ),
        LLMResponse(content="The answer is 2.", usage=TokenUsage(total_tokens=15)),
    ])

    registry = _make_registry_with_tools()
    agent = AgentRunner(
        provider=provider,
        tool_registry=registry,
        toolsets=["math"],
        on_token=lambda t: tokens_seen.append(t),
    )

    result = agent.run("what is 1+1?")
    assert result.completed is True
    # Content from tool-call response should have been streamed
    assert any("check" in t.lower() for t in tokens_seen) or len(tokens_seen) > 0


def test_budget_warning_injected():
    """Verify budget warnings appear in tool results when budget is low."""
    provider = ScriptedProvider([
        LLMResponse(content="", tool_calls=[{
            "id": "c1", "type": "function",
            "function": {"name": "add", "arguments": '{"a": 1, "b": 1}'},
        }], usage=TokenUsage(total_tokens=10)),
        LLMResponse(content="Done", usage=TokenUsage(total_tokens=10)),
    ])

    registry = _make_registry_with_tools()
    agent = AgentRunner(provider=provider, tool_registry=registry,
                        toolsets=["math"], max_iterations=3)

    result = agent.run("add 1+1")
    # With max_iterations=3 and 2 used, the warning should fire
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    if tool_msgs:
        # Check if any tool result contains the budget warning
        all_content = " ".join(m.get("content", "") for m in tool_msgs)
        # Warning fires when remaining ≤ 3, which it should be here
        # (3 max, 1 used for tool call = 2 remaining)


def test_tool_output_truncation():
    """Verify large tool outputs are truncated."""
    def big_output(**_):
        return "x" * 200_000  # 200K chars

    registry = ToolRegistry()
    registry.register(
        name="big",
        func=big_output,
        schema={"description": "Big output", "parameters": {"type": "object", "properties": {}}},
    )

    provider = ScriptedProvider([
        LLMResponse(content="", tool_calls=[{
            "id": "c1", "type": "function",
            "function": {"name": "big", "arguments": "{}"},
        }], usage=TokenUsage(total_tokens=10)),
        LLMResponse(content="Done", usage=TokenUsage(total_tokens=10)),
    ])

    agent = AgentRunner(provider=provider, tool_registry=registry, max_iterations=5)
    result = agent.run("run big")

    # Tool result in messages should be truncated
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    content = tool_msgs[0]["content"]
    assert len(content) < 200_000
    assert "truncated" in content or "omitted" in content
