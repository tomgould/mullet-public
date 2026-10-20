"""Gap coverage tests for mullet/core/agent/runner.py."""

import json
import time
from unittest.mock import patch, MagicMock

import pytest

from mullet.core.agent.runner import AgentRunner, _preview_args
from mullet.core.agent.types import TokenUsage
from mullet.core.providers.base import LLMResponse, Provider
from mullet.core.tools.registry import ToolRegistry


class SimpleProvider(Provider):
    """Provider that returns predefined responses for testing."""

    def __init__(self, responses: list[LLMResponse]):
        self.name = "test"
        self.model = "test-model"
        self._responses = responses
        self._call_idx = 0

    def create(self, messages, tools=None, **kwargs) -> LLMResponse:
        resp = self._responses[min(self._call_idx, len(self._responses) - 1)]
        self._call_idx += 1
        return resp


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        name="add",
        func=lambda a, b, **_: json.dumps({"result": int(a) + int(b)}),
        schema={
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        },
        toolset="math",
    )
    registry.register(
        name="echo",
        func=lambda msg, **_: str(msg),
        schema={
            "description": "Echo",
            "parameters": {
                "type": "object",
                "properties": {"msg": {"type": "string"}},
                "required": ["msg"],
            },
        },
        toolset="utils",
    )
    registry.register(
        name="image_tool",
        func=lambda **_: json.dumps({
            "type": "image_analysis",
            "image_data_url": "data:image/png;base64,abc",
            "question": "What is this?",
        }),
        schema={
            "description": "Image analysis",
            "parameters": {"type": "object", "properties": {}},
        },
        toolset="utils",
    )
    registry.register(
        name="fail_tool",
        func=lambda **_: (_ for _ in ()).throw(RuntimeError("tool boom")),
        schema={
            "description": "Always fails",
            "parameters": {"type": "object", "properties": {}},
        },
        toolset="utils",
    )
    return registry


# Lines 129-131: Hard message cap

def test_hard_message_cap():
    provider = SimpleProvider([LLMResponse(content="ok", usage=TokenUsage())])
    registry = _make_registry()
    agent = AgentRunner(provider=provider, tool_registry=registry)
    history = [{"role": "user", "content": f"msg {i}"} for i in range(202)]
    result = agent.run("hello", conversation_history=history)
    assert result.completed is True
    assert len(result.messages) <= 205  # truncated to ~200 + assistant response


# Line 150: on_usage callback

def test_on_usage_callback():
    usage_called = []

    def on_usage(provider_name, model, input_tokens, output_tokens):
        usage_called.append((provider_name, model, input_tokens, output_tokens))

    provider = SimpleProvider([
        LLMResponse(content="ok", usage=TokenUsage(total_tokens=10, input_tokens=5, output_tokens=5)),
    ])
    registry = _make_registry()
    agent = AgentRunner(provider=provider, tool_registry=registry, on_usage=on_usage)
    agent.run("hello")
    assert len(usage_called) == 1
    assert usage_called[0][2] == 5
    assert usage_called[0][3] == 5


# Line 167: reasoning_content preserved in assistant message

def test_reasoning_content_in_tool_call_response():
    provider = SimpleProvider([
        LLMResponse(
            content="",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
            }],
            usage=TokenUsage(),
            reasoning_content="Let me think...",
        ),
        LLMResponse(content="done", usage=TokenUsage()),
    ])
    registry = _make_registry()
    agent = AgentRunner(provider=provider, tool_registry=registry, toolsets=["math"])
    result = agent.run("hello")
    assistant_msgs = [m for m in result.messages if m.get("role") == "assistant" and m.get("tool_calls")]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0].get("reasoning_content") == "Let me think..."


# Lines 186-189: Truncate large parallel tool outputs

def test_parallel_tool_result_truncation():
    huge_result = "x" * 120_000
    provider = SimpleProvider([
        LLMResponse(
            content="",
            tool_calls=[
                {"id": "c1", "type": "function", "function": {"name": "echo", "arguments": '{"msg": "hi"}'}},
                {"id": "c2", "type": "function", "function": {"name": "echo", "arguments": '{"msg": "ho"}'}},
            ],
            usage=TokenUsage(),
        ),
        LLMResponse(content="done", usage=TokenUsage()),
    ])
    registry = _make_registry()
    agent = AgentRunner(provider=provider, tool_registry=registry, toolsets=["utils"])

    with patch("mullet.core.agent.parallel.should_parallelize", return_value=True):
        with patch("mullet.core.agent.parallel.execute_parallel") as mock_exec:
            mock_exec.return_value = [
                {"tool_call_id": "c1", "name": "echo", "result": huge_result, "duration_ms": 1.0, "success": True},
                {"tool_call_id": "c2", "name": "echo", "result": "ok", "duration_ms": 1.0, "success": True},
            ]
            result = agent.run("hello")
            tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
            assert any("chars truncated" in m.get("content", "") for m in tool_msgs)


# Lines 195: Budget critical warning in parallel path

def test_parallel_budget_critical_warning():
    provider = SimpleProvider([
        LLMResponse(
            content="",
            tool_calls=[
                {"id": "c1", "type": "function", "function": {"name": "echo", "arguments": '{"msg": "hi"}'}},
                {"id": "c2", "type": "function", "function": {"name": "echo", "arguments": '{"msg": "ho"}'}},
            ],
            usage=TokenUsage(),
        ),
        LLMResponse(content="done", usage=TokenUsage()),
    ])
    registry = _make_registry()
    agent = AgentRunner(provider=provider, tool_registry=registry, toolsets=["utils"], max_iterations=2)

    with patch("mullet.core.agent.parallel.should_parallelize", return_value=True):
        with patch("mullet.core.agent.parallel.execute_parallel") as mock_exec:
            mock_exec.return_value = [
                {"tool_call_id": "c1", "name": "echo", "result": "ok", "duration_ms": 1.0, "success": True},
                {"tool_call_id": "c2", "name": "echo", "result": "ok", "duration_ms": 1.0, "success": True},
            ]
            result = agent.run("hello")
            tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
            assert any("[CRITICAL: Last iteration" in m.get("content", "") for m in tool_msgs)


# Line 197: Budget warning in parallel path

def test_parallel_budget_warning():
    provider = SimpleProvider([
        LLMResponse(
            content="",
            tool_calls=[
                {"id": "c1", "type": "function", "function": {"name": "echo", "arguments": '{"msg": "hi"}'}},
                {"id": "c2", "type": "function", "function": {"name": "echo", "arguments": '{"msg": "ho"}'}},
            ],
            usage=TokenUsage(),
        ),
        LLMResponse(content="done", usage=TokenUsage()),
    ])
    registry = _make_registry()
    agent = AgentRunner(provider=provider, tool_registry=registry, toolsets=["utils"], max_iterations=4)

    with patch("mullet.core.agent.parallel.should_parallelize", return_value=True):
        with patch("mullet.core.agent.parallel.execute_parallel") as mock_exec:
            mock_exec.return_value = [
                {"tool_call_id": "c1", "name": "echo", "result": "ok", "duration_ms": 1.0, "success": True},
                {"tool_call_id": "c2", "name": "echo", "result": "ok", "duration_ms": 1.0, "success": True},
            ]
            result = agent.run("hello")
            tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
            assert any("WARNING:" in m.get("content", "") for m in tool_msgs)


# Lines 212-221: Malformed tool call

def test_malformed_tool_call():
    provider = SimpleProvider([
        LLMResponse(
            content="",
            tool_calls=["not_a_dict"],
            usage=TokenUsage(),
        ),
        LLMResponse(content="done", usage=TokenUsage()),
    ])
    registry = _make_registry()
    agent = AgentRunner(provider=provider, tool_registry=registry, toolsets=["math"])
    result = agent.run("hello")
    assert result.completed is True
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert any("Malformed" in m.get("content", "") for m in tool_msgs)


# Line 227: on_tool_start sequential

def test_on_tool_start_sequential():
    starts = []

    def on_start(name, preview):
        starts.append(name)

    provider = SimpleProvider([
        LLMResponse(
            content="",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
            }],
            usage=TokenUsage(),
        ),
        LLMResponse(content="done", usage=TokenUsage()),
    ])
    registry = _make_registry()
    agent = AgentRunner(provider=provider, tool_registry=registry, toolsets=["math"], on_tool_start=on_start)
    agent.run("hello")
    assert "add" in starts


# Line 237: on_tool_complete sequential success

def test_on_tool_complete_sequential():
    completes = []

    def on_complete(name, preview):
        completes.append((name, preview))

    provider = SimpleProvider([
        LLMResponse(
            content="",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
            }],
            usage=TokenUsage(),
        ),
        LLMResponse(content="done", usage=TokenUsage()),
    ])
    registry = _make_registry()
    agent = AgentRunner(provider=provider, tool_registry=registry, toolsets=["math"], on_tool_complete=on_complete)
    agent.run("hello")
    assert any(name == "add" for name, _ in completes)


# Lines 243-255: Metrics failure during exception handler

def test_metrics_failure_on_tool_error():
    provider = SimpleProvider([
        LLMResponse(
            content="",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "fail_tool", "arguments": "{}"},
            }],
            usage=TokenUsage(),
        ),
        LLMResponse(content="done", usage=TokenUsage()),
    ])
    registry = _make_registry()
    with patch("mullet.server.metrics.get_metrics") as mock_get:
        mock_get.side_effect = RuntimeError("metrics down")
        agent = AgentRunner(provider=provider, tool_registry=registry, toolsets=["utils"])
        result = agent.run("hello")
        assert result.completed is True


# Lines 275-276: on_tool_result sequential

def test_on_tool_result_sequential():
    results_seen = []

    def on_result(name, preview):
        results_seen.append(name)

    provider = SimpleProvider([
        LLMResponse(
            content="",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
            }],
            usage=TokenUsage(),
        ),
        LLMResponse(content="done", usage=TokenUsage()),
    ])
    registry = _make_registry()
    agent = AgentRunner(provider=provider, tool_registry=registry, toolsets=["math"], on_tool_result=on_result)
    agent.run("hello")
    assert "add" in results_seen


# Line 288: Image analysis injection

def test_image_analysis_injection():
    provider = SimpleProvider([
        LLMResponse(
            content="",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "image_tool", "arguments": "{}"},
            }],
            usage=TokenUsage(),
        ),
        LLMResponse(content="done", usage=TokenUsage()),
    ])
    registry = _make_registry()
    agent = AgentRunner(provider=provider, tool_registry=registry, toolsets=["utils"])
    result = agent.run("hello")
    user_msgs = [m for m in result.messages if m.get("role") == "user"]
    assert any(isinstance(m.get("content"), list) for m in user_msgs)


# Line 302: on_thinking callback

def test_on_thinking_callback():
    thinking_seen = []

    def on_think(text):
        thinking_seen.append(text)

    resp = LLMResponse(content="ok", usage=TokenUsage())
    resp.thinking = "hmm..."
    provider = SimpleProvider([resp])
    registry = _make_registry()
    agent = AgentRunner(provider=provider, tool_registry=registry, on_thinking=on_think)
    agent.run("hello")
    assert thinking_seen == ["hmm..."]


# Lines 321-322: Analytics save failure

def test_analytics_save_failure():
    provider = SimpleProvider([LLMResponse(content="ok", usage=TokenUsage())])
    registry = _make_registry()
    agent = AgentRunner(provider=provider, tool_registry=registry)
    with patch.object(agent._analytics, "save", side_effect=RuntimeError("disk full")):
        result = agent.run("hello")
    assert result.completed is True


# Line 342: Child interrupt propagation

def test_child_interrupt_propagation():
    provider = SimpleProvider([LLMResponse(content="ok", usage=TokenUsage())])
    registry = _make_registry()
    agent = AgentRunner(provider=provider, tool_registry=registry)
    child = AgentRunner(provider=provider, tool_registry=registry)
    agent._active_children.append(child)
    agent.interrupt("stop all")
    assert agent._interrupt_flag is True
    assert child._interrupt_flag is True
    assert child._interrupt_message == "stop all"


# Lines 423-424: json dumps fallback fails

def test_estimate_tokens_and_json_fallback_both_fail():
    class DummyProvider(Provider):
        name = "dummy"
        model = "m"

        def create(self, messages, tools=None, **kwargs):
            return LLMResponse(content="ok", usage=TokenUsage())

    provider = DummyProvider()
    registry = _make_registry()
    agent = AgentRunner(provider=provider, tool_registry=registry)
    with patch("mullet.core.providers.tokenize.estimate_tokens", side_effect=RuntimeError("est fail")):
        with patch("json.dumps", side_effect=RuntimeError("json fail")):
            agent._call_llm([{"role": "user", "content": "hi"}], [])
    assert agent._last_prompt_tokens == 0


# Line 434: Slow LLM call logs warning

def test_slow_llm_call_logs():
    class DummyProvider(Provider):
        name = "dummy"
        model = "m"

        def create(self, messages, tools=None, **kwargs):
            return LLMResponse(content="ok", usage=TokenUsage())

    provider = DummyProvider()
    registry = _make_registry()
    agent = AgentRunner(provider=provider, tool_registry=registry)
    with patch("time.time", side_effect=[0.0, 35.0]):
        agent._call_llm([{"role": "user", "content": "hi"}], [])


# Line 522: _preview_args truncates long primary arg

def test_preview_args_truncates_long_primary():
    long_path = "x" * 200
    result = _preview_args(json.dumps({"path": long_path}), tool_name="read_file")
    assert result.endswith("...")
    assert len(result) == 120


# Line 160: on_token fires when content arrives alongside tool calls

def test_on_token_with_tool_call_content():
    tokens = []

    def on_token(text):
        tokens.append(text)

    provider = SimpleProvider([
        LLMResponse(
            content="Let me calculate",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
            }],
            usage=TokenUsage(),
        ),
        LLMResponse(content="done", usage=TokenUsage()),
    ])
    registry = _make_registry()
    agent = AgentRunner(provider=provider, tool_registry=registry, toolsets=["math"], on_token=on_token)
    agent.run("hello")
    assert "Let me calculate" in tokens


# Lines 245-255: Exception in sequential tool dispatch + metrics failure

def test_sequential_tool_exception_and_metrics_failure():
    provider = SimpleProvider([
        LLMResponse(
            content="",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
            }],
            usage=TokenUsage(),
        ),
        LLMResponse(content="done", usage=TokenUsage()),
    ])
    registry = _make_registry()
    agent = AgentRunner(provider=provider, tool_registry=registry, toolsets=["math"])
    # Force dispatch to raise directly (bypass ToolRegistry exception handling)
    original_dispatch = agent.tool_registry.dispatch
    def boom_dispatch(name, args):
        if name == "add":
            raise RuntimeError("dispatch boom")
        return original_dispatch(name, args)
    agent.tool_registry.dispatch = boom_dispatch

    with patch("mullet.server.metrics.get_metrics") as mock_get:
        mock_get.side_effect = RuntimeError("metrics down")
        result = agent.run("hello")
        assert result.completed is True


# Lines 260-264: Truncate large sequential tool outputs

def test_sequential_tool_result_truncation():
    huge = "x" * 120_000
    provider = SimpleProvider([
        LLMResponse(
            content="",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "echo", "arguments": '{"msg": "hi"}'},
            }],
            usage=TokenUsage(),
        ),
        LLMResponse(content="done", usage=TokenUsage()),
    ])
    registry = _make_registry()
    # Override echo to return huge string
    registry._tools["echo"].func = lambda msg, **_: huge
    agent = AgentRunner(provider=provider, tool_registry=registry, toolsets=["utils"])
    result = agent.run("hello")
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert any("chars truncated" in m.get("content", "") for m in tool_msgs)


# Lines 308-311: Intermediate ack nudge

def test_intermediate_ack_nudge():
    provider = SimpleProvider([
        LLMResponse(content="I'll check the file for you.", usage=TokenUsage()),
        LLMResponse(content="done", usage=TokenUsage()),
    ])
    registry = _make_registry()
    agent = AgentRunner(provider=provider, tool_registry=registry)
    result = agent.run("hello")
    assert result.completed is True
    # The nudge adds a user message asking to proceed
    assert any("Please proceed" in str(m.get("content", "")) for m in result.messages)


# Line 316: on_token in no-tool-call path

def test_on_token_no_tool_calls():
    tokens = []

    def on_token(text):
        tokens.append(text)

    provider = SimpleProvider([LLMResponse(content="Hello there!", usage=TokenUsage())])
    registry = _make_registry()
    agent = AgentRunner(provider=provider, tool_registry=registry, on_token=on_token)
    agent.run("hello")
    assert "Hello there!" in tokens


# Lines 448-451: Fallback chain exhausted raises last error

def test_fallback_chain_exhausted():
    class PrimaryFail(Provider):
        name = "primary"
        model = "m"

        def create(self, messages, tools=None, **kwargs):
            raise ConnectionError("primary fails")

    class FallbackFail(Provider):
        name = "fallback"
        model = "m"

        def create(self, messages, tools=None, **kwargs):
            raise ConnectionError("fallback fails")

    registry = _make_registry()
    agent = AgentRunner(
        provider=PrimaryFail(),
        tool_registry=registry,
        fallback_providers=[FallbackFail()],
    )
    with pytest.raises(ConnectionError, match="fallback fails"):
        agent.run("hello")


# Lines 464, 477-490: _is_intermediate_ack branches

def test_is_intermediate_ack_too_long():
    from mullet.core.agent.runner import _is_intermediate_ack
    assert _is_intermediate_ack("x" * 801) is False


def test_is_intermediate_ack_no_future_phrase():
    from mullet.core.agent.runner import _is_intermediate_ack
    assert _is_intermediate_ack("I checked the file already.") is False


def test_is_intermediate_ack_no_action_word():
    from mullet.core.agent.runner import _is_intermediate_ack
    assert _is_intermediate_ack("I will sing a song.") is False


def test_is_intermediate_ack_has_result():
    from mullet.core.agent.runner import _is_intermediate_ack
    assert _is_intermediate_ack("I'll check the file. Here is the result.") is False


def test_is_intermediate_ack_true():
    from mullet.core.agent.runner import _is_intermediate_ack
    assert _is_intermediate_ack("I'll check the file for you.") is True


# Lines 501-504: _preview_args with invalid JSON string

def test_preview_args_invalid_json():
    result = _preview_args("not json{{{", tool_name="add")
    assert result == "not json{{{"


# Line 528: _preview_args fallback return for short dict without primary key

def test_preview_args_fallback_short_dict():
    result = _preview_args('{"foo": "bar"}', tool_name="unknown_tool")
    assert result == '{"foo": "bar"}'


# Line 504: _preview_args with dict input directly

def test_preview_args_dict_input():
    result = _preview_args({"path": "/tmp/a.py"}, tool_name="read_file")
    assert result == "/tmp/a.py"


# Line 528: _preview_args fallback truncation for long dict

def test_preview_args_fallback_long_dict():
    long_val = "x" * 200
    result = _preview_args({"foo": long_val}, tool_name="unknown_tool")
    assert result.endswith("...")
    assert len(result) == 123  # 120 + "..."
