"""Integration tests for HarnessEngine tool execution loop."""

from __future__ import annotations

import json

import pytest

from mullet.core.providers.base import LLMResponse, Provider
from mullet.harness.engine.loop import HarnessEngine
from mullet.harness.engine.tool_dispatch import ToolDispatcher
from mullet.harness.hooks.registry import Block, HookRegistry
from mullet.harness.sessions.store import SessionStore


class _MultiResponseProvider(Provider):
    """Mock provider that returns a sequence of responses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.calls: list[tuple[list[dict], list[dict] | None]] = []
        self.responses = responses
        self._idx = 0

    def create(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        reasoning_config: dict | None = None,
    ) -> LLMResponse:
        self.calls.append((list(messages), tools))
        resp = self.responses[self._idx % len(self.responses)]
        self._idx += 1
        return resp


@pytest.fixture
def engine(tmp_path, monkeypatch):
    store = SessionStore(base_dir=tmp_path / "sessions")
    hooks = HookRegistry()

    # Patch ToolDispatcher inside the engine module to inject a safe mock tool
    _original_init = ToolDispatcher.__init__

    def _patched_init(self, registry=None):
        _original_init(self, registry)
        self._registry.register(
            name="mock_tool",
            func=lambda input: f"mock_result: {input}",
            schema={
                "description": "A mock tool for testing",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string", "description": "Input to echo"},
                    },
                    "required": ["input"],
                },
            },
            toolset="test",
        )

    monkeypatch.setattr(
        "mullet.harness.engine.loop.ToolDispatcher.__init__",
        _patched_init,
    )

    return HarnessEngine(store=store, hooks=hooks)


class TestEngineToolExecution:
    def test_engine_executes_one_tool_then_completes(self, engine, monkeypatch):
        provider = _MultiResponseProvider([
            LLMResponse(
                content="",
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "mock_tool",
                        "arguments": json.dumps({"input": "hello"}),
                    },
                }],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content="Done",
                finish_reason="stop",
            ),
        ])
        monkeypatch.setattr(
            "mullet.harness.engine.loop._resolve_provider",
            lambda name, model=None: provider,
        )

        result = engine.run("trigger tool")

        assert result["stop_reason"] == "stop"
        assert result["response"] == "Done"
        assert result["iterations"] == 2

        # Provider should have been called twice
        assert len(provider.calls) == 2

        # First call should include tool schemas
        _first_messages, first_tools = provider.calls[0]
        assert first_tools is not None
        assert any(t["function"]["name"] == "mock_tool" for t in first_tools)

        # Second call should include the tool result
        second_messages, _second_tools = provider.calls[1]
        tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_1"
        assert tool_msgs[0]["content"] == "mock_result: hello"

    def test_engine_max_iterations_stops_loop(self, engine, monkeypatch):
        provider = _MultiResponseProvider([
            LLMResponse(
                content="",
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "mock_tool",
                        "arguments": json.dumps({"input": "hello"}),
                    },
                }],
                finish_reason="tool_calls",
            ),
        ])
        monkeypatch.setattr(
            "mullet.harness.engine.loop._resolve_provider",
            lambda name, model=None: provider,
        )

        result = engine.run("trigger tool", max_iterations=3)

        assert result["iterations"] == 3
        assert result["stop_reason"] == "tool_calls"
        assert len(provider.calls) == 3

    def test_engine_blocked_tool_appends_error_result(self, engine, monkeypatch):
        provider = _MultiResponseProvider([
            LLMResponse(
                content="",
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "mock_tool",
                        "arguments": json.dumps({"input": "hello"}),
                    },
                }],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content="Blocked result",
                finish_reason="stop",
            ),
        ])
        monkeypatch.setattr(
            "mullet.harness.engine.loop._resolve_provider",
            lambda name, model=None: provider,
        )

        # Register a PreToolUse hook that blocks the mock_tool
        engine.hooks.register("PreToolUse", lambda p: Block("denied by test"))

        result = engine.run("trigger tool")

        assert result["response"] == "Blocked result"
        assert result["stop_reason"] == "stop"

        # Second LLM call should contain a synthetic error tool result
        second_messages, _ = provider.calls[1]
        tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_1"
        parsed = json.loads(tool_msgs[0]["content"])
        assert "blocked" in parsed["error"].lower() or "denied" in parsed["error"].lower()

    def test_engine_run_sync_wrapper_works(self, engine, monkeypatch):
        provider = _MultiResponseProvider([
            LLMResponse(content="OK", finish_reason="stop"),
        ])
        monkeypatch.setattr(
            "mullet.harness.engine.loop._resolve_provider",
            lambda name, model=None: provider,
        )

        result = engine.run("hello")

        assert result["session_id"].startswith("sess-")
        assert result["response"] == "OK"
        assert result["stop_reason"] == "stop"
        assert result["iterations"] == 1
        assert len(provider.calls) == 1
