"""Tests for the delegation system."""

from mullet.core.agent.types import TokenUsage
from mullet.core.agent.delegation import delegate_task, delegate_batch, _build_child_system_prompt
from mullet.core.agent.runner import AgentRunner
from mullet.core.providers.base import LLMResponse, Provider
from mullet.core.tools.registry import ToolRegistry


class MockProvider(Provider):
    """A provider that returns canned responses for testing."""

    def __init__(self, responses: list[str] | None = None):
        self.name = "mock"
        self._responses = responses or ["Task completed successfully."]
        self._call_count = 0

    def create(self, messages, tools=None, **kwargs) -> LLMResponse:
        resp = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return LLMResponse(
            content=resp,
            tool_calls=None,
            finish_reason="stop",
            usage=TokenUsage(input_tokens=50, output_tokens=25, total_tokens=75),
            model="mock-model",
            provider="mock",
        )


def _make_parent() -> AgentRunner:
    """Create a real AgentRunner with a mock provider for delegation tests."""
    registry = ToolRegistry()
    registry.register(name="noop", func=lambda **_: "ok", schema={"description": "noop"}, toolset="test")

    return AgentRunner(
        provider=MockProvider(),
        tool_registry=registry,
        toolsets=["test"],
        system_prompt="Parent agent",
        max_iterations=10,
    )


def test_delegate_task_no_parent():
    result = delegate_task(goal="do something")
    assert result.completed is False
    assert "No parent" in result.error


def test_delegate_task_success():
    parent = _make_parent()
    result = delegate_task(goal="summarize this file", parent=parent)

    assert result.completed is True
    assert "Task completed successfully" in result.summary
    assert result.goal == "summarize this file"


def test_delegate_task_with_provider_override():
    parent = _make_parent()
    custom_provider = MockProvider(["Custom provider response."])

    result = delegate_task(
        goal="analyze code",
        provider_override=custom_provider,
        parent=parent,
    )

    assert result.completed is True
    assert "Custom provider response" in result.summary


def test_delegate_batch():
    parent = _make_parent()
    tasks = [
        {"goal": "task 1"},
        {"goal": "task 2"},
        {"goal": "task 3"},
    ]

    results = delegate_batch(tasks, parent=parent)

    assert len(results) == 3
    assert all(r.completed for r in results)
    assert results[0].goal == "task 1"
    assert results[1].goal == "task 2"
    assert results[2].goal == "task 3"


def test_delegate_batch_with_context():
    parent = _make_parent()
    tasks = [
        {"goal": "fix bug", "context": "auth module", "max_iterations": 5},
    ]

    results = delegate_batch(tasks, parent=parent)
    assert len(results) == 1
    assert results[0].completed is True


def test_build_child_system_prompt():
    prompt = _build_child_system_prompt("fix the bug", "in auth.py line 42")
    assert "fix the bug" in prompt
    assert "auth.py line 42" in prompt


def test_build_child_system_prompt_no_context():
    prompt = _build_child_system_prompt("do something", "")
    assert "do something" in prompt
    assert "CONTEXT" not in prompt
