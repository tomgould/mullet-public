"""Tests for the delegate tool."""

import json
from mullet.core.tools.delegate import delegate, set_parent_agent, DELEGATE_SCHEMA
from mullet.core.agent.runner import AgentRunner
from mullet.core.agent.types import TokenUsage
from mullet.core.providers.base import LLMResponse, Provider
from mullet.core.tools.registry import ToolRegistry


class MockProvider(Provider):
    name = "mock"
    def create(self, messages, **kw):
        return LLMResponse(content="Delegated task done.", usage=TokenUsage(total_tokens=50))


def test_delegate_no_parent():
    set_parent_agent(None)
    result = delegate(goal="test")
    data = json.loads(result)
    assert "error" in data


def test_delegate_with_parent():
    agent = AgentRunner(
        provider=MockProvider(),
        tool_registry=ToolRegistry(),
        max_iterations=5,
    )
    set_parent_agent(agent)

    result = delegate(goal="summarize something", context="test context")
    data = json.loads(result)
    assert data["completed"] is True
    assert "done" in data["summary"].lower() or len(data["summary"]) > 0

    set_parent_agent(None)  # Clean up


def test_delegate_schema():
    assert "goal" in DELEGATE_SCHEMA["parameters"]["properties"]
    assert "context" in DELEGATE_SCHEMA["parameters"]["properties"]
    assert DELEGATE_SCHEMA["parameters"]["required"] == ["goal"]
