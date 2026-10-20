"""Tests for core types."""

from mullet.core.agent.types import AgentResult, DelegationResult, TokenUsage


def test_token_usage_defaults():
    t = TokenUsage()
    assert t.input_tokens == 0
    assert t.output_tokens == 0
    assert t.total_tokens == 0
    assert t.cache_read_tokens == 0


def test_token_usage_add():
    a = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
    b = TokenUsage(input_tokens=200, output_tokens=100, total_tokens=300)
    c = a + b
    assert c.input_tokens == 300
    assert c.output_tokens == 150
    assert c.total_tokens == 450


def test_agent_result_defaults():
    r = AgentResult(completed=True, final_response="done")
    assert r.completed is True
    assert r.tool_calls_made == 0
    assert r.iterations_used == 0
    assert r.provider_used == ""
    assert r.tier_used == ""


def test_delegation_result():
    r = DelegationResult(goal="test", completed=True, summary="done")
    assert r.goal == "test"
    assert r.error is None
    assert r.agent_result is None
