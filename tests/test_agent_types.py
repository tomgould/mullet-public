"""Tests for mullet.core.agent.types — TokenUsage, AgentResult, DelegationResult."""


from mullet.core.agent.types import AgentResult, DelegationResult, TokenUsage


# --- TokenUsage ------------------------------------------------------------


def test_token_usage_defaults():
    u = TokenUsage()
    assert u.input_tokens == 0
    assert u.output_tokens == 0
    assert u.total_tokens == 0
    assert u.cache_read_tokens == 0
    assert u.cache_write_tokens == 0


def test_token_usage_fields_set():
    u = TokenUsage(
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        cache_read_tokens=5,
        cache_write_tokens=2,
    )
    assert u.input_tokens == 10
    assert u.output_tokens == 20
    assert u.total_tokens == 30
    assert u.cache_read_tokens == 5
    assert u.cache_write_tokens == 2


def test_token_usage_add_sums_all_fields():
    a = TokenUsage(input_tokens=1, output_tokens=2, total_tokens=3, cache_read_tokens=4, cache_write_tokens=5)
    b = TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30, cache_read_tokens=40, cache_write_tokens=50)
    c = a + b
    assert c.input_tokens == 11
    assert c.output_tokens == 22
    assert c.total_tokens == 33
    assert c.cache_read_tokens == 44
    assert c.cache_write_tokens == 55


def test_token_usage_add_does_not_mutate():
    """__add__ should return a new object, not mutate either operand."""
    a = TokenUsage(input_tokens=1)
    b = TokenUsage(input_tokens=2)
    _c = a + b
    assert a.input_tokens == 1
    assert b.input_tokens == 2


def test_token_usage_add_with_empty():
    a = TokenUsage(input_tokens=5, output_tokens=10)
    zero = TokenUsage()
    c = a + zero
    assert c.input_tokens == 5
    assert c.output_tokens == 10


def test_token_usage_add_chain():
    """Multiple additions compose correctly."""
    a = TokenUsage(input_tokens=1)
    b = TokenUsage(input_tokens=2)
    c = TokenUsage(input_tokens=3)
    total = a + b + c
    assert total.input_tokens == 6


# --- AgentResult -----------------------------------------------------------


def test_agent_result_required_fields():
    r = AgentResult(completed=True, final_response="ok")
    assert r.completed is True
    assert r.final_response == "ok"
    assert r.messages == []
    assert r.tool_calls_made == 0
    assert r.iterations_used == 0
    assert r.provider_used == ""
    assert r.tier_used == ""


def test_agent_result_tokens_default_is_fresh_instance():
    """Each AgentResult gets its own TokenUsage — no shared mutable state."""
    r1 = AgentResult(completed=True, final_response="a")
    r2 = AgentResult(completed=True, final_response="b")
    r1.tokens_used.input_tokens = 100
    assert r2.tokens_used.input_tokens == 0


def test_agent_result_full_construction():
    tokens = TokenUsage(input_tokens=50, output_tokens=20, total_tokens=70)
    r = AgentResult(
        completed=False,
        final_response="timed out",
        messages=[{"role": "user", "content": "hi"}],
        tool_calls_made=3,
        iterations_used=5,
        tokens_used=tokens,
        provider_used="kimi",
        tier_used="STANDARD",
    )
    assert r.completed is False
    assert r.tool_calls_made == 3
    assert r.iterations_used == 5
    assert r.tokens_used.input_tokens == 50
    assert r.provider_used == "kimi"
    assert r.tier_used == "STANDARD"


# --- DelegationResult ------------------------------------------------------


def test_delegation_result_success():
    agent_result = AgentResult(completed=True, final_response="done")
    d = DelegationResult(
        goal="test goal",
        completed=True,
        summary="all done",
        agent_result=agent_result,
    )
    assert d.goal == "test goal"
    assert d.completed is True
    assert d.summary == "all done"
    assert d.agent_result is agent_result
    assert d.error is None


def test_delegation_result_failure():
    d = DelegationResult(
        goal="failing task",
        completed=False,
        summary="",
        error="network timeout",
    )
    assert d.completed is False
    assert d.error == "network timeout"
    assert d.agent_result is None


def test_delegation_result_optional_fields_default_none():
    d = DelegationResult(goal="g", completed=True, summary="s")
    assert d.agent_result is None
    assert d.error is None
