"""Tests for the provider base classes and types."""

import pytest
from mullet.core.providers.base import Provider, LLMResponse, StreamChunk
from mullet.core.agent.types import AgentResult, TokenUsage


# --- TokenUsage ---

def test_token_usage_defaults():
    u = TokenUsage()
    assert u.input_tokens == 0
    assert u.output_tokens == 0
    assert u.total_tokens == 0


def test_token_usage_addition():
    a = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
    b = TokenUsage(input_tokens=200, output_tokens=100, total_tokens=300)
    c = a + b
    assert c.input_tokens == 300
    assert c.output_tokens == 150
    assert c.total_tokens == 450


def test_token_usage_cache_fields():
    u = TokenUsage()
    u.cache_read_tokens = 50
    u.cache_write_tokens = 20
    assert u.cache_read_tokens == 50
    assert u.cache_write_tokens == 20


# --- LLMResponse ---

def test_llm_response_text():
    r = LLMResponse(
        content="Hello",
        tool_calls=None,
        finish_reason="stop",
        usage=TokenUsage(10, 5, 15),
        model="test",
        provider="mock",
    )
    assert r.content == "Hello"
    assert r.tool_calls is None
    assert r.finish_reason == "stop"


def test_llm_response_tool_calls():
    r = LLMResponse(
        content="",
        tool_calls=[{"id": "tc1", "function": {"name": "search", "arguments": "{}"}}],
        finish_reason="tool_calls",
        usage=TokenUsage(20, 10, 30),
        model="test",
        provider="mock",
    )
    assert len(r.tool_calls) == 1
    assert r.finish_reason == "tool_calls"


# --- StreamChunk ---

def test_stream_chunk_content():
    c = StreamChunk(content="hello")
    assert c.content == "hello"
    assert c.finish_reason is None


def test_stream_chunk_finish():
    c = StreamChunk(finish_reason="stop")
    assert c.finish_reason == "stop"


# --- AgentResult ---

def test_agent_result_minimal():
    r = AgentResult(completed=False, final_response="")
    assert r.completed is False
    assert r.final_response == ""
    assert r.tool_calls_made == 0
    assert r.iterations_used == 0


def test_agent_result_completed():
    r = AgentResult(
        completed=True,
        final_response="Done!",
        tool_calls_made=3,
        iterations_used=5,
        tokens_used=TokenUsage(1000, 500, 1500),
        provider_used="local",
        tier_used="LOW",
    )
    assert r.completed is True
    assert r.final_response == "Done!"
    assert r.tool_calls_made == 3
    assert r.provider_used == "local"


def test_agent_result_messages():
    r = AgentResult(
        completed=True,
        final_response="ok",
        messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "ok"},
        ],
    )
    assert len(r.messages) == 2


# --- Provider ABC ---

def test_provider_is_abstract():
    """Provider should not be instantiable directly."""
    with pytest.raises(TypeError):
        Provider()
