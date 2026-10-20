"""Tests for provider implementations."""

from mullet.core.providers.base import Provider, LLMResponse, StreamChunk
from mullet.core.providers.openai_compat import OpenAICompatibleProvider
from mullet.core.agent.types import TokenUsage


def test_provider_abc():
    """Provider ABC has required methods."""
    assert hasattr(Provider, "create")
    assert hasattr(Provider, "create_stream")
    assert hasattr(Provider, "health_check")
    assert hasattr(Provider, "supports_tool_calling")


def test_llm_response_defaults():
    r = LLMResponse()
    assert r.content == ""
    assert r.tool_calls is None
    assert r.finish_reason == "stop"


def test_stream_chunk():
    c = StreamChunk(content="hello")
    assert c.content == "hello"
    assert c.tool_call_delta is None


def test_openai_provider_name_inference():
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    assert p.name == "local"

    p = OpenAICompatibleProvider(base_url="https://openrouter.ai/api/v1")
    assert p.name == "openrouter"

    p = OpenAICompatibleProvider(base_url="https://api.openai.com/v1")
    assert p.name == "openai"

    p = OpenAICompatibleProvider(base_url="https://my.custom.endpoint/v1")
    assert p.name == "custom"


def test_token_usage_addition():
    a = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
    b = TokenUsage(input_tokens=200, output_tokens=100, total_tokens=300)
    c = a + b
    assert c.input_tokens == 300
    assert c.output_tokens == 150
    assert c.total_tokens == 450


def test_token_usage_cache_fields():
    t = TokenUsage(cache_read_tokens=50, cache_write_tokens=25)
    assert t.cache_read_tokens == 50
    assert t.cache_write_tokens == 25
