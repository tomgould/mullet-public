"""Tests for the Anthropic provider message/tool conversion logic.

These tests mock the anthropic SDK to avoid real API calls.
"""

import json
import sys
from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture
def provider():
    """Create AnthropicProvider with mocked anthropic SDK."""
    # Mock the anthropic module since it's imported inside __init__
    mock_anthropic_mod = MagicMock()
    mock_client = MagicMock()
    mock_anthropic_mod.Anthropic.return_value = mock_client

    with patch.dict(sys.modules, {"anthropic": mock_anthropic_mod}):
        from mullet.core.providers.anthropic import AnthropicProvider
        p = AnthropicProvider(api_key="test-key", model="claude-test")
        yield p, mock_client


def test_init(provider):
    p, _ = provider
    assert p.model == "claude-test"
    assert p.name == "anthropic"
    assert p.supports_tool_calling is True


def test_convert_messages_system(provider):
    p, _ = provider
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    system, converted = p._convert_messages(messages)
    assert system == "You are helpful."
    assert len(converted) == 1
    assert converted[0]["role"] == "user"
    assert converted[0]["content"] == "Hello"


def test_convert_messages_tool_results_batching(provider):
    """Multiple consecutive tool results must be batched into one user message."""
    p, _ = provider
    messages = [
        {"role": "user", "content": "Do two things"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "tc1", "type": "function", "function": {"name": "tool_a", "arguments": "{}"}},
            {"id": "tc2", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "tc1", "content": "result A"},
        {"role": "tool", "tool_call_id": "tc2", "content": "result B"},
    ]
    system, converted = p._convert_messages(messages)
    assert system == ""
    assert len(converted) == 3  # user, assistant, batched-tool-results

    # Third message should be a user message with two tool_result content blocks
    tool_msg = converted[2]
    assert tool_msg["role"] == "user"
    assert isinstance(tool_msg["content"], list)
    assert len(tool_msg["content"]) == 2
    assert tool_msg["content"][0]["type"] == "tool_result"
    assert tool_msg["content"][0]["tool_use_id"] == "tc1"
    assert tool_msg["content"][1]["tool_use_id"] == "tc2"


def test_convert_messages_tool_call_args_string(provider):
    """String arguments in tool_calls should be parsed to dict."""
    p, _ = provider
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Let me help", "tool_calls": [
            {"id": "tc1", "type": "function",
             "function": {"name": "search", "arguments": '{"query": "test"}'}},
        ]},
    ]
    _, converted = p._convert_messages(messages)
    assistant_msg = converted[1]
    assert assistant_msg["role"] == "assistant"
    blocks = assistant_msg["content"]
    assert len(blocks) == 2
    assert blocks[0]["type"] == "text"
    assert blocks[0]["text"] == "Let me help"
    assert blocks[1]["type"] == "tool_use"
    assert blocks[1]["input"] == {"query": "test"}


def test_convert_messages_bad_json_args(provider):
    """Invalid JSON in tool arguments should default to empty dict."""
    p, _ = provider
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "tc1", "type": "function",
             "function": {"name": "do_thing", "arguments": "not json"}},
        ]},
    ]
    _, converted = p._convert_messages(messages)
    tool_block = converted[1]["content"][0]
    assert tool_block["input"] == {}


def test_convert_tools(provider):
    p, _ = provider
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }
    ]
    converted = p._convert_tools(tools)
    assert len(converted) == 1
    assert converted[0]["name"] == "search"
    assert converted[0]["description"] == "Search the web"
    assert "properties" in converted[0]["input_schema"]


def test_to_response_text(provider):
    p, _ = provider
    mock_response = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Hello, world!"
    mock_response.content = [text_block]
    mock_response.stop_reason = "end_turn"
    mock_response.usage = MagicMock(
        input_tokens=10, output_tokens=5,
    )
    del mock_response.usage.cache_read_input_tokens
    del mock_response.usage.cache_creation_input_tokens

    result = p._to_response(mock_response)
    assert result.content == "Hello, world!"
    assert result.tool_calls is None
    assert result.finish_reason == "stop"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5
    assert result.usage.total_tokens == 15
    assert result.model == "claude-test"
    assert result.provider == "anthropic"


def test_to_response_tool_use(provider):
    p, _ = provider
    mock_response = MagicMock()
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "tool_123"
    tool_block.name = "search"
    tool_block.input = {"query": "test"}
    mock_response.content = [tool_block]
    mock_response.stop_reason = "tool_use"
    mock_response.usage = MagicMock(
        input_tokens=20, output_tokens=10,
    )
    del mock_response.usage.cache_read_input_tokens
    del mock_response.usage.cache_creation_input_tokens

    result = p._to_response(mock_response)
    assert result.content == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["id"] == "tool_123"
    assert result.tool_calls[0]["function"]["name"] == "search"
    assert json.loads(result.tool_calls[0]["function"]["arguments"]) == {"query": "test"}
    assert result.finish_reason == "tool_calls"


def test_to_response_max_tokens(provider):
    p, _ = provider
    mock_response = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Truncated..."
    mock_response.content = [text_block]
    mock_response.stop_reason = "max_tokens"
    mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
    del mock_response.usage.cache_read_input_tokens
    del mock_response.usage.cache_creation_input_tokens

    result = p._to_response(mock_response)
    assert result.finish_reason == "length"


def test_to_response_cache_tokens(provider):
    p, _ = provider
    mock_response = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Cached response"
    mock_response.content = [text_block]
    mock_response.stop_reason = "end_turn"
    mock_response.usage = MagicMock(
        input_tokens=100, output_tokens=50,
        cache_read_input_tokens=80,
        cache_creation_input_tokens=20,
    )

    result = p._to_response(mock_response)
    assert result.usage.cache_read_tokens == 80
    assert result.usage.cache_write_tokens == 20


def test_health_check_success(provider):
    p, client = provider
    client.messages.create.return_value = MagicMock()
    assert p.health_check() is True


def test_health_check_failure(provider):
    p, client = provider
    client.messages.create.side_effect = Exception("connection refused")
    assert p.health_check() is False


def test_create_calls_api(provider):
    """create() should call the Anthropic API and return LLMResponse."""
    p, client = provider

    mock_response = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "response"
    mock_response.content = [text_block]
    mock_response.stop_reason = "end_turn"
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
    del mock_response.usage.cache_read_input_tokens
    del mock_response.usage.cache_creation_input_tokens

    client.messages.create.return_value = mock_response

    result = p.create(
        messages=[{"role": "user", "content": "test"}],
        max_tokens=100,
    )
    assert result.content == "response"
    client.messages.create.assert_called_once()


def test_retry_on_rate_limit(provider):
    """Provider retries on 429 rate limit errors."""
    from unittest.mock import patch, MagicMock
    p, mock_client = provider

    call_count = 0
    def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("429 rate limit exceeded")
        return MagicMock(
            content=[MagicMock(type="text", text="ok")],
            usage=MagicMock(input_tokens=5, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0),
            stop_reason="end_turn",
        )

    mock_client.messages.create.side_effect = side_effect
    with patch("time.sleep"):
        result = p._call_with_retry({"messages": []})
    assert result.content == "ok"
    assert call_count == 3


def test_no_retry_on_auth_error(provider):
    """Provider does not retry on non-retryable errors."""
    from unittest.mock import patch
    p, mock_client = provider

    mock_client.messages.create.side_effect = Exception("Invalid API key")
    with patch("time.sleep"):
        with pytest.raises(Exception, match="Invalid API key"):
            p._call_with_retry({"messages": []})
    assert mock_client.messages.create.call_count == 1


def test_raises_after_max_retries(provider):
    """Provider raises after all retries are exhausted on rate limit."""
    from unittest.mock import patch
    p, mock_client = provider

    mock_client.messages.create.side_effect = Exception("429 overloaded")
    with patch("time.sleep"):
        with pytest.raises(Exception, match="429 overloaded"):
            p._call_with_retry({"messages": []}, max_retries=2)
    assert mock_client.messages.create.call_count == 2


# --------------------------------------------------------------------------
# Constraint-layer wiring tests
# --------------------------------------------------------------------------


def _build_provider():
    """Build an AnthropicProvider with a mocked SDK client + success response."""
    from mullet.core.providers.anthropic import AnthropicProvider
    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client
        p = AnthropicProvider(api_key="test-key", model="claude-sonnet-4-6")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="hi")]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock(input_tokens=5, output_tokens=3)
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.usage.cache_creation_input_tokens = 0
        mock_client.messages.create.return_value = mock_response
        return p, mock_client


def test_anthropic_name_is_set():
    p, _ = _build_provider()
    assert p.name == "anthropic"


def test_anthropic_shape_request_is_called():
    p, _ = _build_provider()
    with patch("mullet.core.providers.anthropic.shape_request") as mock_shape:
        mock_shape.side_effect = lambda model, kwargs: kwargs
        p.create(messages=[{"role": "user", "content": "hi"}])
        assert mock_shape.called
        args, _kw = mock_shape.call_args
        assert args[0] == "claude-sonnet-4-6"


def test_anthropic_wire_log_is_emitted(caplog):
    import logging as _lg
    p, _ = _build_provider()
    caplog.set_level(_lg.INFO, logger="mullet.core.providers.anthropic")
    p.create(messages=[{"role": "user", "content": "hi"}])
    wire_logs = [r for r in caplog.records if "wire:" in r.message and "provider=anthropic" in r.message]
    assert wire_logs


def test_anthropic_constraint_error_heals_and_retries():
    p, mock_client = _build_provider()
    call_count = {"n": 0}
    first_response = mock_client.messages.create.return_value

    def side_effect(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("unsupported_value: frequency_penalty")
        return first_response

    mock_client.messages.create.side_effect = side_effect

    with patch.object(type(p), "parse_error", return_value=("frequency_penalty", "test reason")):
        with patch("mullet.core.providers.anthropic.get_constraint_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            p.create(messages=[{"role": "user", "content": "hi"}])
            assert call_count["n"] == 2
            mock_reg.return_value.learn.assert_called_once()
