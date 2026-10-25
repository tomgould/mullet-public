"""Tests for the Bedrock provider message/tool conversion logic.

Mocks boto3 to avoid AWS API calls.
"""

import json
import sys
from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture
def provider():
    """Create BedrockProvider with mocked boto3."""
    mock_boto3 = MagicMock()
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    with patch.dict(sys.modules, {"boto3": mock_boto3}):
        from mullet.core.providers.bedrock import BedrockProvider
        p = BedrockProvider(region="us-west-2", model_id="anthropic.claude-test")
        yield p, mock_client


def test_init(provider):
    p, _ = provider
    assert p.model_id == "anthropic.claude-test"
    assert p.name == "bedrock"


def test_convert_messages_system(provider):
    p, _ = provider
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    system_parts, converted = p._convert_messages(messages)
    assert len(system_parts) == 1
    assert system_parts[0]["text"] == "You are helpful."
    assert len(converted) == 1
    assert converted[0]["role"] == "user"
    assert converted[0]["content"] == [{"text": "Hello"}]


def test_convert_messages_assistant_with_tool_calls(provider):
    p, _ = provider
    messages = [
        {"role": "user", "content": "Search for Python"},
        {"role": "assistant", "content": "Let me search.", "tool_calls": [
            {"id": "tc1", "type": "function",
             "function": {"name": "search", "arguments": '{"query": "python"}'}},
        ]},
    ]
    _, converted = p._convert_messages(messages)
    assert len(converted) == 2
    assistant = converted[1]
    assert assistant["role"] == "assistant"
    assert len(assistant["content"]) == 2
    assert assistant["content"][0]["text"] == "Let me search."
    assert assistant["content"][1]["toolUse"]["name"] == "search"
    assert assistant["content"][1]["toolUse"]["input"] == {"query": "python"}


def test_convert_messages_tool_result(provider):
    p, _ = provider
    messages = [
        {"role": "user", "content": "Do something"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "tc1", "type": "function",
             "function": {"name": "search", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "tc1", "content": "Search result here"},
    ]
    _, converted = p._convert_messages(messages)
    assert len(converted) == 3
    tool_result = converted[2]
    assert tool_result["role"] == "user"
    assert tool_result["content"][0]["toolResult"]["toolUseId"] == "tc1"
    assert tool_result["content"][0]["toolResult"]["content"] == [{"text": "Search result here"}]


def test_convert_tools(provider):
    p, _ = provider
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
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
    spec = converted[0]["toolSpec"]
    assert spec["name"] == "web_search"
    assert spec["description"] == "Search the web"
    assert "json" in spec["inputSchema"]
    assert "properties" in spec["inputSchema"]["json"]


def test_to_response_text(provider):
    p, _ = provider
    response = {
        "output": {
            "message": {
                "content": [{"text": "Hello from Bedrock!"}],
            },
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 15,
            "outputTokens": 8,
            "totalTokens": 23,
        },
    }
    result = p._to_response(response)
    assert result.content == "Hello from Bedrock!"
    assert result.tool_calls is None
    assert result.finish_reason == "stop"
    assert result.usage.input_tokens == 15
    assert result.usage.output_tokens == 8
    assert result.provider == "bedrock"


def test_to_response_tool_use(provider):
    p, _ = provider
    response = {
        "output": {
            "message": {
                "content": [
                    {"toolUse": {
                        "toolUseId": "tool_1",
                        "name": "read_file",
                        "input": {"path": "/etc/hosts"},
                    }},
                ],
            },
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 20, "outputTokens": 10, "totalTokens": 30},
    }
    result = p._to_response(response)
    assert result.content == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["function"]["name"] == "read_file"
    assert json.loads(result.tool_calls[0]["function"]["arguments"]) == {"path": "/etc/hosts"}
    assert result.finish_reason == "tool_calls"


def test_health_check_success(provider):
    p, client = provider
    client.converse.return_value = {"output": {"message": {"content": [{"text": "ok"}]}}}
    assert p.health_check() is True


def test_health_check_failure(provider):
    p, client = provider
    client.converse.side_effect = Exception("AWS credentials not configured")
    assert p.health_check() is False


def test_create_calls_converse(provider):
    p, client = provider
    client.converse.return_value = {
        "output": {"message": {"content": [{"text": "response"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    }
    result = p.create(messages=[{"role": "user", "content": "test"}])
    assert result.content == "response"
    client.converse.assert_called_once()


def test_init_without_boto3():
    """Lines 28-29: ImportError raised when boto3 is unavailable."""
    with patch.dict(sys.modules, {"boto3": None}):
        # Force re-import to hit the ImportError path
        if "mullet.core.providers.bedrock" in sys.modules:
            del sys.modules["mullet.core.providers.bedrock"]
        with pytest.raises(ImportError, match="boto3"):
            import mullet.core.providers.bedrock as fresh_bedrock
            fresh_bedrock.BedrockProvider()


def test_create_with_system_message(provider):
    """Line 53: system parts are added when system message present."""
    p, client = provider
    client.converse.return_value = {
        "output": {"message": {"content": [{"text": "ok"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 5, "outputTokens": 3, "totalTokens": 8},
    }
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "hi"},
    ]
    result = p.create(messages=messages)
    assert result.content == "ok"
    call_kwargs = client.converse.call_args[1]
    assert "system" in call_kwargs
    assert call_kwargs["system"][0]["text"] == "Be concise."


def test_create_with_tools(provider):
    """Line 55: toolConfig added when tools list provided."""
    p, client = provider
    client.converse.return_value = {
        "output": {"message": {"content": [{"text": "ok"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 5, "outputTokens": 3, "totalTokens": 8},
    }
    tools = [{"type": "function", "function": {"name": "search", "description": "Search", "parameters": {}}}]
    result = p.create(messages=[{"role": "user", "content": "find"}], tools=tools)
    assert result.content == "ok"
    call_kwargs = client.converse.call_args[1]
    assert "toolConfig" in call_kwargs


def test_create_log_failure_on_success_is_swallowed(provider):
    """Lines 78-79: log_interaction raising after success is caught and logged."""
    p, client = provider
    client.converse.return_value = {
        "output": {"message": {"content": [{"text": "hi"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
    }
    import mullet.core.logging.interaction_log as _il
    original = _il.log_interaction
    _il.log_interaction = MagicMock(side_effect=RuntimeError("db down"))
    try:
        result = p.create(messages=[{"role": "user", "content": "hi"}])
    finally:
        _il.log_interaction = original
    assert result.content == "hi"


def test_create_non_retryable_error_is_raised(provider):
    """Lines 81-106: non-retryable error raised immediately."""
    p, client = provider
    client.converse.side_effect = ValueError("invalid model id")
    with pytest.raises(ValueError, match="invalid model id"):
        p.create(messages=[{"role": "user", "content": "hi"}])


def test_create_non_retryable_error_log_failure_swallowed(provider):
    """Lines 104-105: log_interaction failure during error path is swallowed."""
    p, client = provider
    client.converse.side_effect = ValueError("bad model")
    import mullet.core.logging.interaction_log as _il
    original = _il.log_interaction
    _il.log_interaction = MagicMock(side_effect=RuntimeError("log err"))
    try:
        with pytest.raises(ValueError):
            p.create(messages=[{"role": "user", "content": "hi"}])
    finally:
        _il.log_interaction = original


def test_create_retryable_error_retries_and_succeeds(provider):
    """Lines 85-89: throttle error triggers retry; success on second attempt."""
    p, client = provider
    success_response = {
        "output": {"message": {"content": [{"text": "retry ok"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
    }
    client.converse.side_effect = [
        Exception("ThrottlingException: rate limit"),
        success_response,
    ]
    with patch("time.sleep"):  # skip the wait
        result = p.create(messages=[{"role": "user", "content": "hi"}])
    assert result.content == "retry ok"


def test_create_retryable_error_exhausted_raises(provider):
    """Lines 108-124: retryable error exhausts retries → raises last error."""
    p, client = provider
    client.converse.side_effect = Exception("ThrottlingException: always")
    with patch("time.sleep"):
        with pytest.raises(Exception, match="ThrottlingException"):
            p.create(messages=[{"role": "user", "content": "hi"}])
    assert client.converse.call_count == 4  # initial + 3 retries


def test_convert_messages_malformed_tool_args(provider):
    """Lines 161-163: malformed JSON in tool_calls arguments → empty dict."""
    p, _ = provider
    messages = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "tc1", "type": "function",
             "function": {"name": "f", "arguments": "{not valid json"}},
        ]},
    ]
    _, converted = p._convert_messages(messages)
    # Should not raise; bad JSON args replaced by {}
    assert converted[1]["content"][0]["toolUse"]["input"] == {}


def test_all_retries_exhausted_logs_and_raises(provider):
    """Lines 108-124: retryable error exhausts all retries and re-raises."""
    p, client = provider
    client.converse.side_effect = Exception("ThrottlingException: always")
    with patch("time.sleep"):
        with patch("mullet.core.logging.interaction_log.log_interaction"):
            with pytest.raises(Exception, match="ThrottlingException"):
                p._call_with_retry({"modelId": "test", "messages": []}, [], max_retries=1)
    assert client.converse.call_count == 2  # initial + 1 retry
