"""Tests for the Gemini provider message/tool conversion logic.

Mocks google-genai SDK to avoid real API calls.
"""

import sys
from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture
def provider():
    """Create GeminiProvider with mocked SDK."""
    mock_genai = MagicMock()
    mock_types = MagicMock()
    mock_genai.types = mock_types
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client

    # Mock types module
    mock_types.Content = MagicMock()
    mock_types.Part = MagicMock()
    mock_types.FunctionCall = MagicMock()
    mock_types.FunctionResponse = MagicMock()
    mock_types.FunctionDeclaration = MagicMock()
    mock_types.Tool = MagicMock()
    mock_types.GenerateContentConfig = MagicMock()

    with patch.dict(sys.modules, {
        "google": MagicMock(),
        "google.genai": mock_genai,
        "google.genai.types": mock_types,
    }):
        from mullet.core.providers.gemini import GeminiProvider
        p = GeminiProvider(api_key="test-key", model="gemini-test")
        yield p, mock_client


def test_init(provider):
    p, _ = provider
    assert p.model == "gemini-test"
    assert p.name == "gemini"
    assert p.supports_tool_calling is True


def test_convert_messages_system(provider):
    p, _ = provider
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    system, contents = p._convert_messages(messages)
    assert system == "You are helpful."
    assert len(contents) == 1  # Only user message


def test_convert_messages_multi_turn(provider):
    p, _ = provider
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "How are you?"},
    ]
    system, contents = p._convert_messages(messages)
    assert system == ""
    assert len(contents) == 3


def test_convert_messages_tool_calls(provider):
    p, _ = provider
    messages = [
        {"role": "user", "content": "Search for Python"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "tc1", "type": "function",
             "function": {"name": "search", "arguments": '{"query": "python"}'}},
        ]},
    ]
    system, contents = p._convert_messages(messages)
    assert len(contents) == 2


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
                    "required": ["query"],
                },
            },
        }
    ]
    result = p._convert_tools(tools)
    assert len(result) == 1  # One Tool wrapper


def test_to_response_text(provider):
    p, _ = provider
    mock_response = MagicMock()
    mock_part = MagicMock()
    mock_part.text = "Hello from Gemini!"
    mock_part.function_call = None
    mock_candidate = MagicMock()
    mock_candidate.content.parts = [mock_part]
    mock_candidate.finish_reason = "STOP"
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata = MagicMock(
        prompt_token_count=10, candidates_token_count=5, total_token_count=15,
    )

    result = p._to_response(mock_response)
    assert result.content == "Hello from Gemini!"
    assert result.tool_calls is None
    assert result.finish_reason == "stop"
    assert result.usage.total_tokens == 15


def test_to_response_tool_use(provider):
    p, _ = provider
    mock_fc = MagicMock()
    mock_fc.name = "search"
    mock_fc.args = {"query": "test"}

    mock_part = MagicMock()
    mock_part.text = None
    mock_part.function_call = mock_fc

    mock_candidate = MagicMock()
    mock_candidate.content.parts = [mock_part]
    mock_candidate.finish_reason = None
    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata = MagicMock(
        prompt_token_count=20, candidates_token_count=10, total_token_count=30,
    )

    result = p._to_response(mock_response)
    assert result.content == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["function"]["name"] == "search"
    assert result.finish_reason == "tool_calls"


def test_to_response_none_parts(provider):
    """Gemini can return candidates with None content or None parts."""
    p, _ = provider
    mock_candidate = MagicMock()
    mock_candidate.content.parts = None
    mock_candidate.finish_reason = "STOP"
    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata = MagicMock(
        prompt_token_count=5, candidates_token_count=0, total_token_count=5,
    )

    result = p._to_response(mock_response)
    assert result.content == ""
    assert result.tool_calls is None
    assert result.finish_reason == "stop"


def test_to_response_none_content(provider):
    """Gemini can return candidates with content=None entirely."""
    p, _ = provider
    mock_candidate = MagicMock()
    mock_candidate.content = None
    mock_candidate.finish_reason = None
    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata = None

    result = p._to_response(mock_response)
    assert result.content == ""
    assert result.tool_calls is None


def test_to_response_no_candidates(provider):
    """Gemini can return empty candidates list (safety filter)."""
    p, _ = provider
    mock_response = MagicMock()
    mock_response.candidates = []
    mock_response.usage_metadata = None

    result = p._to_response(mock_response)
    assert result.content == ""
    assert result.tool_calls is None
    assert result.finish_reason == "stop"


def test_health_check_success(provider):
    p, client = provider
    client.models.generate_content.return_value = MagicMock()
    assert p.health_check() is True


def test_health_check_failure(provider):
    p, _ = provider
    # Override the client's method directly on the provider instance
    p.client.models.generate_content.side_effect = Exception("API error")
    assert p.health_check() is False


def test_retry_on_quota_error(provider):
    """Gemini provider retries on 429 quota errors."""
    from unittest.mock import patch, MagicMock
    p, _ = provider

    call_count = 0
    def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("429 quota exceeded")
        mock_resp = MagicMock()
        mock_resp.candidates = [MagicMock(
            content=MagicMock(parts=[MagicMock(text="ok", function_call=None)]),
            finish_reason=MagicMock(name="STOP"),
        )]
        mock_resp.usage_metadata = MagicMock(prompt_token_count=5, candidates_token_count=5)
        return mock_resp

    p.client.models.generate_content.side_effect = side_effect
    with patch("time.sleep"):
        result = p._call_with_retry([], MagicMock())
    assert call_count == 3


def test_no_retry_on_auth_error_gemini(provider):
    """Gemini provider does not retry on auth errors."""
    from unittest.mock import patch
    p, _ = provider

    p.client.models.generate_content.side_effect = Exception("API key invalid")
    with patch("time.sleep"):
        with pytest.raises(Exception, match="API key invalid"):
            p._call_with_retry([], MagicMock(), max_retries=3)
    assert p.client.models.generate_content.call_count == 1


def test_raises_after_max_retries_gemini(provider):
    """Gemini provider raises after all retries exhausted."""
    from unittest.mock import patch
    p, _ = provider

    p.client.models.generate_content.side_effect = Exception("500 resource exhausted")
    with patch("time.sleep"):
        with pytest.raises(Exception, match="500 resource exhausted"):
            p._call_with_retry([], MagicMock(), max_retries=2)
    assert p.client.models.generate_content.call_count == 2


# --------------------------------------------------------------------------
# Constraint-layer wiring tests
# --------------------------------------------------------------------------


def _build_constraint_provider():
    """Build a GeminiProvider with mocked google-genai client + success response."""
    with patch("google.genai.Client") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "hi"
        mock_part.function_call = None
        mock_candidate.content.parts = [mock_part]
        mock_candidate.finish_reason = "STOP"
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=5, candidates_token_count=3, total_token_count=8,
        )
        mock_client.models.generate_content.return_value = mock_response
        from mullet.core.providers.gemini import GeminiProvider
        p = GeminiProvider(api_key="test-key", model="gemini-2.5-flash")
        return p, mock_client


def test_gemini_name_is_set():
    p, _ = _build_constraint_provider()
    assert p.name == "gemini"


def test_gemini_shape_request_is_called():
    p, _ = _build_constraint_provider()
    with patch("mullet.core.providers.gemini.shape_request") as mock_shape:
        mock_shape.side_effect = lambda model, kwargs: kwargs
        p.create(messages=[{"role": "user", "content": "hi"}])
        assert mock_shape.called
        args, _kw = mock_shape.call_args
        assert args[0] == "gemini-2.5-flash"


def test_gemini_wire_log_emitted(caplog):
    import logging as _lg
    p, _ = _build_constraint_provider()
    caplog.set_level(_lg.INFO, logger="mullet.core.providers.gemini")
    p.create(messages=[{"role": "user", "content": "hi"}])
    wire_logs = [r for r in caplog.records if "wire:" in r.message and "provider=gemini" in r.message]
    assert wire_logs


def test_gemini_constraint_error_heals_and_retries():
    p, mock_client = _build_constraint_provider()
    call_count = {"n": 0}
    success_response = mock_client.models.generate_content.return_value

    def side_effect(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("invalid parameter 'foo'")
        return success_response

    mock_client.models.generate_content.side_effect = side_effect

    with patch.object(type(p), "parse_error", return_value=("foo", "test reason")):
        with patch("mullet.core.providers.gemini.get_constraint_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            p.create(messages=[{"role": "user", "content": "hi"}])
            assert call_count["n"] == 2
            mock_reg.return_value.learn.assert_called_once()
