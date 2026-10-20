"""Coverage-gap tests for mullet/core/providers/anthropic.py.

Covers branches missing from test_anthropic_provider.py:
- create(): system/tools/reasoning_config conditionals
- _call_with_retry(): log_interaction exception on success path,
  constraint self-healing retry exception, log_interaction on error paths,
  log_interaction after max retries exhausted
- create_stream(): system/tools conditionals, streaming event types
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def provider():
    """Create AnthropicProvider with mocked SDK."""
    mock_anthropic_mod = MagicMock()
    mock_client = MagicMock()
    mock_anthropic_mod.Anthropic.return_value = mock_client
    with patch.dict(sys.modules, {"anthropic": mock_anthropic_mod}):
        from mullet.core.providers.anthropic import AnthropicProvider
        p = AnthropicProvider(api_key="test-key", model="claude-test")
        yield p, mock_client


def _make_response(text="hi"):
    mock = MagicMock()
    mock.content = [MagicMock(type="text", text=text)]
    mock.stop_reason = "end_turn"
    mock.usage = MagicMock(
        input_tokens=5, output_tokens=3,
        cache_read_input_tokens=0, cache_creation_input_tokens=0,
    )
    return mock


# ---------------------------------------------------------------------------
# create() conditional branches
# ---------------------------------------------------------------------------

def test_create_with_system_message(provider):
    p, client = provider
    client.messages.create.return_value = _make_response()
    p.create(
        messages=[
            {"role": "system", "content": "Sys prompt"},
            {"role": "user", "content": "Hello"},
        ],
    )
    call_kwargs = client.messages.create.call_args[1]
    system = call_kwargs.get("system")
    assert isinstance(system, list)
    assert system[0]["text"] == "Sys prompt"
    assert system[0]["type"] == "text"


def test_create_with_tools(provider):
    p, client = provider
    client.messages.create.return_value = _make_response()
    p.create(
        messages=[{"role": "user", "content": "Hello"}],
        tools=[{"type": "function", "function": {"name": "fn"}}],
    )
    call_kwargs = client.messages.create.call_args[1]
    assert "tools" in call_kwargs


def test_create_with_reasoning_config(provider):
    p, client = provider
    client.messages.create.return_value = _make_response()
    p.create(
        messages=[{"role": "user", "content": "Hello"}],
        reasoning_config={"type": "extended", "budget_tokens": 1000},
    )
    call_kwargs = client.messages.create.call_args[1]
    assert call_kwargs.get("thinking") == {"type": "extended", "budget_tokens": 1000}


# ---------------------------------------------------------------------------
# _call_with_retry() exception handling branches
# ---------------------------------------------------------------------------

def test_log_interaction_exception_on_success_path_is_ignored(provider):
    """Line 100-101: exception inside log_interaction must not bubble up."""
    p, client = provider
    client.messages.create.return_value = _make_response()
    with patch("mullet.core.logging.interaction_log.log_interaction", side_effect=RuntimeError("boom")):
        result = p._call_with_retry({"messages": []})
    assert result.content == "hi"


def test_constraint_heal_retry_raises_is_logged_and_raised(provider):
    """Line 126-127: second call after constraint heal raises → bubble up.

    Note: due to Python ``raise`` semantics the *original* exception is
    re-raised even though the code assigns ``e = e2``.
    """
    p, client = provider
    client.messages.create.side_effect = [
        Exception("unsupported_value: top_p"),
        Exception("still broken"),
    ]
    with patch.object(type(p), "parse_error", return_value=("top_p", "reason")):
        with patch("mullet.core.providers.anthropic.get_constraint_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            with pytest.raises(Exception, match="unsupported_value: top_p"):
                p._call_with_retry({"messages": []})
            # Two calls: first (fails), second (constraint heal retry, also fails)
            assert client.messages.create.call_count == 2
            mock_reg.return_value.learn.assert_called_once()


def test_log_interaction_exception_on_error_path_is_ignored(provider):
    """Line 141-142: log_interaction exception on non-retryable error must not mask real error."""
    p, client = provider
    client.messages.create.side_effect = Exception("bad request")
    with patch("mullet.core.logging.interaction_log.log_interaction", side_effect=RuntimeError("log boom")):
        with pytest.raises(Exception, match="bad request"):
            p._call_with_retry({"messages": []})


def test_log_interaction_exception_after_max_retries_is_ignored(provider):
    """Line 152-153: log_interaction exception after exhausted retries must not mask real error."""
    p, client = provider
    client.messages.create.side_effect = Exception("429 rate limit")
    with patch("time.sleep"):
        with patch("mullet.core.logging.interaction_log.log_interaction", side_effect=RuntimeError("log boom")):
            with pytest.raises(Exception, match="429 rate limit"):
                p._call_with_retry({"messages": []}, max_retries=1)


# ---------------------------------------------------------------------------
# create_stream() conditional branches
# ---------------------------------------------------------------------------

def test_create_stream_with_system_and_tools(provider):
    p, client = provider
    mock_stream_cm = MagicMock()
    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(return_value=iter([]))
    mock_stream_cm.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream_cm.__exit__ = MagicMock(return_value=False)
    client.messages.stream.return_value = mock_stream_cm

    gen = p.create_stream(
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "fn"}}],
    )
    chunks = list(gen)
    assert chunks == []
    call_kwargs = client.messages.stream.call_args[1]
    assert call_kwargs.get("system") == "sys"
    assert "tools" in call_kwargs


# ---------------------------------------------------------------------------
# create_stream() event handling
# ---------------------------------------------------------------------------

def test_create_stream_content_block_delta_text(provider):
    p, client = provider
    event = MagicMock()
    event.type = "content_block_delta"
    event.delta = MagicMock()
    event.delta.text = "hello"

    mock_stream_cm = MagicMock()
    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(return_value=iter([event]))
    mock_stream_cm.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream_cm.__exit__ = MagicMock(return_value=False)
    client.messages.stream.return_value = mock_stream_cm

    chunks = list(p.create_stream(messages=[{"role": "user", "content": "hi"}]))
    assert len(chunks) == 1
    assert chunks[0].content == "hello"


def test_create_stream_content_block_delta_partial_json(provider):
    p, client = provider
    event = MagicMock()
    event.type = "content_block_delta"
    event.delta = MagicMock()
    event.delta.partial_json = '{"arg":1}'
    del event.delta.text

    mock_stream_cm = MagicMock()
    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(return_value=iter([event]))
    mock_stream_cm.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream_cm.__exit__ = MagicMock(return_value=False)
    client.messages.stream.return_value = mock_stream_cm

    chunks = list(p.create_stream(messages=[{"role": "user", "content": "hi"}]))
    assert chunks[0].tool_call_delta == {"partial_json": '{"arg":1}'}


def test_create_stream_content_block_start_tool_use(provider):
    p, client = provider
    event = MagicMock()
    event.type = "content_block_start"
    event.content_block = MagicMock()
    event.content_block.type = "tool_use"
    event.content_block.id = "tool_1"
    event.content_block.name = "search"

    mock_stream_cm = MagicMock()
    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(return_value=iter([event]))
    mock_stream_cm.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream_cm.__exit__ = MagicMock(return_value=False)
    client.messages.stream.return_value = mock_stream_cm

    chunks = list(p.create_stream(messages=[{"role": "user", "content": "hi"}]))
    assert chunks[0].tool_call_delta == {
        "id": "tool_1",
        "type": "function",
        "function": {"name": "search", "arguments": ""},
    }


def test_create_stream_message_delta_stop_reason(provider):
    p, client = provider
    event = MagicMock()
    event.type = "message_delta"
    event.delta = MagicMock()
    event.delta.stop_reason = "end_turn"

    mock_stream_cm = MagicMock()
    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(return_value=iter([event]))
    mock_stream_cm.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream_cm.__exit__ = MagicMock(return_value=False)
    client.messages.stream.return_value = mock_stream_cm

    chunks = list(p.create_stream(messages=[{"role": "user", "content": "hi"}]))
    assert chunks[0].finish_reason == "end_turn"


def test_create_stream_message_delta_no_stop_reason(provider):
    p, client = provider
    event = MagicMock()
    event.type = "message_delta"
    event.delta = MagicMock()
    event.delta.stop_reason = None

    mock_stream_cm = MagicMock()
    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(return_value=iter([event]))
    mock_stream_cm.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream_cm.__exit__ = MagicMock(return_value=False)
    client.messages.stream.return_value = mock_stream_cm

    chunks = list(p.create_stream(messages=[{"role": "user", "content": "hi"}]))
    assert chunks[0].finish_reason == "stop"


def test_create_stream_constraint_heal_on_open(provider):
    """Test constraint self-healing when stream open fails."""
    p, client = provider
    mock_stream_cm = MagicMock()
    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(return_value=iter([]))
    mock_stream_cm.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream_cm.__exit__ = MagicMock(return_value=False)

    call_count = {"n": 0}

    def stream_side_effect(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("unsupported_value: temperature")
        return mock_stream_cm

    client.messages.stream.side_effect = stream_side_effect

    with patch.object(type(p), "parse_error", return_value=("temperature", "test reason")):
        with patch("mullet.core.providers.anthropic.get_constraint_registry") as mock_reg:
            mock_reg.return_value = MagicMock()
            chunks = list(p.create_stream(messages=[{"role": "user", "content": "hi"}]))
    assert chunks == []
    assert call_count["n"] == 2
    mock_reg.return_value.learn.assert_called_once()
