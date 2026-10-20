"""Coverage gap tests for openai_compat — lines missed in existing test suites."""
import sys
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from mullet.core.providers.openai_compat import (
    OpenAICompatibleProvider,
    _brief_task_summary,
    _hud_agent_track,
)

# Use real OpenAI SDK exception classes so isinstance checks inside _call_with_retry match.
try:
    from openai import RateLimitError, APIStatusError
except Exception:  # pragma: no cover
    RateLimitError = type(None)  # type: ignore[misc]
    APIStatusError = type(None)  # type: ignore[misc]


# ── _brief_task_summary ───────────────────────────────────────────────────────

def test_brief_task_summary_multipart_content():
    """Multipart list content is joined and truncated."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]},
    ]
    assert _brief_task_summary(messages) == "hello  world"


def test_brief_task_summary_no_user():
    """No user message falls back to 'LLM call'."""
    assert _brief_task_summary([{"role": "system", "content": "hi"}]) == "LLM call"


def test_brief_task_summary_empty_messages():
    assert _brief_task_summary([]) == "LLM call"
    assert _brief_task_summary(None) == "LLM call"


# ── _hud_agent_track exception handling ───────────────────────────────────────

def test_hud_agent_track_add_fails():
    """If add_agent raises, the context manager still yields."""
    with patch("mullet.hud.agent_state.add_agent", side_effect=RuntimeError("boom")):
        with _hud_agent_track("p", "m", [{"role": "user", "content": "hi"}]):
            pass  # should not raise


def test_hud_agent_track_update_fails():
    """If update_agent_status raises inside except, exception still propagates."""
    with patch("mullet.hud.agent_state.add_agent") as mock_add:
        with patch("mullet.hud.agent_state.update_agent_status", side_effect=RuntimeError("boom")):
            with pytest.raises(ValueError):
                with _hud_agent_track("p", "m", [{"role": "user", "content": "hi"}]):
                    raise ValueError("inner")
    mock_add.assert_called_once()


def test_hud_agent_track_remove_fails():
    """If remove_agent raises, it is swallowed."""
    with patch("mullet.hud.agent_state.add_agent"):
        with patch("mullet.hud.agent_state.remove_agent", side_effect=RuntimeError("boom")):
            with _hud_agent_track("p", "m", [{"role": "user", "content": "hi"}]):
                pass


# ── _is_garbage_content empty non_ws ──────────────────────────────────────────

def test_is_garbage_content_all_whitespace():
    """Whitespace-only string of length >=40: non_ws empty → False."""
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    assert p._is_garbage_content(" " * 50) is False


# ── OpenRouter-specific headers / config ──────────────────────────────────────

def test_openrouter_providers_order_header():
    p = OpenAICompatibleProvider(
        base_url="https://openrouter.ai/api/v1",
        api_key="test",
        openrouter_config={"providers_order": ["together", "fireworks"]},
    )
    assert p.client.default_headers.get("X-OpenRouter-Provider-Order") == "together,fireworks"


def test_create_with_reasoning_config():
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "test-model"

    def capture(**kwargs):
        capture.kwargs = kwargs
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok", tool_calls=None), finish_reason="stop")]
        resp.model = "test-model"
        resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return resp

    with patch.object(p.client.chat.completions, "create", side_effect=capture):
        p.create([{"role": "user", "content": "hi"}], reasoning_config={"effort": "high"})

    assert capture.kwargs.get("reasoning") == {"effort": "high"}


def test_create_with_openrouter_provider_prefs():
    p = OpenAICompatibleProvider(
        base_url="https://openrouter.ai/api/v1",
        api_key="test",
        openrouter_config={
            "providers_allowed": ["together"],
            "providers_ignored": ["fireworks"],
        },
    )
    p.model = "test-model"

    def capture(**kwargs):
        capture.kwargs = kwargs
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok", tool_calls=None), finish_reason="stop")]
        resp.model = "test-model"
        resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return resp

    with patch.object(p.client.chat.completions, "create", side_effect=capture):
        p.create([{"role": "user", "content": "hi"}])

    transforms = capture.kwargs.get("transforms", {})
    assert transforms.get("providers", {}).get("allow") == ["together"]
    assert transforms.get("providers", {}).get("deny") == ["fireworks"]


# ── create() stream path ──────────────────────────────────────────────────────

def test_create_stream_path():
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "test-model"

    mock_resp = MagicMock()
    mock_resp.content = "streamed"
    with patch.object(p, "_handle_stream", return_value=mock_resp) as mock_handle:
        result = p.create([{"role": "user", "content": "hi"}], stream=True)

    assert result.content == "streamed"
    mock_handle.assert_called_once()


# ── create_stream tool_call delta ─────────────────────────────────────────────

def test_create_stream_yields_tool_call_delta():
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "test-model"

    tc = MagicMock()
    tc.index = 0
    tc.id = "tc_1"
    tc.function.name = "search"
    tc.function.arguments = '{"q": "x"}'

    delta = MagicMock()
    delta.content = None
    delta.tool_calls = [tc]

    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=delta, finish_reason=None)]

    with patch.object(p.client.chat.completions, "create", return_value=iter([chunk])):
        chunks = list(p.create_stream([{"role": "user", "content": "hi"}], tools=[{"type": "function", "function": {"name": "search"}}]))

    assert len(chunks) == 1
    assert chunks[0].tool_call_delta is not None
    assert chunks[0].tool_call_delta["function"]["name"] == "search"


# ── _handle_stream ────────────────────────────────────────────────────────────

def _make_stream_chunk(content=None, tool_calls=None, finish_reason=None, usage=None):
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = usage
    return chunk


def test_handle_stream_basic():
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "test-model"

    chunks = [
        _make_stream_chunk(content="hello ", finish_reason=None),
        _make_stream_chunk(content="world", finish_reason="stop"),
    ]
    with patch.object(p.client.chat.completions, "create", return_value=iter(chunks)):
        result = p._handle_stream({"messages": [{"role": "user", "content": "hi"}]})

    assert result.content == "hello world"
    assert result.finish_reason == "stop"
    assert result.model == "test-model"


def test_handle_stream_with_usage():
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "test-model"

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15

    chunks = [
        _make_stream_chunk(content="hi", usage=usage),
    ]
    with patch.object(p.client.chat.completions, "create", return_value=iter(chunks)):
        result = p._handle_stream({"messages": [{"role": "user", "content": "hi"}]})

    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5
    assert result.usage.total_tokens == 15


def test_handle_stream_with_tool_calls():
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "test-model"

    tc1 = MagicMock()
    tc1.index = 0
    tc1.id = "tc_1"
    tc1.function.name = "search"
    tc1.function.arguments = '{"q"'

    tc2 = MagicMock()
    tc2.index = 0
    tc2.id = None
    tc2.function.name = None
    tc2.function.arguments = ': "x"}'

    chunks = [
        _make_stream_chunk(content="", tool_calls=[tc1]),
        _make_stream_chunk(content="", tool_calls=[tc2]),
    ]
    with patch.object(p.client.chat.completions, "create", return_value=iter(chunks)):
        result = p._handle_stream({"messages": [{"role": "user", "content": "hi"}]})

    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["function"]["arguments"] == '{"q": "x"}'


# ── log_interaction exception swallowing ──────────────────────────────────────

def test_call_with_retry_log_interaction_success_raises():
    """Exception in log_interaction on success path is swallowed (lines 324-325)."""
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "test-model"

    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="ok", tool_calls=None), finish_reason="stop")]
    resp.model = "test-model"
    resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)

    with patch.object(p.client.chat.completions, "create", return_value=resp):
        with patch("mullet.core.logging.interaction_log.log_interaction", side_effect=RuntimeError("log fail")):
            result = p._call_with_retry({"messages": [{"role": "user", "content": "hi"}]})
    assert result.content == "ok"


def test_call_with_retry_log_interaction_failure_raises():
    """Exception in log_interaction on failure path is swallowed (lines 442-443)."""
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "test-model"

    with patch.object(p.client.chat.completions, "create", side_effect=ValueError("nope")):
        with patch("mullet.core.logging.interaction_log.log_interaction", side_effect=RuntimeError("log fail")):
            with pytest.raises(ValueError, match="nope"):
                p._call_with_retry({"messages": [{"role": "user", "content": "hi"}]})


def test_call_with_retry_log_interaction_exhausted_raises():
    """Exception in log_interaction when retries exhausted is swallowed (lines 475-476)."""
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "test-model"

    with patch.object(p.client.chat.completions, "create", side_effect=RuntimeError("500")):
        with patch("mullet.core.logging.interaction_log.log_interaction", side_effect=RuntimeError("log fail")):
            with pytest.raises(RuntimeError, match="500"):
                p._call_with_retry({"messages": [{"role": "user", "content": "hi"}]})


def test_call_with_retry_constraint_fix_log_interaction_success_raises():
    """Lines 422-423: log_interaction raises after constraint fix succeeds."""
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "test-model"

    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="ok", tool_calls=None), finish_reason="stop")]
    resp.model = "test-model"
    resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)

    call_count = [0]
    def fail_then_succeed(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise ValueError("Unsupported parameter: temperature")
        return resp

    with patch.object(p.client.chat.completions, "create", side_effect=fail_then_succeed):
        with patch.object(p, "parse_error", return_value=("temperature", "unsupported")):
            with patch("mullet.core.logging.interaction_log.log_interaction", side_effect=RuntimeError("log fail")):
                result = p._call_with_retry({"messages": [{"role": "user", "content": "hi"}], "temperature": 0.7})
    assert result.content == "ok"


def test_call_with_retry_constraint_fix_retry_fails():
    """Lines 425-426: constraint fix retry also fails."""
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "test-model"

    with patch.object(p.client.chat.completions, "create", side_effect=ValueError("bad param")):
        with patch.object(p, "parse_error", return_value=("temperature", "unsupported")):
            with pytest.raises(ValueError, match="bad param"):
                p._call_with_retry({"messages": [{"role": "user", "content": "hi"}], "temperature": 0.7})


# ── context strip exception (lines 382-383) ───────────────────────────────────

def test_call_with_retry_context_strip_fails():
    """Lines 382-383: If _strip_for_small_context raises, exception is swallowed and original error re-raised."""
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "test-model"
    # Replace the openai client with a MagicMock so we control create() directly
    p.client = MagicMock()
    p.client.chat.completions.create.side_effect = Exception("context size has been exceeded for this model")

    with patch("mullet.core.providers.openai_compat._strip_for_small_context", side_effect=RuntimeError("strip fail")), \
         patch("mullet.core.providers.context_gate.assert_context_fits"):
        with pytest.raises(Exception):
            p._call_with_retry({"messages": [{"role": "user", "content": "hi"}]})


# ── RateLimitError / APIStatusError handling ──────────────────────────────────

class FakeRateLimitError(RateLimitError if RateLimitError is not type(None) else Exception):  # type: ignore[misc]
    def __init__(self, message):
        super().__init__(message, response=MagicMock(), body=None)


class FakeAPIStatusError(APIStatusError if APIStatusError is not type(None) else Exception):  # type: ignore[misc]
    def __init__(self, message, status_code):
        super().__init__(message, response=MagicMock(), body=None)
        self.status_code = status_code


def test_rate_limit_free_tier_no_retry():
    """Lines 338-344: free-tier RateLimitError is not retried."""
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "model:free"

    err = FakeRateLimitError("rate limited")
    with patch.object(p.client.chat.completions, "create", side_effect=err):
        with pytest.raises(FakeRateLimitError):
            p._call_with_retry({"messages": [{"role": "user", "content": "hi"}]})


def test_rate_limit_non_free_retries():
    """Lines 338, 344: non-free RateLimitError is retried then succeeds."""
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "model"

    call_count = [0]
    def fail_once(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise FakeRateLimitError("rate limited")
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok", tool_calls=None), finish_reason="stop")]
        resp.model = "test-model"
        resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return resp

    with patch.object(p.client.chat.completions, "create", side_effect=fail_once):
        result = p._call_with_retry({"messages": [{"role": "user", "content": "hi"}]})
        assert result.content == "ok"
        assert call_count[0] == 2


def test_api_status_error_500_retries():
    """Lines 345-346: APIStatusError with 500 is retryable."""
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "model"

    call_count = [0]
    def fail_once(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise FakeAPIStatusError("server error", 500)
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok", tool_calls=None), finish_reason="stop")]
        resp.model = "test-model"
        resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return resp

    with patch.object(p.client.chat.completions, "create", side_effect=fail_once):
        result = p._call_with_retry({"messages": [{"role": "user", "content": "hi"}]})
        assert result.content == "ok"
        assert call_count[0] == 2


def test_fallback_http_codes_in_message():
    """Lines 349-363: non-SDK errors with HTTP codes in message are retryable."""
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "model"

    call_count = [0]
    def fail_once(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("Server returned 503 Service Unavailable")
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok", tool_calls=None), finish_reason="stop")]
        resp.model = "test-model"
        resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return resp

    with patch.object(p.client.chat.completions, "create", side_effect=fail_once):
        result = p._call_with_retry({"messages": [{"role": "user", "content": "hi"}]})
        assert result.content == "ok"
        assert call_count[0] == 2


def test_fallback_429_free_tier_in_message():
    """Lines 352-363: 429 in message on free tier is not retried."""
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "model:free"

    with patch.object(p.client.chat.completions, "create", side_effect=RuntimeError("429 Too Many Requests")):
        with pytest.raises(RuntimeError, match="429"):
            p._call_with_retry({"messages": [{"role": "user", "content": "hi"}]})


def test_import_fallback_for_rate_limit_error():
    """Lines 334-336: ImportError fallback makes RateLimitError = type(None)."""
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "model"

    # Force the local import to fail by temporarily replacing openai module
    class FakeOpenAIMod:
        def __getattr__(self, name):
            raise ImportError

    orig_module = sys.modules.get("openai")
    sys.modules["openai"] = FakeOpenAIMod()

    call_count = [0]
    def fail_once(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("500")
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok", tool_calls=None), finish_reason="stop")]
        resp.model = "test-model"
        resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return resp

    try:
        with patch.object(p.client.chat.completions, "create", side_effect=fail_once):
            result = p._call_with_retry({"messages": [{"role": "user", "content": "hi"}]})
        assert result.content == "ok"
    finally:
        if orig_module is not None:
            sys.modules["openai"] = orig_module
        else:
            del sys.modules["openai"]


# ── remaining easy gaps ───────────────────────────────────────────────────────

def test_escalation_recommended_with_allowed():
    """Line 128: allowed set with a cloud target returns True."""
    from mullet.core.providers import openai_compat as oc
    oc._consecutive_failures["flaky"] = oc._ESCALATION_THRESHOLD
    assert oc.is_escalation_recommended("flaky", allowed={"kimi", "local"}) is True
    oc._consecutive_failures.pop("flaky")


def test_create_passes_tools():
    """Line 184: tools are passed through when supported."""
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "test-model"

    def capture(**kwargs):
        capture.kwargs = kwargs
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok", tool_calls=None), finish_reason="stop")]
        resp.model = "test-model"
        resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return resp

    tool = {"type": "function", "function": {"name": "search"}}
    with patch.object(p.client.chat.completions, "create", side_effect=capture):
        p.create([{"role": "user", "content": "hi"}], tools=[tool])
    assert capture.kwargs.get("tools") == [tool]


def test_create_stream_constraint_self_healing():
    """Lines 237-248: stream-open constraint error is caught, learned, and retried."""
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "test-model"

    call_count = [0]
    def fail_then_succeed(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise ValueError("Unsupported parameter: temperature")
        # Return an iterator for the stream
        chunk = MagicMock()
        chunk.choices = [MagicMock(delta=MagicMock(content="ok", tool_calls=None), finish_reason="stop")]
        return iter([chunk])

    with patch.object(p.client.chat.completions, "create", side_effect=fail_then_succeed):
        with patch.object(p, "parse_error", return_value=("temperature", "unsupported")):
            chunks = list(p.create_stream([{"role": "user", "content": "hi"}], temperature=0.7))

    assert call_count[0] == 2
    assert chunks[0].content == "ok"


def test_create_stream_constraint_no_fix_raises():
    """Line 240: if parse_error returns None, the original exception is raised."""
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "test-model"

    with patch.object(p.client.chat.completions, "create", side_effect=ValueError("bad param")):
        with patch.object(p, "parse_error", return_value=None):
            with pytest.raises(ValueError, match="bad param"):
                list(p.create_stream([{"role": "user", "content": "hi"}]))


def test_create_stream_empty_choices():
    """Line 252: chunks with no choices are skipped."""
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "test-model"

    empty_chunk = MagicMock()
    empty_chunk.choices = []

    content_chunk = MagicMock()
    content_chunk.choices = [MagicMock(delta=MagicMock(content="hi", tool_calls=None), finish_reason="stop")]

    with patch.object(p.client.chat.completions, "create", return_value=iter([empty_chunk, content_chunk])):
        chunks = list(p.create_stream([{"role": "user", "content": "hello"}]))

    assert len(chunks) == 1
    assert chunks[0].content == "hi"


def test_fallback_429_non_free_tier_in_message():
    """Line 359: 429 in message on non-free tier IS retried."""
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "model"

    call_count = [0]
    def fail_once(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("429 Too Many Requests")
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok", tool_calls=None), finish_reason="stop")]
        resp.model = "test-model"
        resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return resp

    with patch.object(p.client.chat.completions, "create", side_effect=fail_once):
        result = p._call_with_retry({"messages": [{"role": "user", "content": "hi"}]})
        assert result.content == "ok"
        assert call_count[0] == 2


def test_handle_stream_empty_choices():
    """Line 499: chunks with no choices are skipped in _handle_stream."""
    p = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
    p.model = "test-model"

    empty_chunk = MagicMock()
    empty_chunk.choices = []
    empty_chunk.usage = None

    content_chunk = _make_stream_chunk(content="hi", finish_reason="stop")

    with patch.object(p.client.chat.completions, "create", return_value=iter([empty_chunk, content_chunk])):
        result = p._handle_stream({"messages": [{"role": "user", "content": "hello"}]})

    assert result.content == "hi"
