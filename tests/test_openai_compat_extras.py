"""Additional tests for openai_compat — strip_for_small_context, infer_name, to_response."""
from unittest.mock import MagicMock, patch

from mullet.core.providers.openai_compat import (
    OpenAICompatibleProvider,
    _strip_for_small_context,
    get_consecutive_failures,
    reset_consecutive_failures,
)


# ── _strip_for_small_context ──────────────────────────────────────────────────

def test_strip_removes_tools():
    kwargs = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "search"}],
        "tool_choice": "auto",
    }
    stripped = _strip_for_small_context(kwargs)
    assert "tools" not in stripped
    assert "tool_choice" not in stripped


def test_strip_truncates_long_system_prompt():
    long_prompt = "x" * 5000
    kwargs = {
        "messages": [
            {"role": "system", "content": long_prompt},
            {"role": "user", "content": "hello"},
        ],
    }
    stripped = _strip_for_small_context(kwargs)
    system_msg = next(m for m in stripped["messages"] if m["role"] == "system")
    assert len(system_msg["content"]) <= 3300
    assert "[truncated" in system_msg["content"]


def test_strip_preserves_short_system_prompt():
    kwargs = {
        "messages": [{"role": "system", "content": "short prompt"}],
    }
    stripped = _strip_for_small_context(kwargs)
    assert stripped["messages"][0]["content"] == "short prompt"


def test_strip_does_not_mutate_original():
    kwargs = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "read_file"}],
    }
    _strip_for_small_context(kwargs)
    assert "tools" in kwargs  # original unchanged


def test_strip_preserves_user_messages():
    kwargs = {
        "messages": [
            {"role": "user", "content": "test message"},
        ],
        "tools": [],
    }
    stripped = _strip_for_small_context(kwargs)
    assert stripped["messages"][0]["content"] == "test message"


# ── _infer_name ───────────────────────────────────────────────────────────────

def _make_provider(base_url: str) -> OpenAICompatibleProvider:
    with patch("openai.OpenAI"):
        return OpenAICompatibleProvider(base_url=base_url, api_key="test", model="x")


def test_infer_name_localhost():
    p = _make_provider("http://localhost:1234/v1")
    assert p.name == "local"


def test_infer_name_127():
    p = _make_provider("http://127.0.0.1:1234/v1")
    assert p.name == "local"


def test_infer_name_openrouter():
    p = _make_provider("https://openrouter.ai/api/v1")
    assert p.name == "openrouter"


def test_infer_name_openai():
    p = _make_provider("https://api.openai.com/v1")
    assert p.name == "openai"


def test_infer_name_custom():
    p = _make_provider("https://my-custom-endpoint.example.com/v1")
    assert p.name == "custom"


# ── _to_response ──────────────────────────────────────────────────────────────

def _make_mock_response(content="hello", tool_calls=None, finish_reason="stop",
                         prompt_tokens=10, completion_tokens=5):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    resp.model = "test-model"
    return resp


def test_to_response_basic():
    p = _make_provider("http://localhost:1234/v1")
    resp = _make_mock_response(content="hello world")
    result = p._to_response(resp)
    assert result.content == "hello world"
    assert result.finish_reason == "stop"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5


def test_to_response_empty_choices():
    p = _make_provider("http://localhost:1234/v1")
    resp = MagicMock()
    resp.choices = []
    result = p._to_response(resp)
    assert result.content == ""
    assert result.finish_reason == "error"


def test_to_response_with_tool_calls():
    p = _make_provider("http://localhost:1234/v1")
    tc = MagicMock()
    tc.id = "tc_1"
    tc.function.name = "search"
    tc.function.arguments = '{"q": "test"}'
    resp = _make_mock_response(tool_calls=[tc])
    result = p._to_response(resp)
    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["function"]["name"] == "search"


# ── thread-safe failure tracking ──────────────────────────────────────────────

def test_failure_tracking_thread_safe():
    """Concurrent reset+get should not raise exceptions."""
    import threading
    errors = []

    def worker():
        try:
            for _ in range(20):
                get_consecutive_failures("test_provider")
                reset_consecutive_failures("test_provider")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Thread safety errors: {errors}"
