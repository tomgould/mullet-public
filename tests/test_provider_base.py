"""Tests for mullet.core.providers.base — Provider base methods."""


from mullet.core.providers.base import LLMResponse, Provider, StreamChunk


class _MinimalProvider(Provider):
    """Minimal Provider for testing default method behaviors."""
    name = "test-provider"

    def __init__(self, model: str = ""):
        self.model = model

    def create(self, messages, tools=None, **kwargs):
        raise NotImplementedError


# --- describe() -----------------------------------------------------------


def test_describe_returns_provider_slash_model():
    """describe() returns the canonical 'provider/model' label."""
    p = _MinimalProvider(model="my-model")
    assert p.describe() == "test-provider/my-model"


def test_describe_empty_model_returns_name_only():
    """When self.model is empty, describe() falls back to the provider name."""
    p = _MinimalProvider(model="")
    assert p.describe() == "test-provider"


def test_describe_missing_model_attr_returns_name_only():
    """If a Provider subclass doesn't set self.model at all, describe() is graceful."""
    class _NoModel(Provider):
        name = "noname"
        def create(self, messages, tools=None, **kwargs):
            raise NotImplementedError

    assert _NoModel().describe() == "noname"


# --- parse_error() default -------------------------------------------------


def test_parse_error_default_returns_none_for_unknown_error():
    """Base Provider.parse_error defers to parse_constraint_error; returns None
    for unknown errors."""
    p = _MinimalProvider(model="unknown-model")
    # An error no parser recognizes
    result = p.parse_error(RuntimeError("unexpected network glitch"))
    assert result is None


def test_parse_error_recognizes_known_pattern():
    """Provider.parse_error dispatches by self.name — an anthropic provider
    with a known anthropic error pattern should return a (param, reason)."""
    class _AnthropicLike(Provider):
        name = "anthropic"
        def __init__(self):
            self.model = "claude-sonnet-4-6"
        def create(self, messages, tools=None, **kwargs):
            raise NotImplementedError

    p = _AnthropicLike()
    # "top_p must be between 0 and 1" was added to the constraint parser
    result = p.parse_error(Exception("invalid_request_error: top_p must be between 0 and 1"))
    assert result is not None
    param, _reason = result
    assert param == "top_p"


# --- create_stream default (fallback) --------------------------------------


def test_create_stream_default_wraps_create():
    """Default create_stream wraps create() and yields one StreamChunk."""
    class _WithCreate(Provider):
        name = "t"
        def __init__(self):
            self.model = "m"
        def create(self, messages, tools=None, max_tokens=4096, temperature=0.7,
                   stream=False, reasoning_config=None):
            return LLMResponse(content="hello world", finish_reason="stop")

    p = _WithCreate()
    chunks = list(p.create_stream(messages=[{"role": "user", "content": "hi"}]))
    assert len(chunks) == 1
    assert chunks[0].content == "hello world"
    assert chunks[0].finish_reason == "stop"


# --- health_check default --------------------------------------------------


def test_health_check_default_true():
    """Default health_check returns True."""
    assert _MinimalProvider().health_check() is True


# --- supports_tool_calling property ---------------------------------------


def test_supports_tool_calling_default_true():
    assert _MinimalProvider().supports_tool_calling is True


# --- dataclass shapes ------------------------------------------------------


def test_llm_response_defaults():
    r = LLMResponse()
    assert r.content == ""
    assert r.tool_calls is None
    assert r.finish_reason == "stop"
    assert r.model == ""
    assert r.provider == ""


def test_llm_response_with_tool_calls():
    r = LLMResponse(content="", tool_calls=[{"id": "c1", "type": "function"}])
    assert r.tool_calls is not None
    assert r.tool_calls[0]["id"] == "c1"


def test_stream_chunk_defaults():
    c = StreamChunk()
    assert c.content == ""
    assert c.tool_call_delta is None
    assert c.finish_reason is None
