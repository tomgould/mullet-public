import os
import subprocess
import sys
from unittest import mock

import pytest
import requests

from mullet.core.providers.openrouter import OpenRouterProvider
from mullet.core.router.complexity import Tier
from mullet.core.router.preflight import PreflightResult
from mullet.core.router.routing_matrix import select_openrouter_model


@mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_api_key"})
def test_openrouter_constructor_base_url():
    provider = OpenRouterProvider()
    assert provider.client.base_url == "https://openrouter.ai/api/v1/"


@mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_api_key"})
def test_openrouter_constructor_api_key_from_env():
    provider = OpenRouterProvider()
    assert provider.client.api_key == "test_api_key"


def test_openrouter_constructor_extra_headers():
    with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_api_key"}):
        provider = OpenRouterProvider()
        expected_headers = {
            "HTTP-Referer": "https://github.com/tomgould/mullet-public",
            "X-Title": "Mullet",
        }
        # The OpenAI client might add other headers, so we check for presence
        assert all(item in provider.client.default_headers.items() for item in expected_headers.items())
        # openai 3.x applies the key per request through auth_headers rather
        # than listing Authorization in default_headers, as 2.x did; both
        # expose it here.
        assert provider.client.auth_headers["Authorization"] == "Bearer test_api_key"


def test_openrouter_list_models_method_exists():
    with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_api_key"}):
        provider = OpenRouterProvider()
        assert hasattr(provider, "list_models")
        assert callable(provider.list_models)


@mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_api_key"})
@mock.patch("requests.get")
def test_openrouter_list_models(mock_requests_get):
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [{"id": "model1"}, {"id": "model2"}]}
    mock_requests_get.return_value = mock_response

    provider = OpenRouterProvider()
    models = provider.list_models()

    mock_requests_get.assert_called_once_with(
        "https://openrouter.ai/api/v1/models",
        headers={
            "Authorization": "Bearer test_api_key",
            "HTTP-Referer": "https://github.com/tomgould/mullet-public",
            "X-Title": "Mullet",
        },
    )
    assert models == ["model1", "model2"]

@mock.patch.dict(os.environ, {}, clear=True)
def test_openrouter_constructor_no_api_key_raises_error():
    with pytest.raises(ValueError, match="OpenRouter API key not found."):
        OpenRouterProvider()


@mock.patch.dict(os.environ, {}, clear=True)
def test_openrouter_api_key_constructor_param():
    provider = OpenRouterProvider(api_key="direct_key")
    assert provider.client.api_key == "direct_key"


@mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_api_key"})
def test_openrouter_name():
    provider = OpenRouterProvider()
    assert provider.name == "openrouter"


@mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_api_key"})
def test_openrouter_create_non_stream():
    provider = OpenRouterProvider(model="test-model")
    mock_usage = mock.Mock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    mock_message = mock.Mock(content="hello", tool_calls=None)
    mock_choice = mock.Mock(message=mock_message, finish_reason="stop")
    mock_completion = mock.Mock(choices=[mock_choice], usage=mock_usage)
    provider.client = mock.Mock()
    provider.client.chat.completions.create.return_value = mock_completion

    result = provider.create([{"role": "user", "content": "hi"}])
    assert result.content == "hello"
    assert result.usage.total_tokens == 15
    assert result.provider == "openrouter"

    call_kwargs = provider.client.chat.completions.create.call_args[1]
    assert call_kwargs["extra_headers"]["X-Title"] == "Mullet"


@mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_api_key"})
def test_openrouter_create_with_stream_true():
    provider = OpenRouterProvider(model="test-model")
    mock_stream = iter([mock.Mock()])
    with mock.patch.object(provider, "_handle_stream", return_value=mock_stream) as mock_handle:
        result = provider.create(messages=[{"role": "user", "content": "hi"}], stream=True)
        assert result is mock_stream
        mock_handle.assert_called_once()


@mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_api_key"})
def test_openrouter_create_with_tools():
    provider = OpenRouterProvider(model="test-model")
    mock_usage = mock.Mock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    mock_func = mock.Mock(name="mock_func")
    mock_func.name = "foo"
    mock_func.arguments = "{}"
    mock_tc = mock.Mock()
    mock_tc.id = "tc1"
    mock_tc.function = mock_func
    mock_message = mock.Mock(content="", tool_calls=[mock_tc])
    mock_choice = mock.Mock(message=mock_message, finish_reason="tool_calls")
    mock_completion = mock.Mock(choices=[mock_choice], usage=mock_usage)
    provider.client = mock.Mock()
    provider.client.chat.completions.create.return_value = mock_completion

    # supports_tool_calling is a property, patch it on the class
    with mock.patch.object(type(provider), "supports_tool_calling", new_callable=mock.PropertyMock, return_value=True):
        tools = [{"type": "function", "function": {"name": "foo"}}]
        result = provider.create([{"role": "user", "content": "hi"}], tools=tools)
        assert result.tool_calls is not None
        assert result.tool_calls[0]["function"]["name"] == "foo"

        call_kwargs = provider.client.chat.completions.create.call_args[1]
        assert "tools" in call_kwargs


@mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_api_key"})
def test_openrouter_create_with_reasoning_config():
    provider = OpenRouterProvider(model="test-model")
    mock_usage = mock.Mock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    mock_message = mock.Mock(content="thought", tool_calls=None)
    mock_choice = mock.Mock(message=mock_message, finish_reason="stop")
    mock_completion = mock.Mock(choices=[mock_choice], usage=mock_usage)
    provider.client = mock.Mock()
    provider.client.chat.completions.create.return_value = mock_completion

    reasoning = {"effort": "high"}
    provider.create([{"role": "user", "content": "think hard"}], reasoning_config=reasoning)

    call_kwargs = provider.client.chat.completions.create.call_args[1]
    assert call_kwargs["reasoning"] == {"effort": "high"}


@mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_api_key"})
def test_openrouter_create_stream_yields_chunks():
    provider = OpenRouterProvider(model="test-model")

    chunk1 = mock.Mock()
    chunk1.choices = [mock.Mock(delta=mock.Mock(content="hel", tool_calls=None), finish_reason=None)]
    chunk2 = mock.Mock()
    chunk2.choices = [mock.Mock(delta=mock.Mock(content="lo", tool_calls=None), finish_reason="stop")]

    provider.client = mock.Mock()
    provider.client.chat.completions.create.return_value = iter([chunk1, chunk2])

    chunks = list(provider.create_stream([{"role": "user", "content": "hi"}]))
    assert len(chunks) == 2
    assert chunks[0].content == "hel"
    assert chunks[1].content == "lo"
    assert chunks[1].finish_reason == "stop"


@mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_api_key"})
def test_openrouter_create_stream_with_tools():
    provider = OpenRouterProvider(model="test-model")
    provider.client = mock.Mock()
    provider.client.chat.completions.create.return_value = iter([])

    tools = [{"type": "function", "function": {"name": "foo"}}]

    with mock.patch.object(type(provider), "supports_tool_calling", new_callable=mock.PropertyMock, return_value=True):
        list(provider.create_stream(messages=[{"role": "user", "content": "hi"}], tools=tools))

    call_kwargs = provider.client.chat.completions.create.call_args[1]
    assert "tools" in call_kwargs


@mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_api_key"})
def test_openrouter_create_stream_constraint_healing():
    """create_stream retries once when parse_error returns a constraint fix."""
    provider = OpenRouterProvider(model="test-model")

    call_count = {"n": 0}

    def fake_create(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("temperature not supported")
        return iter([])

    provider.client = mock.Mock()
    provider.client.chat.completions.create = fake_create

    mock_registry = mock.Mock()

    with mock.patch("mullet.core.providers.constraints.shape_request", side_effect=lambda m, k: k) as mock_shape:
        with mock.patch("mullet.core.providers.constraints.get_constraint_registry", return_value=mock_registry) as mock_get_reg:
            with mock.patch.object(type(provider), "parse_error", return_value=("temperature", "not supported")):
                list(provider.create_stream(messages=[{"role": "user", "content": "hi"}]))

    assert call_count["n"] == 2
    mock_registry.learn.assert_called_once_with("test-model", "temperature", "not supported")
    assert mock_shape.call_count >= 1


@mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_api_key"})
def test_openrouter_create_stream_with_tool_call_delta():
    provider = OpenRouterProvider(model="test-model")

    tc_func = mock.Mock(name="foo", arguments='{"x":1}')
    tc_func.name = "foo"
    tc_func.arguments = '{"x":1}'
    tc = mock.Mock(index=0, id="tc1", function=tc_func)
    chunk = mock.Mock()
    chunk.choices = [mock.Mock(delta=mock.Mock(content="", tool_calls=[tc]), finish_reason=None)]

    provider.client = mock.Mock()
    provider.client.chat.completions.create.return_value = iter([chunk])

    chunks = list(provider.create_stream([{"role": "user", "content": "hi"}]))
    assert chunks[0].tool_call_delta is not None
    assert chunks[0].tool_call_delta["function"]["name"] == "foo"
    assert chunks[0].tool_call_delta["index"] == 0


@mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_api_key"})
def test_openrouter_create_stream_skips_empty_choices():
    provider = OpenRouterProvider(model="test-model")

    chunk1 = mock.Mock()
    chunk1.choices = []
    chunk2 = mock.Mock()
    chunk2.choices = [mock.Mock(delta=mock.Mock(content="ok", tool_calls=None), finish_reason="stop")]

    provider.client = mock.Mock()
    provider.client.chat.completions.create.return_value = iter([chunk1, chunk2])

    chunks = list(provider.create_stream([{"role": "user", "content": "hi"}]))
    assert len(chunks) == 1
    assert chunks[0].content == "ok"


@mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_api_key"})
def test_openrouter_to_response_empty_choices():
    provider = OpenRouterProvider(model="test-model")
    mock_completion = mock.Mock(choices=[])
    result = provider._to_response(mock_completion)
    assert result.content == ""


@mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_api_key"})
@mock.patch("requests.get")
def test_openrouter_list_models_error(mock_get):
    mock_get.side_effect = requests.exceptions.ConnectionError("fail")
    provider = OpenRouterProvider()
    models = provider.list_models()
    assert models == []


@mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_api_key"})
def test_openrouter_default_model():
    provider = OpenRouterProvider()
    assert provider.model == "auto"


# --- Constraint-layer wiring (commit bb90751) --------------------------------


@mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_api_key"})
def test_openrouter_shape_request_called_in_create():
    """OpenRouterProvider.create invokes shape_request before dispatch."""
    provider = OpenRouterProvider(model="moonshot-v1-128k")

    fake_response = mock.Mock()
    fake_response.choices = [mock.Mock(
        message=mock.Mock(content="hi", tool_calls=None),
        finish_reason="stop",
    )]
    fake_response.usage = mock.Mock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    fake_response.model = "moonshot-v1-128k"
    provider.client.chat.completions.create = mock.Mock(return_value=fake_response)

    with mock.patch("mullet.core.providers.constraints.shape_request", side_effect=lambda m, k: k) as mock_shape:
        provider.create(messages=[{"role": "user", "content": "hi"}])

    assert mock_shape.called, "shape_request must be called inside create()"
    args, _kw = mock_shape.call_args
    assert args[0] == "moonshot-v1-128k"


@mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_api_key"})
def test_openrouter_stream_retries_on_constraint_error():
    """create_stream retries stream-open once when parse_error returns a fix."""
    provider = OpenRouterProvider(model="some-model")

    call_count = {"n": 0}

    def fake_create(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("some error")
        return iter([])

    provider.client.chat.completions.create = fake_create

    # Force parse_error to return a fix so the retry path fires
    with mock.patch.object(type(provider), "parse_error",
                           return_value=("temperature", "forced test")):
        list(provider.create_stream(messages=[{"role": "user", "content": "x"}]))

    assert call_count["n"] == 2, f"expected 2 stream.open calls; got {call_count['n']}"


@mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test_api_key"})
def test_openrouter_stream_no_retry_on_unparseable_error():
    """create_stream propagates non-constraint errors without retry."""
    provider = OpenRouterProvider(model="some-model")

    call_count = {"n": 0}

    def fake_create(**kwargs):
        call_count["n"] += 1
        raise RuntimeError("network down")

    provider.client.chat.completions.create = fake_create

    # parse_error returns None → no retry
    with mock.patch.object(type(provider), "parse_error", return_value=None):
        with pytest.raises(RuntimeError, match="network down"):
            list(provider.create_stream(messages=[{"role": "user", "content": "x"}]))

    assert call_count["n"] == 1


# --- Routing matrix tests ----------------------------------------------------


def _make_preflight(
    input_tokens: int = 1000,
    expected_output_tokens: int = 500,
    complexity_tier: Tier = Tier.STANDARD,
    needs_reasoning: bool = False,
) -> PreflightResult:
    return PreflightResult(
        input_tokens=input_tokens,
        expected_output_tokens=expected_output_tokens,
        complexity_tier=complexity_tier,
        needs_reasoning=needs_reasoning,
        confidence=0.9,
        analyzer="local_llm",
        latency_ms=50,
        cached=False,
    )


def test_select_openrouter_model_reasoning():
    pf = _make_preflight(needs_reasoning=True)
    model_id, reason = select_openrouter_model(pf)
    assert model_id == "deepseek/deepseek-v4-pro"
    assert "reasoning" in reason.lower()


def test_select_openrouter_model_standard():
    pf = _make_preflight(complexity_tier=Tier.STANDARD)
    model_id, reason = select_openrouter_model(pf)
    assert model_id == "deepseek/deepseek-v4-flash"
    assert "standard" in reason.lower()


def test_select_openrouter_model_advanced():
    """routing-v3 follow-up: the value picker now wins this pick before the
    static "advanced task" branch is reached. ADVANCED coding floor=54;
    deepseek-v4-flash (59) clears it same as the old default v4-pro (62),
    and is ~4.8x cheaper -- diverges from the old static pick
    deepseek/deepseek-v4-pro. Approved divergence, same rationale as
    _best_openrouter's already-verified real behavior for this tier (see
    test_openrouter_value_picker_integration.py)."""
    pf = _make_preflight(complexity_tier=Tier.ADVANCED)
    model_id, reason = select_openrouter_model(pf)
    assert model_id == "deepseek/deepseek-v4-flash"
    assert "advanced" in reason.lower()


def test_select_openrouter_model_advanced_premium():
    pf = _make_preflight(complexity_tier=Tier.ADVANCED)
    model_id, reason = select_openrouter_model(pf, speed_mode="premium")
    assert model_id == "anthropic/claude-sonnet-5"


def test_select_openrouter_model_critical():
    """routing-v3 follow-up: the value picker now wins this pick before the
    static "critical task, balanced" branch is reached. CRITICAL coding
    floor=65; of the 8 candidates only qwen/qwen3-coder (70),
    anthropic/claude-opus-4.8 (72), and anthropic/claude-sonnet-5 (71) clear
    it, and qwen3-coder is far cheaper (blended 0.00101 vs. sonnet-5's
    0.006) -- diverges from the old static pick anthropic/claude-sonnet-5.
    Approved divergence -- see
    test_select_openrouter_model_real_critical_tier_picks_qwen3_coder in
    test_openrouter_value_picker_integration.py for the full rationale."""
    pf = _make_preflight(complexity_tier=Tier.CRITICAL)
    model_id, reason = select_openrouter_model(pf)
    assert model_id == "qwen/qwen3-coder"
    assert "critical" in reason.lower()


def test_select_openrouter_model_critical_premium():
    pf = _make_preflight(complexity_tier=Tier.CRITICAL)
    model_id, reason = select_openrouter_model(pf, speed_mode="premium")
    assert model_id == "anthropic/claude-opus-4.8"


def test_select_openrouter_model_large_context():
    """routing-v3 follow-up: the value picker is now consulted for this
    STANDARD-tier, balanced-mode pick before the static "large context"
    branch is reached -- same model_id as before (deepseek-v4-flash is both
    the static default AND the cheapest picker survivor), but the reason
    string is now the picker's generic "value picker: domain=coding
    tier=STANDARD" rather than the old "large context — DeepSeek V4
    Flash..." string. Same precedent as
    test_select_kimi_model_large_context in
    tests/test_router_routing_matrix_gaps.py."""
    pf = _make_preflight(input_tokens=150_000)
    model_id, reason = select_openrouter_model(pf)
    assert model_id == "deepseek/deepseek-v4-flash"
    assert "value picker" in reason.lower()


def test_select_openrouter_model_huge_context():
    pf = _make_preflight(input_tokens=700_000)
    model_id, reason = select_openrouter_model(pf)
    assert model_id == "qwen/qwen3-coder"
    assert "very large context" in reason.lower()


def test_select_openrouter_model_huge_context_premium():
    pf = _make_preflight(input_tokens=700_000)
    model_id, reason = select_openrouter_model(pf, speed_mode="premium")
    assert model_id == "anthropic/claude-sonnet-5"


def test_select_openrouter_model_reasoning_premium():
    pf = _make_preflight(needs_reasoning=True)
    model_id, reason = select_openrouter_model(pf, speed_mode="premium")
    assert model_id == "openai/gpt-5"


# --- Edge cases ---


def test_select_openrouter_model_low_tier_is_free():
    """LOW tier now hits the free-tier-first block, picking qwen/qwen3-coder:free
    (alphabetically ahead of tencent/hy3:free at identical zero cost)."""
    pf = _make_preflight(complexity_tier=Tier.LOW)
    model_id, reason = select_openrouter_model(pf)
    assert model_id == "qwen/qwen3-coder:free"
    assert "free-tier" in reason.lower()


def test_select_openrouter_model_mid_large_context():
    """Inputs in the 100K-500K band route to deepseek-v4-flash via the value
    picker (routing-v3 follow-up), NOT the >500K branch. Same model_id as
    the old static "large context" branch, just a different reason string
    -- see test_select_openrouter_model_large_context above."""
    pf = _make_preflight(input_tokens=300_000)
    model_id, reason = select_openrouter_model(pf)
    assert model_id == "deepseek/deepseek-v4-flash"
    assert "value picker" in reason.lower()
    # Important: should NOT trigger the "very large context" >500K branch
    assert "very large" not in reason.lower()


def test_select_openrouter_model_max_speed_equals_premium():
    """``speed_mode='max'`` is an alias for premium routing."""
    pf = _make_preflight(complexity_tier=Tier.CRITICAL)
    model_premium, _ = select_openrouter_model(pf, speed_mode="premium")
    model_max, _ = select_openrouter_model(pf, speed_mode="max")
    assert model_premium == model_max == "anthropic/claude-opus-4.8"


def test_select_openrouter_model_reasoning_trumps_tier():
    """``needs_reasoning=True`` overrides the complexity tier branch."""
    pf = _make_preflight(complexity_tier=Tier.CRITICAL, needs_reasoning=True)
    model_id, _ = select_openrouter_model(pf)  # balanced
    assert model_id == "deepseek/deepseek-v4-pro"  # reasoning wins over critical


def test_select_openrouter_model_huge_context_trumps_tier():
    """Input >500K trumps the complexity tier branch (Sonnet-only in premium)."""
    pf = _make_preflight(input_tokens=700_000, complexity_tier=Tier.CRITICAL)
    model_id, reason = select_openrouter_model(pf)  # balanced
    assert model_id == "qwen/qwen3-coder"
    assert "very large context" in reason.lower()


# --- Doctor tests ------------------------------------------------------------


def _run_doctor(env_overrides: dict, extra_args: list[str] | None = None,
                timeout: int = 20) -> subprocess.CompletedProcess:
    """Run `python -m mullet.cli.main doctor` with a sanitised environment."""
    env = os.environ.copy()
    env["OPENROUTER_API_KEY"] = ""
    env.update(env_overrides)
    args = [sys.executable, "-m", "mullet.cli.main", "doctor"]
    if extra_args:
        args.extend(extra_args)
    return subprocess.run(
        args, capture_output=True, text=True, timeout=timeout,
        cwd="/tmp", env=env,
    )


def test_openrouter_doctor_key_missing():
    result = _run_doctor({})
    assert result.returncode == 0, result.stderr
    assert "OpenRouter API" in result.stdout
    assert "OPENROUTER_API_KEY not set (optional)" in result.stdout
    lines = [line for line in result.stdout.splitlines() if "OpenRouter API" in line]
    assert lines, "OpenRouter line should appear even when key missing"
    assert any("·" in line for line in lines), (
        f"OpenRouter-missing line should use info icon (·): {lines}"
    )
