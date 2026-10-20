"""Tests for Kimi K2.5 default, alias wiring, and moonshot temperature guard."""

from unittest.mock import MagicMock



# ---------------------------------------------------------------------------
# model_info alias tests
# ---------------------------------------------------------------------------

def test_kimi_alias_resolves_to_k2_6():
    """get_model_info("kimi") resolves to kimi-k2.6 pricing ($0.00095/1K in)."""
    from mullet.core.providers.model_info import get_model_info
    info = get_model_info("kimi")
    assert info.input_cost_per_1k == 0.00095, (
        f"Expected 0.00095 (k2.6 pricing) but got {info.input_cost_per_1k}"
    )


def test_kimi_k2_alias():
    """get_model_info("kimi-k2") also resolves to k2.6."""
    from mullet.core.providers.model_info import get_model_info
    assert get_model_info("kimi-k2").input_cost_per_1k == 0.00095


def test_kimi_k2_5_aliases():
    """Both 'kimi-2.5' and 'kimi-k2.5' aliases resolve."""
    from mullet.core.providers.model_info import get_model_info, _ALIASES
    assert _ALIASES["kimi-2.5"] == "kimi-k2.5"
    assert _ALIASES["kimi-k2.5"] == "kimi-k2.5"
    assert get_model_info("kimi-2.5").input_cost_per_1k == 0.0006


# ---------------------------------------------------------------------------
# KimiProvider default model
# ---------------------------------------------------------------------------

def test_kimi_provider_default_model(monkeypatch):
    """KimiProvider() sets model == 'kimi-k2.7-code' when no override given.

    K2.7-Code is the preferred default since 2026-06 — same price as K2.6,
    thinking always ON, ~10% better on agentic benchmarks.
    """
    monkeypatch.setenv("KIMI_API_KEY", "test-key-123")
    from mullet.core.providers.kimi import KimiProvider
    provider = KimiProvider()
    assert provider.model == "kimi-k2.7-code", (
        f"Expected 'kimi-k2.7-code' but got '{provider.model}'"
    )


# ---------------------------------------------------------------------------
# OpenAICompatibleProvider temperature guard
# ---------------------------------------------------------------------------

def _make_mock_response():
    """Build a minimal mock openai ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = "hello"
    choice.message.tool_calls = None
    choice.finish_reason = "stop"
    response = MagicMock()
    response.choices = [choice]
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    response.usage.total_tokens = 15
    response.model = "moonshot-v1-128k"
    return response


def test_moonshot_create_omits_temperature(monkeypatch):
    """create() for moonshot-v1-128k must NOT include temperature in the request."""
    from mullet.core.providers.openai_compat import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider(
        base_url="https://api.moonshot.ai/v1",
        api_key="test-key",
        model="moonshot-v1-128k",
    )

    captured: dict = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _make_mock_response()

    provider.client.chat.completions.create = fake_create

    provider.create(messages=[{"role": "user", "content": "hi"}])

    assert "temperature" not in captured, (
        f"temperature should be omitted for moonshot models but got: {captured.get('temperature')}"
    )


def test_kimi_k2_5_create_includes_temperature(monkeypatch):
    """create() for kimi-k2.5 MUST include temperature in the request."""
    from mullet.core.providers.openai_compat import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider(
        base_url="https://api.moonshot.ai/v1",
        api_key="test-key",
        model="kimi-k2.5",
    )

    captured: dict = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        # Return a valid mock response
        choice = MagicMock()
        choice.message.content = "hello"
        choice.message.tool_calls = None
        choice.finish_reason = "stop"
        response = MagicMock()
        response.choices = [choice]
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 5
        response.usage.total_tokens = 15
        response.model = "kimi-k2.5"
        return response

    provider.client.chat.completions.create = fake_create

    provider.create(messages=[{"role": "user", "content": "hi"}], temperature=0.7)

    assert "temperature" in captured, "temperature should be present for kimi-k2.5"
    assert captured["temperature"] == 0.7


def test_all_moonshot_models_omit_temperature():
    """All three moonshot-v1-* models omit temperature via the constraint registry."""
    from mullet.core.providers.openai_compat import OpenAICompatibleProvider
    from mullet.core.providers.constraints import get_constraint_registry

    registry = get_constraint_registry()

    for model_id in ("moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"):
        assert "temperature" in registry.get(model_id).omit_params, (
            f"{model_id} missing temperature in constraint registry omit_params"
        )

        provider = OpenAICompatibleProvider(
            base_url="https://api.moonshot.ai/v1",
            api_key="test-key",
            model=model_id,
        )
        captured: dict = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return _make_mock_response()

        provider.client.chat.completions.create = fake_create
        provider.create(messages=[{"role": "user", "content": "hi"}])

        assert "temperature" not in captured, (
            f"{model_id}: temperature should be omitted but was present"
        )
        captured.clear()


# ---------------------------------------------------------------------------
# Deprecated model warning & swap
# ---------------------------------------------------------------------------

def test_deprecated_model_triggers_warning_and_swaps(monkeypatch, caplog):
    """Passing a deprecated Kimi model ID swaps to replacement and logs WARNING."""
    monkeypatch.setenv("KIMI_API_KEY", "test-key-123")
    from mullet.core.providers.kimi import KimiProvider

    with caplog.at_level("WARNING", logger="mullet.core.providers.kimi"):
        provider = KimiProvider(model="kimi-k2-0711-preview")

    assert provider.model == "kimi-k2.5", f"Expected 'kimi-k2.5' but got '{provider.model}'"
    assert any(
        "kimi-k2-0711-preview" in rec.message and "deprecated" in rec.message
        for rec in caplog.records
    ), "Expected deprecation WARNING in logs"


def test_non_deprecated_model_no_warning(monkeypatch, caplog):
    """Passing a current Kimi model ID leaves it untouched and emits no warning."""
    monkeypatch.setenv("KIMI_API_KEY", "test-key-123")
    from mullet.core.providers.kimi import KimiProvider

    with caplog.at_level("WARNING", logger="mullet.core.providers.kimi"):
        provider = KimiProvider(model="kimi-k2.6")

    assert provider.model == "kimi-k2.6", f"Expected 'kimi-k2.6' but got '{provider.model}'"
    assert not any("deprecated" in rec.message for rec in caplog.records), (
        "Did not expect any deprecation warnings"
    )
