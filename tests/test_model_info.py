"""Tests for model metadata database."""

from unittest.mock import patch

from mullet.core.providers.model_info import (
    get_model_info,
    get_context_length,
    get_pricing,
    MODEL_DATABASE,
)


def test_database_has_major_models():
    assert "claude-sonnet-4-6" in MODEL_DATABASE
    assert "claude-opus-4-6" in MODEL_DATABASE
    assert "gpt-4o" in MODEL_DATABASE
    assert "gemini-2.5-flash" in MODEL_DATABASE
    assert "deepseek-v3" in MODEL_DATABASE


def test_exact_match():
    info = get_model_info("claude-sonnet-4-6")
    assert info.context_length == 1_000_000
    assert info.input_cost_per_1k == 0.003
    assert info.supports_tools is True


def test_alias_match():
    info = get_model_info("sonnet")
    assert info.context_length == 1_000_000

    info = get_model_info("opus")
    assert info.input_cost_per_1k == 0.005


def test_fuzzy_match():
    info = get_model_info("gpt-4o-2024-05-13")
    assert info.context_length == 128_000


def test_provider_prefix_stripped():
    # Bedrock-style model ID
    info = get_model_info("anthropic.claude-sonnet-4-6-20250514-v1:0")
    assert info.context_length == 1_000_000


def test_openrouter_prefix():
    info = get_model_info("google/gemini-2.5-flash")
    assert info.context_length == 1_000_000


def test_unknown_model_returns_default():
    info = get_model_info("totally-unknown-model-xyz")
    assert info.context_length == 32_000
    assert info.input_cost_per_1k == 0.0


def test_empty_model_returns_default():
    info = get_model_info("")
    assert info.context_length == 32_000


def test_get_context_length():
    assert get_context_length("claude-opus-4-6") == 1_000_000
    assert get_context_length("gemini-2.5-pro") == 1_000_000
    assert get_context_length("gpt-4o-mini") == 128_000


def test_get_pricing():
    inp, out = get_pricing("claude-sonnet-4-6")
    assert inp == 0.003
    assert out == 0.015


def test_local_model_free():
    inp, out = get_pricing("default")
    assert inp == 0.0
    assert out == 0.0


def test_vision_support():
    info = get_model_info("claude-sonnet-4-6")
    assert info.supports_vision is True

    info = get_model_info("gpt-4o")
    assert info.supports_vision is True


def test_all_models_have_valid_data():
    for name, info in MODEL_DATABASE.items():
        assert info.context_length > 0, f"{name} has invalid context_length"
        assert info.input_cost_per_1k >= 0, f"{name} has negative input cost"
        assert info.output_cost_per_1k >= 0, f"{name} has negative output cost"


# --- Coverage gap tests ---

def test_supports_streaming_property():
    """Lines 53-54: ModelInfo.supports_streaming back-compat property."""
    info = get_model_info("claude-sonnet-4-6")
    # Just calling it covers the property body (lines 53-54)
    result = info.supports_streaming
    assert isinstance(result, bool)


def test_with_id_same_model_id_returns_self():
    """Line 256: _with_id returns info unchanged when model_id already matches."""
    from mullet.core.providers.model_info import _with_id, ModelInfo
    # Create a ModelInfo with model_id already set to the target value
    from dataclasses import replace
    base = get_model_info("claude-sonnet-4-6")
    stamped = replace(base, model_id="claude-sonnet-4-6")
    result = _with_id(stamped, "claude-sonnet-4-6")
    assert result is stamped  # same object, not a copy


def test_alias_that_is_model_info_object():
    """Line 280: _ALIASES entry that is a ModelInfo directly → returned as-is."""
    from mullet.core.providers import model_info as mi_mod
    from mullet.core.providers.model_info import ModelInfo
    fake = ModelInfo(context_length=99_999, input_cost_per_1k=0.001, output_cost_per_1k=0.002)
    original = dict(mi_mod._ALIASES)
    mi_mod._ALIASES["test-direct-alias"] = fake
    try:
        result = get_model_info("test-direct-alias")
        assert result is fake
    finally:
        mi_mod._ALIASES.clear()
        mi_mod._ALIASES.update(original)


def test_strip_provider_prefix_slash_and_unknown_model():
    """Lines 293, 309-310: unknown provider/model → prefix stripped, recursive lookup returns default."""
    info = get_model_info("myorg/totally-unknown-xyz-model")
    # Should strip "myorg/" and return default for unknown model
    assert info.context_length == 32_000


def test_strip_provider_prefix_with_tag():
    """Line 314: model:tag format → tag stripped before lookup."""
    info = get_model_info("myorg/totally-unknown-xyz-model:latest")
    assert info.context_length == 32_000


def test_apply_pricing_from_yaml_exception_path():
    """Lines 344-346: when load_pricing raises, function logs and returns early."""
    from mullet.core.providers import model_info as mi_mod
    with patch("mullet.core.providers.pricing.load_pricing", side_effect=Exception("yaml missing")):
        # Should not raise — logs warning and returns
        mi_mod._apply_pricing_from_yaml()
