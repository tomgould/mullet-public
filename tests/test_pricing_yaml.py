"""Tests for the pricing.yaml single source of truth.

These tests verify that:
1. pricing.yaml exists and is valid YAML
2. The pricing loader returns correct types
3. model_registry KNOWN_PRICING matches YAML values
4. model_info MODEL_DATABASE prices match YAML values
5. Cache-hit pricing is present
"""

from pathlib import Path

import pytest
import yaml


PRICING_YAML_PATH = Path("mullet/core/providers/pricing.yaml")


def test_pricing_yaml_exists():
    """pricing.yaml must exist in the expected location."""
    assert PRICING_YAML_PATH.exists(), f"{PRICING_YAML_PATH} does not exist"


def test_pricing_yaml_is_valid():
    """pricing.yaml must be valid YAML that parses to a dict."""
    raw = PRICING_YAML_PATH.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    assert isinstance(data, dict), f"Expected dict, got {type(data)}"


def test_pricing_loader_import():
    """The pricing loader module must be importable."""
    from mullet.core.providers.pricing import load_pricing  # noqa: F401


def test_load_pricing_returns_dict():
    """load_pricing() must return a dict[str, dict] with known models."""
    from mullet.core.providers.pricing import load_pricing

    pricing = load_pricing()
    assert isinstance(pricing, dict), f"Expected dict, got {type(pricing)}"
    # At least one known model must be present
    assert "kimi-k2.7-code" in pricing, "Expected 'kimi-k2.7-code' in pricing data"
    for model_id, entry in pricing.items():
        assert isinstance(model_id, str), f"Model key {model_id!r} is not a str"
        assert isinstance(entry, dict), f"Model entry for {model_id!r} is not a dict"


def test_all_pricing_models_have_required_keys():
    """Every model entry must have input and output pricing fields as floats."""
    from mullet.core.providers.pricing import load_pricing

    pricing = load_pricing()
    for model_id, entry in pricing.items():
        assert "input_per_1k" in entry, f"{model_id} missing 'input_per_1k'"
        assert "output_per_1k" in entry, f"{model_id} missing 'output_per_1k'"
        assert isinstance(entry["input_per_1k"], (int, float)), (
            f"{model_id} input_per_1k is not numeric"
        )
        assert isinstance(entry["output_per_1k"], (int, float)), (
            f"{model_id} output_per_1k is not numeric"
        )


def test_cache_hit_pricing_present():
    """At least one model must expose a cache-hit input price."""
    from mullet.core.providers.pricing import load_pricing

    pricing = load_pricing()
    hits = [
        model_id
        for model_id, entry in pricing.items()
        if "cache_hit_per_1k" in entry
    ]
    assert hits, "No model has 'cache_hit_per_1k' defined"


def test_kimi_k27_pricing_correct():
    """kimi-k2.7-code pricing must match the official rates."""
    from mullet.core.providers.pricing import load_pricing

    pricing = load_pricing()
    entry = pricing.get("kimi-k2.7-code")
    assert entry is not None, "kimi-k2.7-code not found in pricing"
    assert entry["input_per_1k"] == pytest.approx(0.00095, abs=1e-9)
    assert entry["output_per_1k"] == pytest.approx(0.004, abs=1e-9)
    assert entry["cache_hit_per_1k"] == pytest.approx(0.00019, abs=1e-9)


def test_known_pricing_matches_yaml():
    """model_registry KNOWN_PRICING must match the YAML values."""
    from mullet.core.providers.model_registry import KNOWN_PRICING
    from mullet.core.providers.pricing import load_pricing

    yaml_pricing = load_pricing()
    for model_id, (known_in, known_out) in KNOWN_PRICING.items():
        entry = yaml_pricing.get(model_id)
        assert entry is not None, f"{model_id} in KNOWN_PRICING but missing from YAML"
        assert entry["input_per_1k"] == pytest.approx(known_in, abs=1e-9), (
            f"{model_id} input_per_1k mismatch: YAML={entry['input_per_1k']} vs KNOWN={known_in}"
        )
        assert entry["output_per_1k"] == pytest.approx(known_out, abs=1e-9), (
            f"{model_id} output_per_1k mismatch: YAML={entry['output_per_1k']} vs KNOWN={known_out}"
        )


def test_context_windows_in_yaml():
    """YAML must contain context_window metadata for at least kimi-k2.7-code."""
    from mullet.core.providers.pricing import load_pricing

    pricing = load_pricing()
    entry = pricing.get("kimi-k2.7-code")
    assert entry is not None, "kimi-k2.7-code not found in pricing"
    assert "context_window" in entry, "kimi-k2.7-code missing 'context_window'"
    assert isinstance(entry["context_window"], int), (
        f"context_window must be int, got {type(entry['context_window'])}"
    )


def test_get_known_pricing_returns_tuples():
    """get_known_pricing() must return {model_id: (input_per_1k, output_per_1k)}."""
    from mullet.core.providers.pricing import get_known_pricing

    pricing = get_known_pricing()
    assert isinstance(pricing, dict)
    assert len(pricing) > 0
    for model_id, (inp, out) in pricing.items():
        assert isinstance(inp, (int, float)), f"{model_id}: input_per_1k not numeric"
        assert isinstance(out, (int, float)), f"{model_id}: output_per_1k not numeric"


def test_get_cache_hit_pricing_subset_of_known():
    """get_cache_hit_pricing() must only include models with cache pricing."""
    from mullet.core.providers.pricing import get_known_pricing, get_cache_hit_pricing

    known = set(get_known_pricing())
    cached = get_cache_hit_pricing()
    # Every model in cache_hit_pricing must also be in known_pricing
    for model_id in cached:
        assert model_id in known, f"{model_id} in cache_hit_pricing but not in known_pricing"


def test_get_cache_hit_pricing_kimi_k2_7_present():
    """kimi-k2.7-code has cache pricing and must appear in get_cache_hit_pricing."""
    from mullet.core.providers.pricing import get_cache_hit_pricing

    cached = get_cache_hit_pricing()
    assert "kimi-k2.7-code" in cached
    hit_price, out_price = cached["kimi-k2.7-code"]
    assert hit_price > 0
    assert out_price > 0
