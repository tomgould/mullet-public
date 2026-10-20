"""Gap tests for mullet/core/agent/cost.py — target ≥95% coverage."""

from unittest.mock import patch

from mullet.core.agent.cost import _get_pricing, _get_rates, estimate_cost, estimate_savings, format_cost_summary
from mullet.core.agent.types import TokenUsage


# ---------------------------------------------------------------------------
# _get_rates provider-only fallback (lines 77-81)
# ---------------------------------------------------------------------------

def test_provider_only_fallback_with_empty_model():
    """When model is empty the model-only fallback is skipped; provider-only
    fallback (lines 77-81) should match.
    """
    pricing = {"openai-gpt-4": (0.01, 0.03)}
    with patch("mullet.core.agent.cost._get_pricing", return_value=pricing):
        rates = _get_rates(model="", provider="openai")
    assert rates == {"input": 0.01, "output": 0.03}


def test_provider_only_fallback_unknown_model():
    """Model does not match any key; provider does — hit provider-only loop."""
    pricing = {"anthropic-claude": (0.008, 0.024)}
    with patch("mullet.core.agent.cost._get_pricing", return_value=pricing):
        rates = _get_rates(model="unknown-model", provider="anthropic")
    assert rates == {"input": 0.008, "output": 0.024}


# ---------------------------------------------------------------------------
# Default rates when absolutely nothing matches
# ---------------------------------------------------------------------------

def test_default_rates_when_no_match():
    pricing = {"some-key": (0.01, 0.03)}
    with patch("mullet.core.agent.cost._get_pricing", return_value=pricing):
        rates = _get_rates(model="xyz", provider="abc")
    assert rates == {"input": 0.001, "output": 0.005}


# ---------------------------------------------------------------------------
# format_cost_summary edge cases
# ---------------------------------------------------------------------------

def test_format_cost_summary_zero_cost():
    """Local providers cost $0 — special formatting."""
    with patch("mullet.core.agent.cost._get_pricing", return_value={}):
        summary = format_cost_summary("local", "llama", TokenUsage(input_tokens=100, output_tokens=50))
    assert "$0 (local GPU — free)" in summary


def test_format_cost_summary_with_savings():
    pricing = {"gpt-4": (0.003, 0.015)}  # same as PREMIUM_RATE → no savings
    with patch("mullet.core.agent.cost._get_pricing", return_value=pricing):
        summary = format_cost_summary("openai", "gpt-4", TokenUsage(input_tokens=1000, output_tokens=500))
    # cost == premium_cost, so savings == 0 and savings branch is skipped
    assert summary.startswith("Cost: $")
    assert "Saved" not in summary


def test_format_cost_summary_actual_savings():
    pricing = {"gpt-4": (0.001, 0.002)}  # cheaper than premium
    with patch("mullet.core.agent.cost._get_pricing", return_value=pricing):
        summary = format_cost_summary("openai", "gpt-4", TokenUsage(input_tokens=1000, output_tokens=500))
    assert "Saved:" in summary


# ---------------------------------------------------------------------------
# Lines 10-11 — _get_pricing wrapper
# ---------------------------------------------------------------------------

def test_get_pricing_returns_dict():
    result = _get_pricing()
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Line 69 — combined model + provider match
# ---------------------------------------------------------------------------

def test_combined_model_provider_match():
    pricing = {"openai-gpt-4": (0.01, 0.03)}
    with patch("mullet.core.agent.cost._get_pricing", return_value=pricing):
        rates = _get_rates(model="gpt-4", provider="openai")
    assert rates == {"input": 0.01, "output": 0.03}
