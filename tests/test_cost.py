"""Tests for cost estimation and savings calculation."""

from mullet.core.agent.cost import estimate_cost, estimate_savings, format_cost_summary
from mullet.core.agent.types import TokenUsage


def test_local_is_free():
    usage = TokenUsage(input_tokens=10000, output_tokens=5000, total_tokens=15000)
    cost = estimate_cost("local", "default", usage)
    assert cost == 0.0


def test_premium_costs():
    usage = TokenUsage(input_tokens=1000, output_tokens=500, total_tokens=1500)
    cost = estimate_cost("anthropic", "sonnet", usage)
    assert cost > 0
    # 1K input * 0.003 + 0.5K output * 0.015 = 0.003 + 0.0075 = 0.0105
    assert abs(cost - 0.0105) < 0.001


def test_savings_local_vs_premium():
    usage = TokenUsage(input_tokens=10000, output_tokens=5000, total_tokens=15000)
    savings = estimate_savings("local", "default", usage)
    # Local is free, premium would cost: 10K*0.003 + 5K*0.015 = 0.03 + 0.075 = 0.105
    assert savings > 0.1


def test_no_savings_on_premium():
    usage = TokenUsage(input_tokens=1000, output_tokens=500, total_tokens=1500)
    savings = estimate_savings("anthropic", "sonnet", usage)
    assert savings == 0.0  # Same as baseline


def test_format_local():
    usage = TokenUsage(input_tokens=1000, output_tokens=500, total_tokens=1500)
    summary = format_cost_summary("local", "default", usage)
    assert "$0" in summary
    assert "free" in summary.lower()


def test_format_premium():
    usage = TokenUsage(input_tokens=1000, output_tokens=500, total_tokens=1500)
    summary = format_cost_summary("anthropic", "sonnet", usage)
    assert "$" in summary


def test_format_cheap_shows_savings():
    usage = TokenUsage(input_tokens=5000, output_tokens=2000, total_tokens=7000)
    summary = format_cost_summary("openrouter", "haiku", usage)
    assert "Saved" in summary
    assert "%" in summary
