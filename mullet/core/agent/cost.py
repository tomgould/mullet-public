"""Cost estimation and savings calculation.

Shows users how much money Mullet saves compared to routing everything
through premium providers. This is Mullet's value proposition.
"""

from mullet.core.agent.types import TokenUsage

def _get_pricing():
    from mullet.core.providers.model_registry import KNOWN_PRICING
    return KNOWN_PRICING

# What premium would have cost (baseline for savings calculation)
PREMIUM_RATE = {"input": 0.003, "output": 0.015}  # Sonnet pricing


def estimate_cost(provider: str, model: str, usage: TokenUsage) -> float:
    """Estimate the cost of a single LLM call."""
    rates = _get_rates(model, provider)
    return (
        (usage.input_tokens / 1000) * rates["input"]
        + (usage.output_tokens / 1000) * rates["output"]
    )


def estimate_savings(provider: str, model: str, usage: TokenUsage) -> float:
    """Calculate how much was saved vs using premium for everything."""
    actual_cost = estimate_cost(provider, model, usage)
    premium_cost = (
        (usage.input_tokens / 1000) * PREMIUM_RATE["input"]
        + (usage.output_tokens / 1000) * PREMIUM_RATE["output"]
    )
    return max(0, premium_cost - actual_cost)


def format_cost_summary(
    provider: str,
    model: str,
    usage: TokenUsage,
) -> str:
    """Format a cost summary line for CLI verbose output."""
    cost = estimate_cost(provider, model, usage)
    savings = estimate_savings(provider, model, usage)

    parts = [f"Cost: ${cost:.4f}"]
    if savings > 0:
        pct = (savings / (cost + savings)) * 100 if (cost + savings) > 0 else 0
        parts.append(f"Saved: ${savings:.4f} ({pct:.0f}% vs premium)")
    if cost == 0:
        parts = ["Cost: $0 (local GPU — free)"]

    return " | ".join(parts)


def _get_rates(model: str, provider: str = "") -> dict:
    """Get pricing rates for a model/provider combination."""
    model_lower = model.lower()
    provider_lower = provider.lower()

    # Local providers are free
    if any(p in provider_lower for p in ("local", "lmstudio", "ollama")):
        return {"input": 0.0, "output": 0.0}

    # Combined model+provider match first — avoids false positives when provider
    # alone matches multiple keys (e.g. "anthropic" in both opus and sonnet keys).
    if model_lower and provider_lower:
        for key, rates in _get_pricing().items():
            if model_lower in key and provider_lower in key:
                return {"input": rates[0], "output": rates[1]}

    # Model-only fallback
    for key, rates in _get_pricing().items():
        if model_lower in key:
            return {"input": rates[0], "output": rates[1]}

    # Provider-only fallback
    for key, rates in _get_pricing().items():
        if provider_lower and provider_lower in key:
            return {"input": rates[0], "output": rates[1]}

    return {"input": 0.001, "output": 0.005}
