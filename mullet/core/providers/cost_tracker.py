"""Tracks API costs and token usage across different providers."""

from collections import defaultdict
from mullet.core.providers.model_info import get_pricing

class CostTracker:
    """Append-only record of model token usage; computes per-model and total USD."""

    def __init__(self):
        self._records = []

    def record(self, model, tokens_in, tokens_out):
        """
        Records the token usage for a given model and calculates the cost.
        """
        if tokens_in < 0 or tokens_out < 0:
            raise ValueError(f"Token counts must be non-negative, got tokens_in={tokens_in}, tokens_out={tokens_out}")
        input_cost_per_1k, output_cost_per_1k = get_pricing(model)
        if input_cost_per_1k is None or output_cost_per_1k is None:
            raise ValueError(f"Pricing information not found for model: {model}")

        cost_in = tokens_in / 1000 * input_cost_per_1k
        cost_out = tokens_out / 1000 * output_cost_per_1k
        total_record_cost = cost_in + cost_out

        self._records.append({
            "model": model,
            # "provider": provider, # Provider information is not directly available from get_pricing
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost": total_record_cost
        })

    def total_cost(self):
        """
        Returns the total accumulated cost across all records.
        """
        return sum(record["cost"] for record in self._records)

    def breakdown_by_model(self):
        """
        Returns a dictionary with cost breakdown by model.
        """
        breakdown = defaultdict(float)
        for record in self._records:
            breakdown[record["model"]] += record["cost"]
        return dict(breakdown)

