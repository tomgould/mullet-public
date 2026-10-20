"""Single source of truth loader for model pricing.

Reads mullet/core/providers/pricing.yaml and exposes typed accessors.
All pricing is in USD per 1K tokens.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml

_PRICING_YAML = Path(__file__).parent / "pricing.yaml"
_cache: dict | None = None


def load_pricing() -> dict[str, dict[str, Any]]:
    """Load and return the full pricing dict, keyed by model ID."""
    global _cache
    if _cache is None:
        with open(_PRICING_YAML) as f:
            data = yaml.safe_load(f)
        _cache = data.get("models", {})
    return dict(_cache)  # return copy — callers must not mutate the cache


def get_known_pricing() -> dict[str, tuple[float, float]]:
    """Return {model_id: (input_per_1k, output_per_1k)} for all models."""
    return {
        mid: (entry["input_per_1k"], entry["output_per_1k"])
        for mid, entry in load_pricing().items()
    }


def get_cache_hit_pricing() -> dict[str, tuple[float, float]]:
    """Return {model_id: (cache_hit_per_1k, output_per_1k)} for models with cache pricing."""
    return {
        mid: (entry["cache_hit_per_1k"], entry["output_per_1k"])
        for mid, entry in load_pricing().items()
        if "cache_hit_per_1k" in entry
    }
