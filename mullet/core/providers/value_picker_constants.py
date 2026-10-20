"""Shared constants for value-picker candidate lists.

Centralised here so that model_registry.py (no-preflight fallback path)
and routing_matrix.py (preflight-driven path) agree on the same
candidate rosters without duplicated literals.
"""
from __future__ import annotations

# OpenRouter candidates used by both routing_matrix.py::select_openrouter_model
# and model_registry.py::ModelRegistry._best_openrouter.
OPENROUTER_VALUE_PICK_CANDIDATES: list[tuple[str, int]] = [
    ("deepseek/deepseek-v4-flash", 1_000_000),
    ("deepseek/deepseek-v4-pro",   1_000_000),
    ("qwen/qwen3.7-max",           1_000_000),
    ("qwen/qwen3-coder",           1_048_576),
    ("anthropic/claude-opus-4.8",  1_000_000),
    ("anthropic/claude-sonnet-5",  1_000_000),
    ("openai/gpt-5",               400_000),
]

# Free-tier OpenRouter candidates -- tried first for LOW-complexity requests,
# ahead of OPENROUTER_VALUE_PICK_CANDIDATES's paid ladder. Kept as a SEPARATE
# list rather than merged into the paid one: pick_by_value's cost sort
# deliberately sorts zero-cost candidates LAST when mixed with paid ones (see
# value_picker.py's _sort_key comment) -- a genuinely free model would never
# win against paid candidates in the same list. Isolating them here sidesteps
# that guard entirely, since every candidate in THIS list is genuinely free.
#
# Empty for now. OpenRouter withdrew both former entries, qwen/qwen3-coder:free
# and tencent/hy3:free (neither is in https://openrouter.ai/api/v1/models as of
# 2026-10-20; requests for the first fail with 404 "This model is unavailable
# for free"), and none of the free models it lists today is a free variant of
# a model with a sourced QualityScore in quality.py. With the list empty, LOW
# work goes to the paid ladder below (deepseek/deepseek-v4-flash at today's
# prices). To add a free model: check its `:free` slug is in the public list
# above, give it a sourced QualityScore in quality.py (without one it passes
# every tier's quality floor) and a zero-price entry in pricing.yaml (without
# one, a paid model whose name it contains lends it that price). Free slots
# come and go, so OpenRouterProvider also falls back to the next candidate,
# and remembers the dead model (model_availability.py), when one is withdrawn.
OPENROUTER_FREE_TIER_CANDIDATES: list[tuple[str, int]] = []

# Kimi candidates used by both routing_matrix.py::_kimi_value_pick
# and model_registry.py::ModelRegistry._best_kimi.
KIMI_VALUE_PICK_CANDIDATES: list[tuple[str, int]] = [
    ("kimi-k2.7-code", 262_144),
    ("kimi-k2.5",      262_144),
]
