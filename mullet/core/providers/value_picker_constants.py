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
OPENROUTER_FREE_TIER_CANDIDATES: list[tuple[str, int]] = [
    ("qwen/qwen3-coder:free", 1_048_576),
    ("tencent/hy3:free",       262_144),
]

# Kimi candidates used by both routing_matrix.py::_kimi_value_pick
# and model_registry.py::ModelRegistry._best_kimi.
KIMI_VALUE_PICK_CANDIDATES: list[tuple[str, int]] = [
    ("kimi-k2.7-code", 262_144),
    ("kimi-k2.5",      262_144),
]
