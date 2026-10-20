"""Sanity/schema tests for mullet.core.providers.quality.MODEL_QUALITY.

Ensures every model that can be returned by the static ladders in
model_registry.py has a corresponding QualityScore entry, and that all
entries obey structural invariants (int scores in [0, 100], non-empty
source, valid inherited_from references, etc.).
"""
from __future__ import annotations

import ast
import inspect
import re
import textwrap

from mullet.core.providers import quality
from mullet.core.providers.model_registry import ModelRegistry, PREFERRED_DEFAULTS
from mullet.core.router.routing_matrix import select_gemini_model, select_openrouter_model


def _extract_string_literals_from_function(func) -> set[str]:
    """Return all string literals appearing in a function's source code."""
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    strings: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.add(node.value)
    return strings


# Real model ids across all 4 providers share a shape: a known provider
# prefix followed by identifier characters only (no spaces/punctuation like
# reason strings and docstrings have). This is more robust than a hand-
# maintained deny-list of exact noise strings, which breaks every time a
# log message or reason string gets tweaked.
_MODEL_ID_RE = re.compile(r"^(claude-|kimi-|gemini-|deepseek/|qwen/|openai/)[A-Za-z0-9._-]+$")


def _all_required_model_ids() -> set[str]:
    """Union of every model id that can be returned by the static ladders.

    Inspects both model_registry.py's _best_* methods (the live-cache-aware
    pickers) AND routing_matrix.py's select_gemini_model/select_openrouter_model
    (the preflight-aware pickers _best_gemini/_best_kimi defer to) -- both code
    paths can return a model id in production, so both must be covered.
    """
    found: set[str] = set(PREFERRED_DEFAULTS.values())
    for func in (
        ModelRegistry._best_anthropic,
        ModelRegistry._best_kimi,
        ModelRegistry._best_gemini,
        ModelRegistry._best_openrouter,
        select_gemini_model,
        select_openrouter_model,
    ):
        found |= _extract_string_literals_from_function(func)
    return {s for s in found if _MODEL_ID_RE.match(s)}


# ---------------------------------------------------------------------------
# Schema invariants
# ---------------------------------------------------------------------------


def test_every_ladder_model_has_quality_entry() -> None:
    """Every model reachable via PREFERRED_DEFAULTS or a _best_* ladder
    must have a MODEL_QUALITY entry.
    """
    required = _all_required_model_ids()
    missing = required - set(quality.MODEL_QUALITY.keys())
    assert not missing, f"Missing MODEL_QUALITY entries for: {sorted(missing)}"


def test_no_extra_entries_beyond_expected() -> None:
    """The table is intentionally small (Phase 0) -- flag unexpected adds."""
    # Exact expected set per 2026-07-05 research pass
    expected = {
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
        "kimi-k2.7-code",
        "kimi-k2.7-code-highspeed",
        "kimi-k2.6",
        "kimi-k2.5",
        "gemini-3.1-pro",
        "gemini-2.5-pro",
        "gemini-3.5-flash",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4-flash",
        "qwen/qwen3.7-max",
        "qwen/qwen3-coder",
        "qwen/qwen3-coder:free",
        "tencent/hy3:free",
        "openai/gpt-5",
        "anthropic/claude-opus-4.8",
        "anthropic/claude-sonnet-5",
    }
    actual = set(quality.MODEL_QUALITY.keys())
    extra = actual - expected
    assert not extra, f"Unexpected extra MODEL_QUALITY keys: {sorted(extra)}"
    missing = expected - actual
    assert not missing, f"Missing expected MODEL_QUALITY keys: {sorted(missing)}"


def test_all_scores_are_ints_in_range() -> None:
    for model_id, qs in quality.MODEL_QUALITY.items():
        for field in ("coding", "reasoning", "general"):
            val = getattr(qs, field)
            assert isinstance(val, int), (
                f"{model_id}.{field} must be int, got {type(val).__name__}"
            )
            assert 0 <= val <= 100, (
                f"{model_id}.{field}={val} out of [0, 100] range"
            )


def test_all_sources_non_empty() -> None:
    for model_id, qs in quality.MODEL_QUALITY.items():
        assert isinstance(qs.source, str) and qs.source.strip(), (
            f"{model_id} has empty or missing source"
        )


def test_inherited_entries_have_valid_inherited_from() -> None:
    for model_id, qs in quality.MODEL_QUALITY.items():
        if qs.confidence == "inherited":
            assert qs.inherited_from is not None, (
                f"{model_id} confidence='inherited' but inherited_from is None"
            )
            assert qs.inherited_from in quality.MODEL_QUALITY, (
                f"{model_id} inherited_from={qs.inherited_from!r} not in MODEL_QUALITY"
            )


def test_non_inherited_entries_have_none_inherited_from() -> None:
    for model_id, qs in quality.MODEL_QUALITY.items():
        if qs.confidence != "inherited":
            assert qs.inherited_from is None, (
                f"{model_id} confidence='{qs.confidence}' should have inherited_from=None"
            )


def test_variant_penalty_keys_are_known() -> None:
    """VARIANT_PENALTY keys are a closed set documented in the module docstring."""
    known = {"preview", "fast", "highspeed", "lite"}
    assert set(quality.VARIANT_PENALTY.keys()) == known


def test_all_confidence_values_valid() -> None:
    valid = {"direct", "inherited", "estimated"}
    for model_id, qs in quality.MODEL_QUALITY.items():
        assert qs.confidence in valid, (
            f"{model_id} has invalid confidence={qs.confidence!r}"
        )
