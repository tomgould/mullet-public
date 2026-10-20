"""Gap tests for mullet.core.router.routing_matrix uncovered lines."""
from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from mullet.core.router.routing_matrix import (
    select_kimi_model,
    select_openrouter_model,
    select_local_model,
    _classify_local_model,
)
from mullet.core.router.preflight import PreflightResult
from mullet.core.router.complexity import Tier


def _pf(
    input_tokens: int = 1000,
    expected_output_tokens: int = 500,
    complexity_tier: Tier = Tier.STANDARD,
    needs_reasoning: bool = False,
) -> PreflightResult:
    return PreflightResult(
        input_tokens=input_tokens,
        expected_output_tokens=expected_output_tokens,
        complexity_tier=complexity_tier,
        needs_reasoning=needs_reasoning,
        confidence=0.9,
        analyzer="local_llm",
        latency_ms=50,
        cached=False,
    )


# ---------------------------------------------------------------------------
# select_kimi_model — deprecated model replacement (lines 73-79)
# ---------------------------------------------------------------------------

def test_select_kimi_model_replaces_deprecated(caplog):
    """If the chosen model is in _DEPRECATED_KIMI_MODELS, it is swapped out."""
    caplog.set_level(logging.WARNING)
    pf = _pf(complexity_tier=Tier.CRITICAL)
    # Pretend kimi-k2.7-code is deprecated so the replacement path runs.
    fake_deprecated = {"kimi-k2.7-code": ("kimi-k2.5", "2099-12-31")}
    with patch("mullet.core.providers.kimi._DEPRECATED_KIMI_MODELS", fake_deprecated):
        model_id, reason = select_kimi_model(pf)
    assert model_id == "kimi-k2.5"
    assert "deprecated" in caplog.text


# ---------------------------------------------------------------------------
# select_openrouter_model — full branch coverage (lines 137-180)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs,speed_mode,expected_model,expected_snippet",
    [
        # needs_reasoning
        ({"needs_reasoning": True}, "balanced", "deepseek/deepseek-v4-pro", "reasoning"),
        ({"needs_reasoning": True}, "max", "openai/gpt-5", "premium"),
        ({"needs_reasoning": True}, "premium", "openai/gpt-5", "premium"),
        # very large context (>500K, untouched override -- picker never consulted)
        ({"input_tokens": 600_000}, "balanced", "qwen/qwen3-coder", "large context"),
        ({"input_tokens": 600_000}, "max", "anthropic/claude-sonnet-5", "premium"),
        ({"input_tokens": 600_000}, "premium", "anthropic/claude-sonnet-5", "premium"),
        # CRITICAL, balanced -- routing-v3 follow-up: value picker now wins
        # this pick (see test_select_openrouter_model_critical in
        # test_openrouter.py for the full floor/cost rationale). Approved
        # divergence from the old static pick anthropic/claude-sonnet-5.
        ({"complexity_tier": Tier.CRITICAL}, "balanced", "qwen/qwen3-coder", "critical"),
        ({"complexity_tier": Tier.CRITICAL}, "max", "anthropic/claude-opus-4.8", "premium"),
        ({"complexity_tier": Tier.CRITICAL}, "premium", "anthropic/claude-opus-4.8", "premium"),
        # ADVANCED, balanced -- routing-v3 follow-up: value picker now wins
        # this pick. Approved divergence from the old static pick
        # deepseek/deepseek-v4-pro (matches _best_openrouter's
        # already-verified real behavior for this tier).
        ({"complexity_tier": Tier.ADVANCED}, "balanced", "deepseek/deepseek-v4-flash", "advanced"),
        ({"complexity_tier": Tier.ADVANCED}, "max", "anthropic/claude-sonnet-5", "premium"),
        ({"complexity_tier": Tier.ADVANCED}, "premium", "anthropic/claude-sonnet-5", "premium"),
        # large context (>100k), STANDARD tier -- value picker now wins this
        # pick too (same model_id as the old static "large context" branch,
        # but the reason text is now the picker's generic
        # "value picker: domain=coding tier=STANDARD" string -- same
        # precedent as test_select_kimi_model_large_context above).
        ({"input_tokens": 150_000}, "balanced", "deepseek/deepseek-v4-flash", "value picker"),
        # default / STANDARD -> flash via value picker (tier name still
        # appears in the generic reason string for STANDARD)
        ({}, "balanced", "deepseek/deepseek-v4-flash", "standard"),
        # LOW -> free-tier pick (qwen/qwen3-coder:free wins alphabetically
        # over tencent/hy3:free at identical zero cost).
        ({"complexity_tier": Tier.LOW}, "balanced", "qwen/qwen3-coder:free", "free-tier"),
    ],
)
def test_select_openrouter_model_branches(kwargs, speed_mode, expected_model, expected_snippet):
    pf = _pf(**kwargs)
    model_id, reason = select_openrouter_model(pf, speed_mode=speed_mode)
    assert model_id == expected_model, f"expected {expected_model}, got {model_id}"
    assert expected_snippet in reason.lower()


# ---------------------------------------------------------------------------
# _classify_local_model — unknown size defaults to STANDARD (line 223)
# ---------------------------------------------------------------------------

def test_classify_local_model_unknown_size_defaults_standard():
    result = _classify_local_model("my-custom-model-with-no-params")
    assert result["tier"] == "STANDARD"


# ---------------------------------------------------------------------------
# select_local_model — unknown tier & fallback to usable (lines 281, 289)
# ---------------------------------------------------------------------------

def test_select_local_model_unknown_tier_fallback():
    """An unexpected tier value falls through to target_tiers=['STANDARD']."""
    pf = _pf(complexity_tier=Tier.STANDARD)
    # Monkeypatch the tier to something bizarre so the else-branch at line 281 runs.
    object.__setattr__(pf, "complexity_tier", type("FakeTier", (), {"value": "WEIRD"})())
    model, reason = select_local_model(pf, ["qwen2.5-coder-7b-instruct"])
    assert model is not None
    assert "local" in reason.lower()


def test_select_local_model_no_candidates_fallback_to_usable():
    """When no candidates match the target tier, fallback to all usable models (line 289)."""
    # Only tiny models available, but tier is ADVANCED.
    # target_tiers for ADVANCED = ["ADVANCED", "STANDARD"]
    # tiny models are LOW, so no candidates match.
    tiny_models = ["qwen3.5-2b-instruct"]
    pf = _pf(complexity_tier=Tier.ADVANCED)
    model, reason = select_local_model(pf, tiny_models)
    # Should fall back to usable (the tiny model) because line 289 runs.
    assert model is not None
    assert "local" in reason.lower()


# ---------------------------------------------------------------------------
# select_kimi_model — lines 49-69
# ---------------------------------------------------------------------------

def test_select_kimi_model_needs_reasoning():
    """needs_reasoning=True → kimi-k2.7-code."""
    pf = _pf(needs_reasoning=True)
    model_id, reason = select_kimi_model(pf)
    assert model_id.startswith("kimi-k2.7")
    assert "reasoning" in reason.lower()


def test_select_kimi_model_speed_max_standard_tier():
    """speed_mode='max' with STANDARD tier → kimi-k2.7-code."""
    pf = _pf(complexity_tier=Tier.STANDARD)
    model_id, reason = select_kimi_model(pf, speed_mode="max")
    assert model_id.startswith("kimi-k2.7")
    assert "max" in reason.lower()


def test_select_kimi_model_high_output():
    """expected_output_tokens > 8000 → kimi-k2.7-code."""
    pf = _pf(expected_output_tokens=9000)
    model_id, reason = select_kimi_model(pf)
    assert model_id.startswith("kimi-k2.7")
    assert "output" in reason.lower()


def test_select_kimi_model_critical_tier():
    """complexity_tier=CRITICAL → kimi-k2.7-code."""
    pf = _pf(complexity_tier=Tier.CRITICAL)
    model_id, reason = select_kimi_model(pf)
    assert model_id.startswith("kimi-k2.7")
    assert "critical" in reason.lower()


def test_select_kimi_model_advanced_tier():
    """complexity_tier=ADVANCED → kimi-k2.7-code."""
    pf = _pf(complexity_tier=Tier.ADVANCED)
    model_id, reason = select_kimi_model(pf)
    assert model_id.startswith("kimi-k2.7")
    assert "advanced" in reason.lower()


def test_select_kimi_model_large_context():
    """input_tokens > 131072 → kimi-k2.5 (routing-v3 value picker).

    STANDARD floor (coding=18) is cleared by both candidates, and kimi-k2.5
    is cheaper on both axes, so the value picker now wins this pick before
    the static ladder's dedicated large-context branch is ever reached --
    same model_id as the old static "large context, standard model" branch,
    just a different reason string.
    """
    pf = _pf(input_tokens=200_000)
    model_id, reason = select_kimi_model(pf)
    assert model_id == "kimi-k2.5"
    assert "context" in reason.lower() or "value picker" in reason.lower()


def test_select_kimi_model_standard_task():
    """Standard task with no special flags → kimi-k2.5 (else branch, line 68-69)."""
    pf = _pf(
        complexity_tier=Tier.STANDARD,
        needs_reasoning=False,
        input_tokens=1000,
        expected_output_tokens=500,
    )
    model_id, reason = select_kimi_model(pf, speed_mode="balanced")
    assert model_id == "kimi-k2.5"
    assert "standard" in reason.lower()


# ---------------------------------------------------------------------------
# select_gemini_model — lines 96-118
# ---------------------------------------------------------------------------

from mullet.core.router.routing_matrix import select_gemini_model


def test_select_gemini_model_speed_max():
    """speed_mode='max' → gemini-2.5-flash."""
    pf = _pf()
    model_id, reason = select_gemini_model(pf, speed_mode="max")
    assert model_id == "gemini-2.5-flash"
    assert "max" in reason.lower()


def test_select_gemini_model_critical_tier():
    """complexity_tier=CRITICAL → gemini-3.1-pro."""
    pf = _pf(complexity_tier=Tier.CRITICAL)
    model_id, reason = select_gemini_model(pf)
    assert model_id == "gemini-3.1-pro"


def test_select_gemini_model_high_output():
    """expected_output_tokens > 50000 → gemini-3.1-pro."""
    # _GEMINI_HIGH_OUTPUT_THRESHOLD is 8000, so any value above that triggers pro
    pf = _pf(expected_output_tokens=51_000)
    model_id, reason = select_gemini_model(pf)
    assert model_id == "gemini-3.1-pro"
    assert "output" in reason.lower()


def test_select_gemini_model_advanced_tier():
    """complexity_tier=ADVANCED → gemini-3.1-pro (routing-v3 value picker).

    The ADVANCED quality floor (coding=55) is only cleared by gemini-3.1-pro
    (coding=67). This aligns select_gemini_model with _best_gemini's static
    fallback, which already mapped ADVANCED→pro.
    """
    pf = _pf(complexity_tier=Tier.ADVANCED)
    model_id, reason = select_gemini_model(pf)
    assert model_id == "gemini-3.1-pro"
    assert "advanced" in reason.lower() or "value picker" in reason.lower()


def test_select_gemini_model_economy_mode():
    """speed_mode='economy' → gemini-2.5-flash-lite."""
    pf = _pf()
    model_id, reason = select_gemini_model(pf, speed_mode="economy")
    assert model_id == "gemini-2.5-flash-lite"
    assert "economy" in reason.lower()


def test_select_gemini_model_low_tier():
    """complexity_tier=LOW → gemini-2.5-flash-lite."""
    pf = _pf(complexity_tier=Tier.LOW)
    model_id, reason = select_gemini_model(pf)
    assert model_id == "gemini-2.5-flash-lite"


def test_select_gemini_model_standard_balanced():
    """Standard tier, balanced speed → gemini-2.5-flash."""
    pf = _pf(complexity_tier=Tier.STANDARD)
    model_id, reason = select_gemini_model(pf)
    assert model_id == "gemini-2.5-flash"
    assert "standard" in reason.lower()


# ---------------------------------------------------------------------------
# _classify_local_model — lines 203, 215, 217
# ---------------------------------------------------------------------------

def test_classify_local_model_embedding():
    """Embedding model → tier=None."""
    result = _classify_local_model("nomic-embed-text-v1")
    assert result["tier"] is None
    assert result["coder"] is False
    assert result["vision"] is False
    assert result["reasoning"] is False


def test_classify_local_model_70b_critical():
    """Model with '70b' in name → tier='CRITICAL'."""
    result = _classify_local_model("llama-3.1-70b-instruct")
    assert result["tier"] == "CRITICAL"


def test_classify_local_model_13b_advanced():
    """Model with '13b' in name → tier='ADVANCED'."""
    result = _classify_local_model("llama-2-13b-chat")
    assert result["tier"] == "ADVANCED"


# ---------------------------------------------------------------------------
# select_local_model — lines 252, 256-268, 273-279, 286, 296-298, 311
# ---------------------------------------------------------------------------

def test_select_local_model_empty_list():
    """Empty available_models → returns (None, 'no usable...')."""
    pf = _pf()
    model, reason = select_local_model(pf, [])
    assert model is None
    assert "no usable" in reason.lower()


def test_select_local_model_only_embedding_models():
    """Only embedding models → all filtered out → (None, ...)."""
    pf = _pf()
    model, reason = select_local_model(pf, ["nomic-embed-text-v1", "bge-large-en"])
    assert model is None
    assert "no usable" in reason.lower()


def test_select_local_model_needs_reasoning_picks_reasoning_model():
    """needs_reasoning=True with a reasoning model available → picks it."""
    pf = _pf(needs_reasoning=True)
    models = ["qwen2.5-7b-instruct", "qwen3-0.6b-reasoning-distill"]
    model, reason = select_local_model(pf, models)
    assert model == "qwen3-0.6b-reasoning-distill"
    assert "reasoning" in reason.lower()


def test_select_local_model_low_tier_picks_small():
    """LOW tier with a small model → picks small model."""
    pf = _pf(complexity_tier=Tier.LOW)
    models = ["qwen2.5-0.5b-instruct"]
    model, reason = select_local_model(pf, models)
    assert model == "qwen2.5-0.5b-instruct"
    assert "local" in reason.lower()


def test_select_local_model_standard_tier():
    """STANDARD tier with a 7b model → picks it."""
    pf = _pf(complexity_tier=Tier.STANDARD)
    models = ["qwen2.5-7b-instruct"]
    model, reason = select_local_model(pf, models)
    assert model == "qwen2.5-7b-instruct"
    assert "local" in reason.lower()


def test_select_local_model_critical_small_only_returns_none():
    """CRITICAL tier with only tiny/small models → (None, 'too small...')."""
    pf = _pf(complexity_tier=Tier.CRITICAL)
    # target_tiers for CRITICAL = ["ADVANCED", "CRITICAL"]
    # 0.5b and 7b map to LOW and STANDARD — neither matches
    models = ["qwen2.5-0.5b-instruct", "qwen2.5-7b-instruct"]
    model, reason = select_local_model(pf, models)
    assert model is None
    assert "too small" in reason.lower()


def test_select_local_model_critical_large_model():
    """CRITICAL tier with a 70B model → picks it."""
    pf = _pf(complexity_tier=Tier.CRITICAL)
    models = ["llama-3.1-70b-instruct"]
    model, reason = select_local_model(pf, models)
    assert model == "llama-3.1-70b-instruct"
    assert "local" in reason.lower()


def test_select_local_model_code_task_prefers_coder():
    """Code task (STANDARD+) with a coder model available → reason contains 'coder'."""
    pf = _pf(complexity_tier=Tier.STANDARD)
    models = ["qwen2.5-7b-instruct", "qwen2.5-coder-7b-instruct"]
    model, reason = select_local_model(pf, models)
    assert model == "qwen2.5-coder-7b-instruct"
    assert "coder" in reason.lower()


# ---------------------------------------------------------------------------
# estimated_cost_usd — lines 329-335
# ---------------------------------------------------------------------------

from mullet.core.router.routing_matrix import estimated_cost_usd


def test_estimated_cost_usd_known_model():
    """Known model_id → returns a non-zero float."""
    cost = estimated_cost_usd("kimi-k2.5", input_tokens=1000, output_tokens=500)
    assert isinstance(cost, float)
    assert cost > 0.0


def test_estimated_cost_usd_unknown_model():
    """Unknown model_id → returns 0.0."""
    cost = estimated_cost_usd("totally-unknown-model-xyz", input_tokens=1000, output_tokens=500)
    assert cost == 0.0
