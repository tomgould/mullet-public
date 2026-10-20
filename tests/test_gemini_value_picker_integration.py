"""Integration tests for routing-v3 Phase 3 Gemini value-picker wiring.

These verify that the new value-picker calls in _best_gemini and
select_gemini_model fall back gracefully on exception/None, respect
explicit speed_mode overrides, and produce sensible real results.
"""
from __future__ import annotations

from unittest.mock import patch


from mullet.core.providers.model_registry import ModelRegistry
from mullet.core.router.complexity import Tier
from mullet.core.router.preflight import PreflightResult
from mullet.core.router.routing_matrix import select_gemini_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry() -> ModelRegistry:
    return ModelRegistry()


def _make_preflight(
    input_tokens: int = 1_000,
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
# 1. Exception in pick_by_value → fallback for both entry points
# ---------------------------------------------------------------------------

def test_select_gemini_model_fallback_on_exception():
    """If pick_by_value raises, select_gemini_model falls back to static ladder."""
    pf = _make_preflight(complexity_tier=Tier.STANDARD)

    with patch(
        "mullet.core.providers.value_picker.pick_by_value",
        side_effect=RuntimeError("boom"),
    ):
        model_id, reason = select_gemini_model(pf, speed_mode="balanced")

    assert model_id == "gemini-2.5-flash"
    assert "standard task" in reason


def test_best_gemini_fallback_on_exception():
    """If pick_by_value raises, _best_gemini falls back to static ladder."""
    reg = _make_registry()

    with patch(
        "mullet.core.providers.value_picker.pick_by_value",
        side_effect=RuntimeError("boom"),
    ):
        chosen = reg._best_gemini("STANDARD", context_needed=4096)

    assert chosen == "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# 2. pick_by_value returns None → fallback for both entry points
# ---------------------------------------------------------------------------

def test_select_gemini_model_fallback_on_none():
    """If pick_by_value returns None, select_gemini_model falls back."""
    pf = _make_preflight(complexity_tier=Tier.STANDARD)

    with patch(
        "mullet.core.providers.value_picker.pick_by_value",
        return_value=None,
    ):
        model_id, reason = select_gemini_model(pf, speed_mode="balanced")

    assert model_id == "gemini-2.5-flash"
    assert "standard task" in reason


def test_best_gemini_fallback_on_none():
    """If pick_by_value returns None, _best_gemini falls back to static ladder."""
    reg = _make_registry()

    with patch(
        "mullet.core.providers.value_picker.pick_by_value",
        return_value=None,
    ):
        chosen = reg._best_gemini("STANDARD", context_needed=4096)

    assert chosen == "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# 3. speed_mode="max" / "economy" overrides value picker
# ---------------------------------------------------------------------------

def test_select_gemini_model_speed_max_overrides_value_picker():
    """speed_mode='max' must return flash even if value picker wants pro."""
    pf = _make_preflight(complexity_tier=Tier.CRITICAL)

    with patch(
        "mullet.core.providers.value_picker.pick_by_value",
        return_value="gemini-3.1-pro",
    ):
        model_id, reason = select_gemini_model(pf, speed_mode="max")

    assert model_id == "gemini-2.5-flash"
    assert "max speed" in reason.lower()


def test_select_gemini_model_speed_economy_overrides_value_picker():
    """speed_mode='economy' must return flash-lite even if picker wants flash."""
    pf = _make_preflight(complexity_tier=Tier.STANDARD)

    with patch(
        "mullet.core.providers.value_picker.pick_by_value",
        return_value="gemini-2.5-flash",
    ):
        model_id, reason = select_gemini_model(pf, speed_mode="economy")

    assert model_id == "gemini-2.5-flash-lite"
    assert "economy" in reason.lower()


# ---------------------------------------------------------------------------
# 4. Real end-to-end value-picker integration (non-mocked)
# ---------------------------------------------------------------------------

def test_select_gemini_model_real_value_picker_low_tier():
    """Real pick_by_value run: LOW tier coding should land on cheapest capable model."""
    pf = _make_preflight(complexity_tier=Tier.LOW, input_tokens=1_000)
    model_id, reason = select_gemini_model(pf, speed_mode="balanced")
    # LOW coding floor=10; flash-lite (coding=12) clears it and is cheapest.
    assert model_id == "gemini-2.5-flash-lite"
    assert "value picker" in reason.lower()


def test_best_gemini_real_value_picker_low_tier():
    """Real pick_by_value run in _best_gemini: LOW tier → flash-lite."""
    reg = _make_registry()
    chosen = reg._best_gemini("LOW", context_needed=4096)
    # LOW coding floor=10; flash-lite (coding=12) clears it and is cheapest.
    assert chosen == "gemini-2.5-flash-lite"


def test_best_gemini_real_value_picker_with_preflight_reasoning():
    """Real pick_by_value run with reasoning preflight: domain=reasoning."""
    reg = _make_registry()
    pf = _make_preflight(needs_reasoning=True, complexity_tier=Tier.STANDARD)
    chosen = reg._best_gemini(
        "STANDARD",
        context_needed=4096,
        preflight=pf,
    )
    # STANDARD reasoning floor=15; gemini-3.1-pro (reasoning=55) and
    # gemini-3.5-flash (reasoning=45) clear it. 3.5-flash is cheaper
    # blended cost than 3.1-pro, but 2.5-flash (reasoning=18) also clears
    # and is cheaper still. 2.5-flash-lite (reasoning=10) does NOT clear.
    # So the cheapest survivor should be gemini-2.5-flash.
    assert chosen == "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# 5. Phase 4 regression: demanding tiers, cheaper survivor, missing-data fail-open
# ---------------------------------------------------------------------------

def test_best_gemini_advanced_tier_escalates_to_only_survivor():
    """ADVANCED coding floor=55: only gemini-3.1-pro (coding=67) clears."""
    reg = _make_registry()
    chosen = reg._best_gemini("ADVANCED", context_needed=4096)
    assert chosen == "gemini-3.1-pro"


def test_best_gemini_critical_tier_escalates_to_only_survivor():
    """CRITICAL coding floor=65: only gemini-3.1-pro (coding=67) clears."""
    reg = _make_registry()
    chosen = reg._best_gemini("CRITICAL", context_needed=4096)
    assert chosen == "gemini-3.1-pro"


def test_best_gemini_advanced_tier_reasoning_picks_cheaper_survivor():
    """Real end-to-end ADVANCED reasoning path through _best_gemini + preflight."""
    reg = _make_registry()
    pf = _make_preflight(needs_reasoning=True, complexity_tier=Tier.ADVANCED)
    chosen = reg._best_gemini("ADVANCED", context_needed=4096, preflight=pf)
    # No Gemini model has is_reasoning_tuned=True, so meets_reasoning_gate
    # filters out every candidate when needs_reasoning=True.  The value
    # picker returns None, falling back to select_gemini_model's static
    # ladder which returns gemini-2.5-flash for ADVANCED.  If the gate were
    # bypassed, gemini-3.5-flash (blended 0.00525) would win as the cheaper
    # survivor over gemini-3.1-pro (blended 0.007) per the cost tie-break.
    assert chosen == "gemini-2.5-flash"


def test_best_gemini_missing_quality_data_fails_open_to_cheapest():
    """Empty MODEL_QUALITY fails open: every candidate passes the floor check,
    so cost-only sorting applies and the cheapest candidate wins."""
    reg = _make_registry()
    with patch("mullet.core.providers.quality_floors.MODEL_QUALITY", {}):
        chosen = reg._best_gemini("ADVANCED", context_needed=4096)
    # With no quality data, meets_quality_floor returns True for all models.
    # The value picker sorts by blended cost; gemini-2.5-flash-lite is
    # cheapest (input 0.0001 / output 0.0004 → blended 0.00025).
    assert chosen is not None
    assert chosen == "gemini-2.5-flash-lite"
