"""Integration tests for routing-v3 Phase 3b Kimi value-picker wiring.

These verify that the new value-picker calls in _best_kimi and
select_kimi_model fall back gracefully on exception/None, that the
unconditional needs_reasoning early-return still short-circuits before the
value picker is ever consulted, and that real (non-mocked) picks match the
plan's predicted "provable no-op" outcome across the 2-candidate Kimi set.
"""
from __future__ import annotations

from unittest.mock import patch


from mullet.core.providers.model_registry import ModelRegistry
from mullet.core.router.complexity import Tier
from mullet.core.router.preflight import PreflightResult
from mullet.core.router.routing_matrix import select_kimi_model


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

def test_select_kimi_model_fallback_on_exception():
    """If pick_by_value raises, select_kimi_model falls back to static ladder."""
    pf = _make_preflight(complexity_tier=Tier.STANDARD)

    with patch(
        "mullet.core.providers.value_picker.pick_by_value",
        side_effect=RuntimeError("boom"),
    ):
        model_id, reason = select_kimi_model(pf, speed_mode="balanced")

    assert model_id == "kimi-k2.5"
    assert "standard task" in reason


def test_best_kimi_fallback_on_exception():
    """If pick_by_value raises, _best_kimi falls back to static ladder."""
    reg = _make_registry()

    with patch(
        "mullet.core.providers.value_picker.pick_by_value",
        side_effect=RuntimeError("boom"),
    ):
        chosen = reg._best_kimi("STANDARD", context_needed=4096)

    assert chosen == "kimi-k2.5"


# ---------------------------------------------------------------------------
# 2. pick_by_value returns None → fallback for both entry points
# ---------------------------------------------------------------------------

def test_select_kimi_model_fallback_on_none():
    """If pick_by_value returns None, select_kimi_model falls back."""
    pf = _make_preflight(complexity_tier=Tier.STANDARD)

    with patch(
        "mullet.core.providers.value_picker.pick_by_value",
        return_value=None,
    ):
        model_id, reason = select_kimi_model(pf, speed_mode="balanced")

    assert model_id == "kimi-k2.5"
    assert "standard task" in reason


def test_best_kimi_fallback_on_none():
    """If pick_by_value returns None, _best_kimi falls back to static ladder."""
    reg = _make_registry()

    with patch(
        "mullet.core.providers.value_picker.pick_by_value",
        return_value=None,
    ):
        chosen = reg._best_kimi("STANDARD", context_needed=4096)

    assert chosen == "kimi-k2.5"


# ---------------------------------------------------------------------------
# 3. needs_reasoning early-return must short-circuit before the value picker
# ---------------------------------------------------------------------------

def test_select_kimi_model_needs_reasoning_bypasses_value_picker():
    """needs_reasoning=True must return kimi-k2.7-code via the unconditional
    early-return, without ever consulting the value picker -- regardless of
    complexity tier. Patch pick_by_value to raise unconditionally: if the
    early-return didn't truly short-circuit, this test would blow up instead
    of passing."""
    pf = _make_preflight(needs_reasoning=True, complexity_tier=Tier.STANDARD)

    with patch(
        "mullet.core.providers.value_picker.pick_by_value",
        side_effect=RuntimeError("boom"),
    ) as mock_pick:
        model_id, reason = select_kimi_model(pf, speed_mode="balanced")

    assert model_id == "kimi-k2.7-code"
    assert "reasoning task" in reason
    mock_pick.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Real end-to-end value-picker integration (non-mocked)
# ---------------------------------------------------------------------------

def test_select_kimi_model_real_standard_tier_picks_k25():
    """Real pick_by_value run: STANDARD tier coding should land on kimi-k2.5.

    Floor=18; both kimi-k2.7-code (coding=35) and kimi-k2.5 (coding=25)
    clear it. kimi-k2.5 is cheaper on both input ($0.0006 vs $0.00095) and
    output ($0.003 vs $0.004) per pricing.yaml, so it wins the cost
    tie-break -- unchanged from today's static ladder pick for STANDARD.
    """
    pf = _make_preflight(complexity_tier=Tier.STANDARD, needs_reasoning=False)
    model_id, reason = select_kimi_model(pf, speed_mode="balanced")
    assert model_id == "kimi-k2.5"
    assert "value picker" in reason.lower()


def test_best_kimi_real_advanced_tier_falls_back_to_k27code():
    """Real pick_by_value run in _best_kimi: ADVANCED tier, no preflight
    (domain=coding path).

    Floor=55; neither kimi-k2.5 (coding=25) nor kimi-k2.7-code (coding=35)
    clears it even after the one-tier escalation to CRITICAL (floor=65).
    pick_by_value must return None, and _best_kimi falls back to the static
    ladder's ADVANCED pick (most-capable-first) -- kimi-k2.7-code.
    """
    reg = _make_registry()
    chosen = reg._best_kimi("ADVANCED", context_needed=4096)
    assert chosen == "kimi-k2.7-code"


def test_best_kimi_missing_quality_data_fails_open():
    """Empty MODEL_QUALITY fails open: every candidate passes the floor
    check, so cost-only sorting applies and the cheapest candidate wins."""
    reg = _make_registry()
    with patch("mullet.core.providers.quality_floors.MODEL_QUALITY", {}):
        chosen = reg._best_kimi("ADVANCED", context_needed=4096)
    assert chosen is not None
    assert chosen == "kimi-k2.5"
