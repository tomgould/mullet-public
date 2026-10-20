"""Integration tests for the routing-v3 Anthropic value-picker wiring.

These verify that the new value-picker call in _best_anthropic falls back
gracefully on exception/None, and that real (non-mocked) picks match the
verified real outcomes across the 6-candidate Anthropic set -- including
the biggest divergence in the whole routing-v3 rollout: ADVANCED and
CRITICAL both move off the flagship claude-opus-4-8 onto the mid-tier
claude-sonnet-5, on the strength of sonnet-5's quality.py coding score
(71 -- higher than 2 of the 3 opus variants, one point below opus-4-8's 72)
combined with its much lower cost (blended 0.006 vs. opus-4-8's 0.015).
This was explicitly reviewed and approved based on these exact predicted
outcomes before implementation.
"""
from __future__ import annotations

from unittest.mock import patch

from mullet.core.providers.model_registry import ModelRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry() -> ModelRegistry:
    return ModelRegistry()


# ---------------------------------------------------------------------------
# 1. Exception in pick_by_value → fallback to static ladder
# ---------------------------------------------------------------------------

def test_best_anthropic_fallback_on_exception():
    """If pick_by_value raises, _best_anthropic falls back to the static
    mapping's STANDARD default (today's pre-value-picker behavior:
    claude-sonnet-5)."""
    reg = _make_registry()

    with patch(
        "mullet.core.providers.value_picker.pick_by_value",
        side_effect=RuntimeError("boom"),
    ):
        chosen = reg._best_anthropic("STANDARD")

    assert chosen == "claude-sonnet-5"


# ---------------------------------------------------------------------------
# 2. pick_by_value returns None → fallback to static ladder
# ---------------------------------------------------------------------------

def test_best_anthropic_fallback_on_none():
    """If pick_by_value returns None, _best_anthropic falls back to the
    static mapping's STANDARD default (today's pre-value-picker behavior:
    claude-sonnet-5)."""
    reg = _make_registry()

    with patch(
        "mullet.core.providers.value_picker.pick_by_value",
        return_value=None,
    ):
        chosen = reg._best_anthropic("STANDARD")

    assert chosen == "claude-sonnet-5"


# ---------------------------------------------------------------------------
# 3. Real end-to-end value-picker integration (non-mocked)
# ---------------------------------------------------------------------------

def test_best_anthropic_real_low_tier_matches_static():
    """Real pick_by_value run: LOW tier lands on
    claude-haiku-4-5-20251001 -- matches today's static mapping exactly
    (no-op). Floor=10; haiku's coding=35 clears it easily, and haiku is the
    cheapest of all 6 candidates (blended (0.001+0.005)/2=0.003)."""
    reg = _make_registry()
    chosen = reg._best_anthropic("LOW")
    assert chosen == "claude-haiku-4-5-20251001"


def test_best_anthropic_real_standard_tier_diverges_to_haiku():
    """Real pick_by_value run: STANDARD tier lands on
    claude-haiku-4-5-20251001, NOT the old static mapping's claude-sonnet-5.

    STANDARD coding floor=18. All 6 candidates clear it (haiku's coding=35
    is the lowest score in the set and still clears 18 comfortably), so
    cost-only sorting applies -- haiku is the cheapest of all 6 (blended
    0.003 vs. sonnet-5's 0.006), so it wins. Approved, intentional
    divergence -- not a bug. Real historical STANDARD-tier Anthropic
    traffic in .mullet/routing_history.jsonl already resolves to haiku (via
    the live-cache path, untouched by this wiring), so this brings the
    static-fallback path's stated intent in line with observed reality.
    """
    reg = _make_registry()
    chosen = reg._best_anthropic("STANDARD")
    assert chosen == "claude-haiku-4-5-20251001"


def test_best_anthropic_real_advanced_tier_diverges_to_sonnet_5():
    """Real pick_by_value run: ADVANCED tier lands on claude-sonnet-5, NOT
    the old static mapping's claude-opus-4-8.

    ADVANCED coding floor=54. Survivors: claude-sonnet-4-6 (coding=54),
    claude-sonnet-5 (71), claude-opus-4-6 (60), claude-opus-4-7 (65),
    claude-opus-4-8 (72) -- claude-haiku-4-5-20251001 (35) does NOT clear
    the floor. Per pricing.yaml, claude-sonnet-5's blended cost is
    (0.002+0.01)/2=0.006 -- the cheapest of the 5 survivors (claude-opus-4-8
    is (0.005+0.025)/2=0.015, ~2.5x more expensive) -- so sonnet-5 wins the
    cost tie-break.

    This is a much bigger claim than a marginal cost optimization between
    similar-tier models: it replaces Anthropic's flagship (opus-4-8) with
    its mid-tier model (sonnet-5) for a high-stakes complexity tier, on the
    strength of sonnet-5's coding score of 71 -- higher than 2 of the 3 opus
    variants (opus-4-6=60, opus-4-7=65), just one point below opus-4-8's
    72. Explicitly reviewed and approved based on this exact predicted
    outcome; zero real ADVANCED-tier Anthropic traffic exists today in
    .mullet/routing_history.jsonl, so nothing observable changes
    immediately, but future ADVANCED-tier Anthropic requests will pick
    sonnet-5 instead of opus-4-8.
    """
    reg = _make_registry()
    chosen = reg._best_anthropic("ADVANCED")
    assert chosen == "claude-sonnet-5"


def test_best_anthropic_real_critical_tier_diverges_to_sonnet_5():
    """Real pick_by_value run: CRITICAL tier lands on claude-sonnet-5, NOT
    the old static mapping's claude-opus-4-8.

    CRITICAL coding floor=65. Survivors: claude-sonnet-5 (coding=71),
    claude-opus-4-7 (65), claude-opus-4-8 (72) -- claude-sonnet-4-6 (54) and
    claude-opus-4-6 (60) do NOT clear this higher floor. Of the 3
    survivors, claude-sonnet-5's blended cost (0.006) is the cheapest --
    claude-opus-4-7 and claude-opus-4-8 are both (0.005+0.025)/2=0.015,
    ~2.5x more expensive -- so sonnet-5 wins the cost tie-break.

    Same rationale and same magnitude of divergence as the ADVANCED tier
    above -- this is the biggest divergence in the whole routing-v3
    rollout: Anthropic's flagship model (opus-4-8) is replaced by its
    mid-tier model (sonnet-5) for the highest-stakes complexity tier.
    Explicitly reviewed and approved based on this exact predicted outcome.
    Zero real CRITICAL-tier Anthropic traffic exists today in
    .mullet/routing_history.jsonl.
    """
    reg = _make_registry()
    chosen = reg._best_anthropic("CRITICAL")
    assert chosen == "claude-sonnet-5"


def test_best_anthropic_missing_quality_data_fails_open_to_cheapest():
    """Empty MODEL_QUALITY fails open: every candidate passes the floor
    check, so cost-only sorting applies and the cheapest of all 6
    candidates wins -- claude-haiku-4-5-20251001 (blended
    (0.001+0.005)/2=0.003, cheapest among the 6-candidate set)."""
    reg = _make_registry()
    with patch("mullet.core.providers.quality_floors.MODEL_QUALITY", {}):
        chosen = reg._best_anthropic("ADVANCED")
    assert chosen is not None
    assert chosen == "claude-haiku-4-5-20251001"
