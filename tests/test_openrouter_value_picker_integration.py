"""Integration tests for routing-v3 Phase 3a OpenRouter value-picker wiring.

These verify that the value-picker calls in `_best_openrouter` AND
`select_openrouter_model` (routing_matrix.py) fall back gracefully on
exception/None, that the unconditional overrides (`prefer_reasoning`/
CRITICAL in `_best_openrouter`; `needs_reasoning`/>500K-context/explicit
premium-speed-mode CRITICAL+ADVANCED in `select_openrouter_model`) still
short-circuit before the value picker is ever consulted, and that real
(non-mocked) picks match verified real outcomes across the 8-candidate
OpenRouter set -- including the approved behavior changes in this phase:
ADVANCED now lands on deepseek/deepseek-v4-flash instead of
deepseek/deepseek-v4-pro (both entry points), and -- newly verified for
`select_openrouter_model` specifically, since its CRITICAL branch was never
gated behind a reasoning-style override the way `_best_openrouter`'s is --
CRITICAL now lands on qwen/qwen3-coder instead of anthropic/claude-sonnet-5.

Follow-up (see the routing-v3 follow-up plan, Part A): `_best_openrouter`
now accepts `preflight`/`speed_mode` and delegates to
`select_openrouter_model` when a preflight is supplied, mirroring
`_best_gemini`/`_best_kimi` exactly -- so `select_openrouter_model` is no
longer dead code; both entry points are exercised below.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


from mullet.core.providers.model_registry import ModelRegistry
from mullet.core.router.complexity import Tier
from mullet.core.router.preflight import PreflightResult
from mullet.core.router.routing_matrix import select_openrouter_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry() -> ModelRegistry:
    return ModelRegistry()


# ---------------------------------------------------------------------------
# 1. Exception in pick_by_value → fallback to static ladder
# ---------------------------------------------------------------------------

def test_best_openrouter_fallback_on_exception():
    """If pick_by_value raises, _best_openrouter falls back to the static
    ladder's STANDARD default (today's behavior)."""
    reg = _make_registry()

    with patch(
        "mullet.core.providers.value_picker.pick_by_value",
        side_effect=RuntimeError("boom"),
    ):
        chosen = reg._best_openrouter("STANDARD")

    assert chosen == "deepseek/deepseek-v4-flash"


# ---------------------------------------------------------------------------
# 2. pick_by_value returns None → fallback to static ladder
# ---------------------------------------------------------------------------

def test_best_openrouter_fallback_on_none():
    """If pick_by_value returns None, _best_openrouter falls back to the
    static ladder's STANDARD default (today's behavior)."""
    reg = _make_registry()

    with patch(
        "mullet.core.providers.value_picker.pick_by_value",
        return_value=None,
    ):
        chosen = reg._best_openrouter("STANDARD")

    assert chosen == "deepseek/deepseek-v4-flash"


# ---------------------------------------------------------------------------
# 3. prefer_reasoning / CRITICAL overrides must short-circuit before the
#    value picker is ever consulted
# ---------------------------------------------------------------------------

def test_best_openrouter_reasoning_bypasses_value_picker():
    """prefer_reasoning=True must return deepseek/deepseek-v4-pro via the
    unconditional override, without ever consulting the value picker --
    regardless of complexity tier. Patch pick_by_value to raise
    unconditionally: if the override didn't truly short-circuit, this test
    would blow up instead of passing."""
    reg = _make_registry()

    with patch(
        "mullet.core.providers.value_picker.pick_by_value",
        side_effect=RuntimeError("boom"),
    ) as mock_pick:
        chosen = reg._best_openrouter("STANDARD", prefer_reasoning=True)

    assert chosen == "deepseek/deepseek-v4-pro"
    mock_pick.assert_not_called()


def test_best_openrouter_critical_bypasses_value_picker():
    """complexity="CRITICAL" (with prefer_reasoning=False) must ALSO return
    deepseek/deepseek-v4-pro via the same unconditional override, without ever
    consulting the value picker -- proving CRITICAL is protected by the
    override just as much as explicit prefer_reasoning requests are."""
    reg = _make_registry()

    with patch(
        "mullet.core.providers.value_picker.pick_by_value",
        side_effect=RuntimeError("boom"),
    ) as mock_pick:
        chosen = reg._best_openrouter("CRITICAL", prefer_reasoning=False)

    assert chosen == "deepseek/deepseek-v4-pro"
    mock_pick.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Real end-to-end value-picker integration (non-mocked)
# ---------------------------------------------------------------------------

def test_best_openrouter_real_standard_tier_picks_v4_flash():
    """Real pick_by_value run: STANDARD tier coding lands on
    deepseek/deepseek-v4-flash -- matches today's static default exactly.
    Floor=18; all 8 candidates clear it, and v4-flash is cheapest (blended
    0.000135 vs. the next-cheapest v4-pro at 0.0006525 per pricing.yaml)."""
    reg = _make_registry()
    chosen = reg._best_openrouter("STANDARD")
    assert chosen == "deepseek/deepseek-v4-flash"


def test_best_openrouter_real_low_tier_picks_v4_flash():
    """Real pick_by_value run: LOW tier coding lands on deepseek/deepseek-v4-flash.
    The free-tier list is empty since OpenRouter withdrew qwen/qwen3-coder:free
    and tencent/hy3:free (2026-10-20), so LOW goes to the paid ladder, where
    v4-flash is the cheapest candidate over the LOW floor (10)."""
    reg = _make_registry()
    chosen = reg._best_openrouter("LOW")
    assert chosen == "deepseek/deepseek-v4-flash"


def test_best_openrouter_real_advanced_tier_picks_v4_flash_not_v4_pro():
    """Real pick_by_value run: ADVANCED tier coding lands on
    deepseek/deepseek-v4-flash, NOT the old static default
    deepseek/deepseek-v4-pro.

    This is an intentional, user-approved divergence from today's behavior
    -- the first genuine (not merely hypothetical) behavior change in the
    whole routing-v3 rollout (Gemini/Kimi wiring were both provable no-ops
    on every real historical tier). ADVANCED's coding floor is 54
    (post floor-recalibration); deepseek-v4-flash's coding score is 59, so
    it clears the floor same as deepseek-v4-pro (62) does. Per
    pricing.yaml, v4-flash's blended cost is (0.00009 + 0.00018) / 2 =
    0.000135 vs. v4-pro's (0.000435 + 0.00087) / 2 = 0.0006525 -- v4-flash
    is ~4.8x cheaper, so it wins the cost tie-break. This is the value
    picker doing exactly its job -- "cheapest candidate that clears the
    quality bar" -- not a bug. Zero real ADVANCED-tier OpenRouter traffic
    exists in .mullet/routing_history.jsonl today, so nothing observable
    changes yet, but future ADVANCED-tier OpenRouter requests will pick
    v4-flash instead of v4-pro.
    """
    reg = _make_registry()
    chosen = reg._best_openrouter("ADVANCED")
    assert chosen == "deepseek/deepseek-v4-flash"


def test_best_openrouter_missing_quality_data_fails_open():
    """Empty MODEL_QUALITY fails open: every candidate passes the floor and
    reasoning-gate checks, so cost-only sorting applies and the cheapest of
    all 8 candidates wins -- deepseek/deepseek-v4-flash (blended 0.000135
    per pricing.yaml, cheapest among the 8-candidate set)."""
    reg = _make_registry()
    with patch("mullet.core.providers.quality_floors.MODEL_QUALITY", {}):
        chosen = reg._best_openrouter("ADVANCED")
    assert chosen is not None
    assert chosen == "deepseek/deepseek-v4-flash"


# ---------------------------------------------------------------------------
# 5. select_openrouter_model wiring (now reachable via _best_openrouter's
#    preflight param -- see the routing-v3 follow-up plan, Part A)
# ---------------------------------------------------------------------------

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


def test_select_openrouter_model_fallback_on_exception():
    """If pick_by_value raises, select_openrouter_model falls back to the
    static ladder's STANDARD default (today's behavior)."""
    pf = _make_preflight(complexity_tier=Tier.STANDARD)

    with patch(
        "mullet.core.providers.value_picker.pick_by_value",
        side_effect=RuntimeError("boom"),
    ):
        model_id, reason = select_openrouter_model(pf, speed_mode="balanced")

    assert model_id == "deepseek/deepseek-v4-flash"
    assert "standard" in reason.lower()


def test_select_openrouter_model_fallback_on_none():
    """If pick_by_value returns None, select_openrouter_model falls back to
    the static ladder's STANDARD default (today's behavior)."""
    pf = _make_preflight(complexity_tier=Tier.STANDARD)

    with patch(
        "mullet.core.providers.value_picker.pick_by_value",
        return_value=None,
    ):
        model_id, reason = select_openrouter_model(pf, speed_mode="balanced")

    assert model_id == "deepseek/deepseek-v4-flash"
    assert "standard" in reason.lower()


def test_select_openrouter_model_needs_reasoning_bypasses_value_picker():
    """needs_reasoning=True must short-circuit to deepseek/deepseek-v4-pro
    (balanced) without ever consulting the value picker. pick_by_value is
    patched to raise unconditionally: if the override didn't truly
    short-circuit, this test would blow up instead of passing."""
    pf = _make_preflight(needs_reasoning=True, complexity_tier=Tier.CRITICAL)

    with patch(
        "mullet.core.providers.value_picker.pick_by_value",
        side_effect=RuntimeError("boom"),
    ) as mock_pick:
        model_id, reason = select_openrouter_model(pf, speed_mode="balanced")

    assert model_id == "deepseek/deepseek-v4-pro"
    mock_pick.assert_not_called()


def test_select_openrouter_model_huge_context_bypasses_value_picker():
    """input_tokens > 500K must short-circuit to qwen/qwen3-coder (balanced)
    without ever consulting the value picker."""
    pf = _make_preflight(input_tokens=700_000, complexity_tier=Tier.CRITICAL)

    with patch(
        "mullet.core.providers.value_picker.pick_by_value",
        side_effect=RuntimeError("boom"),
    ) as mock_pick:
        model_id, reason = select_openrouter_model(pf, speed_mode="balanced")

    assert model_id == "qwen/qwen3-coder"
    mock_pick.assert_not_called()


def test_select_openrouter_model_premium_critical_bypasses_value_picker():
    """speed_mode='premium' with CRITICAL must short-circuit to the
    preserved premium override (anthropic/claude-opus-4.8) without ever
    consulting the value picker."""
    pf = _make_preflight(complexity_tier=Tier.CRITICAL)

    with patch(
        "mullet.core.providers.value_picker.pick_by_value",
        side_effect=RuntimeError("boom"),
    ) as mock_pick:
        model_id, reason = select_openrouter_model(pf, speed_mode="premium")

    assert model_id == "anthropic/claude-opus-4.8"
    mock_pick.assert_not_called()


def test_select_openrouter_model_premium_advanced_bypasses_value_picker():
    """speed_mode='premium' with ADVANCED must short-circuit to the
    preserved premium override (anthropic/claude-sonnet-5) without ever
    consulting the value picker."""
    pf = _make_preflight(complexity_tier=Tier.ADVANCED)

    with patch(
        "mullet.core.providers.value_picker.pick_by_value",
        side_effect=RuntimeError("boom"),
    ) as mock_pick:
        model_id, reason = select_openrouter_model(pf, speed_mode="premium")

    assert model_id == "anthropic/claude-sonnet-5"
    mock_pick.assert_not_called()


def test_select_openrouter_model_real_low_tier():
    """Real pick_by_value run: with no free-tier candidates left, LOW tier
    lands on deepseek/deepseek-v4-flash through the paid value picker."""
    pf = _make_preflight(complexity_tier=Tier.LOW)
    model_id, reason = select_openrouter_model(pf, speed_mode="balanced")
    assert model_id == "deepseek/deepseek-v4-flash"
    assert "free-tier" not in reason.lower()


def test_select_openrouter_model_real_standard_tier():
    """Real pick_by_value run: STANDARD tier lands on
    deepseek/deepseek-v4-flash -- matches today's static default exactly."""
    pf = _make_preflight(complexity_tier=Tier.STANDARD)
    model_id, reason = select_openrouter_model(pf, speed_mode="balanced")
    assert model_id == "deepseek/deepseek-v4-flash"
    assert "value picker" in reason.lower()


def test_select_openrouter_model_real_advanced_tier_picks_v4_flash_not_v4_pro():
    """Real pick_by_value run: ADVANCED tier lands on
    deepseek/deepseek-v4-flash, NOT the old static default
    deepseek/deepseek-v4-pro -- same approved divergence already verified
    for _best_openrouter this session (see
    test_best_openrouter_real_advanced_tier_picks_v4_flash_not_v4_pro
    above). ADVANCED coding floor=54; v4-flash (59) clears it same as
    v4-pro (62), and is ~4.8x cheaper, so it wins the cost tie-break."""
    pf = _make_preflight(complexity_tier=Tier.ADVANCED)
    model_id, reason = select_openrouter_model(pf, speed_mode="balanced")
    assert model_id == "deepseek/deepseek-v4-flash"
    assert "value picker" in reason.lower()


def test_select_openrouter_model_real_critical_tier_picks_qwen3_coder():
    """Real pick_by_value run: CRITICAL tier lands on qwen/qwen3-coder, NOT
    the old static default anthropic/claude-sonnet-5.

    This is a NEW divergence, not previously exercised anywhere this
    session: `_best_openrouter`'s own CRITICAL handling always
    short-circuits to deepseek/deepseek-v4-pro before the value picker is ever
    consulted (see test_best_openrouter_critical_bypasses_value_picker
    above), so the picker has never actually run for CRITICAL until now.
    `select_openrouter_model`'s CRITICAL branch has no such override --
    only `needs_reasoning` bypasses the picker -- so CRITICAL (balanced,
    non-reasoning) reaches the picker here.

    CRITICAL coding floor=65. Of the 8 candidates, only qwen/qwen3-coder
    (coding=70), anthropic/claude-opus-4.8 (72), and anthropic/claude-sonnet-5
    (71) clear it. Per pricing.yaml, qwen3-coder's blended cost is
    (0.00022 + 0.0018) / 2 = 0.00101 -- far cheaper than sonnet-5's
    (0.002 + 0.01) / 2 = 0.006 or opus-4.8's (0.005 + 0.025) / 2 = 0.015 --
    so qwen3-coder wins the cost tie-break. Verified by running the real
    picker logic directly, not assumed from the plan.
    """
    pf = _make_preflight(complexity_tier=Tier.CRITICAL)
    model_id, reason = select_openrouter_model(pf, speed_mode="balanced")
    assert model_id == "qwen/qwen3-coder"
    assert "value picker" in reason.lower()


def test_select_openrouter_model_missing_quality_data_fails_open():
    """Empty MODEL_QUALITY fails open: every candidate passes the floor and
    reasoning-gate checks, so cost-only sorting applies and the cheapest of
    all 8 candidates wins -- deepseek/deepseek-v4-flash."""
    pf = _make_preflight(complexity_tier=Tier.ADVANCED)
    with patch("mullet.core.providers.quality_floors.MODEL_QUALITY", {}):
        model_id, reason = select_openrouter_model(pf, speed_mode="balanced")
    assert model_id == "deepseek/deepseek-v4-flash"


# ---------------------------------------------------------------------------
# 6. _best_openrouter + preflight delegation (mirrors _best_gemini/_best_kimi)
# ---------------------------------------------------------------------------

def test_best_openrouter_preflight_delegates_to_routing_matrix():
    """When a preflight is supplied, _best_openrouter delegates to
    select_openrouter_model and returns its pick when that model id is
    present in the (static-fallback) candidate list."""
    reg = _make_registry()
    pf = _make_preflight(complexity_tier=Tier.STANDARD)
    chosen = reg._best_openrouter("STANDARD", preflight=pf, speed_mode="balanced")
    assert chosen == "deepseek/deepseek-v4-flash"


def test_best_openrouter_preflight_falls_back_when_pick_not_in_candidates():
    """When select_openrouter_model's pick isn't present in
    _openrouter_candidates() (no live cache in this test, and the static
    fallback list doesn't carry that exact id), _best_openrouter falls
    through to its own existing ladder rather than returning a model id it
    can't confirm is available -- exactly the CRITICAL case:
    select_openrouter_model picks qwen/qwen3-coder (see above), but the
    OpenRouter static fallback list only has qwen/qwen3-coder-next, not
    qwen/qwen3-coder, so the match fails and _best_openrouter's own
    unconditional CRITICAL override (deepseek/deepseek-v4-pro) applies
    instead."""
    reg = _make_registry()
    pf = _make_preflight(complexity_tier=Tier.CRITICAL)
    chosen = reg._best_openrouter("CRITICAL", preflight=pf, speed_mode="balanced")
    assert chosen == "deepseek/deepseek-v4-pro"


# ---------------------------------------------------------------------------
# 7. Free-tier candidates -- LOW-only, never leak to higher tiers
#
# The real free-tier list is empty (OpenRouter withdrew both former entries),
# so these run against a stand-in free model to keep the mechanism covered.
# ---------------------------------------------------------------------------

FAKE_FREE = "example/free-coder:free"


@pytest.fixture()
def fake_free_tier(monkeypatch):
    from mullet.core.providers import model_registry as _registry
    from mullet.core.providers.quality import MODEL_QUALITY, QualityScore
    from mullet.core.router import routing_matrix as _matrix

    roster = [(FAKE_FREE, 1_048_576)]
    monkeypatch.setattr(_matrix, "OPENROUTER_FREE_TIER_CANDIDATES", roster)
    monkeypatch.setattr(_registry, "OPENROUTER_FREE_TIER_CANDIDATES", roster)
    monkeypatch.setitem(MODEL_QUALITY, FAKE_FREE, QualityScore(
        coding=70, reasoning=45, general=45, confidence="direct", source="test stand-in",
    ))
    return FAKE_FREE


def test_free_tier_low_preflight_picks_the_free_model(fake_free_tier):
    """LOW + preflight → free-tier block wins."""
    pf = _make_preflight(complexity_tier=Tier.LOW)
    model_id, reason = select_openrouter_model(pf, speed_mode="balanced")
    assert model_id == fake_free_tier
    assert "free-tier" in reason.lower()


def test_free_tier_low_no_preflight_picks_the_free_model(fake_free_tier):
    """LOW without preflight → _best_openrouter's free-tier block wins."""
    reg = _make_registry()
    chosen = reg._best_openrouter("LOW")
    assert chosen == fake_free_tier


def test_free_tier_skips_a_model_recorded_as_unavailable(fake_free_tier, monkeypatch, tmp_path):
    """A free model OpenRouter has refused is left out until its record expires."""
    from mullet.core.providers import model_availability

    monkeypatch.setattr(model_availability, "STORE_PATH", tmp_path / "unavailable.json")
    model_availability.mark_unavailable("openrouter", fake_free_tier, "404 unavailable for free")
    pf = _make_preflight(complexity_tier=Tier.LOW)
    model_id, reason = select_openrouter_model(pf, speed_mode="balanced")
    assert model_id == "deepseek/deepseek-v4-flash"
    assert _make_registry()._best_openrouter("LOW") == "deepseek/deepseek-v4-flash"


def test_free_tier_never_leaks_to_standard(fake_free_tier):
    """STANDARD tier must NOT pick a free model -- unchanged from before."""
    pf = _make_preflight(complexity_tier=Tier.STANDARD)
    model_id, reason = select_openrouter_model(pf, speed_mode="balanced")
    assert model_id == "deepseek/deepseek-v4-flash"
    assert "free-tier" not in reason.lower()


def test_free_tier_never_leaks_to_advanced(fake_free_tier):
    """ADVANCED tier must NOT pick a free model -- unchanged from before."""
    pf = _make_preflight(complexity_tier=Tier.ADVANCED)
    model_id, reason = select_openrouter_model(pf, speed_mode="balanced")
    assert model_id == "deepseek/deepseek-v4-flash"
    assert "free-tier" not in reason.lower()


def test_free_tier_never_leaks_to_critical(fake_free_tier):
    """CRITICAL tier must NOT pick a free model -- unchanged from before."""
    pf = _make_preflight(complexity_tier=Tier.CRITICAL)
    model_id, reason = select_openrouter_model(pf, speed_mode="balanced")
    assert model_id == "qwen/qwen3-coder"
    assert "free-tier" not in reason.lower()


def test_free_tier_skipped_when_needs_reasoning_at_low(fake_free_tier):
    """needs_reasoning=True at LOW must bypass the free-tier block and fall
    through to the existing reasoning override (deepseek/deepseek-v4-pro)."""
    pf = _make_preflight(complexity_tier=Tier.LOW, needs_reasoning=True)
    model_id, reason = select_openrouter_model(pf, speed_mode="balanced")
    assert model_id == "deepseek/deepseek-v4-pro"
    assert "reasoning" in reason.lower()


def test_free_tier_skipped_when_needs_reasoning_at_low_no_preflight(fake_free_tier):
    """prefer_reasoning=True at LOW (no preflight) must bypass the free-tier
    block in _best_openrouter and fall through to deepseek/deepseek-v4-pro."""
    reg = _make_registry()
    chosen = reg._best_openrouter("LOW", prefer_reasoning=True)
    assert chosen == "deepseek/deepseek-v4-pro"
