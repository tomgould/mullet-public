"""Tests for mullet.core.providers.quality_floors (routing-v3 Phase 1)."""
from __future__ import annotations


from mullet.core.providers.quality_floors import (
    TIER_QUALITY_FLOORS,
    meets_quality_floor,
    meets_reasoning_gate,
)


# ---------------------------------------------------------------------------
# TIER_QUALITY_FLOORS structural invariants
# ---------------------------------------------------------------------------


def test_tier_keys_exact() -> None:
    assert set(TIER_QUALITY_FLOORS.keys()) == {"LOW", "STANDARD", "ADVANCED", "CRITICAL"}


def test_each_tier_has_exact_domain_keys() -> None:
    for tier, floors in TIER_QUALITY_FLOORS.items():
        assert set(floors.keys()) == {"coding", "reasoning", "general"}, (
            f"{tier} has unexpected domain keys"
        )


def test_all_floors_are_ints_in_range() -> None:
    for tier, floors in TIER_QUALITY_FLOORS.items():
        for domain, val in floors.items():
            assert isinstance(val, int), (
                f"{tier}.{domain} must be int, got {type(val).__name__}"
            )
            assert 0 <= val <= 100, (
                f"{tier}.{domain}={val} out of [0, 100] range"
            )


def test_floors_monotonically_non_decreasing() -> None:
    """Floors must not accidentally decrease from LOW -> CRITICAL."""
    tiers = ["LOW", "STANDARD", "ADVANCED", "CRITICAL"]
    for domain in ("coding", "reasoning", "general"):
        prev = -1
        for tier in tiers:
            curr = TIER_QUALITY_FLOORS[tier][domain]
            assert curr >= prev, (
                f"{domain} floor decreased from previous tier to {tier}: "
                f"{prev} -> {curr}"
            )
            prev = curr


# ---------------------------------------------------------------------------
# meets_quality_floor
# ---------------------------------------------------------------------------


def test_meets_quality_floor_known_model_clears() -> None:
    """claude-opus-4-8 coding=72 clears CRITICAL coding floor=65."""
    assert meets_quality_floor("claude-opus-4-8", "CRITICAL", "coding") is True


def test_meets_quality_floor_known_model_fails() -> None:
    """gemini-2.5-flash coding=20 does NOT clear ADVANCED coding floor=55."""
    assert meets_quality_floor("gemini-2.5-flash", "ADVANCED", "coding") is False


def test_meets_quality_floor_unknown_model_fails_open() -> None:
    """A model not in MODEL_QUALITY always returns True."""
    assert meets_quality_floor("unknown-model-xyz", "CRITICAL", "coding") is True
    assert meets_quality_floor("unknown-model-xyz", "LOW", "general") is True


# ---------------------------------------------------------------------------
# meets_reasoning_gate
# ---------------------------------------------------------------------------


def test_meets_reasoning_gate_false_always_passes() -> None:
    """needs_reasoning=False never excludes anything."""
    assert meets_reasoning_gate("deepseek/deepseek-v4-pro", needs_reasoning=False) is True
    assert meets_reasoning_gate("gemini-2.5-flash", needs_reasoning=False) is True


def test_meets_reasoning_gate_true_reasoning_tuned_models_pass() -> None:
    assert meets_reasoning_gate("deepseek/deepseek-v4-pro", needs_reasoning=True) is True
    assert meets_reasoning_gate("kimi-k2.7-code", needs_reasoning=True) is True


def test_meets_reasoning_gate_true_non_reasoning_tuned_fails() -> None:
    assert meets_reasoning_gate("gemini-2.5-flash", needs_reasoning=True) is False


def test_meets_reasoning_gate_unknown_model_fails_open() -> None:
    assert meets_reasoning_gate("unknown-model-xyz", needs_reasoning=True) is True


# ---------------------------------------------------------------------------
# Anchor-model regression tests
# ---------------------------------------------------------------------------


def test_advanced_floors_anchor_to_sonnet_4_6() -> None:
    """claude-sonnet-4-6 is the ADVANCED-tier anchor model and must clear
    ALL THREE domain floors, not just two -- regression test for the
    off-by-one that had ADVANCED coding=55 (sonnet-4-6 scores 54)."""
    assert meets_quality_floor("claude-sonnet-4-6", "ADVANCED", "coding") is True
    assert meets_quality_floor("claude-sonnet-4-6", "ADVANCED", "reasoning") is True
    assert meets_quality_floor("claude-sonnet-4-6", "ADVANCED", "general") is True


def test_critical_coding_floor_anchors_to_opus_4_7() -> None:
    """claude-opus-4-7 is the CRITICAL-coding anchor model (coding=65 exactly
    matches the floor) -- documents the already-correct pattern so it can't
    silently drift later."""
    assert meets_quality_floor("claude-opus-4-7", "CRITICAL", "coding") is True
