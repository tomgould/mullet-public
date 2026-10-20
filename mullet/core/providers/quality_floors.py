"""Complexity tier -> minimum quality-score floor mapping (routing-v3 Phase 1).

Defines, per Tier, the minimum domain sub-score (coding/reasoning/general)
a model must clear to be a value-picker candidate (Phase 2). Floors are a
first-pass calibration against the Phase 0 score distribution -- expect
these to be tuned once Phase 4 validates against real routing_history.jsonl
traffic via `mullet routing simulate`. See docs/design/routing-v3-value-picker.md.
"""
from __future__ import annotations

from mullet.core.providers.quality import MODEL_QUALITY

# Minimum score per domain a model must clear for a given tier. Deliberately
# permissive at LOW/STANDARD (the whole point of those tiers is "cheapest
# capable option" -- a floor that's too strict there would exclude models
# already serving as a provider's own curated default, e.g. gemini-2.5-flash
# at coding=20 must still clear STANDARD). Tightens meaningfully at
# ADVANCED/CRITICAL where quality matters more than unit cost. Where possible,
# ADVANCED/CRITICAL floors are anchored to a reference model's exact score
# (claude-sonnet-4-6 for ADVANCED; claude-opus-4-7 for CRITICAL coding;
# deepseek/deepseek-v4-pro and deepseek/deepseek-v4-flash for CRITICAL reasoning;
# qwen/qwen3-coder for CRITICAL general) rather than picked by feel.
TIER_QUALITY_FLOORS: dict[str, dict[str, int]] = {
    "LOW": {"coding": 10, "reasoning": 10, "general": 5},
    "STANDARD": {"coding": 18, "reasoning": 15, "general": 12},
    "ADVANCED": {"coding": 54, "reasoning": 45, "general": 40},
    "CRITICAL": {"coding": 65, "reasoning": 50, "general": 45},
}


def meets_quality_floor(model_id: str, tier: str, domain: str) -> bool:
    """Return True if model_id's <domain> score clears <tier>'s floor.

    Fails open (returns True) when model_id has no MODEL_QUALITY entry --
    Phase 2's picker must fall back to the current static ladder for
    providers/models with no quality data yet (see design doc Phase 0
    "no blind inherit" note -- but for MISSING entries entirely, rather
    than inherited variants, the invariant is: don't block routing on
    missing research data).
    """
    qs = MODEL_QUALITY.get(model_id)
    if qs is None:
        return True
    floor = TIER_QUALITY_FLOORS.get(tier, TIER_QUALITY_FLOORS["STANDARD"]).get(domain, 0)
    return getattr(qs, domain) >= floor


def meets_reasoning_gate(model_id: str, needs_reasoning: bool) -> bool:
    """Hard capability gate for needs_reasoning=True requests.

    When needs_reasoning is True, only models with is_reasoning_tuned=True
    qualify at all -- a high 'reasoning' score alone is not sufficient,
    per the Phase 1 design note (some models test well on reasoning
    benchmarks without being architecturally reasoning-tuned). When
    needs_reasoning is False, this gate never excludes anything.
    Fails open (returns True) when model_id has no MODEL_QUALITY entry,
    same invariant as meets_quality_floor.
    """
    if not needs_reasoning:
        return True
    qs = MODEL_QUALITY.get(model_id)
    if qs is None:
        return True
    return qs.is_reasoning_tuned
