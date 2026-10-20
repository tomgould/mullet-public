"""Cover the last lines of routing-v2 new code (Phase A confidence_breakdown
signals + Phase B pick_provider_for_tier no-allowlist scan)."""

from __future__ import annotations

from mullet.core.router.complexity import (
    confidence_breakdown,
    extract_signals,
    score_complexity,
)
from mullet.core.router.decision import ProviderEntry, Router


def test_confidence_breakdown_includes_phase_a_signals():
    """confidence_breakdown reports the Phase A reasoning-keyword (+15) and
    prose-sophistication (+10) contributions, and stays in lockstep with
    score_complexity (the total must match)."""
    # ~124 words, contains reasoning keywords ("review"/"critique"), no code/paths,
    # so it lands in the 120–200 word prose band.
    prose = "Please review and critique " + "the design and approach in depth " * 20
    wc = len(prose.split())
    assert 120 < wc <= 200  # guard the test prompt stays in the prose band

    sig = extract_signals(prose)
    assert sig.lexical.has_reasoning_keywords is True

    bd = confidence_breakdown(sig)
    assert bd["lexical_signals"].get("reasoning_keywords") == 15
    assert bd["lexical_signals"].get("prose_sophistication") == 10
    # Lockstep invariant: breakdown total equals score_complexity.
    assert bd["total_score"] == score_complexity(sig)


def test_pick_provider_for_tier_no_allowlist_scans_order():
    """With allowed_providers=None and recommend_provider's pick absent from the
    registered providers, pick_provider_for_tier falls through to the
    _ALLOWLIST_TIER_PICK_ORDER scan (decision.py last line)."""
    # Only kimi is registered; CRITICAL recommends opus (not present) at low pressure,
    # so the scan picks the first tier-capable registered provider → kimi.
    router = Router(
        providers={"kimi": ProviderEntry(name="kimi", tier="cheap", models={"default": "k"})},
        use_pressure_routing=False,
    )
    assert router.pick_provider_for_tier("CRITICAL", None, pressure=0.05) == "kimi"
