"""Delegation safety D5 — auto-escalation within provider when tier too small.

When ``get_best_model(provider, tier=X, context_tokens=N)`` finds that no
candidate at tier X can serve the prompt, D5 transparently retries at the
next-higher tier in the same provider before raising
``ModelContextExceeded(all_candidates_exhausted=True)``.

This is the difference between "STANDARD has only 8K-128K models, the user's
14K prompt is doomed" (today: raise; what users see) and "STANDARD's
candidates can't serve this, but ADVANCED's kimi-k2.5 has 262K — pick that"
(D5: silent escalation, INFO-logged for observability).

Invariants:
  1. No escalation when the picker succeeds at the requested tier.
  2. Escalation only fires on ``ModelContextExceeded(all_candidates_exhausted=False)``.
     If the picker says "even my biggest model is too small", we stop —
     climbing tiers won't help and just wastes cycles.
  3. Escalation walks the canonical tier order
     (LOW → STANDARD → ADVANCED → CRITICAL) starting from the requested tier.
  4. After all tiers are exhausted in the same provider, raise MCE with
     ``all_candidates_exhausted=True``.
  5. Auto-escalation is logged at INFO so callers can see the tier change.
"""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from mullet.core.providers.errors import ModelContextExceeded
from mullet.core.providers.model_registry import ModelRegistry


# ---------------------------------------------------------------------------
# 1. Tier-chain helper
# ---------------------------------------------------------------------------


def test_tier_chain_starting_from_standard():
    """Red: helper must return [STANDARD, ADVANCED, CRITICAL] from STANDARD.

    Failure mode: a typo in TIER_ORDER means escalation goes the wrong
    direction (STANDARD → LOW), or it never escalates because the chain
    is empty after the start tier.
    """
    from mullet.core.providers.model_registry import _tier_chain_starting_from

    assert _tier_chain_starting_from("STANDARD") == ["STANDARD", "ADVANCED", "CRITICAL"]


def test_tier_chain_starting_from_low():
    """Red: from LOW, chain spans the full tier ladder."""
    from mullet.core.providers.model_registry import _tier_chain_starting_from

    assert _tier_chain_starting_from("LOW") == ["LOW", "STANDARD", "ADVANCED", "CRITICAL"]


def test_tier_chain_starting_from_critical_is_singleton():
    """Red: from the top tier there is nowhere to escalate to."""
    from mullet.core.providers.model_registry import _tier_chain_starting_from

    assert _tier_chain_starting_from("CRITICAL") == ["CRITICAL"]


def test_tier_chain_handles_unknown_tier():
    """Red: an unknown tier returns just itself — never KeyError."""
    from mullet.core.providers.model_registry import _tier_chain_starting_from

    # Defensive: a typo or new-string tier shouldn't crash the chain helper.
    result = _tier_chain_starting_from("WUT")
    assert result == ["WUT"]


# ---------------------------------------------------------------------------
# 2. No escalation for small prompts
# ---------------------------------------------------------------------------


def test_get_best_model_no_escalation_for_small_prompt(caplog):
    """Red: STANDARD + ~1K tokens picks at STANDARD; no escalation log.

    Failure mode: D5 fires unconditionally even when no escalation is
    required, polluting INFO logs and possibly degrading model selection.
    """
    reg = ModelRegistry()
    # Empty live cache → static fallback runs.
    reg._cache["kimi"] = []
    reg._cache_ts["kimi"] = time.time()

    with caplog.at_level("INFO", logger="mullet.core.providers.model_registry"):
        chosen = reg.get_best_model(
            "kimi",
            complexity="STANDARD",
            context_needed=4096,
            context_tokens=1_000,
        )

    assert chosen, "expected a non-empty model id"
    # The escalation message starts with "auto-escalated"; confirm it's NOT present.
    assert not any(
        "auto-escalated" in rec.getMessage() for rec in caplog.records
    ), (
        f"unexpected escalation log when none was needed; got "
        f"{[r.getMessage() for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# 3. Escalation when tier-static-fallback raises non-exhausted MCE
# ---------------------------------------------------------------------------


def test_get_best_model_escalates_when_pick_raises_non_exhausted(caplog):
    """Red: when ``_pick_model`` raises MCE(exhausted=False) for the requested
    tier, ``get_best_model`` must retry at the next tier and succeed there.

    Simulates: a provider whose STANDARD-tier picker artificially refuses
    (e.g., a future tier-allowlist rejected the candidates) but whose
    ADVANCED-tier picker succeeds with a bigger model.

    Failure mode: today the MCE leaks straight to the caller and the user
    sees a refusal even though the same provider has a model that could
    have served the prompt.
    """
    reg = ModelRegistry()

    call_log: list[str] = []
    success_model = "kimi-k2.5"

    def fake_pick_model(*, provider, complexity, context_needed, context_tokens, **_kw):
        call_log.append(complexity)
        if complexity == "STANDARD":
            raise ModelContextExceeded(
                provider="kimi",
                tier="STANDARD",
                context_tokens=context_tokens or 0,
                tried_candidates=["moonshot-v1-8k"],
                max_candidate_context=8_192,
                all_candidates_exhausted=False,  # bigger sibling exists
            )
        return success_model

    # Bypass the post-pick assertion so we test the escalation logic in isolation.
    with patch.object(reg, "_pick_model", side_effect=fake_pick_model), \
         patch.object(reg, "_assert_can_serve", side_effect=lambda p, m, c, t: m), \
         caplog.at_level("INFO", logger="mullet.core.providers.model_registry"):
        chosen = reg.get_best_model(
            "kimi",
            complexity="STANDARD",
            context_needed=4096,
            context_tokens=200_000,
        )

    assert chosen == success_model, (
        f"expected escalated pick {success_model!r}, got {chosen!r}"
    )
    assert call_log == ["STANDARD", "ADVANCED"], (
        f"escalation should have walked STANDARD → ADVANCED; got {call_log!r}"
    )
    assert any(
        "auto-escalated" in rec.getMessage() and "ADVANCED" in rec.getMessage()
        for rec in caplog.records
    ), (
        f"expected an INFO 'auto-escalated' log mentioning ADVANCED; got "
        f"{[r.getMessage() for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# 4. NO escalation when MCE has all_candidates_exhausted=True
# ---------------------------------------------------------------------------


def test_get_best_model_does_not_escalate_when_exhausted(caplog):
    """Red: when ``_pick_model`` raises MCE(exhausted=True), do NOT escalate.

    The provider's biggest model is already too small — climbing tiers
    just wastes cycles and ends in the same refusal.

    Failure mode: D5 escalates blindly, calls _pick_model 4 times for the
    same provider, returns a stale max_candidate_context to the caller.
    """
    reg = ModelRegistry()
    call_log: list[str] = []

    def always_exhausted(*, provider, complexity, context_needed, context_tokens, **_kw):
        call_log.append(complexity)
        raise ModelContextExceeded(
            provider="kimi",
            tier=complexity,
            context_tokens=context_tokens or 0,
            tried_candidates=["kimi-k2.5"],
            max_candidate_context=262_144,
            all_candidates_exhausted=True,
        )

    with patch.object(reg, "_pick_model", side_effect=always_exhausted):
        with pytest.raises(ModelContextExceeded) as exc_info:
            reg.get_best_model(
                "kimi",
                complexity="STANDARD",
                context_needed=4096,
                context_tokens=10_000_000,
            )

    assert exc_info.value.all_candidates_exhausted is True
    assert call_log == ["STANDARD"], (
        f"exhausted flag must short-circuit escalation; got {call_log!r}"
    )


# ---------------------------------------------------------------------------
# 5. Escalation walks multiple tiers
# ---------------------------------------------------------------------------


def test_get_best_model_escalates_through_multiple_tiers(caplog):
    """Red: LOW request must walk LOW → STANDARD → ADVANCED before picking.

    Failure mode: D5 only tries the next tier and gives up if that also
    fails — the user wants the chain walked end-to-end.
    """
    reg = ModelRegistry()
    call_log: list[str] = []

    def fake_pick_model(*, provider, complexity, context_needed, context_tokens, **_kw):
        call_log.append(complexity)
        if complexity in ("LOW", "STANDARD"):
            raise ModelContextExceeded(
                provider=provider,
                tier=complexity,
                context_tokens=context_tokens or 0,
                tried_candidates=["small-model"],
                max_candidate_context=8_192,
                all_candidates_exhausted=False,
            )
        # ADVANCED succeeds.
        return "kimi-k2.5"

    with patch.object(reg, "_pick_model", side_effect=fake_pick_model), \
         patch.object(reg, "_assert_can_serve", side_effect=lambda p, m, c, t: m), \
         caplog.at_level("INFO", logger="mullet.core.providers.model_registry"):
        chosen = reg.get_best_model(
            "kimi",
            complexity="LOW",
            context_needed=4096,
            context_tokens=200_000,
        )

    assert chosen == "kimi-k2.5"
    assert call_log == ["LOW", "STANDARD", "ADVANCED"], (
        f"chain must walk LOW → STANDARD → ADVANCED; got {call_log!r}"
    )


# ---------------------------------------------------------------------------
# 6. All tiers exhausted → MCE with all_candidates_exhausted=True
# ---------------------------------------------------------------------------


def test_get_best_model_raises_after_all_tiers_exhausted(caplog):
    """Red: when every tier in the chain refuses with non-exhausted MCE,
    the final raise must carry ``all_candidates_exhausted=True``.

    Failure mode: D5 leaks the LAST tier's MCE (which still says
    exhausted=False) and the caller wrongly thinks escalation is still
    possible — looping forever or trying a doomed retry.
    """
    reg = ModelRegistry()

    def always_non_exhausted(*, provider, complexity, context_needed, context_tokens, **_kw):
        # Each tier raises non-exhausted, claiming a bigger sibling exists —
        # but no tier succeeds. After walking all tiers, get_best_model must
        # convert the trail of non-exhausted MCEs into a single exhausted MCE.
        raise ModelContextExceeded(
            provider=provider,
            tier=complexity,
            context_tokens=context_tokens or 0,
            tried_candidates=[f"fake-{complexity.lower()}"],
            max_candidate_context=8_192,
            all_candidates_exhausted=False,
        )

    with patch.object(reg, "_pick_model", side_effect=always_non_exhausted):
        with pytest.raises(ModelContextExceeded) as exc_info:
            reg.get_best_model(
                "kimi",
                complexity="STANDARD",
                context_needed=4096,
                context_tokens=10_000_000,
            )

    err = exc_info.value
    assert err.all_candidates_exhausted is True, (
        "after walking every tier without success, exhausted must be True"
    )
    assert err.provider == "kimi"


# ---------------------------------------------------------------------------
# 7. Escalated pick is logged at INFO with all key fields
# ---------------------------------------------------------------------------


def test_escalation_log_includes_requested_tier_and_picked_tier(caplog):
    """Red: the INFO log line must include the original requested tier,
    the tier the picker actually succeeded at, the context_tokens count,
    and the picked model name — so operators can see what happened.

    Failure mode: an opaque "auto-escalated" log without context makes
    debugging impossible.
    """
    reg = ModelRegistry()

    def fake_pick_model(*, provider, complexity, context_needed, context_tokens, **_kw):
        if complexity == "STANDARD":
            raise ModelContextExceeded(
                provider="kimi",
                tier="STANDARD",
                context_tokens=context_tokens or 0,
                tried_candidates=["moonshot-v1-128k"],
                max_candidate_context=131_072,
                all_candidates_exhausted=False,
            )
        return "kimi-k2.5"

    with patch.object(reg, "_pick_model", side_effect=fake_pick_model), \
         patch.object(reg, "_assert_can_serve", side_effect=lambda p, m, c, t: m), \
         caplog.at_level("INFO", logger="mullet.core.providers.model_registry"):
        reg.get_best_model(
            "kimi",
            complexity="STANDARD",
            context_needed=4096,
            context_tokens=200_000,
        )

    matching = [
        rec.getMessage() for rec in caplog.records
        if "auto-escalated" in rec.getMessage()
    ]
    assert matching, "expected at least one auto-escalation INFO log"
    msg = matching[0]
    assert "STANDARD" in msg, "log must mention requested tier"
    assert "ADVANCED" in msg, "log must mention picked tier"
    assert "200000" in msg or "200_000" in msg, "log must mention context_tokens"
    assert "kimi-k2.5" in msg, "log must mention the picked model"
