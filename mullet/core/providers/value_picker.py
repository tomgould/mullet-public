"""The value-based model picker (routing-v3 Phase 2).

Given a candidate set for one provider, picks the cheapest model that
clears both a hard capability gate (reasoning-tuning, context fit) and a
quality floor for the task's complexity tier -- instead of a single
hardcoded "preferred" pick. See docs/design/routing-v3-value-picker.md
Phase 2 for the full design and plan-critique history.
"""
from __future__ import annotations

import logging

from mullet.core.providers.model_registry import (
    _filter_candidates_by_context,
    _pricing_for,
    _tier_chain_starting_from,
)
from mullet.core.providers.quality_floors import meets_quality_floor, meets_reasoning_gate

logger = logging.getLogger(__name__)


def pick_by_value(
    candidates: list[tuple[str, int]],
    tier: str,
    context_needed: int,
    domain: str,
    context_tokens: int | None = None,
    prefer_reasoning: bool = False,
) -> str | None:
    """Pick the cheapest candidate that clears the tier's quality floor.

    candidates: (model_id, max_context) pairs -- the same shape
        model_registry.py's _best_anthropic/_best_gemini already build.
    domain: which QualityScore field to filter on ("coding"/"reasoning"/
        "general") -- REQUIRED, no default, so callers can't silently
        mis-score a non-coding task through the coding lens.

    Returns None if nothing survives even after one tier of escalation --
    caller must fall back to its own static ladder. Never invents a model id.
    """
    def _survivors(t: str) -> list[tuple[str, int]]:
        fitting = [(mid, ctx) for mid, ctx in candidates if ctx >= context_needed]
        fitting = _filter_candidates_by_context(fitting, context_tokens)
        fitting = [(mid, ctx) for mid, ctx in fitting if meets_reasoning_gate(mid, prefer_reasoning)]
        fitting = [(mid, ctx) for mid, ctx in fitting if meets_quality_floor(mid, t, domain)]
        return fitting

    survivors = _survivors(tier)
    if not survivors:
        chain = _tier_chain_starting_from(tier)
        if len(chain) > 1:
            next_tier = chain[1]
            escalated = _survivors(next_tier)
            if escalated:
                logger.warning(
                    "pick_by_value: no candidate cleared tier=%s domain=%s floor; "
                    "escalated one tier to %s and found %d candidate(s)",
                    tier, domain, next_tier, len(escalated),
                )
                survivors = escalated
    if not survivors:
        return None

    def _blended_cost(model_id: str) -> float:
        inp, out = _pricing_for(model_id)
        # Reuses the 1:1 input/output cost-blending convention from
        # decision.py::_estimate_cost (line ~780): when no explicit
        # input/output token split is available, the router averages the
        # per-1K input and output rates. This is the only existing
        # single-number cost estimate in the codebase.
        return (inp + out) / 2.0

    def _sort_key(item: tuple[str, int]) -> tuple[bool, float, str]:
        model_id, _ctx = item
        cost = _blended_cost(model_id)
        is_zero_cost = cost == 0.0
        # zero/missing pricing sorts LAST (worst), not first -- an unpriced
        # model must never win the cost tie-break just because _pricing_for's
        # fail-open default looks like "free". Still selectable if it's the
        # only survivor.
        return (is_zero_cost, cost, model_id)

    survivors.sort(key=_sort_key)
    return survivors[0][0]
