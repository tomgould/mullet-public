"""Eco modifier — downtiers everything to save tokens.

Applied as a wrapper around any mode. CRITICAL→ADVANCED, ADVANCED→STANDARD, STANDARD→LOW.
Quality floor: verification stays at least STANDARD.
"""

import logging

from mullet.core.router.complexity import Tier
from mullet.core.router.decision import RoutingDecision

logger = logging.getLogger(__name__)

TIER_DOWN = {
    Tier.CRITICAL: Tier.ADVANCED,
    Tier.ADVANCED: Tier.STANDARD,
    Tier.STANDARD: Tier.LOW,
    Tier.LOW: Tier.LOW,
}


def apply_eco(decision: RoutingDecision, quality_floor: Tier = Tier.STANDARD) -> RoutingDecision:
    """Downgrade a routing decision by one tier.

    The quality_floor prevents critical operations (like verification)
    from being downgraded below STANDARD.
    """
    new_tier = TIER_DOWN.get(decision.tier, decision.tier)

    # Enforce quality floor
    tier_order = {Tier.LOW: 0, Tier.STANDARD: 1, Tier.ADVANCED: 2, Tier.CRITICAL: 3}
    if tier_order.get(new_tier, 0) < tier_order.get(quality_floor, 0):
        new_tier = quality_floor

    return RoutingDecision(
        provider=decision.provider,
        model=decision.model,
        tier=new_tier,
        confidence=decision.confidence,
        reasons=decision.reasons + [f"eco: {decision.tier.value} → {new_tier.value}"],
        estimated_cost=decision.estimated_cost * 0.3,  # rough estimate
        fallback_chain=decision.fallback_chain,
    )


# Per-mode token tracking
_mode_tokens: dict[str, int] = {}


def record_token_usage(mode: str, tokens: int) -> None:
    """Track token usage per orchestration mode."""
    _mode_tokens[mode] = _mode_tokens.get(mode, 0) + tokens


def get_mode_usage(mode: str) -> int:
    """Get total tokens used by a mode."""
    return _mode_tokens.get(mode, 0)


def check_budget_alert(mode: str, budget: int) -> bool:
    """Return True if mode has exceeded its token budget."""
    used = _mode_tokens.get(mode, 0)
    if used >= budget:
        logger.warning(
            "Budget alert: %s used %d tokens (budget: %d)", mode, used, budget
        )
        return True
    return False


def reset_tracking() -> None:
    """Reset all token tracking."""
    _mode_tokens.clear()
