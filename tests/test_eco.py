"""Tests for eco mode modifier."""

from mullet.core.orchestration.eco import apply_eco, Tier, TIER_DOWN
from mullet.core.router.decision import RoutingDecision


def test_downtier_high_to_medium():
    decision = RoutingDecision(provider="cloud", model="opus", tier=Tier.ADVANCED)
    result = apply_eco(decision, quality_floor=Tier.LOW)
    assert result.tier == Tier.STANDARD


def test_downtier_medium_to_low():
    decision = RoutingDecision(provider="cloud", model="sonnet", tier=Tier.STANDARD)
    result = apply_eco(decision, quality_floor=Tier.LOW)
    assert result.tier == Tier.LOW


def test_low_stays_low():
    decision = RoutingDecision(provider="local", model="mistral", tier=Tier.LOW)
    result = apply_eco(decision, quality_floor=Tier.LOW)
    assert result.tier == Tier.LOW


def test_quality_floor_prevents_below_medium():
    decision = RoutingDecision(provider="cloud", model="sonnet", tier=Tier.STANDARD)
    result = apply_eco(decision, quality_floor=Tier.STANDARD)
    assert result.tier == Tier.STANDARD  # Floor prevents going to LOW


def test_eco_adds_reason():
    decision = RoutingDecision(provider="p", model="m", tier=Tier.ADVANCED, reasons=["original"])
    result = apply_eco(decision, quality_floor=Tier.LOW)
    assert any("eco" in r for r in result.reasons)


def test_eco_reduces_cost_estimate():
    decision = RoutingDecision(provider="p", model="m", tier=Tier.ADVANCED, estimated_cost=1.0)
    result = apply_eco(decision, quality_floor=Tier.LOW)
    assert result.estimated_cost < 1.0


def test_tier_down_mapping():
    assert TIER_DOWN[Tier.ADVANCED] == Tier.STANDARD
    assert TIER_DOWN[Tier.STANDARD] == Tier.LOW
    assert TIER_DOWN[Tier.LOW] == Tier.LOW


from mullet.core.orchestration.eco import (
    record_token_usage, get_mode_usage, check_budget_alert, reset_tracking
)


def test_record_and_get_token_usage():
    reset_tracking()
    record_token_usage("ralph", 1000)
    record_token_usage("ralph", 500)
    assert get_mode_usage("ralph") == 1500


def test_budget_alert_triggered():
    reset_tracking()
    record_token_usage("autopilot", 5000)
    assert check_budget_alert("autopilot", budget=4000) is True


def test_budget_alert_not_triggered():
    reset_tracking()
    record_token_usage("autopilot", 1000)
    assert check_budget_alert("autopilot", budget=5000) is False


def test_reset_tracking_clears_state():
    reset_tracking()
    record_token_usage("eco", 9999)
    reset_tracking()
    assert get_mode_usage("eco") == 0


def test_critical_downtier_to_advanced():
    decision = RoutingDecision(provider="cloud", model="opus", tier=Tier.CRITICAL)
    result = apply_eco(decision, quality_floor=Tier.LOW)
    assert result.tier == Tier.ADVANCED
