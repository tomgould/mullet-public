"""Tests for the budget tracker."""

import pytest
from mullet.core.router.budget import BudgetTracker, BudgetLimits


@pytest.fixture
def tracker():
    limits = BudgetLimits(
        per_task_max_tokens=10_000,
        session_max_tokens=50_000,
        daily_max_dollars=5.0,
    )
    return BudgetTracker(limits=limits)


def test_initial_state(tracker):
    assert tracker.session_tokens == 0
    assert tracker.session_cost == 0.0
    assert tracker.daily_cost == 0.0


def test_estimate_cost(tracker):
    cost = tracker.estimate_cost("anthropic", "sonnet", 1000)
    assert cost > 0


def test_estimate_cost_local_cheap(tracker):
    cost = tracker.estimate_cost("local", "local", 1000)
    # Local models should be free or near-free
    assert cost == 0.0


def test_record_usage(tracker):
    tracker.record_usage("anthropic", "sonnet", 500, 200)
    assert tracker.session_tokens == 700
    assert tracker.session_cost > 0


def test_can_afford_within_limits(tracker):
    assert tracker.can_afford("local", "default", 5000) is True


def test_cannot_afford_exceeds_per_task(tracker):
    assert tracker.can_afford("anthropic", "sonnet", 20_000) is False


def test_cannot_afford_exceeds_session(tracker):
    # Fill up session near limit
    for _ in range(5):
        tracker.record_usage("anthropic", "sonnet", 9000, 1000)
    assert tracker.can_afford("anthropic", "sonnet", 5000) is False


def test_cannot_afford_exceeds_daily_budget(tracker):
    from mullet.core.router.budget import UsageRecord
    for _ in range(20):
        tracker._records.append(UsageRecord(
            provider="anthropic", model="opus",
            input_tokens=10000, output_tokens=5000,
            estimated_cost=0.3,
        ))
    assert tracker.daily_cost >= 5.0
    assert tracker.can_afford("anthropic", "opus", 1000) is False


def test_suggest_downgrade_near_daily_limit(tracker):
    from mullet.core.router.budget import UsageRecord
    for _ in range(15):
        tracker._records.append(UsageRecord(
            provider="anthropic", model="opus",
            input_tokens=10000, output_tokens=5000,
            estimated_cost=0.35,
        ))
    suggestion = tracker.suggest_downgrade("opus")
    assert suggestion == "local"  # Near daily limit → cheapest


def test_suggest_downgrade_near_session_limit(tracker):
    from mullet.core.router.budget import UsageRecord
    # Fill up session tokens near limit (80%)
    for _ in range(4):
        tracker._records.append(UsageRecord(
            provider="anthropic", model="opus",
            input_tokens=9000, output_tokens=1000,
            estimated_cost=0.01,
        ))
    suggestion = tracker.suggest_downgrade("opus")
    assert suggestion == "sonnet"


def test_suggest_downgrade_sonnet_to_haiku(tracker):
    from mullet.core.router.budget import UsageRecord
    for _ in range(4):
        tracker._records.append(UsageRecord(
            provider="anthropic", model="sonnet",
            input_tokens=9000, output_tokens=1000,
            estimated_cost=0.01,
        ))
    suggestion = tracker.suggest_downgrade("sonnet")
    assert suggestion == "haiku"


def test_suggest_no_downgrade_when_ok(tracker):
    assert tracker.suggest_downgrade("sonnet") is None


def test_status(tracker):
    tracker.record_usage("local", "default", 100, 50)
    status = tracker.status()
    assert status["session_tokens"] == 150
    assert "limits" in status
    assert "remaining" in status
    assert status["remaining"]["session_tokens"] == 50_000 - 150


def test_model_info_integration(tracker):
    """Budget tracker should use model_info database for accurate pricing."""
    cost_sonnet = tracker.estimate_cost("anthropic", "claude-sonnet-4-6", 1000)
    cost_opus = tracker.estimate_cost("anthropic", "claude-opus-4-6", 1000)
    # Opus should be more expensive than Sonnet
    assert cost_opus > cost_sonnet


def test_daily_tokens_counts_today(tracker):
    """daily_tokens property sums input+output for records from today only."""
    tracker.record_usage("local", "default", 100, 50)
    tracker.record_usage("kimi", "kimi-k2.6", 200, 100)
    assert tracker.daily_tokens == 450  # 150 + 300


def test_daily_tokens_zero_initially(tracker):
    assert tracker.daily_tokens == 0
