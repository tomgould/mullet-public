"""Tests targeting reported coverage gaps in mullet/core/router/budget.py."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from mullet.core.router.budget import (
    BudgetTracker,
    BudgetLimits,
    UsageRecord,
    _start_of_day,
)


# ---------------------------------------------------------------------------
# can_afford
# ---------------------------------------------------------------------------

def test_can_afford_per_task_limit_exceeded():
    """Line 83: can_afford returns False when est_tokens > per_task_max_tokens."""
    limits = BudgetLimits(per_task_max_tokens=100)
    tracker = BudgetTracker(limits=limits)
    assert tracker.can_afford("local", "default", 101) is False


def test_can_afford_session_limit_exceeded():
    """Line 87: can_afford returns False when session would be exceeded."""
    limits = BudgetLimits(session_max_tokens=100)
    tracker = BudgetTracker(limits=limits)
    tracker.record_usage("local", "default", 80, 20)  # 100 tokens used
    assert tracker.can_afford("local", "default", 1) is False


def test_can_afford_daily_dollar_limit_exceeded():
    """Line 92: daily dollar check is reached (session tokens must not short-circuit)."""
    limits = BudgetLimits(session_max_tokens=1_000_000, daily_max_dollars=1.0)
    tracker = BudgetTracker(limits=limits)
    tracker._records.append(
        UsageRecord(
            provider="anthropic",
            model="opus",
            input_tokens=10,
            output_tokens=10,
            estimated_cost=1.5,
        )
    )
    assert tracker.can_afford("local", "default", 1) is False


def test_can_afford_returns_true_when_within_limits():
    """Line 94: can_afford returns True when all checks pass."""
    limits = BudgetLimits(
        per_task_max_tokens=1000,
        session_max_tokens=1000,
        daily_max_dollars=10.0,
    )
    tracker = BudgetTracker(limits=limits)
    assert tracker.can_afford("local", "default", 500) is True


# ---------------------------------------------------------------------------
# suggest_downgrade
# ---------------------------------------------------------------------------

def test_suggest_downgrade_near_daily_limit():
    """Lines 98-100: near daily limit -> 'local'."""
    limits = BudgetLimits(daily_max_dollars=10.0)
    tracker = BudgetTracker(limits=limits)
    tracker._records.append(
        UsageRecord(
            provider="anthropic",
            model="opus",
            input_tokens=100,
            output_tokens=100,
            estimated_cost=9.5,
        )
    )
    assert tracker.suggest_downgrade("opus") == "local"


def test_suggest_downgrade_opus_to_sonnet():
    """Lines 104-106: near session limit with opus -> sonnet."""
    limits = BudgetLimits(session_max_tokens=100)
    tracker = BudgetTracker(limits=limits)
    tracker._records.append(
        UsageRecord(
            provider="anthropic",
            model="opus",
            input_tokens=80,
            output_tokens=10,
            estimated_cost=0.01,
        )
    )
    assert tracker.suggest_downgrade("claude-opus") == "sonnet"


def test_suggest_downgrade_sonnet_to_haiku():
    """Lines 107-108: near session limit with sonnet -> haiku."""
    limits = BudgetLimits(session_max_tokens=100)
    tracker = BudgetTracker(limits=limits)
    tracker._records.append(
        UsageRecord(
            provider="anthropic",
            model="sonnet",
            input_tokens=80,
            output_tokens=10,
            estimated_cost=0.01,
        )
    )
    assert tracker.suggest_downgrade("claude-sonnet") == "haiku"


def test_suggest_downgrade_haiku_to_local():
    """Lines 109-110: near session limit with haiku -> local."""
    limits = BudgetLimits(session_max_tokens=100)
    tracker = BudgetTracker(limits=limits)
    tracker._records.append(
        UsageRecord(
            provider="anthropic",
            model="haiku",
            input_tokens=80,
            output_tokens=10,
            estimated_cost=0.01,
        )
    )
    assert tracker.suggest_downgrade("claude-haiku") == "local"


def test_suggest_downgrade_none_when_ok():
    """Line 112: no downgrade needed."""
    tracker = BudgetTracker()
    assert tracker.suggest_downgrade("opus") is None


# ---------------------------------------------------------------------------
# daily_tokens
# ---------------------------------------------------------------------------

def test_daily_tokens_property():
    """Lines 133-134: daily_tokens sums today's records."""
    tracker = BudgetTracker()
    tracker.record_usage("local", "default", 100, 50)
    assert tracker.daily_tokens == 150


def test_daily_tokens_zero_when_no_records():
    tracker = BudgetTracker()
    assert tracker.daily_tokens == 0


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def test_status_method():
    """Line 138: status returns expected dict shape."""
    tracker = BudgetTracker()
    tracker.record_usage("local", "local-model", 100, 50)
    status = tracker.status()
    assert status["session_tokens"] == 150
    assert status["session_cost"] == 0.0
    assert status["daily_tokens"] == 150
    assert status["daily_cost"] == 0.0
    assert "limits" in status
    assert "remaining" in status
    assert status["remaining"]["session_tokens"] == tracker.limits.session_max_tokens - 150
    assert status["remaining"]["daily_dollars"] == tracker.limits.daily_max_dollars


# ---------------------------------------------------------------------------
# _get_rates
# ---------------------------------------------------------------------------

def test_get_rates_import_error_fallback():
    """Lines 170-172: ImportError in _get_rates logs warning and falls through."""
    tracker = BudgetTracker()

    class BadModule:
        def __getattr__(self, name):
            raise ImportError("no model_info")

    original = sys.modules.get("mullet.core.providers.model_info")
    sys.modules["mullet.core.providers.model_info"] = BadModule()
    try:
        rates = tracker._get_rates("unknown-model-xyz")
    finally:
        if original is not None:
            sys.modules["mullet.core.providers.model_info"] = original
        else:
            sys.modules.pop("mullet.core.providers.model_info", None)

    assert rates == {"input": 0.001, "output": 0.005}


def test_get_rates_model_info_returns_zero():
    """Lines 169-170: _get_rates falls through when model_info returns zero costs."""
    tracker = BudgetTracker()
    with patch("mullet.core.providers.model_info.get_pricing", return_value=(0.0, 0.0)):
        rates = tracker._get_rates("some-unknown-model")
    assert rates == {"input": 0.001, "output": 0.005}


def test_get_rates_model_info_returns_positive():
    """Line 169: _get_rates returns model_info pricing when costs are positive."""
    tracker = BudgetTracker()
    with patch(
        "mullet.core.providers.model_info.get_pricing", return_value=(0.002, 0.010)
    ):
        rates = tracker._get_rates("some-unknown-model")
    assert rates == {"input": 0.002, "output": 0.010}


# ---------------------------------------------------------------------------
# _start_of_day
# ---------------------------------------------------------------------------

def test_start_of_day():
    """_start_of_day returns a timestamp at midnight UTC."""
    ts = _start_of_day()
    import datetime

    now = datetime.datetime.now(datetime.timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    assert abs(ts - start.timestamp()) < 1.0
