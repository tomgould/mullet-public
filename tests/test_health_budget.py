"""Tests for provider health tracking and budget enforcement."""

import time
from mullet.core.providers.health import (
    ProviderHealth, HealthTracker, classify_error,
    MAX_CONSECUTIVE_FAILURES,
)
from mullet.core.router.budget import BudgetTracker, BudgetLimits


# --- Health tracking ---

def test_provider_starts_healthy():
    h = ProviderHealth(provider_name="test")
    assert h.is_available is True
    assert h.consecutive_failures == 0


def test_failure_triggers_backoff():
    h = ProviderHealth(provider_name="test")
    h.record_failure(ConnectionError("refused"))
    assert h.consecutive_failures == 1
    assert h.backoff_until > time.time()
    assert h.is_available is False


def test_success_resets_failures():
    h = ProviderHealth(provider_name="test")
    h.record_failure(ConnectionError("down"))
    h.record_failure(ConnectionError("down"))
    assert h.consecutive_failures == 2
    h.record_success()
    assert h.consecutive_failures == 0
    assert h.is_available is True


def test_max_consecutive_failures():
    h = ProviderHealth(provider_name="test")
    for _ in range(MAX_CONSECUTIVE_FAILURES):
        h.record_failure(RuntimeError("error"))
    assert h.is_available is False
    # Even after backoff expires, max failures blocks it
    h.backoff_until = 0
    assert h.is_available is False


def test_classify_rate_limit():
    assert classify_error(Exception("429 Too Many Requests")) == "rate_limit"
    assert classify_error(Exception("rate limit exceeded")) == "rate_limit"


def test_classify_connection():
    assert classify_error(ConnectionError("refused")) == "connection"
    assert classify_error(TimeoutError("timed out")) == "connection"


def test_classify_auth():
    assert classify_error(Exception("401 Unauthorized")) == "auth"


def test_classify_server_error():
    assert classify_error(Exception("500 Internal Server Error")) == "server_error"


def test_health_tracker():
    tracker = HealthTracker()
    assert tracker.is_available("provider_a") is True

    tracker.record_failure("provider_a", ConnectionError("down"))
    assert tracker.is_available("provider_a") is False

    tracker.record_success("provider_a")
    assert tracker.is_available("provider_a") is True


def test_get_available_filters():
    tracker = HealthTracker()
    tracker.record_failure("bad", ConnectionError("down"))
    tracker.record_success("good")

    available = tracker.get_available(["good", "bad"])
    assert available == ["good"]


def test_health_to_dict():
    h = ProviderHealth(provider_name="test")
    h.record_success()
    d = h.to_dict()
    assert d["provider"] == "test"
    assert d["available"] is True
    assert d["success_rate"] == 1.0


# --- Budget tracking ---

def test_budget_starts_empty():
    budget = BudgetTracker()
    assert budget.session_tokens == 0
    assert budget.session_cost == 0.0


def test_record_usage():
    budget = BudgetTracker()
    budget.record_usage("local", "mistral", 100, 50)
    assert budget.session_tokens == 150


def test_can_afford_under_limit():
    budget = BudgetTracker(limits=BudgetLimits(per_task_max_tokens=10_000))
    assert budget.can_afford("local", "mistral", 5_000) is True


def test_cannot_afford_over_per_task():
    budget = BudgetTracker(limits=BudgetLimits(per_task_max_tokens=1_000))
    assert budget.can_afford("local", "mistral", 5_000) is False


def test_cannot_afford_over_session():
    budget = BudgetTracker(limits=BudgetLimits(session_max_tokens=1_000))
    budget.record_usage("cloud", "sonnet", 500, 400)
    assert budget.can_afford("cloud", "sonnet", 200) is False


def test_suggest_downgrade_near_limit():
    budget = BudgetTracker(limits=BudgetLimits(daily_max_dollars=1.0))
    # Simulate expensive usage
    for _ in range(50):
        budget.record_usage("cloud", "opus", 1000, 500)
    suggestion = budget.suggest_downgrade("opus")
    assert suggestion is not None  # Should suggest something cheaper


def test_estimate_cost():
    budget = BudgetTracker()
    cost = budget.estimate_cost("cloud", "opus", 1000, 500)
    assert cost > 0


def test_budget_status():
    budget = BudgetTracker()
    budget.record_usage("local", "mistral", 100, 50)
    status = budget.status()
    assert status["session_tokens"] == 150
    assert "remaining" in status
    assert "limits" in status


def test_local_is_free():
    budget = BudgetTracker()
    budget.record_usage("local", "local-model", 10000, 5000)
    assert budget.session_cost == 0.0  # Local is free
