"""Tests for provider health tracking and error classification."""

import time
import pytest

from mullet.core.providers.health import (
    ProviderHealth,
    HealthTracker,
    classify_error,
    MAX_CONSECUTIVE_FAILURES,
)


# --- Error classification ---

def test_classify_rate_limit_429():
    assert classify_error(Exception("HTTP 429 Too Many Requests")) == "rate_limit"


def test_classify_rate_limit_text():
    assert classify_error(Exception("rate limit exceeded")) == "rate_limit"


def test_classify_quota():
    assert classify_error(Exception("quota exceeded for model")) == "rate_limit"


def test_classify_overloaded():
    assert classify_error(Exception("HTTP 503 Service Unavailable")) == "overloaded"


def test_classify_overloaded_text():
    assert classify_error(Exception("server overloaded, try later")) == "overloaded"


def test_classify_server_error():
    assert classify_error(Exception("500 Internal Server Error")) == "server_error"


def test_classify_auth():
    assert classify_error(Exception("401 Unauthorized")) == "auth"


def test_classify_auth_forbidden():
    assert classify_error(Exception("403 Forbidden")) == "auth"


def test_classify_connection():
    assert classify_error(ConnectionError("connection refused")) == "connection"


def test_classify_timeout():
    assert classify_error(TimeoutError("timed out")) == "connection"


def test_classify_unknown():
    assert classify_error(Exception("something went wrong")) == "unknown"


# --- ProviderHealth ---

def test_health_starts_available():
    h = ProviderHealth(provider_name="test")
    assert h.is_available is True
    assert h.consecutive_failures == 0
    assert h.success_rate == 1.0


def test_health_failure_triggers_backoff():
    h = ProviderHealth(provider_name="test")
    h.record_failure(Exception("429"))
    assert h.consecutive_failures == 1
    assert h.backoff_until > time.time()
    assert h.last_failure_type == "rate_limit"


def test_health_success_resets():
    h = ProviderHealth(provider_name="test")
    h.record_failure(Exception("429"))
    assert h.consecutive_failures == 1
    h.record_success()
    assert h.consecutive_failures == 0
    assert h.backoff_until == 0


def test_health_max_failures_kills_provider():
    h = ProviderHealth(provider_name="test")
    for i in range(MAX_CONSECUTIVE_FAILURES):
        h.record_failure(Exception("500"))
    assert h.is_available is False


def test_health_backoff_exponential():
    h = ProviderHealth(provider_name="test")
    # First failure
    h.record_failure(Exception("500"))
    first_backoff = h.backoff_until - time.time()

    # Reset and fail again (simulating a second consecutive failure)
    h.record_failure(Exception("500"))
    second_backoff = h.backoff_until - time.time()

    # Second backoff should be longer due to exponential growth
    assert second_backoff > first_backoff


def test_health_backoff_remaining():
    h = ProviderHealth(provider_name="test")
    assert h.backoff_remaining == 0

    h.record_failure(Exception("429"))
    assert h.backoff_remaining > 0


def test_health_success_rate():
    h = ProviderHealth(provider_name="test")
    h.record_success()
    h.record_success()
    h.record_failure(Exception("500"))
    assert h.success_rate == pytest.approx(2/3, abs=0.01)


def test_health_to_dict():
    h = ProviderHealth(provider_name="test")
    h.record_success()
    d = h.to_dict()
    assert d["provider"] == "test"
    assert d["available"] is True
    assert d["consecutive_failures"] == 0
    assert d["success_rate"] == 1.0


def test_health_available_after_backoff():
    h = ProviderHealth(provider_name="test")
    h.record_failure(Exception("429"))
    # Manually expire the backoff
    h.backoff_until = time.time() - 1
    assert h.is_available is True


# --- HealthTracker ---

def test_tracker_auto_creates():
    tracker = HealthTracker()
    assert tracker.is_available("new_provider") is True


def test_tracker_record_failure():
    tracker = HealthTracker()
    tracker.record_failure("provider1", Exception("429"))
    assert tracker.get("provider1").consecutive_failures == 1


def test_tracker_record_success():
    tracker = HealthTracker()
    tracker.record_failure("provider1", Exception("500"))
    tracker.record_success("provider1")
    assert tracker.get("provider1").consecutive_failures == 0


def test_tracker_get_available():
    tracker = HealthTracker()
    tracker.record_success("good1")
    tracker.record_success("good2")
    # Kill a provider with max failures
    for _ in range(MAX_CONSECUTIVE_FAILURES):
        tracker.record_failure("bad", Exception("500"))

    available = tracker.get_available(["good1", "bad", "good2"])
    assert "good1" in available
    assert "good2" in available
    assert "bad" not in available


def test_tracker_status():
    tracker = HealthTracker()
    tracker.record_success("p1")
    tracker.record_failure("p2", Exception("429"))
    status = tracker.status()
    assert len(status) == 2
    assert any(s["provider"] == "p1" for s in status)
    assert any(s["provider"] == "p2" for s in status)


# --- Latency tracking tests ---

def test_record_success_with_latency():
    """record_success stores latency samples."""
    h = ProviderHealth(provider_name="test")
    h.record_success(latency_ms=120.5)
    h.record_success(latency_ms=80.0)
    assert h.avg_latency_ms == pytest.approx(100.25)


def test_avg_latency_none_when_no_samples():
    h = ProviderHealth(provider_name="test")
    assert h.avg_latency_ms is None


def test_p90_latency_none_when_too_few_samples():
    h = ProviderHealth(provider_name="test")
    h.record_success(latency_ms=100.0)
    h.record_success(latency_ms=200.0)
    assert h.p90_latency_ms is None  # need at least 3


def test_p90_latency_computed():
    h = ProviderHealth(provider_name="test")
    for ms in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        h.record_success(latency_ms=float(ms))
    p90 = h.p90_latency_ms
    assert p90 is not None
    assert p90 >= 900


def test_latency_window_capped():
    """Only the last _LATENCY_WINDOW samples are kept."""
    from mullet.core.providers.health import _LATENCY_WINDOW
    h = ProviderHealth(provider_name="test")
    for i in range(_LATENCY_WINDOW + 5):
        h.record_success(latency_ms=float(i))
    assert len(h._latency_samples) == _LATENCY_WINDOW


def test_to_dict_includes_latency():
    h = ProviderHealth(provider_name="test")
    for ms in [100, 200, 300, 400, 500]:
        h.record_success(latency_ms=float(ms))
    d = h.to_dict()
    assert "avg_latency_ms" in d
    assert "p90_latency_ms" in d
    assert d["avg_latency_ms"] == pytest.approx(300.0)


def test_record_success_without_latency_ok():
    """record_success without latency_ms still resets failures."""
    h = ProviderHealth(provider_name="test")
    h.record_failure(Exception("err"))
    h.record_success()
    assert h.consecutive_failures == 0
    assert h.avg_latency_ms is None
