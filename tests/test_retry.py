"""Tests for the universal retry module."""

import pytest

from mullet.core.providers.retry import (
    RetryConfig, RetryResult,
    is_retryable, with_retry, _compute_delay,
)


# ── is_retryable ────────────────────────────────────────────────────────────

def test_retryable_rate_limit():
    assert is_retryable(Exception("429 rate limit exceeded"))


def test_retryable_server_error():
    assert is_retryable(Exception("500 Internal Server Error"))


def test_retryable_timeout():
    assert is_retryable(Exception("Connection timeout"))


def test_retryable_overloaded():
    assert is_retryable(Exception("Server overloaded, try again"))


def test_not_retryable_auth():
    assert not is_retryable(Exception("401 Unauthorized"))


def test_not_retryable_bad_request():
    assert not is_retryable(Exception("400 Bad Request: invalid JSON"))


def test_retryable_by_type():
    class TimeoutError(Exception):
        pass
    assert is_retryable(TimeoutError("timed out"))


# ── _compute_delay ───────────────────────────────────────────────────────────

def test_delay_increases():
    config = RetryConfig(initial_delay=1.0, backoff_factor=2.0, jitter=False)
    d0 = _compute_delay(0, config)
    d1 = _compute_delay(1, config)
    d2 = _compute_delay(2, config)
    assert d0 == 1.0
    assert d1 == 2.0
    assert d2 == 4.0


def test_delay_capped():
    config = RetryConfig(initial_delay=1.0, backoff_factor=10.0, max_delay=5.0, jitter=False)
    d = _compute_delay(5, config)
    assert d == 5.0


def test_delay_with_jitter():
    config = RetryConfig(initial_delay=1.0, backoff_factor=2.0, jitter=True)
    delays = [_compute_delay(0, config) for _ in range(10)]
    # With jitter, delays should vary
    assert len(set(round(d, 3) for d in delays)) > 1


# ── with_retry ───────────────────────────────────────────────────────────────

def test_success_first_try():
    result = with_retry(lambda: "ok")
    assert result.success
    assert result.value == "ok"
    assert result.attempts == 1
    assert result.total_delay == 0.0


def test_success_after_retries():
    call_count = [0]
    def flaky():
        call_count[0] += 1
        if call_count[0] < 3:
            raise Exception("429 rate limit")
        return "ok"

    config = RetryConfig(max_retries=3, initial_delay=0.01, jitter=False)
    result = with_retry(flaky, config)
    assert result.success
    assert result.value == "ok"
    assert result.attempts == 3


def test_failure_after_max_retries():
    config = RetryConfig(max_retries=2, initial_delay=0.01, jitter=False)
    result = with_retry(
        lambda: (_ for _ in ()).throw(Exception("500 server error")),
        config,
    )
    assert not result.success
    assert result.attempts == 3
    assert "500" in result.last_error


def test_non_retryable_fails_fast():
    call_count = [0]
    def auth_error():
        call_count[0] += 1
        raise Exception("401 Unauthorized")

    config = RetryConfig(max_retries=3, initial_delay=0.01)
    result = with_retry(auth_error, config)
    assert not result.success
    assert call_count[0] == 1  # No retries for non-retryable


def test_on_retry_callback():
    callbacks = []
    def on_retry(attempt, error, delay):
        callbacks.append((attempt, str(error)))

    call_count = [0]
    def flaky():
        call_count[0] += 1
        if call_count[0] < 2:
            raise Exception("429 rate limit")
        return "ok"

    config = RetryConfig(max_retries=3, initial_delay=0.01, jitter=False)
    with_retry(flaky, config, on_retry=on_retry)
    assert len(callbacks) == 1
    assert callbacks[0][0] == 1


def test_default_config():
    config = RetryConfig()
    assert config.max_retries == 3
    assert config.initial_delay == 1.0
    assert config.backoff_factor == 2.0


def test_retry_result_defaults():
    r = RetryResult()
    assert r.success
    assert r.attempts == 1


def test_retry_config_negative_max_retries():
    with pytest.raises(ValueError):
        RetryConfig(max_retries=-1)


def test_retry_config_negative_initial_delay():
    with pytest.raises(ValueError):
        RetryConfig(initial_delay=-0.1)


def test_retry_config_backoff_factor_too_low():
    with pytest.raises(ValueError):
        RetryConfig(backoff_factor=0.5)
