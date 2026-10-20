"""Race-condition tests for CircuitBreaker.allow_request half-open state."""

import threading
import time

from mullet.knowledge.engine.health import CircuitBreaker


def _trip_to_open(cb: CircuitBreaker):
    """Force the breaker into OPEN state by recording failures."""
    for _ in range(cb.failure_threshold):
        cb.record_failure()


def test_two_threads_in_half_open_only_one_probe_passes():
    """When cooldown expires and N threads all call allow_request, only one
    should return True (the probe slot). The rest should be rejected."""
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.05)
    _trip_to_open(cb)
    # Wait for cooldown to expire
    time.sleep(0.1)

    results: list[bool] = []
    results_lock = threading.Lock()

    def caller():
        ok = cb.allow_request()
        with results_lock:
            results.append(ok)

    threads = [threading.Thread(target=caller) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    passes = sum(1 for r in results if r)
    assert passes == 1, f"expected exactly 1 probe pass, got {passes} of 10 threads"


def test_closed_breaker_allows_all_threads():
    """Closed breaker should let every thread through."""
    cb = CircuitBreaker(failure_threshold=5)
    results: list[bool] = []
    results_lock = threading.Lock()

    def caller():
        ok = cb.allow_request()
        with results_lock:
            results.append(ok)

    threads = [threading.Thread(target=caller) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(results), "closed breaker should allow all callers"
    assert len(results) == 20


def test_open_before_cooldown_rejects_all():
    """While OPEN and within cooldown, no requests pass."""
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=10.0)
    _trip_to_open(cb)
    # Don't wait — cooldown has not expired

    results = [cb.allow_request() for _ in range(5)]
    assert not any(results), f"open-within-cooldown should reject; got {results}"


def test_probe_success_resets_breaker():
    """A successful probe should close the breaker and allow subsequent traffic."""
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.05)
    _trip_to_open(cb)
    time.sleep(0.1)

    # First caller gets the probe slot
    assert cb.allow_request() is True
    cb.record_success()

    # Now the breaker should be closed — subsequent callers pass
    assert cb.allow_request() is True
    assert cb.is_closed() is True
