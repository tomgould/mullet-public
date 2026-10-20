"""Tests for `mullet.core.providers.backoff.compute_backoff`.

Covers:
- the pure-math helper (basic exponentiation, max_attempt clamp, max_delay
  cap, OverflowError handling, jitter range, base case)
- parity locks: `health.ProviderHealth.record_failure` and
  `retry._compute_delay` must produce the SAME numeric value as the
  pre-refactor formulas they used to inline.
"""

import time

import pytest

from mullet.core.providers.backoff import compute_backoff
from mullet.core.providers.health import (
    BACKOFF_DURATIONS,
    BACKOFF_MULTIPLIER,
    ProviderHealth,
)
from mullet.core.providers.retry import RetryConfig, _compute_delay


# ── compute_backoff: pure math ──────────────────────────────────────────────


def test_compute_backoff_basic():
    """Sanity check: base * factor**attempt, no caps, no jitter."""
    assert compute_backoff(base=1.0, factor=2.0, attempt=3) == 8.0


def test_compute_backoff_max_attempt_clamp():
    """attempt clamped to max_attempt before exponentiation.

    attempt=10 with max_attempt=5 must equal attempt=5 (no further growth).
    Failure here means the cap on consecutive growth has been lost.
    """
    clamped = compute_backoff(base=1.0, factor=2.0, attempt=10, max_attempt=5)
    expected = compute_backoff(base=1.0, factor=2.0, attempt=5)
    assert clamped == expected
    assert clamped == 32.0


def test_compute_backoff_max_delay_cap():
    """Output capped at max_delay regardless of how big the raw product is."""
    delay = compute_backoff(base=1.0, factor=10.0, attempt=10, max_delay=10.0)
    assert delay == 10.0


def test_compute_backoff_overflow_returns_max_delay():
    """Astronomical exponents must not crash — they fall back to max_delay."""
    delay = compute_backoff(
        base=1.0, factor=10.0, attempt=10000, max_delay=60.0
    )
    assert delay == 60.0


def test_compute_backoff_jitter_in_half_to_full_range():
    """With jitter, every sample lies in [0.5*expected, expected]."""
    expected = 8.0  # base=1, factor=2, attempt=3
    samples = [
        compute_backoff(base=1.0, factor=2.0, attempt=3, jitter=True)
        for _ in range(200)
    ]
    for s in samples:
        assert 0.5 * expected <= s <= expected, f"out-of-range jitter sample: {s}"
    # Jitter should produce variability — fail if all identical (RNG broken
    # or jitter not applied).
    assert len({round(s, 4) for s in samples}) > 1


def test_compute_backoff_zero_attempt_returns_base():
    """attempt=0 must always return exactly `base` (no growth on first try)."""
    assert compute_backoff(base=2.5, factor=4.0, attempt=0) == 2.5


# ── parity: health.record_failure ──────────────────────────────────────────


def test_health_record_failure_matches_old_formula():
    """ProviderHealth backoff must equal the pre-refactor inline formula.

    Old formula:
        base = BACKOFF_DURATIONS[error_type]
        multiplier = BACKOFF_MULTIPLIER ** min(consecutive_failures - 1, 5)
        duration = base * multiplier

    Failure here means the refactor changed user-visible backoff durations.
    """
    h = ProviderHealth(provider_name="parity")
    err = Exception("HTTP 500 Internal Server Error")  # → server_error
    base = BACKOFF_DURATIONS["server_error"]

    for n in (1, 2, 3, 4, 5, 6, 7):
        before = time.time()
        h.record_failure(err)
        actual = h.backoff_until - before

        old = base * (BACKOFF_MULTIPLIER ** min(n - 1, 5))

        # Allow tiny drift from the two `time.time()` calls inside record_failure.
        assert abs(actual - old) < 0.05, (
            f"parity drift @ failure #{n}: actual={actual} old={old} "
            f"diff={actual - old}"
        )


# ── parity: retry._compute_delay ───────────────────────────────────────────


def test_retry_compute_delay_matches_old_formula():
    """_compute_delay (jitter off) must equal the pre-refactor inline formula.

    Old formula:
        delay = min(initial_delay * (factor ** attempt), max_delay)

    Failure here means the refactor changed retry timings under load.
    """
    config = RetryConfig(
        initial_delay=0.5,
        backoff_factor=2.0,
        max_delay=10.0,
        jitter=False,
    )
    for attempt in range(0, 6):
        actual = _compute_delay(attempt, config)
        raw = config.initial_delay * (config.backoff_factor ** attempt)
        expected = min(raw, config.max_delay)
        assert actual == pytest.approx(expected, abs=1e-9), (
            f"divergence @ attempt={attempt}: actual={actual} expected={expected}"
        )
