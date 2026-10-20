"""Tests for mullet.core.knowledge.ingest_progress.IngestProgress tracker."""

import time
from unittest.mock import patch


from mullet.core.knowledge.ingest_progress import IngestProgress


def test_initial_state():
    p = IngestProgress()
    assert p._done == 0
    assert p._total == 0
    assert p._rate == 0.0
    assert p.percent == 0.0
    assert p.eta_seconds == 0.0


def test_update_sets_counts():
    p = IngestProgress()
    p.update(10, 100)
    assert p._done == 10
    assert p._total == 100


def test_percent_calculation():
    p = IngestProgress()
    p.update(25, 100)
    assert p.percent == 25.0


def test_percent_zero_total_returns_zero():
    """Avoid div-by-zero on empty total."""
    p = IngestProgress()
    p.update(0, 0)
    assert p.percent == 0.0


def test_percent_exact_half():
    p = IngestProgress()
    p.update(50, 100)
    assert p.percent == 50.0


def test_rate_computed_over_elapsed_time():
    """Rate = done / elapsed_seconds."""
    p = IngestProgress()
    # Simulate known elapsed time
    p._start_time = time.time() - 10.0  # 10 seconds ago
    p.update(50, 100)
    # rate ~= 50 / 10 = 5.0
    assert 4.5 <= p._rate <= 5.5


def test_eta_seconds_positive_when_rate_known():
    """ETA = remaining / rate."""
    p = IngestProgress()
    p._start_time = time.time() - 10.0
    p.update(50, 100)
    # remaining = 50, rate ~= 5 → ETA ~= 10s
    assert 8.0 <= p.eta_seconds <= 12.0


def test_eta_zero_when_complete():
    """Complete work (done == total) → ETA 0."""
    p = IngestProgress()
    p._start_time = time.time() - 10.0
    p.update(100, 100)
    assert p.eta_seconds == 0.0


def test_eta_zero_when_no_rate():
    """No rate (0.0) → ETA 0, no div-by-zero."""
    p = IngestProgress()
    p.update(0, 100)  # 0 done → rate 0
    assert p.eta_seconds == 0.0


def test_multiple_updates_progress_forward():
    p = IngestProgress()
    p.update(10, 100)
    p.update(25, 100)
    p.update(75, 100)
    assert p._done == 75
    assert p.percent == 75.0


def test_update_zero_elapsed_sets_rate_zero():
    """Line 25: total_elapsed == 0 → rate stays 0.0 (no div-by-zero)."""
    fixed_time = 1_700_000_000.0
    with patch("mullet.core.knowledge.ingest_progress.time.time", return_value=fixed_time):
        p = IngestProgress()
        p.update(10, 100)
    assert p._rate == 0.0
