"""Tests for auth.py rate-limit helpers (_record_failure + _rate_limit_check)."""

import time

import pytest


@pytest.fixture(autouse=True)
def _clear_rate_limit_state():
    """Wipe the module-level _failed_attempts between tests."""
    from mullet.server import auth
    auth._failed_attempts.clear()
    yield
    auth._failed_attempts.clear()


def test_empty_identifier_has_zero_wait():
    from mullet.server.auth import _rate_limit_check
    assert _rate_limit_check("192.168.1.1") == 0.0


def test_five_failures_gives_5s_wait():
    from mullet.server.auth import _record_failure, _rate_limit_check
    for _ in range(5):
        _record_failure("ip-a")
    assert _rate_limit_check("ip-a") == 5.0


def test_ten_failures_gives_30s_wait():
    from mullet.server.auth import _record_failure, _rate_limit_check
    for _ in range(10):
        _record_failure("ip-b")
    assert _rate_limit_check("ip-b") == 30.0


def test_twenty_failures_gives_60s_wait():
    from mullet.server.auth import _record_failure, _rate_limit_check
    for _ in range(20):
        _record_failure("ip-c")
    assert _rate_limit_check("ip-c") == 60.0


def test_old_entries_are_discarded():
    """Entries older than 60s should not count toward the rate limit."""
    from mullet.server import auth
    now = time.time()
    # Seed with 30 ancient timestamps + 2 recent ones
    auth._failed_attempts["ip-d"] = [now - 120 for _ in range(30)] + [now - 5, now - 2]
    # _rate_limit_check cleans and evaluates
    wait = auth._rate_limit_check("ip-d")
    # Only 2 recent remain — below all thresholds
    assert wait == 0.0
    assert len(auth._failed_attempts["ip-d"]) == 2


def test_record_failure_trims_old_entries():
    from mullet.server import auth
    now = time.time()
    auth._failed_attempts["ip-e"] = [now - 200, now - 150, now - 70]
    auth._record_failure("ip-e")
    attempts = auth._failed_attempts["ip-e"]
    # All entries older than 60s trimmed; only the fresh one remains
    assert len(attempts) == 1
    assert attempts[0] >= now - 1


def test_different_ips_are_independent():
    from mullet.server.auth import _record_failure, _rate_limit_check
    for _ in range(10):
        _record_failure("ip-f")
    # ip-g should have no accumulated failures
    assert _rate_limit_check("ip-g") == 0.0
    assert _rate_limit_check("ip-f") == 30.0
