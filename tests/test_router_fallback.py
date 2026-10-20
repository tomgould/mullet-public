"""Tests for mullet.core.router.fallback."""

import logging
from unittest.mock import MagicMock

import pytest

from mullet.core.router.fallback import FallbackManager


def _make_provider(name: str):
    """Create a minimal mock provider with a name attribute."""
    p = MagicMock()
    p.name = name
    return p


def _make_health_tracker():
    """Create a mock health tracker with record_failure, record_success, is_available."""
    h = MagicMock()
    h.is_available.return_value = True
    return h


# --- __init__ / properties ---------------------------------------------------

def test_init_defaults():
    primary = _make_provider("primary")
    fb = _make_provider("fb1")
    mgr = FallbackManager(primary=primary, fallbacks=[fb])
    assert mgr.current == primary
    assert mgr.is_fallback_active is False


def test_init_with_health_and_allowed():
    primary = _make_provider("primary")
    fb = _make_provider("fb1")
    health = _make_health_tracker()
    mgr = FallbackManager(primary=primary, fallbacks=[fb], health=health,
                          allowed_providers=frozenset({"primary", "fb1"}))
    assert mgr._health is health
    assert mgr._allowed == frozenset({"primary", "fb1"})


# --- on_turn_start -----------------------------------------------------------

def test_on_turn_start_noop_when_not_activated():
    primary = _make_provider("primary")
    fb = _make_provider("fb1")
    mgr = FallbackManager(primary=primary, fallbacks=[fb])
    mgr._current = fb
    mgr._activated = False
    mgr.on_turn_start()
    assert mgr.current == fb  # unchanged because _activated is False


def test_on_turn_start_restores_primary():
    primary = _make_provider("primary")
    fb = _make_provider("fb1")
    mgr = FallbackManager(primary=primary, fallbacks=[fb])
    mgr._current = fb
    mgr._activated = True
    mgr._tried.add("fb1")
    mgr.on_turn_start()
    assert mgr.current == primary
    assert mgr.is_fallback_active is False
    assert len(mgr._tried) == 0


# --- on_failure --------------------------------------------------------------

def test_on_failure_records_failure_and_falls_back():
    primary = _make_provider("primary")
    fb1 = _make_provider("fb1")
    health = _make_health_tracker()
    mgr = FallbackManager(primary=primary, fallbacks=[fb1], health=health)

    err = RuntimeError("boom")
    next_provider = mgr.on_failure(err)

    assert next_provider == fb1
    assert mgr.current == fb1
    assert mgr.is_fallback_active is True
    health.record_failure.assert_called_once_with("primary", err)


def test_on_failure_exhausted_returns_none():
    primary = _make_provider("primary")
    fb1 = _make_provider("fb1")
    mgr = FallbackManager(primary=primary, fallbacks=[fb1])

    mgr.on_failure(RuntimeError("boom"))
    # second failure: fb1 already tried
    result = mgr.on_failure(RuntimeError("boom again"))
    assert result is None
    assert mgr.is_fallback_active is True


def test_on_failure_skips_not_allowed():
    primary = _make_provider("primary")
    fb1 = _make_provider("fb1")
    fb2 = _make_provider("fb2")
    mgr = FallbackManager(primary=primary, fallbacks=[fb1, fb2],
                          allowed_providers=frozenset({"primary", "fb2"}))

    result = mgr.on_failure(RuntimeError("boom"))
    assert result == fb2
    assert mgr.current == fb2


def test_on_failure_skips_unhealthy():
    primary = _make_provider("primary")
    fb1 = _make_provider("fb1")
    fb2 = _make_provider("fb2")
    health = _make_health_tracker()
    health.is_available.side_effect = lambda name: name != "fb1"

    mgr = FallbackManager(primary=primary, fallbacks=[fb1, fb2], health=health)
    result = mgr.on_failure(RuntimeError("boom"))
    assert result == fb2


def test_on_failure_skips_already_tried():
    primary = _make_provider("primary")
    fb1 = _make_provider("fb1")
    fb2 = _make_provider("fb2")
    mgr = FallbackManager(primary=primary, fallbacks=[fb1, fb2])

    mgr.on_failure(RuntimeError("boom"))  # tries fb1
    mgr.on_failure(RuntimeError("boom2"))  # tries fb2
    result = mgr.on_failure(RuntimeError("boom3"))
    assert result is None


def test_on_failure_no_health():
    primary = _make_provider("primary")
    fb1 = _make_provider("fb1")
    mgr = FallbackManager(primary=primary, fallbacks=[fb1])
    result = mgr.on_failure(RuntimeError("boom"))
    assert result == fb1


def test_on_failure_no_fallbacks():
    primary = _make_provider("primary")
    mgr = FallbackManager(primary=primary, fallbacks=[])
    result = mgr.on_failure(RuntimeError("boom"))
    assert result is None


def test_on_failure_allowed_empty_frozenset():
    primary = _make_provider("primary")
    fb1 = _make_provider("fb1")
    mgr = FallbackManager(primary=primary, fallbacks=[fb1],
                          allowed_providers=frozenset())
    result = mgr.on_failure(RuntimeError("boom"))
    assert result is None


# --- on_success --------------------------------------------------------------

def test_on_success_with_health():
    primary = _make_provider("primary")
    health = _make_health_tracker()
    mgr = FallbackManager(primary=primary, fallbacks=[], health=health)
    mgr.on_success()
    health.record_success.assert_called_once_with("primary")


def test_on_success_without_health():
    primary = _make_provider("primary")
    mgr = FallbackManager(primary=primary, fallbacks=[])
    # Should not raise
    mgr.on_success()


# --- chain_status ------------------------------------------------------------

def test_chain_status_no_health():
    primary = _make_provider("primary")
    fb1 = _make_provider("fb1")
    mgr = FallbackManager(primary=primary, fallbacks=[fb1])
    status = mgr.chain_status
    assert len(status) == 2
    assert status[0] == {"name": "primary", "role": "primary", "active": True}
    assert status[1] == {"name": "fb1", "role": "fallback", "active": False, "healthy": True}


def test_chain_status_with_health():
    primary = _make_provider("primary")
    fb1 = _make_provider("fb1")
    health = _make_health_tracker()
    health.is_available.return_value = False
    mgr = FallbackManager(primary=primary, fallbacks=[fb1], health=health)
    status = mgr.chain_status
    assert status[1]["healthy"] is False


def test_chain_status_active_fallback():
    primary = _make_provider("primary")
    fb1 = _make_provider("fb1")
    mgr = FallbackManager(primary=primary, fallbacks=[fb1])
    mgr._current = fb1
    mgr._activated = True
    status = mgr.chain_status
    assert status[0]["active"] is False
    assert status[1]["active"] is True


def test_chain_status_multiple_fallbacks():
    primary = _make_provider("primary")
    fb1 = _make_provider("fb1")
    fb2 = _make_provider("fb2")
    mgr = FallbackManager(primary=primary, fallbacks=[fb1, fb2])
    status = mgr.chain_status
    assert len(status) == 3
    assert status[0]["role"] == "primary"
    assert status[1]["role"] == "fallback"
    assert status[2]["role"] == "fallback"


# --- integration-style scenarios ---------------------------------------------

def test_full_fallback_chain_then_restore():
    primary = _make_provider("primary")
    fb1 = _make_provider("fb1")
    fb2 = _make_provider("fb2")
    mgr = FallbackManager(primary=primary, fallbacks=[fb1, fb2])

    # first failure → fb1
    assert mgr.on_failure(RuntimeError("e1")) == fb1
    # second failure → fb2
    assert mgr.on_failure(RuntimeError("e2")) == fb2
    # third failure → exhausted
    assert mgr.on_failure(RuntimeError("e3")) is None

    # new turn → restore primary
    mgr.on_turn_start()
    assert mgr.current == primary
    assert mgr.is_fallback_active is False

    # can fail over again
    assert mgr.on_failure(RuntimeError("e4")) == fb1


def test_logs_on_restore(caplog):
    primary = _make_provider("primary")
    fb1 = _make_provider("fb1")
    mgr = FallbackManager(primary=primary, fallbacks=[fb1])
    mgr._current = fb1
    mgr._activated = True

    with caplog.at_level(logging.DEBUG, logger="mullet"):
        mgr.on_turn_start()
    assert "Restored primary" in caplog.text


def test_logs_on_failover(caplog):
    primary = _make_provider("primary")
    fb1 = _make_provider("fb1")
    mgr = FallbackManager(primary=primary, fallbacks=[fb1])

    with caplog.at_level(logging.WARNING):
        mgr.on_failure(RuntimeError("exploded"))
    assert "Failing over" in caplog.text


def test_logs_on_exhausted(caplog):
    primary = _make_provider("primary")
    fb1 = _make_provider("fb1")
    mgr = FallbackManager(primary=primary, fallbacks=[fb1])
    mgr.on_failure(RuntimeError("e1"))

    with caplog.at_level(logging.ERROR):
        mgr.on_failure(RuntimeError("e2"))
    assert "Fallback chain exhausted" in caplog.text
