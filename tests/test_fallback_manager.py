"""Tests for FallbackManager — turn-scoped provider failover."""

from unittest.mock import MagicMock



class _P:
    """Minimal Provider stub for FallbackManager tests."""
    def __init__(self, name: str):
        self.name = name


def test_initial_current_is_primary():
    from mullet.core.router.fallback import FallbackManager
    primary = _P("primary")
    fbs = [_P("fb1"), _P("fb2")]
    mgr = FallbackManager(primary, fbs)
    assert mgr.current is primary
    assert mgr.is_fallback_active is False


def test_on_failure_walks_chain():
    from mullet.core.router.fallback import FallbackManager
    primary = _P("primary")
    fb1, fb2 = _P("fb1"), _P("fb2")
    mgr = FallbackManager(primary, [fb1, fb2])

    # First failure → falls back to fb1
    next1 = mgr.on_failure(Exception("boom"))
    assert next1 is fb1
    assert mgr.current is fb1
    assert mgr.is_fallback_active is True

    # Second failure → falls back to fb2
    next2 = mgr.on_failure(Exception("boom2"))
    assert next2 is fb2
    assert mgr.current is fb2

    # Third failure → chain exhausted
    next3 = mgr.on_failure(Exception("boom3"))
    assert next3 is None


def test_on_turn_start_restores_primary():
    from mullet.core.router.fallback import FallbackManager
    primary = _P("primary")
    fb = _P("fb")
    mgr = FallbackManager(primary, [fb])

    mgr.on_failure(Exception("x"))
    assert mgr.current is fb

    mgr.on_turn_start()
    assert mgr.current is primary
    assert mgr.is_fallback_active is False


def test_on_success_notifies_health_tracker():
    from mullet.core.router.fallback import FallbackManager
    primary = _P("primary")
    health = MagicMock()
    mgr = FallbackManager(primary, [], health=health)

    mgr.on_success()
    health.record_success.assert_called_once_with("primary")


def test_on_failure_notifies_health_and_skips_unhealthy_providers():
    from mullet.core.router.fallback import FallbackManager
    primary = _P("primary")
    fb1, fb2 = _P("fb1"), _P("fb2")

    health = MagicMock()
    # fb1 is unhealthy; fb2 is healthy
    health.is_available.side_effect = lambda name: name != "fb1"

    mgr = FallbackManager(primary, [fb1, fb2], health=health)
    next_ = mgr.on_failure(Exception("oops"))
    assert next_ is fb2  # fb1 skipped because is_available("fb1") → False
    health.record_failure.assert_called_once()


def test_chain_status_lists_all_providers():
    from mullet.core.router.fallback import FallbackManager
    primary = _P("primary")
    fb1, fb2 = _P("fb1"), _P("fb2")
    mgr = FallbackManager(primary, [fb1, fb2])

    status = mgr.chain_status
    assert len(status) == 3
    names = [s["name"] for s in status]
    assert names == ["primary", "fb1", "fb2"]
    roles = [s["role"] for s in status]
    assert roles == ["primary", "fallback", "fallback"]


def test_already_tried_provider_not_reselected():
    from mullet.core.router.fallback import FallbackManager
    primary = _P("primary")
    fb = _P("fb")
    mgr = FallbackManager(primary, [fb])

    # Simulate primary fails, fall back to fb
    mgr.on_failure(Exception("x1"))
    assert mgr.current is fb

    # fb fails too — no more providers, and primary is already-tried
    next_ = mgr.on_failure(Exception("x2"))
    assert next_ is None


# ---------------------------------------------------------------------------
# Allowlist enforcement — G13
# ---------------------------------------------------------------------------


def test_fallback_manager_respects_allowlist() -> None:
    from mullet.core.router.fallback import FallbackManager
    primary = _P("kimi")
    fallbacks = [_P("local"), _P("gemini"), _P("anthropic")]
    mgr = FallbackManager(
        primary=primary,
        fallbacks=fallbacks,
        allowed_providers=frozenset({"kimi"}),
    )
    result = mgr.on_failure(RuntimeError("boom"))
    assert result is None


def test_fallback_manager_allows_listed_fallback() -> None:
    from mullet.core.router.fallback import FallbackManager
    primary = _P("kimi")
    fallbacks = [_P("local"), _P("gemini"), _P("haiku")]
    mgr = FallbackManager(
        primary=primary,
        fallbacks=fallbacks,
        allowed_providers=frozenset({"kimi", "haiku"}),
    )
    result = mgr.on_failure(RuntimeError("boom"))
    assert result is not None
    assert result.name == "haiku"


def test_fallback_manager_no_allowlist_preserves_behaviour() -> None:
    from mullet.core.router.fallback import FallbackManager
    primary = _P("kimi")
    fallbacks = [_P("local"), _P("gemini")]
    mgr = FallbackManager(primary=primary, fallbacks=fallbacks)
    result = mgr.on_failure(RuntimeError("boom"))
    assert result is not None
    assert result.name == "local"
