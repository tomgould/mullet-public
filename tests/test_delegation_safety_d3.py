"""Delegation safety D3 — structured ``ModelContextExceeded`` error class.

Distinguishes "model too small" (a bigger sibling in same provider/tier
could serve — caller can escalate) from "all candidates too small"
(caller must split the task or refuse). Subclasses ``ValueError`` for
back-compat with existing ``except ValueError`` handlers in the router
and orchestrator.

Invariants:
  1. ``issubclass(ModelContextExceeded, ValueError)`` — legacy catchers
     (resolver, swarm) must continue to see these as ValueErrors.
  2. Every raise carries ``provider``, ``tier``, ``context_tokens``,
     ``tried_candidates`` and the ``all_candidates_exhausted`` flag so
     callers can route on it programmatically.
  3. ``_best_kimi`` / ``_first_fitting`` / ``_assert_can_serve`` all
     raise ``ModelContextExceeded`` (not bare ValueError) with the flag
     set correctly — True when no candidate in the table can serve,
     False when a fit-able candidate was filtered out upstream (tier
     limits, allowlist, etc).
  4. ``handle_mullet_run`` surfaces the exhausted variant on the
     ``filtered_providers`` key in the response so the user can see
     which provider was refused and why.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. Subclass + attributes
# ---------------------------------------------------------------------------

def test_model_context_exceeded_is_value_error_subclass():
    """Red: ``ModelContextExceeded`` must subclass ValueError for back-compat.

    Failure mode: if MCE is raised from inside ``get_best_model`` and the
    outer ``except ValueError`` in ``resolver._create_provider`` or
    ``_build_fallback_chain`` doesn't catch it, the whole dispatch
    crashes instead of falling through to the next provider.
    """
    from mullet.core.providers.errors import ModelContextExceeded

    assert issubclass(ModelContextExceeded, ValueError), (
        "ModelContextExceeded must subclass ValueError so existing "
        "`except ValueError:` catchers continue to work."
    )


def test_mce_attributes_populated():
    """Red: MCE must expose provider/tier/context_tokens/tried/exhausted programmatically.

    Failure mode: callers that want to surface rejection reasons on the
    response (``filtered_providers``) need to *inspect* the error, not
    string-match on ``str(exc)``.
    """
    from mullet.core.providers.errors import ModelContextExceeded

    err = ModelContextExceeded(
        provider="kimi",
        tier="STANDARD",
        context_tokens=14_485,
        tried_candidates=["moonshot-v1-8k", "moonshot-v1-32k"],
        max_candidate_context=32_768,
        all_candidates_exhausted=False,
    )
    assert err.provider == "kimi"
    assert err.tier == "STANDARD"
    assert err.context_tokens == 14_485
    assert err.tried_candidates == ["moonshot-v1-8k", "moonshot-v1-32k"]
    assert err.max_candidate_context == 32_768
    assert err.all_candidates_exhausted is False
    msg = str(err)
    assert "kimi" in msg
    assert "14485" in msg
    assert "moonshot-v1-8k" in msg


def test_mce_all_candidates_exhausted_flag():
    """Red: the ``all_candidates_exhausted`` flag must round-trip.

    Failure mode: bridge/swarm dispatcher uses this flag to decide
    between "try next tier" (False) and "refuse + log ERROR" (True).
    """
    from mullet.core.providers.errors import ModelContextExceeded

    err = ModelContextExceeded(
        provider="anthropic",
        tier="CRITICAL",
        context_tokens=2_000_000,
        tried_candidates=["claude-opus-4-7", "claude-sonnet-4-6"],
        all_candidates_exhausted=True,
    )
    assert err.all_candidates_exhausted is True
    assert "exhausted=True" in str(err)


# ---------------------------------------------------------------------------
# 2. _best_kimi raises MCE not plain ValueError
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_registry():
    """Return a ModelRegistry with empty live cache so the static fallback path runs."""
    from mullet.core.providers.model_registry import ModelRegistry

    reg = ModelRegistry()
    for provider in ("anthropic", "kimi", "gemini"):
        reg._cache[provider] = []
        reg._cache_ts[provider] = time.time()
    return reg


def test_best_kimi_raises_mce_on_exhaustion(fresh_registry):
    """Red: 300K prompt against kimi (max 262K) must raise MCE with exhausted=True.

    Failure mode: today D2 raises a plain ValueError; callers can only
    string-match. MCE lets ``handle_mullet_run`` add the provider to
    ``filtered_providers`` with a structured reason.
    """
    from mullet.core.providers.errors import ModelContextExceeded

    with pytest.raises(ModelContextExceeded) as exc_info:
        fresh_registry.get_best_model(
            "kimi",
            complexity="STANDARD",
            context_needed=4096,
            context_tokens=300_000,  # exceeds every kimi candidate
        )
    err = exc_info.value
    assert err.provider == "kimi"
    assert err.all_candidates_exhausted is True, (
        "300K exceeds max kimi candidate (262K) → exhausted must be True"
    )
    assert err.context_tokens == 300_000
    assert "moonshot-v1-8k" in err.tried_candidates or any(
        "moonshot" in c or "kimi" in c for c in err.tried_candidates
    )


def test_best_kimi_raises_exhausted_when_post_pick_rejects_at_every_tier(
    monkeypatch, fresh_registry
):
    """When can_serve() rejects the picked model at every tier in the
    provider, ``get_best_model`` raises ``ModelContextExceeded`` with
    ``all_candidates_exhausted=True``.

    D5 (auto-escalation within provider) walks LOW → STANDARD → ADVANCED
    → CRITICAL on a non-exhausted MCE before re-raising. When every tier
    refuses with the same can_serve=False, the chain has been fully
    walked and the provider truly has nothing for this prompt — so
    exhausted=True is the honest answer. Callers see "this provider
    failed at every tier; try a different provider".

    Pre-D5 this test asserted exhausted=False because the picker only
    raised once at STANDARD and the bigger sibling at ADVANCED/CRITICAL
    was theoretical. D5 makes the walk explicit, so the per-tier
    "bigger sibling" check has been performed in-process.

    Failure mode: today the post-pick assertion raises a plain
    ValueError with no way to tell exhaustion from 'just this tier'.
    """
    from mullet.core.providers import constraints as _c
    from mullet.core.providers.errors import ModelContextExceeded

    # Force every can_serve() call to say False — simulates a stale
    # ConstraintRegistry cache or a wrapper that rejects tools-capable
    # models. Candidate table still contains 262K models, but every
    # tier-walk attempt is rejected.
    fake = MagicMock()
    fake.can_serve.return_value = False
    fake.max_context = 262_144
    monkeypatch.setattr(
        _c.ConstraintRegistry, "get", lambda self, model_id: fake
    )

    with pytest.raises(ModelContextExceeded) as exc_info:
        fresh_registry.get_best_model(
            "kimi",
            complexity="STANDARD",
            context_needed=4096,
            context_tokens=100_000,  # 100K ≪ 262K — bigger sibling theoretical
        )
    err = exc_info.value
    assert err.provider == "kimi"
    assert err.all_candidates_exhausted is True, (
        "D5 walks LOW → STANDARD → ADVANCED → CRITICAL on non-exhausted MCE; "
        "when every tier still refuses, the final raise must mark exhausted=True"
    )


def test_best_anthropic_raises_mce_exhausted_on_oversize(fresh_registry):
    """Red: 1.5M prompt against anthropic (max 1M) must raise MCE exhausted=True."""
    from mullet.core.providers.errors import ModelContextExceeded

    with pytest.raises(ModelContextExceeded) as exc_info:
        fresh_registry.get_best_model(
            "anthropic",
            complexity="STANDARD",
            context_needed=4096,
            context_tokens=1_500_000,  # > every anthropic candidate (1M)
        )
    err = exc_info.value
    assert err.provider == "anthropic"
    assert err.all_candidates_exhausted is True
    assert any("claude" in c for c in err.tried_candidates)


def test_legacy_except_value_error_still_catches_mce(fresh_registry):
    """Red: back-compat — existing ``except ValueError:`` must catch MCE.

    This locks the contract: MCE must subclass ValueError and callers
    that haven't yet been migrated (resolver._create_provider, swarm
    dispatcher's broad try/except) continue to fall through cleanly.
    """
    caught = None
    try:
        fresh_registry.get_best_model(
            "kimi",
            complexity="STANDARD",
            context_needed=4096,
            context_tokens=5_000_000,
        )
    except ValueError as exc:  # MCE subclasses ValueError
        caught = exc

    assert caught is not None, "plain `except ValueError` must catch MCE"
    # Yet it's still the structured subclass, not a bare ValueError.
    from mullet.core.providers.errors import ModelContextExceeded
    assert isinstance(caught, ModelContextExceeded)


# ---------------------------------------------------------------------------
# 3. Agent-mode integration — MCE attached to filtered_providers
# ---------------------------------------------------------------------------

def _make_provider(name: str, model: str) -> MagicMock:
    p = MagicMock()
    p.name = name
    p.model = model
    return p


@patch("mullet.server.mcp_handlers._write_agent_state_fn")
@patch("mullet.server.mcp_handlers._clear_agent_state")
def test_agent_mode_surfaces_mce_on_exhausted(
    mock_clear, mock_write, monkeypatch
):
    """Red: when _build_fallback_chain raises with ``filtered_providers``
    containing an MCE-derived reason, the agent-mode response must
    list the provider under ``filtered_providers`` with a max_context /
    context_tokens marker so the user can see why the refusal happened.
    """
    from mullet.server.mcp_handlers import handle_mullet_run
    from mullet.core.providers.errors import ModelContextExceeded

    kimi = _make_provider("kimi", "kimi-k2.5")
    monkeypatch.setattr(
        "mullet.server.mcp_handlers._resolve_provider",
        lambda **kw: kimi,
    )

    def _raising_fb(**kw):
        filtered = kw.get("filtered_out")
        # Simulate that _build_fallback_chain intercepted an MCE from
        # get_best_model and pushed a structured reason into filtered_out.
        mce = ModelContextExceeded(
            provider="kimi",
            tier="STANDARD",
            context_tokens=500_000,
            tried_candidates=["kimi-k2.5", "moonshot-v1-128k"],
            max_candidate_context=262_144,
            all_candidates_exhausted=True,
        )
        if filtered is not None:
            filtered["kimi"] = str(mce)
        err = ValueError("All providers in allowlist {'kimi'} filtered out")
        err.filtered_providers = dict(filtered or {})
        raise err

    monkeypatch.setattr(
        "mullet.server.mcp_handlers._build_fallback_chain",
        _raising_fb,
    )

    result = handle_mullet_run({
        "task": "do something huge",
        "providers": "kimi",
    })

    assert isinstance(result, dict)  # D7: dict-not-string contract
    assert "error" in result
    assert "filtered_providers" in result
    reason = result["filtered_providers"].get("kimi", "")
    assert "kimi" in reason
    assert "exhausted=True" in reason or "500000" in reason
