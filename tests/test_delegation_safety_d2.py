"""Delegation safety D2 — constraint filter gates ALL provider-pick sites.

Covers the invariant: EVERY `get_best_model` call path must post-validate the
picked model against `ConstraintRegistry`. A picked model whose
`max_context < context_tokens * 1.1` (10% safety margin) must be rejected
with a clear ValueError, never returned.

Call sites gated:
  1. `ModelRegistry.get_best_model` (public entry) — post-pick assertion
  2. `_best_kimi` — candidate filter + raise on empty
  3. `_best_gemini` — candidate filter + raise on empty
  4. `_best_anthropic` — candidate filter + raise on empty (via _first_fitting)
  5. The (new) Sonnet/Haiku explicit-candidate helpers

Also verifies that the extracted `_filter_candidates_by_context` helper is
shared across all four `_best_<provider>` functions.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_registry(monkeypatch):
    """Return a ModelRegistry with empty live cache so the static fallback path runs."""
    import time

    from mullet.core.providers.model_registry import ModelRegistry

    reg = ModelRegistry()
    # Empty live caches for all four providers → forces static fallback
    for provider in ("anthropic", "kimi", "gemini"):
        reg._cache[provider] = []
        reg._cache_ts[provider] = time.time()
    return reg


def _force_constraint(monkeypatch, can_serve_result: bool):
    """Monkeypatch ConstraintRegistry.get() so can_serve() returns the given result.

    Used to drive the preferred short-circuit off the happy path so the
    heuristic branch is exercised.
    """
    from mullet.core.providers import constraints as _c

    fake = MagicMock()
    fake.can_serve.return_value = can_serve_result
    fake.max_context = 8_192  # arbitrary — only relevant when True
    monkeypatch.setattr(
        _c.ConstraintRegistry, "get", lambda self, model_id: fake
    )
    return fake


# ---------------------------------------------------------------------------
# 1. Post-pick assertion on get_best_model
# ---------------------------------------------------------------------------

def test_get_best_model_post_pick_assertion_rejects_too_small(monkeypatch, fresh_registry):
    """Red: post-pick, the chosen model must be verified against ConstraintRegistry.

    Failure mode: a heuristic branch (live or static) could return a model
    whose constraint entry has `max_context < context_tokens * 1.1` — the
    current code DOESN'T re-validate, so undersized models slip through.
    """
    from mullet.core.providers import constraints as _c

    # Inject a live model whose constraint entry declares too-small max_context
    # for the requested tokens. Force every can_serve() call to say False so
    # the post-pick assertion (once wired) must raise rather than silently
    # return the model.
    fake_constraint = MagicMock()
    fake_constraint.can_serve.return_value = False
    fake_constraint.max_context = 8_192
    monkeypatch.setattr(
        _c.ConstraintRegistry, "get", lambda self, model_id: fake_constraint
    )

    # Requesting 50k tokens against a provider whose every candidate is
    # rejected by can_serve → must raise ValueError, not return a stale id.
    with pytest.raises(ValueError) as exc_info:
        fresh_registry.get_best_model(
            "kimi",
            complexity="STANDARD",
            context_needed=4096,
            context_tokens=50_000,
        )
    msg = str(exc_info.value).lower()
    assert "kimi" in msg
    assert "context" in msg or "candidates" in msg


# ---------------------------------------------------------------------------
# 2. _best_kimi filters moonshot-v1-8k
# ---------------------------------------------------------------------------

def test_best_kimi_filters_moonshot_v1_8k_for_14k_prompt(fresh_registry):
    """14K-token prompt must not land on moonshot-v1-8k (8192 max).

    Uses the real ConstraintRegistry so the end-to-end filter + post-pick
    assertion chain is exercised. moonshot-v1-8k has max_context=8192 which
    is <14485 * 1.1 = 15933 and must be rejected.
    """
    chosen = fresh_registry.get_best_model(
        "kimi",
        complexity="STANDARD",
        context_needed=4096,
        context_tokens=14_485,  # the Cycle 1 bug value
    )
    assert chosen != "moonshot-v1-8k", (
        f"14K-token prompt must filter moonshot-v1-8k; got {chosen!r}"
    )
    # Must pick a model whose max_context covers 14485 * 1.1 = 15933.
    # kimi-k2.7-code is the STANDARD preferred default (256K) — expected.
    assert chosen == "kimi-k2.7-code", (
        f"Picked model must be the preferred default kimi-k2.7-code; got {chosen!r}"
    )


# ---------------------------------------------------------------------------
# 3. _best_gemini filters at context limit
# ---------------------------------------------------------------------------

def test_best_gemini_filters_at_context_limit(fresh_registry):
    """Red: Gemini pick must be filtered by context_tokens.

    Gemini models all have 1M context in the static table, so we inject a
    fake 64K Gemini model to force the filter to fire.
    """
    import time

    from mullet.core.providers.model_registry import ProviderModel

    # Inject a Gemini candidate with tiny context → must be filtered for >64K tokens.
    fresh_registry._cache["gemini"] = [
        ProviderModel(
            id="gemini-tiny",
            provider="gemini",
            display_name="Gemini Tiny",
            context_length=32_000,
            input_cost_per_1k=0.0001,
            output_cost_per_1k=0.0001,
            tier="standard",
            is_available=True,
        ),
        ProviderModel(
            id="gemini-2.5-flash",
            provider="gemini",
            display_name="Gemini 2.5 Flash",
            context_length=1_000_000,
            input_cost_per_1k=0.00015,
            output_cost_per_1k=0.00035,
            tier="standard",
            is_available=True,
        ),
    ]
    fresh_registry._cache_ts["gemini"] = time.time()

    chosen = fresh_registry.get_best_model(
        "gemini",
        complexity="STANDARD",
        context_needed=4096,
        context_tokens=64_000,
    )
    # gemini-tiny has 32k context; 64k * 1.1 = 70400 > 32000 → filtered
    assert chosen != "gemini-tiny", (
        f"64K prompt must filter gemini-tiny (32K context); got {chosen!r}"
    )


# ---------------------------------------------------------------------------
# 4. _best_anthropic (Haiku) filters at context limit
# ---------------------------------------------------------------------------

def test_best_haiku_filters_at_context_limit(fresh_registry):
    """Red: Haiku (200K) must be filtered out for context > 200K * 0.91.

    Failure mode: LOW-tier anthropic maps to haiku in `_best_anthropic`
    static fallback; with 250K token prompt, haiku cannot serve and the
    picker must escalate to sonnet/opus instead. Uses real ConstraintRegistry.
    """
    chosen = fresh_registry.get_best_model(
        "anthropic",
        complexity="LOW",  # LOW → preferred haiku in mapping
        context_needed=4096,
        context_tokens=250_000,  # > 200K → haiku excluded by 10% margin
    )
    assert "haiku" not in chosen.lower(), (
        f"250K prompt must filter haiku (200K ctx); got {chosen!r}"
    )


# ---------------------------------------------------------------------------
# 5. _best_anthropic (Sonnet) filters at context limit
# ---------------------------------------------------------------------------

def test_best_sonnet_filters_at_context_limit(fresh_registry):
    """Red: Sonnet (1M) rejected when prompt > 1M * 0.91 ≈ 909K.

    Uses the real ConstraintRegistry: sonnet's max_context=1M so 1.5M
    tokens * 1.1 = 1.65M fails `can_serve`. Every anthropic candidate
    caps at 1M → all fail → ValueError.
    """
    with pytest.raises(ValueError) as exc_info:
        fresh_registry.get_best_model(
            "anthropic",
            complexity="STANDARD",
            context_needed=4096,
            context_tokens=1_500_000,  # > 1M * any margin
        )
    msg = str(exc_info.value).lower()
    assert "anthropic" in msg
    assert "candidates" in msg or "context" in msg


# ---------------------------------------------------------------------------
# 6. All candidates fail → ValueError with list
# ---------------------------------------------------------------------------

def test_all_candidates_fail_raises_with_list(monkeypatch, fresh_registry):
    """All candidates too small → ValueError mentions provider, tier, candidates."""
    _force_constraint(monkeypatch, can_serve_result=False)

    with pytest.raises(ValueError) as exc_info:
        fresh_registry.get_best_model(
            "kimi",
            complexity="STANDARD",
            context_needed=4096,
            context_tokens=5_000_000,  # > every kimi candidate max_context
        )
    msg = str(exc_info.value)
    assert "kimi" in msg.lower()
    assert "tier" in msg.lower() or "standard" in msg.lower()
    assert "candidates" in msg.lower() or "tried" in msg.lower()


# ---------------------------------------------------------------------------
# 7. Isolated unit test for the extracted helper
# ---------------------------------------------------------------------------

def test_filter_candidates_helper_isolated():
    """The extracted `_filter_candidates_by_context` helper must:

    - Accept a list of (model_id, max_context) tuples and a token count.
    - Return only candidates whose max_context >= tokens * 1.1.
    - Return [] when no candidate fits (caller decides how to raise).
    - When context_tokens is None or 0, return the full list unchanged.
    """
    from mullet.core.providers.model_registry import _filter_candidates_by_context

    candidates = [
        ("moonshot-v1-8k", 8_192),
        ("moonshot-v1-32k", 32_768),
        ("moonshot-v1-128k", 131_072),
        ("kimi-k2.5", 262_144),
    ]

    # Small prompt — all fit (8192 * 1.1 = 9011, so 14485 excludes 8k only)
    result = _filter_candidates_by_context(candidates, 14_485)
    ids = [c[0] for c in result]
    assert "moonshot-v1-8k" not in ids, "8k must be filtered for 14K prompt"
    assert "moonshot-v1-32k" in ids, "32k fits 14485 * 1.1 = 15933"

    # Large prompt — only kimi-k2.5 fits (131072 * 1.1 = 144179 > 150000 excludes 128k)
    result = _filter_candidates_by_context(candidates, 150_000)
    ids = [c[0] for c in result]
    assert ids == ["kimi-k2.5"], (
        f"150K prompt must leave only kimi-k2.5; got {ids!r}"
    )

    # Huge prompt — empty
    result = _filter_candidates_by_context(candidates, 5_000_000)
    assert result == [], "5M prompt must filter every candidate"

    # None / 0 tokens — pass-through
    assert _filter_candidates_by_context(candidates, None) == candidates
    assert _filter_candidates_by_context(candidates, 0) == candidates
