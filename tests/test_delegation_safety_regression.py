"""Delegation safety D6 — regression suite for the Cycle 1 picker bug.

This suite pins the canonical failure pattern that triggered the entire
delegation-safety workstream (D1-D5):

  ``handle_mullet_run({providers: "kimi", tier: "ADVANCED",
                       files: ["scripts/reindex.py"], apply: true})``
  with a ~14,485-token prompt was routed to ``moonshot-v1-8k`` (8192 max
  context) → 400: ``"request exceeded model token limit: 8192
  (requested: 14485)"``.

The underlying gates were closed by:
  - D1 (e7bc1fc): pre-dispatch token measurement on every call path
  - D2 (385e877): constraint filter post-validates every picker return
  - D3 (7b86ba7): structured ``ModelContextExceeded`` with an
    ``all_candidates_exhausted`` discriminator
  - D4 (b9850c8): wire-level assertion in every provider's
    ``_call_with_retry`` (last-chance safety net)
  - D5 (03213f3): auto-escalation up the tier ladder when the
    requested tier's candidates are too small

D6 (this file) does NOT add new behaviour — it locks the seven
invariants that, together, make the Cycle 1 reproduction impossible to
recur. Each test docstring names the originating cycle / commit / ticket
and the failure mode it would catch.

Validation: ``python3 -m pytest tests/test_delegation_safety_regression.py
-q --tb=short`` — all 7 tests must pass on first run.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mullet.core.providers.errors import ModelContextExceeded


# ---------------------------------------------------------------------------
# Helpers shared with the D1 suite (pattern parity).
# ---------------------------------------------------------------------------


def _mock_provider(name: str = "kimi", model: str = "kimi-k2.5") -> MagicMock:
    """Construct a duck-typed provider with the attrs bridge mode reads."""
    p = MagicMock()
    p.name = name
    p.model = model
    return p


def _mock_response(content: str = "ok") -> MagicMock:
    r = MagicMock()
    r.content = content
    r.usage = MagicMock(total_tokens=10, input_tokens=5, output_tokens=5)
    r.tool_calls = []
    return r


def _clear_env(monkeypatch) -> None:
    """Clear competing provider keys so the test only exercises one branch."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# 1. Canonical Cycle 1 reproduction — ADVANCED tier, ~14K tokens.
# ---------------------------------------------------------------------------


def test_cycle1_kimi_advanced_14k_does_not_pick_moonshot_v1_8k(monkeypatch):
    """Red (pre-D1/D2): a 14K-token prompt routed to Kimi/ADVANCED selects
    moonshot-v1-8k (8192 ctx) and explodes 400 at the wire.

    Failure mode (Cycle 1, 2026-04-23 23:24 UTC, commit reference: D1
    fix e7bc1fc): ``handle_mullet_run({providers:"kimi", tier:"ADVANCED",
    files:["scripts/reindex.py"]})`` returned a 400 from Moonshot's
    upstream because the picker handed the bridge moonshot-v1-8k.

    This test pins the post-D2 invariant: when a 14K prompt is composed,
    the constraint filter must reject every model whose
    ``max_context * 1.1 < prompt_tokens``. moonshot-v1-8k (8192) and
    moonshot-v1-32k (32768) both pass on-paper, but only moonshot-v1-8k
    is small enough to break Cycle 1 reproducibly — assert it never
    appears in the constructed Kimi fallback chain.
    """
    from mullet.server.mcp_handlers import _build_fallback_chain

    monkeypatch.setenv("KIMI_API_KEY", "fake")
    _clear_env(monkeypatch)

    # Force the registry to hand back moonshot-v1-8k for the kimi branch —
    # this is the exact bug the production picker had.
    with patch("mullet.core.providers.model_registry.get_registry") as mock_reg, \
         patch("mullet.core.providers.kimi.KimiProvider") as mock_kimi_cls:
        mock_reg.return_value.get_best_model.return_value = "moonshot-v1-8k"
        # D2 invariant — when the only allowed provider's only candidate
        # cannot serve the prompt, _build_fallback_chain raises a ValueError
        # carrying ``filtered_providers`` so the bridge can surface why.
        # This is far better than the Cycle 1 failure mode (silent 400 at
        # the wire). EITHER an empty chain OR a ValueError-with-reasons is
        # acceptable; what is NOT acceptable is moonshot-v1-8k surviving
        # to a real KimiProvider construction.
        try:
            chain = _build_fallback_chain(
                allowed={"kimi"},
                requested_tier="ADVANCED",
                prompt_tokens=14_485,  # the exact Cycle 1 token count
            )
            raised: ValueError | None = None
        except ValueError as exc:
            chain = []
            raised = exc

    # The cardinal Cycle 1 invariant: KimiProvider was NEVER instantiated
    # with moonshot-v1-8k for a 14K prompt.
    mock_kimi_cls.assert_not_called()
    assert chain == [], (
        f"moonshot-v1-8k must be filtered for 14K-token prompt; got "
        f"{[getattr(p, 'model', '?') for p in chain]!r}"
    )
    if raised is not None:
        # When the chain raises, it must carry filtered_providers with the
        # moonshot-v1-8k reason so the user sees WHY the request was refused.
        filtered = getattr(raised, "filtered_providers", None)
        assert isinstance(filtered, dict) and "kimi" in filtered, (
            f"ValueError must carry filtered_providers['kimi'] for observability; "
            f"got {filtered!r}"
        )
        assert "moonshot-v1-8k" in filtered["kimi"], (
            f"filtered_providers['kimi'] must name the rejected model; "
            f"got {filtered['kimi']!r}"
        )


# ---------------------------------------------------------------------------
# 2. STANDARD tier auto-escalates to ADVANCED for a 14K-token prompt.
# ---------------------------------------------------------------------------


def test_cycle1_kimi_standard_14k_auto_escalates_to_advanced(caplog):
    """Red (pre-D5): a 14K-token prompt at STANDARD tier raises ValueError
    because STANDARD's defaults can't fit, even though kimi-k2.5 at
    ADVANCED has 262K context.

    Failure mode: D1+D2 alone refuse the prompt; user has to re-ask with
    tier=ADVANCED. D5 (03213f3) makes that escalation transparent — the
    picker walks STANDARD → ADVANCED automatically and logs at INFO.

    This test pins the D5 contract: STANDARD-tier kimi requests for a
    14K prompt land on an ADVANCED-tier model (kimi-k2.5 / similar)
    AND emit a single ``auto-escalated`` INFO log so operators can see
    what happened.
    """
    from mullet.core.providers.model_registry import ModelRegistry

    reg = ModelRegistry()

    call_log: list[str] = []

    def fake_pick_model(*, provider, complexity, context_needed, context_tokens, **_kw):
        call_log.append(complexity)
        if complexity == "STANDARD":
            # STANDARD's defaults (moonshot-v1-8k / moonshot-v1-128k) can't
            # serve 14K with the 1.1 safety margin → raise non-exhausted MCE
            # so D5 escalates.
            raise ModelContextExceeded(
                provider="kimi",
                tier="STANDARD",
                context_tokens=context_tokens or 0,
                tried_candidates=["moonshot-v1-8k"],
                max_candidate_context=8_192,
                all_candidates_exhausted=False,  # bigger sibling exists
            )
        # ADVANCED tier picks kimi-k2.5 (262K).
        return "kimi-k2.5"

    with patch.object(reg, "_pick_model", side_effect=fake_pick_model), \
         patch.object(reg, "_assert_can_serve", side_effect=lambda p, m, c, t: m), \
         caplog.at_level("INFO", logger="mullet.core.providers.model_registry"):
        chosen = reg.get_best_model(
            "kimi",
            complexity="STANDARD",
            context_needed=4096,
            context_tokens=14_485,
        )

    assert chosen == "kimi-k2.5", (
        f"STANDARD + 14K prompt must auto-escalate to kimi-k2.5; got {chosen!r}"
    )
    assert call_log == ["STANDARD", "ADVANCED"], (
        f"escalation must walk STANDARD → ADVANCED exactly once; got {call_log!r}"
    )
    auto_msgs = [
        rec.getMessage() for rec in caplog.records
        if "auto-escalated" in rec.getMessage()
    ]
    assert auto_msgs, (
        f"expected at least one INFO 'auto-escalated' log; got "
        f"{[r.getMessage() for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# 3. 1M-token prompt to Kimi raises clean MCE (exhausted=True).
# ---------------------------------------------------------------------------


def test_kimi_request_exceeds_all_tiers_raises_clean_mce():
    """Red (pre-D3): a 1M-token prompt to Kimi raises a bare ValueError;
    callers can't tell ``escalate vs split-the-task`` apart.

    Failure mode (commit ref: D3 7b86ba7): every ``except ValueError:``
    site treats the refusal as transient and retries forever, or
    surfaces a useless string error.

    This test pins the post-D3+D5 contract: a prompt larger than the
    biggest model's ``max_context`` (kimi-k2.5 = 262K) raises
    ``ModelContextExceeded`` with ``all_candidates_exhausted=True`` —
    the unambiguous "split the task or refuse" signal.
    """
    from mullet.core.providers.model_registry import ModelRegistry

    reg = ModelRegistry()

    def always_exhausted(*, provider, complexity, context_needed, context_tokens, **_kw):
        # Simulate a picker that has exhausted its candidate list at the
        # current tier; the post-pick assertion converts this to a non-
        # exhausted MCE at lower tiers but the chain walks all tiers.
        raise ModelContextExceeded(
            provider="kimi",
            tier=complexity,
            context_tokens=context_tokens or 0,
            tried_candidates=["kimi-k2.5"],
            max_candidate_context=262_144,
            all_candidates_exhausted=True,
        )

    with patch.object(reg, "_pick_model", side_effect=always_exhausted):
        with pytest.raises(ModelContextExceeded) as exc_info:
            reg.get_best_model(
                "kimi",
                complexity="ADVANCED",
                context_needed=4096,
                context_tokens=1_000_000,
            )

    err = exc_info.value
    assert err.all_candidates_exhausted is True, (
        "1M-token prompt must surface as exhausted=True so callers see "
        "'split the task or switch provider'"
    )
    assert err.provider == "kimi"
    # Subclass-of-ValueError parity for legacy callers.
    assert isinstance(err, ValueError), (
        "ModelContextExceeded must remain a ValueError subclass for "
        "back-compat with `except ValueError:` catchers"
    )


# ---------------------------------------------------------------------------
# 4. D4 dispatch-time assertion blocks BEFORE HTTP if D1/D2/D3/D5 are bypassed.
# ---------------------------------------------------------------------------


def test_d4_dispatch_assertion_blocks_if_picker_is_bypassed():
    """Red (pre-D4): a buggy picker (or a future regression) hands the
    OpenAI-compat provider moonshot-v1-8k for a 14K prompt; the prior
    safety nets (D1/D2/D3/D5) don't fire because they live in the picker
    layer; the request goes out and Moonshot returns 400.

    Failure mode (commit ref: D4 b9850c8): the only thing standing
    between the picker and the wire is the per-provider
    ``_call_with_retry`` — which previously trusted the picker.

    This test pins the D4 invariant: when a too-small model arrives at
    the wire, the gate raises ``ModelContextExceeded(exhausted=True)``
    BEFORE ``client.chat.completions.create`` is touched. Bypasses
    every higher-layer gate by directly invoking ``_call_with_retry``.
    """
    from mullet.core.providers.openai_compat import OpenAICompatibleProvider

    p = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    p.client = MagicMock()
    p.client.chat.completions.create = MagicMock(
        side_effect=AssertionError(
            "HTTP call was reached — D4 gate failed to block oversize prompt"
        )
    )
    p.model = "moonshot-v1-8k"
    p.name = "kimi"
    p._openrouter_config = {}

    # ~14K tokens via cl100k (and ~13K via the 6 chars/token fallback).
    big_content = "A" * 90_000
    kwargs = {
        "model": "moonshot-v1-8k",
        "messages": [{"role": "user", "content": big_content}],
        "max_tokens": 4096,
    }

    with pytest.raises(ModelContextExceeded) as exc_info:
        p._call_with_retry(kwargs)

    err = exc_info.value
    assert err.all_candidates_exhausted is True, (
        "At dispatch time there is no fallback left — exhausted must be True"
    )
    assert err.provider == "kimi"
    assert err.tried_candidates == ["moonshot-v1-8k"]
    assert err.max_candidate_context == 8_192
    # The whole point of D4: the wire was never touched.
    p.client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# 5. handle_mullet_run returns dict, never a string-of-JSON, on failure.
# ---------------------------------------------------------------------------


def test_handle_mullet_run_returns_parseable_dict_on_failure(monkeypatch):
    """D7 (Cycle 17, this commit): ``handle_mullet_run`` MUST return a
    native ``dict`` on every path (success and failure). Pre-D7 it
    returned ``json.dumps(...)`` (string), which broke the Cycle 1
    Python adapter with ``'str' object has no attribute 'get'``.

    Failure mode (pre-D7): the adapter called ``result.get("error")``
    on a string and crashed. Bridge mode is the canonical failure path
    because it carries the most non-trivial early-exit branches
    (constraint conflict, provider exhaustion, no-provider-available).

    Tightened from D6: where the previous version did
    ``json.loads(raw)`` to accept either string-or-dict, this version
    asserts the native dict contract directly.
    """
    from mullet.server import mcp_handlers

    monkeypatch.setenv("KIMI_API_KEY", "fake")
    _clear_env(monkeypatch)

    kimi_provider = _mock_provider("kimi", "kimi-k2.5")
    monkeypatch.setattr(mcp_handlers, "_resolve_provider", lambda **kw: kimi_provider)
    monkeypatch.setattr(mcp_handlers, "_write_agent_state_fn", lambda *a, **kw: None)
    monkeypatch.setattr(mcp_handlers, "_clear_agent_state", lambda: None)

    # Force the fallback-chain builder to raise the canonical "all
    # providers filtered out" ValueError so we hit the failure-return path.
    def boom_chain(**kwargs):
        err = ValueError("All providers in allowlist {'kimi'} filtered out by registry")
        err.filtered_providers = {  # type: ignore[attr-defined]
            "kimi": "context_tokens=14485 exceeds moonshot-v1-8k max_context=8192",
        }
        raise err

    monkeypatch.setattr(mcp_handlers, "_build_fallback_chain", boom_chain)

    result = mcp_handlers.handle_mullet_run({
        "task": "do something",
        "tier": "ADVANCED",
        "providers": "kimi",
    })

    # D7 (this cycle): tightened from `json.loads(raw)` to a direct
    # `isinstance(result, dict)` assertion. The handler must return a
    # native dict, not a JSON-encoded string.
    assert isinstance(result, dict), (
        f"D7 contract: handle_mullet_run must return dict, got "
        f"{type(result).__name__!r}"
    )
    assert "error" in result, (
        f"failure dict must include an 'error' key; got keys={list(result)!r}"
    )
    # Filtered-provider reason must surface (the whole point of P1-1).
    assert "filtered_providers" in result, (
        f"failure dict should surface filtered_providers; got keys={list(result)!r}"
    )
    assert result["filtered_providers"]["kimi"], (
        "filtered_providers['kimi'] must include a non-empty reason"
    )


# ---------------------------------------------------------------------------
# 6. Swarm dispatcher threads context_tokens (D1 regression guard).
# ---------------------------------------------------------------------------


def test_swarm_dispatcher_passes_context_tokens(monkeypatch):
    """Red (pre-D1): SwarmDispatcher used the ad-hoc heuristic
    ``prompt_len // 4 * 2`` and never threaded ``context_tokens=`` into
    ``get_best_model`` — so the picker's constraint filter saw 0 tokens
    and happily returned moonshot-v1-8k for a 14K prompt.

    Failure mode (commit ref: D1 e7bc1fc): swarm dispatch path was the
    second-most-common Cycle 1 leak (after the bridge).

    This test pins the D1 contract: ``_make_provider`` calls
    ``estimate_tokens`` on the actual prompt and threads the result
    through every ``get_best_model`` call.
    """
    # Import order matters — kimi_async must resolve KimiProvider before
    # we monkeypatch it to a fake.
    import mullet.core.providers.kimi_async as _kimi_async_mod  # noqa: F401
    from mullet.core.orchestration.swarm_dispatcher import SwarmDispatcher

    captured: dict = {"calls": []}

    class FakeRegistry:
        def get_best_model(self, provider, tier, **kwargs):
            captured["calls"].append({"provider": provider, "tier": tier, **kwargs})
            return "kimi-k2.5"

    fake_reg = FakeRegistry()
    monkeypatch.setattr(
        "mullet.core.providers.model_registry.get_registry",
        lambda: fake_reg,
    )

    fake_kimi_instance = _mock_provider("kimi", "kimi-k2.5")

    class FakeKimiProvider:
        def __init__(self, *args, **kwargs):
            self.model = kwargs.get("model", "kimi-k2.5")

    class FakeKimiAsyncProvider:
        def __init__(self, provider=None):
            self._provider = provider or fake_kimi_instance
            self.name = "kimi"
            self.model = getattr(self._provider, "model", "kimi-k2.5")

    monkeypatch.setattr(
        "mullet.core.providers.kimi.KimiProvider", FakeKimiProvider, raising=True
    )
    monkeypatch.setattr(
        "mullet.core.providers.kimi_async.KimiAsyncProvider",
        FakeKimiAsyncProvider,
        raising=True,
    )

    # Patch tokenize so we control the exact context_tokens value passed.
    expected_tokens = 14_485

    def fake_estimate(obj):
        return expected_tokens

    monkeypatch.setattr(
        "mullet.core.providers.tokenize.estimate_tokens", fake_estimate
    )

    dispatcher = SwarmDispatcher(concurrency=1)
    sub_prompt = "do work please"  # any string — estimator is stubbed
    dispatcher._make_provider("STANDARD", len(sub_prompt), prompt=sub_prompt)

    assert captured["calls"], "get_best_model was not called"
    with_ctx = [c for c in captured["calls"] if "context_tokens" in c]
    assert with_ctx, (
        f"swarm _make_provider must pass context_tokens= to every "
        f"get_best_model call; calls observed: {captured['calls']!r}"
    )
    # D1 contract: context_tokens reflects the estimator's output, not the
    # ad-hoc prompt_len // 4 * 2 fallback.
    assert with_ctx[0]["context_tokens"] == expected_tokens, (
        f"context_tokens must equal estimate_tokens(prompt) ({expected_tokens}); "
        f"got {with_ctx[0]['context_tokens']!r}"
    )



