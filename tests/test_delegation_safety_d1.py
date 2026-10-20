"""Delegation safety D1 — pre-dispatch prompt-token measurement.

Covers the invariant: before every cloud dispatch, measure the composed
prompt and reject models where ``max_context < prompt_tokens * 1.1``.

Three gate points must honour this invariant:

1. Bridge mode (``mullet/server/mcp_handlers.handle_mullet_run`` with ``files``)
2. Agent mode (``mullet/core/agent/runner.AgentRunner._call_llm``)
3. Swarm dispatch (``mullet/core/orchestration/swarm_dispatcher._make_provider``)

Each test docstring names the expected Red-phase failure so the developer
running this suite can see why a regression matters.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_provider(name: str = "kimi", model: str = "kimi-k2.5") -> MagicMock:
    p = MagicMock()
    p.name = name
    p.model = model
    p._mcp_resolved_alias = name
    return p


def _mock_response(content: str = "ok") -> MagicMock:
    r = MagicMock()
    r.content = content
    r.usage = MagicMock(total_tokens=10, input_tokens=5, output_tokens=5)
    r.tool_calls = []
    return r


# ---------------------------------------------------------------------------
# 1. Bridge mode
# ---------------------------------------------------------------------------

def test_agent_mode_estimates_composed_prompt(monkeypatch):
    """Red: agent mode must call estimate_tokens() on the task text
    AND pass the result to _build_fallback_chain as prompt_tokens.
    Failure mode: agent ships bytes to router without measuring → router
    picks moonshot-v1-8k for a 14K-token prompt → 400.
    """
    from mullet.server import mcp_handlers

    monkeypatch.setenv("KIMI_API_KEY", "fake")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    kimi = _mock_provider("kimi", "kimi-k2.5")
    kimi.create.return_value = _mock_response("done")

    captured: dict = {}

    def fake_estimate(obj):
        captured["estimate_called_with"] = obj
        return 14_485  # the Cycle 1 bug value

    def fake_build_fallback(**kwargs):
        captured["fallback_kwargs"] = kwargs
        return []

    monkeypatch.setattr(
        "mullet.core.providers.tokenize.estimate_tokens", fake_estimate
    )
    monkeypatch.setattr(mcp_handlers, "_resolve_provider", lambda **kw: kimi)
    monkeypatch.setattr(mcp_handlers, "_build_fallback_chain", fake_build_fallback)
    monkeypatch.setattr(mcp_handlers, "_write_agent_state_fn", lambda *a, **kw: None)
    monkeypatch.setattr(mcp_handlers, "_clear_agent_state", lambda: None)

    mcp_handlers.handle_mullet_run({
        "task": "do something",
        "tier": "ADVANCED",
        "providers": "kimi",
    })

    assert "estimate_called_with" in captured, (
        "estimate_tokens was never called — agent mode is shipping without measurement"
    )
    assert "fallback_kwargs" in captured, "_build_fallback_chain was not called"
    assert captured["fallback_kwargs"].get("prompt_tokens") == 14_485, (
        f"prompt_tokens must be threaded to _build_fallback_chain, got "
        f"{captured['fallback_kwargs'].get('prompt_tokens')!r}"
    )


def test_bridge_mode_rejects_undersized_model(monkeypatch):
    """Red: for a 14K-token prompt, moonshot-v1-8k (8192 max) must be filtered
    out of the Kimi fallback branch even if get_best_model returned it.
    Failure mode: the Cycle 1 bug — moonshot-v1-8k selected → 400 context_exceeded.
    """
    from mullet.server.mcp_handlers import _build_fallback_chain

    monkeypatch.setenv("KIMI_API_KEY", "fake")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    # Force the registry to hand back moonshot-v1-8k for the kimi branch.
    with patch("mullet.core.providers.model_registry.get_registry") as mock_reg:
        mock_reg.return_value.get_best_model.return_value = "moonshot-v1-8k"
        with patch("mullet.core.providers.kimi.KimiProvider") as mock_kimi_cls:
            chain = _build_fallback_chain(
                requested_tier="STANDARD",
                prompt_tokens=14_485,
            )

    # moonshot-v1-8k has max_context=8192; 14485 > 8192 → must be rejected.
    mock_kimi_cls.assert_not_called()
    models_in_chain = [getattr(p, "model", "") for p in chain]
    assert not any(m.startswith("moonshot-v1-8k") for m in models_in_chain), (
        f"moonshot-v1-8k must be filtered out for 14K-token prompt, got {models_in_chain!r}"
    )


# ---------------------------------------------------------------------------
# 2. Agent mode
# ---------------------------------------------------------------------------

def test_agent_mode_estimates_current_messages_before_call(monkeypatch):
    """Red: AgentRunner._call_llm must estimate the outgoing message list's
    token count before dispatch. Required so that, in a follow-up cycle, we
    can thread the estimate to routing.
    Failure mode: the agent blindly calls provider.create() regardless of
    whether the model can hold the conversation → opaque 400 mid-loop.
    """
    from mullet.core.agent.runner import AgentRunner
    from mullet.core.providers.base import Provider as _Provider  # noqa: F401

    provider = _mock_provider("kimi", "kimi-k2.5")
    provider.create.return_value = _mock_response()

    registry = MagicMock()
    registry.get_schemas.return_value = []
    registry.dispatch.return_value = ""

    captured: dict = {"calls": []}

    def fake_estimate(obj):
        captured["calls"].append(obj)
        return 1234

    monkeypatch.setattr(
        "mullet.core.providers.tokenize.estimate_tokens", fake_estimate
    )

    runner = AgentRunner(
        provider=provider,
        tool_registry=registry,
        toolsets=[],
        system_prompt="sys",
        max_iterations=1,
    )
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "x" * 5_000},
    ]
    runner._call_llm(messages, tools=[])

    assert captured["calls"], (
        "AgentRunner._call_llm must call estimate_tokens on outgoing messages; "
        "none observed"
    )
    # estimate_tokens was called with exactly the outgoing messages list.
    assert any(c == messages for c in captured["calls"]), (
        f"estimate_tokens should be called with the exact outgoing messages, "
        f"got {[type(c).__name__ for c in captured['calls']]}"
    )


# ---------------------------------------------------------------------------
# 3. Swarm dispatch
# ---------------------------------------------------------------------------

def test_swarm_dispatch_threads_context_tokens(monkeypatch):
    """Red: SwarmDispatcher._make_provider must pass context_tokens= to
    get_best_model based on estimate_tokens(prompt), not the ad-hoc
    ``prompt_len // 4 * 2`` heuristic.
    Failure mode: swarm picks kimi-k2-0711-preview for oversize prompt
    because prompt_len//4*2 is a bad estimator, or blindly returns the
    first tier without context_tokens filtering.
    """
    # Import kimi_async first so its class body is evaluated with the real
    # KimiProvider in scope — otherwise monkeypatching the class triggers
    # re-import and `KimiProvider | None` blows up.
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

    # Patch the KimiProvider / KimiAsyncProvider names used inside the
    # swarm_dispatcher module's _make_provider via local imports.
    # The local imports resolve `from mullet.core.providers.kimi import
    # KimiProvider` at call time — so we patch the module attrs directly.
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

    dispatcher = SwarmDispatcher(concurrency=1)
    prompt = "x" * 8_000  # ~ 1300 tokens via cl100k
    dispatcher._make_provider("STANDARD", len(prompt))

    assert captured["calls"], "get_best_model was not called"
    # At least one call must include context_tokens kwarg.
    with_ctx = [c for c in captured["calls"] if "context_tokens" in c]
    assert with_ctx, (
        f"swarm _make_provider must pass context_tokens= to get_best_model; "
        f"calls observed: {captured['calls']!r}"
    )
    # context_tokens should be a positive integer reflecting the prompt size
    ctok = with_ctx[0]["context_tokens"]
    assert isinstance(ctok, int) and ctok > 0, (
        f"context_tokens must be positive int, got {ctok!r}"
    )


# ---------------------------------------------------------------------------
# 4. No regression — tiny prompts
# ---------------------------------------------------------------------------

def test_no_regression_on_small_prompts(monkeypatch):
    """Green-guard: a 50-token prompt must flow through bridge mode
    without the constraint filter rejecting every provider."""
    from mullet.server import mcp_handlers

    monkeypatch.setenv("KIMI_API_KEY", "fake")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    kimi = _mock_provider("kimi", "kimi-k2.5")
    kimi.create.return_value = _mock_response("FILE a.py\npass\n/FILE")

    monkeypatch.setattr(mcp_handlers, "_resolve_provider", lambda **kw: kimi)
    # Real _build_fallback_chain with small prompt should not raise
    monkeypatch.setattr(mcp_handlers, "_write_agent_state_fn", lambda *a, **kw: None)
    monkeypatch.setattr(mcp_handlers, "_clear_agent_state", lambda: None)

    out = mcp_handlers.handle_mullet_run({
        "task": "hi",
        "tier": "STANDARD",
        "providers": "kimi",
    })

    assert out.get("completed") is True, f"small prompt should succeed: {out}"


# ---------------------------------------------------------------------------
# 5. Estimator failure fallback
# ---------------------------------------------------------------------------

def test_estimator_failure_does_not_break_agent_dispatch(monkeypatch):
    """Green: if estimate_tokens itself raises (tiktoken cold-start / OOM),
    the agent must fall back to a rough char-based estimate and proceed,
    not abort the whole dispatch.
    """
    from mullet.server import mcp_handlers

    monkeypatch.setenv("KIMI_API_KEY", "fake")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    kimi = _mock_provider("kimi", "kimi-k2.5")
    kimi.create.return_value = _mock_response("done")

    def boom(_obj):
        raise RuntimeError("tiktoken cold-start OOM")

    monkeypatch.setattr(
        "mullet.core.providers.tokenize.estimate_tokens", boom
    )
    monkeypatch.setattr(mcp_handlers, "_resolve_provider", lambda **kw: kimi)
    monkeypatch.setattr(mcp_handlers, "_write_agent_state_fn", lambda *a, **kw: None)
    monkeypatch.setattr(mcp_handlers, "_clear_agent_state", lambda: None)

    out = mcp_handlers.handle_mullet_run({
        "task": "hi",
        "tier": "STANDARD",
        "providers": "kimi",
    })

    assert out.get("completed") is True, (
        f"estimator failure must not break dispatch, got {out}"
    )
