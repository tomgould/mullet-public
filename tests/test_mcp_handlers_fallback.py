"""Tests for MCP handler fallback chain and temperature handling."""
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_provider(name: str, model: str) -> MagicMock:
    p = MagicMock()
    p.name = name
    p.model = model
    return p


def _make_response(content: str = "ok") -> MagicMock:
    r = MagicMock()
    r.content = content
    r.usage = MagicMock(total_tokens=10)
    r.tool_calls = []
    return r


# ---------------------------------------------------------------------------
# Fix B: moonshot-v1 excluded from CRITICAL/ADVANCED fallback chains
# ---------------------------------------------------------------------------

def test_critical_tier_excludes_moonshot(monkeypatch):
    """_build_fallback_chain with requested_tier=CRITICAL must not include moonshot-v1-* models."""
    from mullet.server.mcp_handlers import _build_fallback_chain

    moonshot_provider = _make_provider("kimi", "moonshot-v1-8k")

    monkeypatch.setenv("KIMI_API_KEY", "fake-kimi-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with patch("mullet.core.providers.model_registry.get_registry") as mock_reg:
        mock_reg.return_value.get_best_model.return_value = "moonshot-v1-8k"
        with patch("mullet.core.providers.kimi.KimiProvider", return_value=moonshot_provider):
            chain = _build_fallback_chain(requested_tier="CRITICAL")

    # moonshot-v1-8k must NOT appear in the fallback chain for CRITICAL
    models_in_chain = [getattr(p, "model", "") for p in chain]
    assert not any(m.startswith("moonshot-v1") for m in models_in_chain), (
        f"moonshot-v1 should be excluded from CRITICAL fallback chain, got: {models_in_chain}"
    )


def test_advanced_tier_excludes_moonshot(monkeypatch):
    """_build_fallback_chain with requested_tier=ADVANCED must not include moonshot-v1-* models."""
    from mullet.server.mcp_handlers import _build_fallback_chain

    monkeypatch.setenv("KIMI_API_KEY", "fake-kimi-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with patch("mullet.core.providers.model_registry.get_registry") as mock_reg:
        mock_reg.return_value.get_best_model.return_value = "moonshot-v1-32k"
        with patch("mullet.core.providers.kimi.KimiProvider") as mock_kimi:
            chain = _build_fallback_chain(requested_tier="ADVANCED")

    # KimiProvider should not have been instantiated (moonshot-v1-32k was skipped)
    mock_kimi.assert_not_called()


# ---------------------------------------------------------------------------
# Fix D: Kimi timeout falls back to Gemini, not moonshot
# ---------------------------------------------------------------------------

@patch("mullet.server.mcp_handlers._write_agent_state_fn")
@patch("mullet.server.mcp_handlers._clear_agent_state")
def test_kimi_timeout_falls_back_to_gemini(mock_clear, mock_write, monkeypatch):
    """When kimi raises TimeoutError, agent mode falls back to gemini, not moonshot."""
    from mullet.server.mcp_handlers import handle_mullet_run

    kimi_provider = _make_provider("kimi", "kimi-k2.5")
    kimi_provider.create.side_effect = TimeoutError("kimi timed out")

    gemini_provider = _make_provider("gemini", "gemini-2.0-flash")
    gemini_provider.create.return_value = _make_response("done")

    monkeypatch.setattr(
        "mullet.server.mcp_handlers._resolve_provider",
        lambda **kw: kimi_provider,
    )
    monkeypatch.setattr(
        "mullet.server.mcp_handlers._build_fallback_chain",
        lambda **kw: [gemini_provider],
    )

    result = handle_mullet_run({
        "task": "do something",
        "tier": "ADVANCED",
    })

    assert result.get("completed") is True
    assert result.get("actual_provider") == "gemini"
    assert result.get("actual_model") == "gemini-2.0-flash"


# ---------------------------------------------------------------------------
# Fix C: actual_provider / actual_model in response
# ---------------------------------------------------------------------------

@patch("mullet.server.mcp_handlers._write_agent_state_fn")
@patch("mullet.server.mcp_handlers._clear_agent_state")
def test_response_includes_actual_model(mock_clear, mock_write, monkeypatch):
    """Response dict must contain actual_provider and actual_model from post-dispatch provider."""
    from mullet.server.mcp_handlers import handle_mullet_run

    real_provider = _make_provider("kimi", "kimi-k2.5")
    real_provider.create.return_value = _make_response("done")

    monkeypatch.setattr("mullet.server.mcp_handlers._resolve_provider", lambda **kw: real_provider)
    monkeypatch.setattr("mullet.server.mcp_handlers._build_fallback_chain", lambda **kw: [])

    result = handle_mullet_run({
        "task": "do something",
    })

    assert "actual_provider" in result, "Response must include actual_provider"
    assert "actual_model" in result, "Response must include actual_model"
    assert result["actual_provider"] == "kimi"
    assert result["actual_model"] == "kimi-k2.5"


# ---------------------------------------------------------------------------
# Fix A: no hardcoded temperature=0.0 passed to provider.create()
# ---------------------------------------------------------------------------

@patch("mullet.server.mcp_handlers._write_agent_state_fn")
@patch("mullet.server.mcp_handlers._clear_agent_state")
def test_no_hardcoded_temperature(mock_clear, mock_write, monkeypatch):
    """provider.create() must be called WITHOUT a temperature kwarg."""
    from mullet.server.mcp_handlers import handle_mullet_run

    provider = _make_provider("kimi", "kimi-k2.5")
    provider.create.return_value = _make_response("done")

    monkeypatch.setattr("mullet.server.mcp_handlers._resolve_provider", lambda **kw: provider)
    monkeypatch.setattr("mullet.server.mcp_handlers._build_fallback_chain", lambda **kw: [])

    result = handle_mullet_run({
        "task": "do something",
    })

    for c in provider.create.call_args_list:
        assert "temperature" not in c.kwargs, (
            f"temperature must not be hardcoded in provider.create() call; got kwargs={c.kwargs}"
        )


# ---------------------------------------------------------------------------
# P1-1 — Fallback-chain error transparency
# ---------------------------------------------------------------------------

@patch("mullet.server.mcp_handlers._write_agent_state_fn")
@patch("mullet.server.mcp_handlers._clear_agent_state")
def test_all_providers_fail_returns_structured_errors(mock_clear, mock_write, monkeypatch):
    """When every provider raises, response must include tried_providers and errors_by_provider."""
    from mullet.server.mcp_handlers import handle_mullet_run

    kimi = _make_provider("kimi", "kimi-k2.5")
    kimi.create.side_effect = TimeoutError("context window exceeded")

    gemini = _make_provider("gemini", "gemini-2.5-flash")
    gemini.create.side_effect = RuntimeError("GOOGLE_API_KEY not set")

    haiku = _make_provider("haiku", "claude-haiku-4-5-20251001")
    haiku.create.side_effect = RuntimeError("rate_limit: 429")

    monkeypatch.setattr(
        "mullet.server.mcp_handlers._resolve_provider",
        lambda **kw: kimi,
    )
    monkeypatch.setattr(
        "mullet.server.mcp_handlers._build_fallback_chain",
        lambda **kw: [gemini, haiku],
    )

    result = handle_mullet_run({
        "task": "do something",
        "tier": "ADVANCED",
    })

    assert "error" in result
    assert "Agent execution failed" in result["error"]
    assert set(result["tried_providers"]) == {"kimi", "gemini", "haiku"}


@patch("mullet.server.mcp_handlers._write_agent_state_fn")
@patch("mullet.server.mcp_handlers._clear_agent_state")
def test_success_after_retry_shows_tried_providers(mock_clear, mock_write, monkeypatch):
    """First provider raises, second succeeds → response lists both providers, no errors dict."""
    from mullet.server.mcp_handlers import handle_mullet_run

    kimi = _make_provider("kimi", "kimi-k2.5")
    kimi.create.side_effect = RuntimeError("kimi down")

    gemini = _make_provider("gemini", "gemini-2.5-flash")
    gemini.create.return_value = _make_response("done")

    monkeypatch.setattr(
        "mullet.server.mcp_handlers._resolve_provider",
        lambda **kw: kimi,
    )
    monkeypatch.setattr(
        "mullet.server.mcp_handlers._build_fallback_chain",
        lambda **kw: [gemini],
    )

    result = handle_mullet_run({
        "task": "do something",
        "tier": "ADVANCED",
    })

    assert result.get("completed") is True
    assert set(result.get("tried_providers", [])) == {"kimi", "gemini"}, (
        f"expected both providers listed; got {result.get('tried_providers')!r}"
    )
    # Success path must NOT include errors_by_provider
    assert "errors_by_provider" not in result


@patch("mullet.server.mcp_handlers._write_agent_state_fn")
@patch("mullet.server.mcp_handlers._clear_agent_state")
def test_first_attempt_success_omits_tried_providers(mock_clear, mock_write, monkeypatch):
    """First provider succeeds on first try → response does NOT include tried_providers."""
    from mullet.server.mcp_handlers import handle_mullet_run

    kimi = _make_provider("kimi", "kimi-k2.5")
    kimi.create.return_value = _make_response("done")

    monkeypatch.setattr(
        "mullet.server.mcp_handlers._resolve_provider",
        lambda **kw: kimi,
    )
    monkeypatch.setattr(
        "mullet.server.mcp_handlers._build_fallback_chain",
        lambda **kw: [],
    )

    result = handle_mullet_run({
        "task": "do something",
    })

    assert result.get("completed") is True
    # Legacy shape: no tried_providers key on first-try success.
    assert "tried_providers" not in result
    assert "errors_by_provider" not in result


@patch("mullet.server.mcp_handlers._write_agent_state_fn")
@patch("mullet.server.mcp_handlers._clear_agent_state")
def test_constraint_conflict_lists_filtered_providers(mock_clear, mock_write, monkeypatch):
    """When the allowlist + constraint registry eliminate every candidate,
    the response must include filtered_providers with rejection reasons.
    """
    from mullet.server.mcp_handlers import handle_mullet_run

    # _resolve_provider succeeds (preferred available).
    kimi = _make_provider("kimi", "kimi-k2.5")
    monkeypatch.setattr(
        "mullet.server.mcp_handlers._resolve_provider",
        lambda **kw: kimi,
    )

    def _raising_fb(**kw):
        filtered = kw.get("filtered_out")
        if filtered is not None:
            filtered["kimi"] = "context_tokens=300000 exceeds kimi-k2.5 max_context=262144"
            filtered["gemini"] = "context_tokens=300000 exceeds gemini-2.5-flash max_context=200000"
        err = ValueError("All providers in allowlist {'kimi'} filtered out by constraint registry")
        err.filtered_providers = dict(filtered or {})
        raise err

    monkeypatch.setattr(
        "mullet.server.mcp_handlers._build_fallback_chain",
        _raising_fb,
    )

    result = handle_mullet_run({
        "task": "do something",
        "providers": "kimi",
    })

    assert "error" in result
    assert "allowlist/constraint conflict" in result["error"]
    assert "filtered_providers" in result
    assert "kimi" in result["filtered_providers"]
    assert "max_context" in result["filtered_providers"]["kimi"]
