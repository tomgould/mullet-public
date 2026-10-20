"""Cycle 24 — obedience polish bundle.

Three small follow-ups closing the residual TODO under
"Orchestration Obedience — 2026-04-23 / Follow-ups (future)":

  1. ``Router.route_delegation`` auto-infers ``allowed_tiers`` from
     ``allowed_providers`` when only the latter is given.
  2. WebSocket ``/ws`` first-frame ``providers`` field is forwarded to
     ``_build_server_agent`` (parity with REST ``/v1/chat/completions``).
  3. ``ObedienceError`` exception class — typed surface for
     allowlist/constraint conflicts; subclass of ``ValueError`` so the
     existing ``except ValueError:`` catchers keep working.

Each test docstring names the failure mode the test would catch on
regression.
"""
from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Item 1 — Router.route_delegation auto-infers allowed_tiers
# ---------------------------------------------------------------------------


def _build_test_router():
    """Construct a Router with a kimi entry so ``route()`` doesn't fail."""
    from mullet.core.router.decision import Router, ProviderEntry

    kimi = ProviderEntry(
        name="kimi",
        tier="STANDARD",
        models={"default": "kimi-k2.5", "standard": "kimi-k2.5"},
        cost_per_1k_input=0.0006,
        cost_per_1k_output=0.0025,
    )
    return Router(providers={"kimi": kimi})


def test_route_delegation_infers_allowed_tiers_from_providers(caplog):
    """Red: pre-Cycle-24 ``route_delegation`` ignored allowed_providers
    when computing the child tier — a kimi-only allowlist could see a
    CRITICAL child analysed first, then bounced by the downstream
    allowlist filter. Green: tiers inferred from
    ``PROVIDER_TIER_CAPABILITIES`` cap the child tier up-front and an
    INFO line is emitted naming the inference."""
    router = _build_test_router()

    captured = {}

    def _fake_score_to_tier(score, thresholds=None, *, allowed_tiers=None):
        from mullet.core.router.complexity import Tier
        captured["allowed_tiers"] = allowed_tiers
        return Tier.STANDARD

    with caplog.at_level(logging.INFO, logger="mullet.core.router.decision"), \
         patch("mullet.core.router.decision.score_to_tier", side_effect=_fake_score_to_tier):
        router.route_delegation(
            parent_tier="STANDARD",
            child_task="hello world",
            allowed_providers=frozenset({"kimi"}),
        )

    assert captured["allowed_tiers"] is not None, \
        "allowed_tiers should be auto-inferred from allowed_providers"
    # PROVIDER_TIER_CAPABILITIES[kimi] = {LOW, STANDARD, ADVANCED, CRITICAL}
    assert set(captured["allowed_tiers"]) == {"LOW", "STANDARD", "ADVANCED", "CRITICAL"}
    assert any("inferred allowed_tiers" in r.message for r in caplog.records), \
        "expected INFO log naming the inferred tier set"


def test_route_delegation_respects_explicit_allowed_tiers():
    """Red: auto-infer would clobber an explicit caller-supplied
    ``allowed_tiers``. Green: explicit always wins; no inference runs
    when the caller already passed a tier list."""
    router = _build_test_router()

    captured = {}

    def _fake_score_to_tier(score, thresholds=None, *, allowed_tiers=None):
        from mullet.core.router.complexity import Tier
        captured["allowed_tiers"] = allowed_tiers
        return Tier.STANDARD

    with patch("mullet.core.router.decision.score_to_tier", side_effect=_fake_score_to_tier):
        router.route_delegation(
            parent_tier="STANDARD",
            child_task="hello world",
            allowed_providers=frozenset({"kimi"}),
            allowed_tiers=["LOW"],  # explicit cap, narrower than inferred
        )

    assert captured["allowed_tiers"] == ["LOW"], \
        "explicit allowed_tiers must not be overridden by auto-inference"


def test_route_delegation_no_inference_without_providers():
    """Red: inference would run even when no allowlist was given.
    Green: no allowed_providers means no auto-inference — caller gets
    the same tier scoring as before this cycle."""
    router = _build_test_router()

    captured = {}

    def _fake_score_to_tier(score, thresholds=None, *, allowed_tiers=None):
        from mullet.core.router.complexity import Tier
        captured["allowed_tiers"] = allowed_tiers
        return Tier.STANDARD

    with patch("mullet.core.router.decision.score_to_tier", side_effect=_fake_score_to_tier):
        router.route_delegation(parent_tier="STANDARD", child_task="hello world")

    assert captured["allowed_tiers"] is None


# ---------------------------------------------------------------------------
# Item 2 — WebSocket first-frame `providers` field
# ---------------------------------------------------------------------------


def _ws_payload(providers=None, options=None):
    """Build a `start_task` payload mirroring the WS protocol."""
    payload = {"prompt": "hello", "project": None}
    if providers is not None:
        payload["providers"] = providers
    if options is not None:
        payload["options"] = options
    return payload


def _run_ws_execute(payload):
    """Drive `WebSocketHandler._execute_task` once and capture the opts
    that reach `_build_server_agent`."""
    from mullet.server.websocket import WebSocketHandler

    captured = {}

    def _fake_build(_user_message, opts, on_token=None, on_tool_call=None):
        captured["opts"] = dict(opts)
        return None, None  # short-circuits the rest of _execute_task

    fake_ws = MagicMock()
    fake_ws.closed = False

    async def _send(*_a, **_kw):  # noqa: D401 — async sender stub
        return None

    fake_ws.send_json.side_effect = _send

    handler = WebSocketHandler()

    with patch("mullet.server.api._build_server_agent", side_effect=_fake_build):
        asyncio.run(handler._execute_task(fake_ws, "task-id", payload))

    return captured


def test_websocket_first_frame_providers_csv_string():
    """Red: pre-Cycle-24 the WS handler ignored top-level
    ``providers``; only ``options.providers`` was honoured. Green:
    top-level CSV string is forwarded verbatim into ``opts['providers']``
    so ``_build_server_agent`` parses it into the allowlist."""
    captured = _run_ws_execute(_ws_payload(providers="kimi,haiku"))
    assert captured["opts"].get("providers") == "kimi,haiku"


def test_websocket_first_frame_providers_list():
    """Red: only CSV strings would be accepted, breaking JSON-native
    array clients. Green: list form is forwarded as-is — the parser
    inside ``_build_server_agent`` already accepts list/tuple/set."""
    captured = _run_ws_execute(_ws_payload(providers=["kimi", "haiku"]))
    assert captured["opts"].get("providers") == ["kimi", "haiku"]


def test_websocket_top_level_providers_overrides_options():
    """Red: top-level and nested could both be set; ambiguity. Green:
    top-level wins so a single-task override doesn't require a nested
    options dict."""
    captured = _run_ws_execute(_ws_payload(
        providers="kimi",
        options={"providers": "haiku"},
    ))
    assert captured["opts"].get("providers") == "kimi"


def test_websocket_omitted_providers_leaves_options_untouched():
    """Red: a defensive write could clobber ``options.providers`` with
    ``None`` when the top-level field was omitted. Green: when no
    top-level ``providers`` is sent, the nested options pass through
    unchanged."""
    captured = _run_ws_execute(_ws_payload(options={"providers": "haiku"}))
    assert captured["opts"].get("providers") == "haiku"


# ---------------------------------------------------------------------------
# Item 3 — ObedienceError class + handler integration
# ---------------------------------------------------------------------------


def test_obedience_error_subclasses_value_error():
    """Red: a sibling ``Exception`` would break every existing
    ``except ValueError:`` catcher (CLI surface, MCP handler, router
    resolver, fallback-chain builder). Green: the new exception is a
    ``ValueError`` subclass so back-compat is preserved."""
    from mullet.core.providers.errors import ObedienceError
    assert issubclass(ObedienceError, ValueError)


def test_obedience_error_attributes_round_trip():
    """Red: required keyword args silently accepted but not stored.
    Green: ``allowed_providers``, ``filtered_providers``,
    ``requested_tier`` all round-trip through ``__init__``."""
    from mullet.core.providers.errors import ObedienceError

    err = ObedienceError(
        message="all blocked",
        allowed_providers=["kimi"],
        filtered_providers={"kimi": "context too large"},
        requested_tier="ADVANCED",
    )
    assert str(err) == "all blocked"
    assert err.allowed_providers == ["kimi"]
    assert err.filtered_providers == {"kimi": "context too large"}
    assert err.requested_tier == "ADVANCED"


def test_obedience_error_defaults_to_empty_collections():
    """Red: ``None`` defaults could surface as ``None`` and crash
    callers that iterate / serialise the fields. Green: missing
    optional fields default to ``[]`` and ``{}`` respectively."""
    from mullet.core.providers.errors import ObedienceError

    err = ObedienceError(message="x")
    assert err.allowed_providers == []
    assert err.filtered_providers == {}
    assert err.requested_tier is None


def test_obedience_error_caught_by_legacy_value_error_handler():
    """Red: a regression to ``Exception`` would slip past existing
    ``except ValueError:`` sites silently. Green: legacy catcher still
    fires."""
    from mullet.core.providers.errors import ObedienceError

    caught = False
    try:
        raise ObedienceError(message="legacy catcher")
    except ValueError:
        caught = True
    assert caught


def test_build_fallback_chain_raises_obedience_error_on_conflict(monkeypatch):
    """Red: pre-Cycle-24 the bridge raised a bare ``ValueError`` with
    only an attached ``filtered_providers`` attribute. Green: the
    raise is now an ``ObedienceError`` carrying ``allowed_providers``,
    ``filtered_providers``, ``requested_tier`` as first-class fields."""
    from mullet.core.providers.errors import ObedienceError
    from mullet.server import mcp_handlers

    monkeypatch.setenv("KIMI_API_KEY", "k")
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    never_can_serve = MagicMock()
    never_can_serve.can_serve.return_value = False
    fake_registry = MagicMock()
    fake_registry.get.return_value = never_can_serve

    with patch("mullet.core.providers.constraints.get_constraint_registry",
               return_value=fake_registry):
        with pytest.raises(ObedienceError) as excinfo:
            mcp_handlers._build_fallback_chain(
                allowed={"kimi"},
                requested_tier="ADVANCED",
                prompt_tokens=999_999_999,
            )

    err = excinfo.value
    assert err.allowed_providers == ["kimi"]
    assert "kimi" in err.filtered_providers
    assert err.requested_tier == "ADVANCED"
    # Back-compat: a `except ValueError:` handler still catches it.
    assert isinstance(err, ValueError)


def test_build_fallback_chain_empty_allowlist_raises_obedience_error():
    """Red: empty allowlist used to raise a bare ValueError lacking the
    structured fields. Green: ObedienceError with empty
    ``allowed_providers`` and the requested tier preserved."""
    from mullet.core.providers.errors import ObedienceError
    from mullet.server import mcp_handlers

    with pytest.raises(ObedienceError) as excinfo:
        mcp_handlers._build_fallback_chain(allowed=set(), requested_tier="STANDARD")
    assert excinfo.value.allowed_providers == []
    assert excinfo.value.requested_tier == "STANDARD"


def test_handle_mullet_run_response_includes_obedience_error_fields(monkeypatch):
    """Red: the bridge error response only carried ``error`` +
    ``filtered_providers``; callers had to regex-parse the message
    string to recover the allowlist. Green: the response dict carries
    structured ``allowed_providers``, ``filtered_providers``,
    ``requested_tier`` lifted directly from the ObedienceError."""
    from mullet.server import mcp_handlers

    monkeypatch.setenv("KIMI_API_KEY", "k")
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    never_can_serve = MagicMock()
    never_can_serve.can_serve.return_value = False
    fake_registry = MagicMock()
    fake_registry.get.return_value = never_can_serve

    with patch("mullet.core.providers.constraints.get_constraint_registry",
               return_value=fake_registry):
        result = mcp_handlers.handle_mullet_run({
            "task": "x" * 200,
            "providers": "kimi",
            "tier": "ADVANCED",
            "files": ["nonexistent_for_bridge_route.py"],
        })

    assert isinstance(result, dict), f"expected dict, got {type(result).__name__}"
    # On the bridge path with a kimi allowlist + always-rejecting
    # registry, the handler must surface every structured field.
    assert "error" in result
    # Either the bridge path (files list resolved) or agent path will
    # surface these — accept whichever fired by checking the key
    # presence in the dict, not the path.
    assert result.get("filtered_providers"), \
        f"expected filtered_providers in result, got {result!r}"
    assert "kimi" in result["filtered_providers"]
    assert result.get("allowed_providers") == ["kimi"], \
        f"expected allowed_providers=['kimi'] in result, got {result!r}"
    assert result.get("requested_tier") == "ADVANCED", \
        f"expected requested_tier='ADVANCED' in result, got {result!r}"
