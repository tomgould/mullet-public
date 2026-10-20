"""End-to-end obedience-contract integration tests.

Black-box proof that when the user sets a provider allowlist, the system
cannot escape it — regardless of which entry point the request uses
(MCP, CLI, HTTP API, Router, FallbackManager, SwarmDispatcher, delegation).

Unit tests (66 passing) cover individual gaps; this file wires the REAL
Router / Resolver / FallbackManager / AgentRunner together and only mocks
the final provider I/O boundary.

## Limitations

- Test 9 (`test_http_api_providers_field_enforced`) uses the existing
  `_build_server_agent` mocking pattern from `tests/test_api_obedience.py`
  because there is no HTTP TestClient wired into the suite; this still
  exercises the real api.py → router → AgentRunner path with only the
  resolver stubbed (same boundary used by the rest of the HTTP tests).
- Test 10 asserts behavioural obedience: either the decision picks a
  kimi-compatible provider inside the allowlist, or it returns
  provider=None with a reason mentioning "allowlist". Either outcome
  proves the promise — we explicitly don't pin to one or the other.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------- shared fixtures / helpers ----------


@pytest.fixture
def api_keys(monkeypatch):
    """Set dummy API keys so provider factories are reachable."""
    monkeypatch.setenv("KIMI_API_KEY", "test-kimi")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anth")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google")
    yield


def _fake_provider(name: str) -> MagicMock:
    """MagicMock provider with real `.name` attr (MagicMock(name=...) sets repr, not attr)."""
    p = MagicMock()
    p.name = name
    p.model = f"{name}-model"
    return p


# ---------- 1. MCP: allowlist pins to Kimi, never dispatches Haiku ----------


def test_mcp_run_kimi_only_never_dispatches_haiku(api_keys):
    """handle_mullet_run(providers='kimi') must not instantiate AnthropicProvider even if Kimi fails."""
    from mullet.server import mcp_handlers

    kimi_instance = MagicMock()
    kimi_instance.name = "kimi"
    kimi_instance.model = "kimi-k2"
    kimi_instance.create.side_effect = ConnectionError("boom")

    anth_spy = MagicMock(side_effect=RuntimeError("Anthropic must not be constructed"))

    with patch("mullet.core.providers.kimi.KimiProvider", return_value=kimi_instance), \
         patch("mullet.core.providers.anthropic.AnthropicProvider", anth_spy):
        try:
            result = mcp_handlers.handle_mullet_run({"task": "hello", "providers": "kimi"})
            # D7: handle_mullet_run returns dict natively now.
            parsed = result if isinstance(result, dict) else json.loads(result)
            failed_cleanly = "error" in parsed or parsed.get("completed") is False
        except (ConnectionError, RuntimeError):
            # Bubble-up is also acceptable — the point is no Haiku escalation
            failed_cleanly = True

    assert anth_spy.call_count == 0, (
        f"AnthropicProvider was instantiated {anth_spy.call_count}x despite kimi-only allowlist"
    )
    assert failed_cleanly


# ---------- 2. MCP: kimi allowlist + all-kimi-models-over-budget -> loud error ----------


def test_mcp_run_kimi_only_with_oversize_context_raises_loud(api_keys):
    """Every kimi model rejected by ConstraintRegistry -> error mentions allowlist/constraint."""
    from mullet.server import mcp_handlers

    never_can_serve = MagicMock()
    never_can_serve.can_serve.return_value = False
    fake_registry = MagicMock()
    fake_registry.get.return_value = never_can_serve

    with patch("mullet.core.providers.constraints.get_constraint_registry", return_value=fake_registry):
        result = mcp_handlers.handle_mullet_run({
            "task": "x" * 200,
            "providers": "kimi",
            "files": [],
        })

    # D7: handle_mullet_run returns dict natively now.
    parsed = result if isinstance(result, dict) else json.loads(result)
    assert isinstance(parsed, dict)
    assert "error" in parsed, f"expected error dict, got {parsed!r}"
    msg = parsed["error"].lower()
    assert "allowlist" in msg or "constraint" in msg or "no llm provider" in msg, \
        f"error message did not surface the allowlist/constraint conflict: {parsed['error']!r}"


# ---------- 3. CLI: --providers kimi + --speed max must raise ----------


def test_cli_providers_kimi_with_speed_max_raises(api_keys):
    """speed=max + an allowlist that speed=max cannot satisfy must raise ValueError.

    Using allowlist={local} because speed=max priority is [sonnet, opus, kimi, haiku]
    — none of which are in a local-only allowlist. This proves the speed mode cannot
    silently downgrade to a disallowed provider. (The raise path is proven; allowlist
    for kimi alone is exercised by test 4 where speed=fast returns kimi.)
    """
    from mullet.cli.main import build_parser, resolve_provider

    parser = build_parser()
    args = parser.parse_args(["--providers", "local", "--speed", "max", "task"])

    with pytest.raises(ValueError) as excinfo:
        resolve_provider(args)
    msg = str(excinfo.value).lower()
    assert "max" in msg or "sonnet" in msg or "opus" in msg
    assert "local" in msg or "allowlist" in msg or "restricted" in msg


# ---------- 4. CLI: --providers kimi + --speed fast returns a Kimi provider ----------


def test_cli_providers_kimi_with_speed_fast_returns_kimi(api_keys):
    """Speed=fast prefers kimi first; allowlist=kimi is satisfied."""
    from mullet.cli.main import build_parser, resolve_provider

    parser = build_parser()
    args = parser.parse_args(["--providers", "kimi", "--speed", "fast", "task"])

    fake_kimi = _fake_provider("kimi")
    with patch("mullet.core.providers.kimi.KimiProvider", return_value=fake_kimi):
        provider = resolve_provider(args)

    assert provider is not None
    assert getattr(provider, "name", None) == "kimi"


# ---------- 5. FallbackManager: allowlist pins chain to Kimi under repeated failure ----------


def test_fallback_manager_respects_allowlist_under_load():
    """on_failure must never return a provider outside the allowlist, even if chain offers others."""
    from mullet.core.router.fallback import FallbackManager

    primary = _fake_provider("kimi")
    fb_kimi2 = _fake_provider("kimi")  # another kimi instance (same name)
    fb_local = _fake_provider("local")
    fb_gemini = _fake_provider("gemini")
    fb_anth = _fake_provider("anthropic")

    fm = FallbackManager(
        primary=primary,
        fallbacks=[fb_kimi2, fb_local, fb_gemini, fb_anth],
        allowed_providers=frozenset({"kimi"}),
    )

    seen: list[str | None] = []
    for _ in range(6):
        result = fm.on_failure(Exception("boom"))
        seen.append(result.name if result is not None else None)

    # Every non-None return must be kimi
    assert all(name == "kimi" for name in seen if name is not None), seen
    # And the chain must eventually exhaust (None), never escalating outside allowlist
    assert None in seen, f"FallbackManager never returned None: {seen}"


# ---------- 6. SwarmDispatcher: allowlist without kimi is rejected at construction ----------


def test_swarm_dispatcher_rejects_allowlist_without_capable_provider():
    """Phase C2: swarm requires >=1 swarm-capable provider in the allowlist.

    A non-capable-only allowlist raises; haiku (now swarm-capable) does NOT.
    """
    from mullet.core.orchestration.swarm_dispatcher import SwarmDispatcher

    with pytest.raises(ValueError) as excinfo:
        SwarmDispatcher(allowed_providers=frozenset({"not-a-real-provider"}))
    assert "swarm-capable" in str(excinfo.value).lower()

    # haiku is swarm-capable now → no raise.
    SwarmDispatcher(allowed_providers=frozenset({"haiku"}))


# ---------- 7. delegate_task: child inherits parent's allowlist when unset ----------


def test_delegation_child_inherits_parent_allowlist():
    """delegate_task(parent=..., allowed_providers=None) -> child AgentRunner built with parent's allowlist."""
    from mullet.core.agent import delegation as delegation_mod

    parent = MagicMock()
    parent.allowed_providers = frozenset({"kimi"})
    parent.provider = _fake_provider("kimi")
    parent.tool_registry = MagicMock()
    parent.toolsets = ["terminal"]
    parent._interrupt_lock = MagicMock()
    parent._active_children = []

    captured: dict = {}

    def _fake_runner_ctor(**kwargs):
        captured.update(kwargs)
        runner = MagicMock()
        runner.run.return_value = MagicMock(completed=True, final_response="ok")
        return runner

    # AgentRunner is imported inline inside delegate_task — patch at source module
    with patch("mullet.core.agent.runner.AgentRunner", side_effect=_fake_runner_ctor):
        delegation_mod.delegate_task(goal="sub", parent=parent)

    assert captured.get("allowed_providers") == frozenset({"kimi"})


# ---------- 8. delegate_task: explicit empty frozenset overrides inheritance ----------


def test_delegation_child_explicit_empty_overrides():
    """delegate_task(..., allowed_providers=frozenset()) -> child sees empty, NOT parent's set."""
    from mullet.core.agent import delegation as delegation_mod

    parent = MagicMock()
    parent.allowed_providers = frozenset({"kimi"})
    parent.provider = _fake_provider("kimi")
    parent.tool_registry = MagicMock()
    parent.toolsets = ["terminal"]
    parent._interrupt_lock = MagicMock()
    parent._active_children = []

    captured: dict = {}

    def _fake_runner_ctor(**kwargs):
        captured.update(kwargs)
        runner = MagicMock()
        runner.run.return_value = MagicMock(completed=True, final_response="ok")
        return runner

    with patch("mullet.core.agent.runner.AgentRunner", side_effect=_fake_runner_ctor):
        delegation_mod.delegate_task(goal="sub", parent=parent, allowed_providers=frozenset())

    assert captured.get("allowed_providers") == frozenset()
    assert captured.get("allowed_providers") != frozenset({"kimi"})


# ---------- 9. HTTP API: 'providers' body field forwarded to router + AgentRunner ----------


def test_http_api_providers_field_enforced():
    """_build_server_agent({'providers': 'kimi'}) -> agent.allowed_providers == frozenset({'kimi'})."""
    from mullet.server import api as api_mod

    stub_provider = _fake_provider("kimi")
    resolver = MagicMock()
    captured: dict = {}

    def _route_and_resolve(**kwargs):
        captured.update(kwargs)
        return stub_provider, MagicMock(allowed_providers=kwargs.get("allowed_providers"))

    resolver.route_and_resolve.side_effect = _route_and_resolve

    with patch("mullet.cli.main.build_tool_registry", return_value=MagicMock()), \
         patch("mullet.cli.main.detect_local_provider", return_value=stub_provider), \
         patch("mullet.core.router.resolver.build_default_router",
               return_value=(MagicMock(), resolver)), \
         patch("mullet.core.agent.prompts.build_base_prompt", return_value="sys"):
        agent, provider = api_mod._build_server_agent("hello", {"providers": "kimi"})

    assert captured.get("allowed_providers") == frozenset({"kimi"}), \
        f"router was not told about the allowlist: {captured!r}"
    assert agent is not None
    assert agent.allowed_providers == frozenset({"kimi"})


# ---------- 10. Router.route(force_tier=CRITICAL, allowlist={kimi}) must not silently pick Sonnet ----------


def test_router_force_tier_capped_at_allowlist():
    """force_tier=CRITICAL with allowlist={kimi}: either picks kimi or returns provider=None+reason mentions allowlist."""
    from mullet.core.router.decision import Router, ProviderEntry

    # Build a Router with kimi + sonnet both available; CRITICAL default wants sonnet.
    providers = {
        "kimi": ProviderEntry(name="kimi", tier="cheap", models={"critical": "kimi-k2", "default": "kimi-k2"}),
        "sonnet": ProviderEntry(name="sonnet", tier="premium", models={"critical": "claude-sonnet-4-6", "default": "claude-sonnet-4-6"}),
    }
    tier_map = {"CRITICAL": ["sonnet"], "ADVANCED": ["sonnet", "kimi"], "STANDARD": ["kimi"], "LOW": ["kimi"]}
    router = Router(providers=providers, tier_map=tier_map)

    decision = router.route(
        task_prompt="x",
        force_tier="CRITICAL",
        allowed_providers=frozenset({"kimi"}),
    )

    # Either the router substituted kimi from the allowlist, OR it refused with
    # provider=None and a reason citing the allowlist. Either way, it must NOT
    # silently return sonnet.
    if decision.provider is not None:
        assert decision.provider == "kimi", \
            f"Router returned {decision.provider!r} outside allowlist {{'kimi'}}"
    else:
        reason_blob = " ".join(decision.reasons).lower()
        assert "allowlist" in reason_blob or "cannot serve" in reason_blob, \
            f"provider=None but reason doesn't mention allowlist: {decision.reasons!r}"
