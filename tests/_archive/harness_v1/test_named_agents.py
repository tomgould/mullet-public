"""Tests for the named-subagent registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from mullet.core.providers.base import LLMResponse, Provider
from mullet.harness.coordinator.agents import (
    DEFAULT_AGENTS_YAML,
    AgentRegistry,
    AgentSpec,
    default_registry,
)
from mullet.harness.engine.loop import HarnessEngine
from mullet.harness.hooks.registry import HookRegistry
from mullet.harness.sessions.store import SessionStore

pytestmark = [pytest.mark.harness]


# ---------------------------------------------------------------------------
# Bundled defaults
# ---------------------------------------------------------------------------


def test_loads_4_default_agents() -> None:
    registry = AgentRegistry(builtin_path=DEFAULT_AGENTS_YAML)
    agents = registry.load()
    assert set(agents.keys()) == {"sisyphus", "hephaestus", "oracle", "librarian"}
    assert all(isinstance(a, AgentSpec) for a in agents.values())


def test_get_returns_agent_spec() -> None:
    registry = AgentRegistry(builtin_path=DEFAULT_AGENTS_YAML)
    spec = registry.get("sisyphus")
    assert spec is not None
    assert spec.name == "sisyphus"
    assert spec.primary_model == "claude-opus-4-7"
    assert spec.todo_driven is True
    assert spec.extended_thinking_budget == 32000
    assert "anthropic/claude-opus-4-7" in spec.fallback_chain


def test_get_unknown_returns_none() -> None:
    registry = AgentRegistry(builtin_path=DEFAULT_AGENTS_YAML)
    assert registry.get("does-not-exist") is None


def test_oracle_is_read_only() -> None:
    registry = AgentRegistry(builtin_path=DEFAULT_AGENTS_YAML)
    spec = registry.get("oracle")
    assert spec is not None
    assert spec.read_only is True


def test_hephaestus_requires_openai() -> None:
    registry = AgentRegistry(builtin_path=DEFAULT_AGENTS_YAML)
    spec = registry.get("hephaestus")
    assert spec is not None
    assert spec.requires_provider == "openai"


def test_librarian_is_knowledge_focused() -> None:
    registry = AgentRegistry(builtin_path=DEFAULT_AGENTS_YAML)
    spec = registry.get("librarian")
    assert spec is not None
    assert spec.knowledge_focused is True


def test_list_names_sorted() -> None:
    registry = AgentRegistry(builtin_path=DEFAULT_AGENTS_YAML)
    assert registry.list_names() == sorted(registry.list_names())
    assert "sisyphus" in registry.list_names()


# ---------------------------------------------------------------------------
# Resolve chain
# ---------------------------------------------------------------------------


def test_resolve_chain_excludes_current_model() -> None:
    registry = AgentRegistry(builtin_path=DEFAULT_AGENTS_YAML)
    chain = registry.resolve_chain("sisyphus", current_model="kimi/kimi-k2.6")
    assert "kimi/kimi-k2.6" not in chain
    assert "kimi/kimi-k2.5" in chain  # rest of the chain preserved


def test_resolve_chain_unknown_agent_returns_empty() -> None:
    registry = AgentRegistry(builtin_path=DEFAULT_AGENTS_YAML)
    assert registry.resolve_chain("does-not-exist") == []


def test_resolve_chain_no_current_model_returns_full_chain() -> None:
    registry = AgentRegistry(builtin_path=DEFAULT_AGENTS_YAML)
    spec = registry.get("librarian")
    assert spec is not None
    chain = registry.resolve_chain("librarian")
    assert chain == spec.fallback_chain


# ---------------------------------------------------------------------------
# User overlay
# ---------------------------------------------------------------------------


def test_user_overlay_overrides_default(tmp_path: Path) -> None:
    user_yaml = tmp_path / "agents.yaml"
    user_yaml.write_text(
        "agents:\n"
        "  sisyphus:\n"
        "    role: 'Custom orchestrator override'\n"
        "    primary_model: 'kimi/kimi-k2.6'\n"
        "    fallback_chain:\n"
        "      - 'kimi/kimi-k2.6'\n"
        "      - 'openai/gpt-4o-mini'\n"
        "    todo_driven: false\n"
    )
    registry = AgentRegistry(builtin_path=DEFAULT_AGENTS_YAML, user_path=user_yaml)
    spec = registry.get("sisyphus")
    assert spec is not None
    assert spec.primary_model == "kimi/kimi-k2.6"
    assert spec.role == "Custom orchestrator override"
    assert spec.todo_driven is False
    # Other defaults still present.
    assert registry.get("hephaestus") is not None


def test_user_overlay_can_add_new_agent(tmp_path: Path) -> None:
    user_yaml = tmp_path / "agents.yaml"
    user_yaml.write_text(
        "agents:\n"
        "  newcomer:\n"
        "    role: 'a brand new agent'\n"
        "    primary_model: 'kimi/kimi-k2.6'\n"
    )
    registry = AgentRegistry(builtin_path=DEFAULT_AGENTS_YAML, user_path=user_yaml)
    assert registry.get("newcomer") is not None
    assert "newcomer" in registry.list_names()


def test_missing_user_overlay_is_silently_ignored(tmp_path: Path) -> None:
    user_yaml = tmp_path / "missing.yaml"
    registry = AgentRegistry(builtin_path=DEFAULT_AGENTS_YAML, user_path=user_yaml)
    assert "sisyphus" in registry.list_names()


def test_default_registry_module_singleton_loads() -> None:
    # The module-level registry should at minimum surface the bundled
    # defaults — user overlay may or may not be present.
    names = default_registry.list_names()
    assert "sisyphus" in names
    assert "hephaestus" in names
    assert "oracle" in names
    assert "librarian" in names


# ---------------------------------------------------------------------------
# Engine wiring
# ---------------------------------------------------------------------------


class _MockProvider(Provider):
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def create(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        reasoning_config: dict | None = None,
    ) -> LLMResponse:
        self.calls.append((messages, tools))
        return LLMResponse(content="OK", finish_reason="stop")


def test_engine_run_with_agent_uses_chain(tmp_path, monkeypatch) -> None:
    mock = _MockProvider()
    captured: dict = {}

    def fake_resolve(name, model=None):
        captured["name"] = name
        captured["model"] = model
        return mock

    monkeypatch.setattr(
        "mullet.harness.engine.loop._resolve_provider",
        fake_resolve,
    )
    store = SessionStore(base_dir=tmp_path / "sessions")
    engine = HarnessEngine(store=store, hooks=HookRegistry())
    result = engine.run("hello", agent="sisyphus")

    assert result["response"] == "OK"
    # Sisyphus's primary_model should propagate as the resolved model.
    assert captured["model"] == "claude-opus-4-7"


def test_engine_run_with_unknown_agent_raises(tmp_path) -> None:
    store = SessionStore(base_dir=tmp_path / "sessions")
    engine = HarnessEngine(store=store, hooks=HookRegistry())
    with pytest.raises(ValueError, match="Unknown agent 'ghost'"):
        engine.run("hi", agent="ghost")


def test_engine_run_with_explicit_model_overrides_agent_primary(
    tmp_path, monkeypatch
) -> None:
    mock = _MockProvider()
    captured: dict = {}

    def fake_resolve(name, model=None):
        captured["model"] = model
        return mock

    monkeypatch.setattr(
        "mullet.harness.engine.loop._resolve_provider",
        fake_resolve,
    )
    store = SessionStore(base_dir=tmp_path / "sessions")
    engine = HarnessEngine(store=store, hooks=HookRegistry())
    engine.run("hi", agent="sisyphus", model="kimi/kimi-k2.5")
    assert captured["model"] == "kimi/kimi-k2.5"
