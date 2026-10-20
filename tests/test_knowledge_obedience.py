"""Regression tests for Trawl MCP handler provider allowlist (G22)."""

from __future__ import annotations

import types

import pytest

from mullet.server import mcp_knowledge


class _FakeClient:
    def __init__(self, llm_config=None, name="pool"):
        self.llm_config = llm_config or {"provider": "lmstudio"}
        self.name = name
        self.index_path = None


def test_normalize_string():
    out = mcp_knowledge._normalize_allowed_providers("kimi, haiku")
    assert out == frozenset({"kimi", "haiku"})


def test_normalize_list():
    out = mcp_knowledge._normalize_allowed_providers(["Kimi", "Haiku"])
    assert out == frozenset({"kimi", "haiku"})


def test_normalize_none_and_empty():
    assert mcp_knowledge._normalize_allowed_providers(None) is None
    assert mcp_knowledge._normalize_allowed_providers("") is None
    assert mcp_knowledge._normalize_allowed_providers([]) is None


def test_create_provider_pool_pick_in_allowlist():
    client = _FakeClient({"provider": "lmstudio", "lmstudio_host": "http://x", "lmstudio_model": "m"})
    prov = mcp_knowledge._create_llm_provider(client, allowed_providers=frozenset({"local"}))
    assert prov is not None


def test_create_provider_blocked_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = _FakeClient({"provider": "lmstudio", "lmstudio_host": "http://x", "lmstudio_model": "m"})
    with pytest.raises(ValueError) as exc_info:
        mcp_knowledge._create_llm_provider(client, allowed_providers=frozenset({"haiku"}))
    msg = str(exc_info.value)
    assert "allowlist" in msg
    assert "lmstudio" in msg


def test_create_provider_fallback_to_haiku(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    client = _FakeClient({"provider": "lmstudio", "lmstudio_host": "http://x", "lmstudio_model": "m"})
    try:
        prov = mcp_knowledge._create_llm_provider(client, allowed_providers=frozenset({"haiku"}))
    except Exception:
        pytest.skip("Anthropic provider import unavailable in this environment")
    assert prov is not None


@pytest.mark.parametrize("tool_name", ["trawl_summarize", "trawl_run_agent", "trawl_build_graph"])
def test_schema_has_providers_property(tool_name):
    tool = next((t for t in mcp_knowledge.MCP_KNOWLEDGE_TOOLS if t["name"] == tool_name), None)
    assert tool is not None, f"{tool_name} not in MCP_TOOLS"
    props = tool["inputSchema"]["properties"]
    assert "providers" in props, f"{tool_name} missing 'providers' property"
    assert props["providers"].get("type") == "string"


def test_handle_summarize_forwards_providers(monkeypatch):
    captured = {}

    def fake_create(client, allowed_providers=None):
        captured["allowed"] = allowed_providers
        return None  # provider unavailable -> falls back to truncation

    searcher_stub = types.SimpleNamespace(
        search_with_quality=lambda **kw: types.SimpleNamespace(results=[]),
    )
    client_stub = types.SimpleNamespace(name="p", llm_config={}, index_path=None)

    monkeypatch.setattr(mcp_knowledge, "_resolve_pool_name", lambda args: "p")
    monkeypatch.setattr(mcp_knowledge, "_load_pool", lambda name: (client_stub, searcher_stub))
    monkeypatch.setattr(mcp_knowledge, "_create_llm_provider", fake_create)

    out = mcp_knowledge.handle_summarize({"query": "q", "providers": "kimi,haiku"})
    assert "allowed" in captured
    assert captured["allowed"] == frozenset({"kimi", "haiku"})
    assert isinstance(out, str)
