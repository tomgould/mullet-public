"""Tests for mullet.core.router.resolver — comprehensive unit tests."""

from __future__ import annotations

import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mullet.core.router.complexity import Tier
from mullet.core.router.decision import ProviderEntry, Router, RoutingDecision
from mullet.core.router.resolver import ProviderResolver, _pricing, build_default_router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_router(providers: dict[str, ProviderEntry] | None = None) -> Router:
    """Build a minimal Router for testing."""
    if providers is None:
        providers = {
            "local": ProviderEntry(name="local", tier="free",
                                   models={"default": "llama-8b"}),
            "cheap_cloud": ProviderEntry(name="kimi", tier="cheap",
                                         models={"default": "kimi-k2.7-code"}),
            "premium_cloud": ProviderEntry(name="anthropic", tier="premium",
                                           models={"default": "claude-sonnet-4-6"}),
        }
    return Router(providers=providers)


def _make_decision(
    provider: str = "local",
    model: str = "llama-8b",
    tier: Tier = Tier.LOW,
    fallback_chain: list[str] | None = None,
    allowed_providers: frozenset[str] | None = None,
) -> RoutingDecision:
    return RoutingDecision(
        provider=provider,
        model=model,
        tier=tier,
        fallback_chain=fallback_chain or [],
        allowed_providers=allowed_providers,
    )


# ---------------------------------------------------------------------------
# ProviderResolver.__init__
# ---------------------------------------------------------------------------


def test_resolver_init():
    router = _make_router()
    resolver = ProviderResolver(router)
    assert resolver.router is router
    assert resolver._provider_cache == {}


# ---------------------------------------------------------------------------
# ProviderResolver.resolve
# ---------------------------------------------------------------------------


@patch("mullet.core.router.resolver.OpenAICompatibleProvider")
def test_resolve_cache_miss_and_create(mock_oai, tmp_path):
    router = _make_router()
    resolver = ProviderResolver(router)
    decision = _make_decision(provider="local", model="llama-8b", tier=Tier.LOW)
    mock_inst = MagicMock()
    mock_oai.return_value = mock_inst

    result = resolver.resolve(decision)
    assert result is mock_inst
    assert "local:LOW" in resolver._provider_cache
    mock_oai.assert_called_once_with(base_url="http://localhost:1234/v1", model="llama-8b")


@patch("mullet.core.router.resolver.OpenAICompatibleProvider")
def test_resolve_cache_hit(mock_oai):
    router = _make_router()
    resolver = ProviderResolver(router)
    cached = MagicMock()
    resolver._provider_cache["local:LOW"] = cached
    decision = _make_decision(provider="local", model="llama-8b", tier=Tier.LOW)

    result = resolver.resolve(decision)
    assert result is cached
    mock_oai.assert_not_called()


def test_resolve_missing_provider():
    router = _make_router()
    resolver = ProviderResolver(router)
    decision = _make_decision(provider="missing")
    assert resolver.resolve(decision) is None


# ---------------------------------------------------------------------------
# ProviderResolver.route_and_resolve
# ---------------------------------------------------------------------------


@patch("mullet.core.routing_history.append_decision_log")
@patch("mullet.core.router.resolver.OpenAICompatibleProvider")
def test_route_and_resolve_primary_success(mock_oai, mock_log):
    router = _make_router()
    resolver = ProviderResolver(router)
    mock_inst = MagicMock()
    mock_inst.model = "llama-8b"
    mock_oai.return_value = mock_inst

    provider, out_decision = resolver.route_and_resolve("hello world")
    assert provider is mock_inst
    assert out_decision.provider == "local"
    mock_log.assert_called_once()


@patch("mullet.core.routing_history.append_decision_log")
@patch("mullet.core.router.resolver.OpenAICompatibleProvider")
def test_route_and_resolve_primary_none_fallback_chain(mock_oai, mock_log):
    """Primary provider creation fails, fallback chain succeeds."""
    router = _make_router()
    resolver = ProviderResolver(router)
    # Patch router.route to return a decision with a fallback chain
    decision = _make_decision(
        provider="local",
        tier=Tier.LOW,
        fallback_chain=["cheap_cloud"],
    )
    with patch.object(router, "route", return_value=decision):
        with patch.object(resolver, "_create_provider", side_effect=[None, MagicMock()]):
            provider, out_decision = resolver.route_and_resolve("task")
            assert provider is not None
            assert out_decision.provider == "cheap_cloud"
            assert any("fallback" in r for r in out_decision.reasons)


@patch("mullet.core.routing_history.append_decision_log", side_effect=RuntimeError("boom"))
@patch("mullet.core.router.resolver.OpenAICompatibleProvider")
def test_route_and_resolve_decision_log_failure(mock_oai, mock_log):
    """Decision log append failure should be caught, not raised."""
    router = _make_router()
    resolver = ProviderResolver(router)
    mock_inst = MagicMock()
    mock_inst.model = "llama-8b"
    mock_oai.return_value = mock_inst

    provider, out_decision = resolver.route_and_resolve("hello")
    assert provider is mock_inst


@patch("mullet.core.routing_history.append_decision_log")
@patch("mullet.core.router.resolver.OpenAICompatibleProvider")
def test_route_and_resolve_with_allowed_providers(mock_oai, mock_log):
    router = _make_router()
    resolver = ProviderResolver(router)
    mock_inst = MagicMock()
    mock_inst.model = "llama-8b"
    mock_oai.return_value = mock_inst

    provider, decision = resolver.route_and_resolve(
        "task", allowed_providers=frozenset({"local"})
    )
    assert provider is mock_inst
    assert decision.allowed_providers == frozenset({"local"})


# ---------------------------------------------------------------------------
# ProviderResolver._create_provider — local / lmstudio / ollama
# ---------------------------------------------------------------------------


@patch("mullet.core.providers.openai_compat.is_escalation_recommended", return_value=False)
@patch("mullet.core.router.resolver.OpenAICompatibleProvider")
def test_create_provider_local_basic(mock_oai, mock_esc):
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="local", tier="free", models={"default": "llama-8b"})
    resolver._create_provider(entry, model="llama-8b")
    mock_oai.assert_called_once_with(base_url="http://localhost:1234/v1", model="llama-8b")


@patch("mullet.core.providers.openai_compat.is_escalation_recommended", return_value=True)
@patch("mullet.core.router.resolver.OpenAICompatibleProvider")
def test_create_provider_local_escalation_skipped(mock_oai, mock_esc):
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="local", tier="free", models={"default": "llama-8b"})
    result = resolver._create_provider(entry, model="llama-8b")
    assert result is None
    mock_oai.assert_not_called()


@patch("mullet.core.providers.lmstudio_manager.LMStudioManager")
@patch("mullet.core.providers.openai_compat.is_escalation_recommended", return_value=False)
@patch("mullet.core.router.resolver.OpenAICompatibleProvider")
def test_create_provider_local_lmstudio_manager(mock_oai, mock_esc, mock_mgr_cls):
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="lmstudio", tier="free", models={"default": "default"})
    mock_mgr = MagicMock()
    mock_mgr.pick_best_model.return_value = {"id": "picked-model"}
    mock_mgr_cls.return_value = mock_mgr

    resolver._create_provider(entry, model="default")
    mock_oai.assert_called_once_with(base_url="http://localhost:1234/v1", model="picked-model")


@patch("mullet.core.providers.lmstudio_manager.LMStudioManager", side_effect=ImportError)
@patch("mullet.core.providers.openai_compat.is_escalation_recommended", return_value=False)
@patch("mullet.core.router.resolver.OpenAICompatibleProvider")
def test_create_provider_local_lmstudio_manager_import_error(mock_oai, mock_esc, mock_mgr):
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="ollama", tier="free", models={"default": "llama-8b"})
    resolver._create_provider(entry, model="default")
    mock_oai.assert_called_once_with(base_url="http://localhost:1234/v1", model="default")


@patch("mullet.core.providers.openai_compat.is_escalation_recommended", return_value=False)
@patch("mullet.core.router.resolver.OpenAICompatibleProvider")
def test_create_provider_local_custom_base_url(mock_oai, mock_esc):
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="local", tier="free", models={"default": "llama-8b"})
    entry.base_url = "http://custom:8080/v1"
    resolver._create_provider(entry, model="llama-8b")
    mock_oai.assert_called_once_with(base_url="http://custom:8080/v1", model="llama-8b")


# ---------------------------------------------------------------------------
# ProviderResolver._create_provider — anthropic
# ---------------------------------------------------------------------------


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False)
@patch("mullet.core.providers.model_registry.get_registry")
@patch("mullet.core.providers.anthropic.AnthropicProvider")
def test_create_provider_anthropic_with_api_key(mock_anthropic, mock_reg):
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="anthropic", tier="premium",
                          models={"default": "claude-sonnet-4-6"})
    mock_anthropic.return_value = MagicMock()
    result = resolver._create_provider(entry, model="claude-sonnet-4-6")
    assert result is mock_anthropic.return_value
    mock_anthropic.assert_called_once_with(api_key="sk-ant-test", model="claude-sonnet-4-6")


@patch.dict(os.environ, {}, clear=True)
@patch("mullet.core.providers.model_registry.get_registry")
@patch("mullet.core.providers.anthropic.AnthropicProvider")
def test_create_provider_anthropic_no_auth(mock_anthropic, mock_reg):
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="anthropic", tier="premium",
                          models={"default": "claude-sonnet-4-6"})
    result = resolver._create_provider(entry, model="default")
    assert result is None


# ---------------------------------------------------------------------------
# ProviderResolver._create_provider — haiku / sonnet / opus aliases
# ---------------------------------------------------------------------------


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False)
@patch("mullet.core.providers.anthropic.AnthropicProvider")
def test_create_provider_haiku_with_api_key(mock_anthropic):
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="haiku", tier="cheap",
                          models={"default": "claude-haiku-4-5"})
    mock_anthropic.return_value = MagicMock()
    result = resolver._create_provider(entry, model="claude-haiku-4-5")
    assert result is mock_anthropic.return_value
    mock_anthropic.assert_called_once_with(api_key="sk-ant-test", model="claude-haiku-4-5")


@patch.dict(os.environ, {}, clear=True)
@patch("mullet.core.providers.anthropic.AnthropicProvider")
def test_create_provider_opus_no_auth(mock_anthropic):
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="opus", tier="premium",
                          models={"default": "claude-opus-4-8"})
    assert resolver._create_provider(entry, model="claude-opus-4-8") is None


@patch.dict(os.environ, {}, clear=True)
@patch("mullet.core.providers.anthropic.AnthropicProvider")
def test_create_provider_sonnet_no_auth(mock_anthropic):
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="sonnet", tier="premium",
                          models={"default": "claude-sonnet-4-6"})
    assert resolver._create_provider(entry, model="claude-sonnet-4-6") is None


# ---------------------------------------------------------------------------
# ProviderResolver._create_provider — gemini
# ---------------------------------------------------------------------------


@patch.dict(os.environ, {"GOOGLE_API_KEY": "gkey"}, clear=True)
@patch("mullet.core.providers.gemini.GeminiProvider")
def test_create_provider_gemini_with_key(mock_gemini):
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="gemini", tier="cheap",
                          models={"default": "gemini-2.5-flash"})
    mock_gemini.return_value = MagicMock()
    result = resolver._create_provider(entry, model="default")
    assert result is mock_gemini.return_value


@patch.dict(os.environ, {"GEMINI_API_KEY": "gkey2"}, clear=True)
@patch("mullet.core.providers.gemini.GeminiProvider")
def test_create_provider_gemini_alt_env(mock_gemini):
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="gemini", tier="cheap",
                          models={"default": "gemini-2.5-flash"})
    mock_gemini.return_value = MagicMock()
    result = resolver._create_provider(entry, model="default")
    assert result is mock_gemini.return_value


@patch.dict(os.environ, {"GOOGLE_API_KEY": "gkey"}, clear=True)
@patch("mullet.core.providers.gemini.GeminiProvider", side_effect=ImportError)
def test_create_provider_gemini_import_error(mock_gemini):
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="gemini", tier="cheap",
                          models={"default": "gemini-2.5-flash"})
    result = resolver._create_provider(entry, model="default")
    assert result is None


@patch.dict(os.environ, {}, clear=True)
@patch("mullet.core.providers.gemini.GeminiProvider")
def test_create_provider_gemini_no_key(mock_gemini):
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="gemini", tier="cheap",
                          models={"default": "gemini-2.5-flash"})
    assert resolver._create_provider(entry, model="default") is None


# ---------------------------------------------------------------------------
# ProviderResolver._create_provider — kimi
# ---------------------------------------------------------------------------


@patch.dict(os.environ, {"KIMI_API_KEY": "kkey"}, clear=True)
@patch("mullet.core.providers.kimi.KimiProvider")
def test_create_provider_kimi_with_key(mock_kimi):
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="kimi", tier="cheap",
                          models={"default": "kimi-k2.7-code"})
    mock_kimi.return_value = MagicMock()
    result = resolver._create_provider(entry, model="default")
    assert result is mock_kimi.return_value


@patch.dict(os.environ, {"MOONSHOT_API_KEY": "mkey"}, clear=True)
@patch("mullet.core.providers.kimi.KimiProvider")
def test_create_provider_kimi_moonshot_env(mock_kimi):
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="kimi", tier="cheap",
                          models={"default": "kimi-k2.7-code"})
    mock_kimi.return_value = MagicMock()
    result = resolver._create_provider(entry, model="default")
    assert result is mock_kimi.return_value


@patch.dict(os.environ, {"KIMI_API_KEY": "kkey"}, clear=True)
@patch("mullet.core.providers.kimi.KimiProvider", side_effect=ImportError)
def test_create_provider_kimi_import_error(mock_kimi):
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="kimi", tier="cheap",
                          models={"default": "kimi-k2.7-code"})
    result = resolver._create_provider(entry, model="default")
    assert result is None


@patch.dict(os.environ, {}, clear=True)
def test_create_provider_kimi_no_key():
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="kimi", tier="cheap",
                          models={"default": "kimi-k2.7-code"})
    assert resolver._create_provider(entry, model="default") is None


# ---------------------------------------------------------------------------
# ProviderResolver._create_provider — openrouter
# ---------------------------------------------------------------------------


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "orkey"}, clear=True)
@patch("mullet.core.providers.openrouter.OpenRouterProvider")
def test_create_provider_openrouter(mock_or):
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="openrouter", tier="cheap",
                          models={"default": "deepseek/deepseek-v4-flash"})
    mock_or.return_value = MagicMock()
    result = resolver._create_provider(entry, model="default")
    assert result is mock_or.return_value
    mock_or.assert_called_once_with(model="deepseek/deepseek-v4-flash", api_key="orkey")


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "orkey"}, clear=True)
@patch("mullet.core.router.resolver.OpenAICompatibleProvider")
@patch("mullet.core.providers.openrouter.OpenRouterProvider", side_effect=Exception("fail"))
def test_create_provider_openrouter_fallback_oai(mock_or, mock_oai):
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="openrouter", tier="cheap",
                          models={"default": "deepseek/deepseek-v4-flash"})
    mock_oai.return_value = MagicMock()
    result = resolver._create_provider(entry, model="default")
    assert result is mock_oai.return_value
    mock_oai.assert_called_once_with(
        base_url="https://openrouter.ai/api/v1", api_key="orkey", model="deepseek/deepseek-v4-flash"
    )


@patch.dict(os.environ, {}, clear=True)
def test_create_provider_openrouter_no_key():
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="openrouter", tier="cheap",
                          models={"default": "deepseek/deepseek-v4-flash"})
    assert resolver._create_provider(entry, model="default") is None


# ---------------------------------------------------------------------------
# ProviderResolver._create_provider — generic OpenAI-compatible
# ---------------------------------------------------------------------------


@patch("mullet.core.router.resolver.OpenAICompatibleProvider")
def test_create_provider_generic_openai(mock_oai):
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="custom", tier="premium",
                          models={"default": "gpt-4"})
    entry.base_url = "https://api.custom.com/v1"
    mock_oai.return_value = MagicMock()
    result = resolver._create_provider(entry, model="gpt-4")
    assert result is mock_oai.return_value
    mock_oai.assert_called_once_with(base_url="https://api.custom.com/v1", api_key="no-key-required", model="gpt-4")


@patch("mullet.core.router.resolver.OpenAICompatibleProvider")
def test_create_provider_generic_openai_with_api_key_env(mock_oai):
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="custom", tier="premium",
                          models={"default": "gpt-4"})
    entry.base_url = "https://api.custom.com/v1"
    entry.api_key_env = "CUSTOM_API_KEY"
    with patch.dict(os.environ, {"CUSTOM_API_KEY": "ckey"}, clear=False):
        mock_oai.return_value = MagicMock()
        result = resolver._create_provider(entry, model="gpt-4")
        assert result is mock_oai.return_value
        mock_oai.assert_called_once_with(base_url="https://api.custom.com/v1", api_key="ckey", model="gpt-4")


def test_create_provider_unknown_entry():
    """Provider names that don't match any branch should return None."""
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="unknown_provider", tier="cheap",
                          models={"default": "model-x"})
    assert resolver._create_provider(entry, model="model-x") is None


# ---------------------------------------------------------------------------
# ProviderResolver._create_provider — _dynamic_model
# ---------------------------------------------------------------------------


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key"}, clear=False)
@patch("mullet.core.providers.anthropic.AnthropicProvider")
@patch("mullet.core.providers.model_registry.get_registry")
def test_dynamic_model_uses_registry(mock_reg, mock_anthropic):
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="anthropic", tier="premium",
                          models={"default": "claude-sonnet-4-6"})
    mock_registry = MagicMock()
    mock_registry.get_best_model.return_value = "dynamic-model"
    mock_reg.return_value = mock_registry

    mock_anthropic.return_value = MagicMock()
    result = resolver._create_provider(entry, model="default", tier="ADVANCED")
    mock_registry.get_best_model.assert_called_once_with("anthropic", "ADVANCED", 4096, context_tokens=None)
    mock_anthropic.assert_called_once_with(api_key="key", model="dynamic-model")


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key"}, clear=False)
@patch("mullet.core.providers.anthropic.AnthropicProvider")
@patch("mullet.core.providers.model_registry.get_registry", side_effect=RuntimeError)
def test_dynamic_model_registry_exception(mock_reg, mock_anthropic):
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="anthropic", tier="premium",
                          models={"default": "claude-sonnet-4-6"})
    mock_anthropic.return_value = MagicMock()
    result = resolver._create_provider(entry, model="default", tier="ADVANCED")
    mock_anthropic.assert_called_once_with(api_key="key", model="claude-sonnet-5")


# ---------------------------------------------------------------------------
# _pricing
# ---------------------------------------------------------------------------


@patch("mullet.core.providers.pricing.get_known_pricing")
def test_pricing_hit(mock_get_pricing):
    mock_get_pricing.return_value = {"model-a": (0.01, 0.02)}
    assert _pricing("model-a", 0.1, 0.2) == (0.01, 0.02)


@patch("mullet.core.providers.pricing.get_known_pricing")
def test_pricing_miss(mock_get_pricing):
    mock_get_pricing.return_value = {"other": (0.01, 0.02)}
    assert _pricing("model-a", 0.1, 0.2) == (0.1, 0.2)


@patch("mullet.core.providers.pricing.get_known_pricing", side_effect=RuntimeError)
def test_pricing_exception(mock_get_pricing):
    assert _pricing("model-a", 0.1, 0.2) == (0.1, 0.2)


# ---------------------------------------------------------------------------
# build_default_router
# ---------------------------------------------------------------------------


@patch("mullet.core.providers.pricing.get_known_pricing")
@patch("mullet.core.providers.claude_max._read_oauth_token")
@patch("mullet.cli.main.detect_local_provider")
def test_build_default_router_no_providers(mock_detect, mock_oauth, mock_pricing):
    mock_detect.return_value = None
    mock_oauth.return_value = None
    mock_pricing.return_value = (0.001, 0.005)
    with patch.dict(os.environ, {}, clear=True):
        router, resolver = build_default_router()
    assert isinstance(router, Router)
    assert isinstance(resolver, ProviderResolver)
    assert resolver.router is router
    assert "local" not in router.providers


@patch("mullet.core.providers.pricing.get_known_pricing")
@patch("mullet.cli.main.detect_local_provider")
def test_build_default_router_with_local(mock_detect, mock_pricing):
    mock_detect.return_value = MagicMock(model="llama-8b")
    mock_pricing.return_value = (0.001, 0.005)
    with patch.dict(os.environ, {}, clear=True):
        router, resolver = build_default_router()
    assert "local" in router.providers
    assert router.providers["local"].name == "local"


@patch("mullet.core.providers.pricing.get_known_pricing")
@patch("mullet.cli.main.detect_local_provider")
def test_build_default_router_with_anthropic(mock_detect, mock_pricing):
    mock_detect.return_value = None
    mock_pricing.return_value = (0.003, 0.015)
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key"}, clear=True):
        router, resolver = build_default_router()
    assert "premium_cloud" in router.providers
    assert "haiku" in router.providers
    assert "sonnet" in router.providers
    assert "opus" in router.providers
    assert "opus-4-8" in router.providers
    assert "opus-4-7" in router.providers


@patch("mullet.core.providers.pricing.get_known_pricing")
@patch("mullet.cli.main.detect_local_provider")
def test_build_default_router_with_kimi(mock_detect, mock_pricing):
    mock_detect.return_value = None
    mock_pricing.return_value = (0.00095, 0.004)
    with patch.dict(os.environ, {"KIMI_API_KEY": "kkey"}, clear=True):
        router, resolver = build_default_router()
    assert "cheap_cloud" in router.providers
    assert "kimi" in router.providers
    assert router.providers["cheap_cloud"].name == "kimi"


@patch("mullet.core.providers.pricing.get_known_pricing")
@patch("mullet.cli.main.detect_local_provider")
def test_build_default_router_with_moonshot(mock_detect, mock_pricing):
    mock_detect.return_value = None
    mock_pricing.return_value = (0.00095, 0.004)
    with patch.dict(os.environ, {"MOONSHOT_API_KEY": "mkey"}, clear=True):
        router, resolver = build_default_router()
    assert "kimi" in router.providers


@patch("mullet.core.providers.pricing.get_known_pricing")
@patch("mullet.cli.main.detect_local_provider")
def test_build_default_router_with_gemini(mock_detect, mock_pricing):
    mock_detect.return_value = None
    mock_pricing.return_value = (0.0003, 0.0025)
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "gkey"}, clear=True):
        router, resolver = build_default_router()
    assert "gemini" in router.providers
    # Phase E: gemini no longer fills cheap_cloud (it's last-resort, not cheap).
    assert "cheap_cloud" not in router.providers


@patch("mullet.core.providers.pricing.get_known_pricing")
@patch("mullet.cli.main.detect_local_provider")
def test_build_default_router_with_openrouter(mock_detect, mock_pricing):
    mock_detect.return_value = None
    mock_pricing.return_value = (0.000098, 0.000196)
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "orkey"}, clear=True):
        router, resolver = build_default_router()
    assert "openrouter" in router.providers


@patch("mullet.core.providers.pricing.get_known_pricing")
@patch("mullet.cli.main.detect_local_provider")
def test_build_default_router_openrouter_fills_cheap_cloud(mock_detect, mock_pricing):
    """OpenRouter fills cheap_cloud when no other cheap provider exists."""
    mock_detect.return_value = None
    mock_pricing.return_value = (0.000098, 0.000196)
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "orkey"}, clear=True):
        router, resolver = build_default_router()
    assert "cheap_cloud" in router.providers
    assert router.providers["cheap_cloud"].name == "openrouter"


@patch("mullet.core.providers.pricing.get_known_pricing")
@patch("mullet.cli.main.detect_local_provider")
def test_build_default_router_kimi_beats_openrouter_for_cheap_cloud(mock_detect, mock_pricing):
    """Kimi takes precedence for cheap_cloud over openrouter."""
    mock_detect.return_value = None
    mock_pricing.return_value = (0.00095, 0.004)
    with patch.dict(os.environ, {"KIMI_API_KEY": "kkey", "OPENROUTER_API_KEY": "orkey"}, clear=True):
        router, resolver = build_default_router()
    assert router.providers["cheap_cloud"].name == "kimi"
    assert "openrouter" in router.providers


@patch("mullet.core.providers.pricing.get_known_pricing")
@patch("mullet.cli.main.detect_local_provider")
def test_build_default_router_openrouter_beats_gemini_for_cheap_cloud(mock_detect, mock_pricing):
    """Phase E: openrouter (genuinely cheap) fills cheap_cloud over gemini (last-resort)."""
    mock_detect.return_value = None
    mock_pricing.return_value = (0.0003, 0.0025)
    with patch.dict(os.environ, {"GEMINI_API_KEY": "gkey", "OPENROUTER_API_KEY": "orkey"}, clear=True):
        router, resolver = build_default_router()
    assert router.providers["cheap_cloud"].name == "openrouter"


@patch("mullet.core.providers.pricing.get_known_pricing")
@patch("mullet.cli.main.detect_local_provider")
def test_build_default_router_local_only(mock_detect, mock_pricing):
    mock_detect.return_value = MagicMock(model="llama-8b")
    mock_pricing.return_value = (0.001, 0.005)
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key", "KIMI_API_KEY": "kkey"}, clear=True):
        router, resolver = build_default_router(local_only=True)
    assert router.local_only is True


@patch("mullet.core.providers.pricing.get_known_pricing")
@patch("mullet.cli.main.detect_local_provider")
def test_build_default_router_use_pressure_routing(mock_detect, mock_pricing):
    mock_detect.return_value = None
    mock_pricing.return_value = (0.001, 0.005)
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key", "KIMI_API_KEY": "kkey"}, clear=True):
        router, resolver = build_default_router()
    assert router.use_pressure_routing is True


@patch("mullet.core.providers.pricing.get_known_pricing")
@patch("mullet.core.providers.claude_max._read_oauth_token", side_effect=Exception)
@patch("mullet.cli.main.detect_local_provider")
def test_build_default_router_oauth_exception(mock_detect, mock_oauth, mock_pricing):
    """OAuth token read failure should be caught."""
    mock_detect.return_value = None
    mock_pricing.return_value = (0.003, 0.015)
    with patch.dict(os.environ, {}, clear=True):
        router, resolver = build_default_router()
    assert "premium_cloud" not in router.providers


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@patch("mullet.core.router.resolver.OpenAICompatibleProvider")
def test_resolve_tier_without_value_attribute(mock_oai):
    """Test resolve when decision.tier has no .value attribute."""
    router = _make_router()
    resolver = ProviderResolver(router)
    decision = MagicMock()
    decision.provider = "local"
    decision.model = "llama-8b"
    decision.tier = "STANDARD"  # str, no .value
    decision.allowed_providers = None
    mock_inst = MagicMock()
    mock_oai.return_value = mock_inst
    result = resolver.resolve(decision)
    assert result is mock_inst
    assert "local:STANDARD" in resolver._provider_cache


@patch("mullet.core.routing_history.append_decision_log")
@patch("mullet.core.router.resolver.OpenAICompatibleProvider")
def test_route_and_resolve_no_provider_logs_model(mock_oai, mock_log):
    """When provider is None, decision.model should still be logged."""
    router = _make_router()
    resolver = ProviderResolver(router)
    decision = _make_decision(provider="missing", model="missing-model", tier=Tier.LOW)
    with patch.object(router, "route", return_value=decision):
        provider, out = resolver.route_and_resolve("task")
        assert provider is None
        mock_log.assert_called_once()
        _, kwargs = mock_log.call_args
        assert kwargs["model"] == "missing-model"


@patch("mullet.core.providers.lmstudio_manager.LMStudioManager")
@patch("mullet.core.providers.openai_compat.is_escalation_recommended", return_value=False)
@patch("mullet.core.router.resolver.OpenAICompatibleProvider")
def test_create_provider_local_no_model_uses_default(mock_oai, mock_esc, mock_mgr_cls):
    """Local provider with empty model uses default path but still passes model to OpenAICompatibleProvider."""
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="local", tier="free", models={"default": "llama-8b"})
    mock_mgr = MagicMock()
    mock_mgr.pick_best_model.return_value = None
    mock_mgr_cls.return_value = mock_mgr
    resolver._create_provider(entry, model="")
    mock_oai.assert_called_once_with(base_url="http://localhost:1234/v1", model="")


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key"}, clear=False)
@patch("mullet.core.providers.anthropic.AnthropicProvider")
@patch("mullet.core.providers.model_registry.get_registry")
def test_create_provider_context_tokens_threaded(mock_reg, mock_anthropic):
    """_dynamic_model should pass context_tokens to get_best_model."""
    router = _make_router()
    resolver = ProviderResolver(router)
    entry = ProviderEntry(name="anthropic", tier="premium",
                          models={"default": "claude-sonnet-4-6"})
    mock_registry = MagicMock()
    mock_registry.get_best_model.return_value = "dynamic-model"
    mock_reg.return_value = mock_registry
    mock_anthropic.return_value = MagicMock()

    resolver._create_provider(entry, model="default", context_tokens=8192)
    mock_registry.get_best_model.assert_called_once_with(
        "anthropic", "STANDARD", 4096, context_tokens=8192,
    )
