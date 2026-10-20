"""Tests for provider resolver."""

import logging
import sys
from unittest.mock import patch, MagicMock

from mullet.core.router.resolver import ProviderResolver, build_default_router, _pricing
from mullet.core.router.decision import Router, ProviderEntry, RoutingDecision
from mullet.core.router.complexity import Tier


def test_resolver_returns_none_for_missing():
    router = Router(providers={})
    resolver = ProviderResolver(router)
    decision = RoutingDecision(provider="nonexistent", model="x", tier=Tier.LOW)
    result = resolver.resolve(decision)
    assert result is None


def test_resolver_caches():
    router = Router(providers={
        "local": ProviderEntry(name="local", tier="free", models={"default": "m"}),
    })
    resolver = ProviderResolver(router)

    decision = RoutingDecision(provider="local", model="m", tier=Tier.LOW)
    result = resolver.resolve(decision)
    # When a real Provider instance is constructed, the resolver caches it
    # under ``"{provider}:{tier_value}"`` (NOT provider:model). When local
    # is unavailable (no LM Studio / Ollama running in CI), resolve returns
    # None and the cache stays empty.
    if result is not None:
        assert "local:LOW" in resolver._provider_cache
    else:
        assert resolver._provider_cache == {}


def test_build_default_router():
    """build_default_router should not crash."""
    router, resolver = build_default_router()
    assert router is not None
    assert resolver is not None
    assert isinstance(router, Router)


def test_route_and_resolve():
    router, resolver = build_default_router()
    provider, decision = resolver.route_and_resolve("list files")
    # May or may not find a provider depending on environment
    assert decision is not None
    assert decision.tier in (Tier.LOW, Tier.STANDARD, Tier.ADVANCED)


def test_kimi_provider_created_when_key_set(monkeypatch):
    """ProviderResolver must create KimiProvider for kimi entries — not fall through to None."""
    monkeypatch.setenv("KIMI_API_KEY", "test-key-xyz")
    from mullet.core.router.decision import ProviderEntry
    from mullet.core.router.resolver import ProviderResolver
    from mullet.core.router.decision import Router

    router = Router(providers={
        "kimi": ProviderEntry(name="kimi", tier="cheap", models={"default": "kimi-k2-0711-preview"}),
    })
    resolver = ProviderResolver(router)
    from mullet.core.router.decision import RoutingDecision
    from mullet.core.router.complexity import Tier
    decision = RoutingDecision(provider="kimi", model="kimi-k2-0711-preview", tier=Tier.STANDARD)
    provider = resolver.resolve(decision)
    assert provider is not None, "kimi entry must not fall through to None (regression: no kimi handler)"
    assert "kimi" in type(provider).__name__.lower() or hasattr(provider, "_api_key")


def test_kimi_not_created_without_key(monkeypatch):
    """KimiProvider must not be created when no API key is present."""
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    from mullet.core.router.decision import ProviderEntry, Router, RoutingDecision
    from mullet.core.router.resolver import ProviderResolver
    from mullet.core.router.complexity import Tier

    router = Router(providers={
        "kimi": ProviderEntry(name="kimi", tier="cheap", models={"default": "kimi-k2-0711-preview"}),
    })
    resolver = ProviderResolver(router)
    decision = RoutingDecision(provider="kimi", model="kimi-k2-0711-preview", tier=Tier.STANDARD)
    provider = resolver.resolve(decision)
    assert provider is None


def test_route_and_resolve_logs_decision(monkeypatch):
    """route_and_resolve must call append_decision_log on every decision."""
    from unittest.mock import patch as _patch
    router, resolver = build_default_router()
    logged = []
    with _patch("mullet.core.routing_history.append_decision_log", side_effect=lambda **kw: logged.append(kw)):
        resolver.route_and_resolve("write a test for the search module")
    assert len(logged) == 1
    log = logged[0]
    assert "provider" in log
    assert "tier" in log
    assert "reasons" in log
    assert isinstance(log["reasons"], list)
    assert "write a test" in log["task_preview"]


class TestResolverMissingLines:
    """Cover previously-untested branches in resolver.py."""

    # Line 29 — cache hit
    def test_resolve_cache_hit_returns_same_instance(self):
        entry = ProviderEntry(name="generic", tier="free", models={"default": "m"})
        entry.base_url = "http://localhost:1234/v1"
        router = Router(providers={"generic": entry})
        resolver = ProviderResolver(router)
        decision = RoutingDecision(provider="generic", model="m", tier=Tier.LOW)
        first = resolver.resolve(decision)
        assert first is not None
        second = resolver.resolve(decision)
        assert first is second
        assert "generic:LOW" in resolver._provider_cache

    def test_resolve_cache_hit_different_model_same_tier(self):
        entry = ProviderEntry(name="generic", tier="free", models={"default": "m"})
        entry.base_url = "http://localhost:1234/v1"
        router = Router(providers={"generic": entry})
        resolver = ProviderResolver(router)
        d1 = RoutingDecision(provider="generic", model="alpha", tier=Tier.LOW)
        d2 = RoutingDecision(provider="generic", model="beta", tier=Tier.LOW)
        r1 = resolver.resolve(d1)
        r2 = resolver.resolve(d2)
        assert r1 is r2
        assert r1.model == "alpha"

    # Lines 58-65 — fallback chain walk
    def test_fallback_chain_primary_none_fallback_resolves(self, monkeypatch):
        fb_entry = ProviderEntry(name="fallback", tier="cheap", models={"default": "m"})
        fb_entry.base_url = "http://localhost:1234/v1"
        router = Router(providers={"fallback": fb_entry})
        resolver = ProviderResolver(router)
        decision = RoutingDecision(
            provider="missing",
            model="m",
            tier=Tier.LOW,
            fallback_chain=["fallback"],
            reasons=["primary missing"],
        )
        monkeypatch.setattr(router, "route", lambda *a, **kw: decision)
        provider, out_decision = resolver.route_and_resolve("task")
        assert provider is not None
        assert out_decision.provider == "fallback"
        assert "fallback: fallback" in out_decision.reasons

    def test_fallback_chain_all_fallbacks_none(self, monkeypatch):
        router = Router(providers={})
        resolver = ProviderResolver(router)
        decision = RoutingDecision(
            provider="missing",
            model="m",
            tier=Tier.LOW,
            fallback_chain=["also_missing"],
            reasons=["primary missing"],
        )
        monkeypatch.setattr(router, "route", lambda *a, **kw: decision)
        provider, _ = resolver.route_and_resolve("task")
        assert provider is None

    def test_fallback_chain_entry_missing_from_router(self, monkeypatch):
        router = Router(providers={})
        resolver = ProviderResolver(router)
        decision = RoutingDecision(
            provider="missing",
            model="m",
            tier=Tier.LOW,
            fallback_chain=["not_in_router"],
            reasons=["primary missing"],
        )
        monkeypatch.setattr(router, "route", lambda *a, **kw: decision)
        provider, _ = resolver.route_and_resolve("task")
        assert provider is None

    # Lines 78-79 — append_decision_log raises
    def test_route_and_resolve_decision_log_exception(self, monkeypatch):
        entry = ProviderEntry(name="generic", tier="free", models={"default": "m"})
        entry.base_url = "http://localhost:1234/v1"
        router = Router(providers={"generic": entry})
        resolver = ProviderResolver(router)
        decision = RoutingDecision(
            provider="generic",
            model="m",
            tier=Tier.LOW,
            reasons=["ok"],
        )
        monkeypatch.setattr(router, "route", lambda *a, **kw: decision)
        with patch("mullet.core.routing_history.append_decision_log", side_effect=ValueError("boom")):
            provider, out_decision = resolver.route_and_resolve("task")
        assert provider is not None
        assert out_decision.provider == "generic"

    # Lines 104-111 — _dynamic_model closure
    def test_dynamic_model_anthropic_default(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "akey")
        mock_reg = MagicMock()
        mock_reg.get_best_model.return_value = "claude-dynamic"
        with patch("mullet.core.providers.model_registry.get_registry", return_value=mock_reg):
            with patch("mullet.core.providers.anthropic.AnthropicProvider") as MockProv:
                router = Router(providers={
                    "anthropic": ProviderEntry(name="anthropic", tier="premium", models={"default": "claude-sonnet-4-6"}),
                })
                resolver = ProviderResolver(router)
                decision = RoutingDecision(provider="anthropic", model="default", tier=Tier.STANDARD)
                resolver.resolve(decision)
        assert MockProv.call_args.kwargs.get("model") == "claude-dynamic"

    def test_dynamic_model_exception_returns_fallback(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "akey")
        mock_reg = MagicMock()
        mock_reg.get_best_model.side_effect = ValueError("boom")
        with patch("mullet.core.providers.model_registry.get_registry", return_value=mock_reg):
            with patch("mullet.core.providers.anthropic.AnthropicProvider") as MockProv:
                router = Router(providers={
                    "anthropic": ProviderEntry(name="anthropic", tier="premium", models={"default": "claude-sonnet-4-6"}),
                })
                resolver = ProviderResolver(router)
                decision = RoutingDecision(provider="anthropic", model="default", tier=Tier.STANDARD)
                resolver.resolve(decision)
        assert MockProv.call_args.kwargs.get("model") == "claude-sonnet-5"

    # Lines 116-117 — is_escalation_recommended returns True
    def test_local_escalation_recommended_skips(self):
        with patch("mullet.core.providers.openai_compat.is_escalation_recommended", return_value=True):
            router = Router(providers={
                "local": ProviderEntry(name="local", tier="free", models={"default": "m"}),
            })
            resolver = ProviderResolver(router)
            decision = RoutingDecision(provider="local", model="m", tier=Tier.LOW)
            assert resolver.resolve(decision) is None

    # Lines 121-128 — LMStudioManager picks best model
    def test_local_lmstudio_picks_best_model(self):
        mock_mgr = MagicMock()
        mock_mgr.pick_best_model.return_value = {"id": "picked-model"}
        with patch("mullet.core.providers.lmstudio_manager.LMStudioManager", return_value=mock_mgr):
            router = Router(providers={
                "local": ProviderEntry(name="local", tier="free", models={"default": "m"}),
            })
            resolver = ProviderResolver(router)
            decision = RoutingDecision(provider="local", model="", tier=Tier.LOW)
            provider = resolver.resolve(decision)
        assert provider is not None
        assert provider.model == "picked-model"

    def test_local_lmstudio_raises_uses_default(self):
        with patch("mullet.core.providers.lmstudio_manager.LMStudioManager", side_effect=Exception("boom")):
            router = Router(providers={
                "local": ProviderEntry(name="local", tier="free", models={"default": "m"}),
            })
            resolver = ProviderResolver(router)
            decision = RoutingDecision(provider="local", model="default", tier=Tier.LOW)
            provider = resolver.resolve(decision)
        assert provider is not None
        assert provider.model == "default"

    def test_anthropic_no_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        router = Router(providers={
            "anthropic": ProviderEntry(name="anthropic", tier="premium", models={"default": "claude-sonnet-4-6"}),
        })
        resolver = ProviderResolver(router)
        decision = RoutingDecision(provider="anthropic", model="default", tier=Tier.STANDARD)
        assert resolver.resolve(decision) is None

    def test_haiku_no_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        router = Router(providers={
            "haiku": ProviderEntry(name="haiku", tier="cheap", models={"default": "claude-haiku"}),
        })
        resolver = ProviderResolver(router)
        decision = RoutingDecision(provider="haiku", model="claude-haiku", tier=Tier.LOW)
        assert resolver.resolve(decision) is None

    def test_sonnet_no_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        router = Router(providers={
            "sonnet": ProviderEntry(name="sonnet", tier="premium", models={"default": "claude-sonnet"}),
        })
        resolver = ProviderResolver(router)
        decision = RoutingDecision(provider="sonnet", model="claude-sonnet", tier=Tier.STANDARD)
        assert resolver.resolve(decision) is None

    # Lines 160-168 — gemini with key + ImportError
    def test_gemini_with_key_success(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "gkey")
        with patch("mullet.core.providers.gemini.GeminiProvider") as MockProv:
            router = Router(providers={
                "gemini": ProviderEntry(name="gemini", tier="cheap", models={"default": "gemini-2.5-flash"}),
            })
            resolver = ProviderResolver(router)
            decision = RoutingDecision(provider="gemini", model="default", tier=Tier.LOW)
            provider = resolver.resolve(decision)
        assert provider is not None
        assert MockProv.call_args.kwargs.get("api_key") == "gkey"

    def test_gemini_import_error_returns_none_warns(self, monkeypatch, caplog):
        monkeypatch.setenv("GOOGLE_API_KEY", "gkey")
        with caplog.at_level(logging.WARNING, logger="mullet.core.router.resolver"):
            with patch.dict(sys.modules, {"mullet.core.providers.gemini": None}):
                router = Router(providers={
                    "gemini": ProviderEntry(name="gemini", tier="cheap", models={"default": "gemini-2.5-flash"}),
                })
                resolver = ProviderResolver(router)
                decision = RoutingDecision(provider="gemini", model="default", tier=Tier.LOW)
                assert resolver.resolve(decision) is None
        assert any("Gemini provider requested but" in r.message for r in caplog.records)

    # Lines 178-179 — kimi ImportError
    def test_kimi_import_error_returns_none_warns(self, monkeypatch, caplog):
        monkeypatch.setenv("KIMI_API_KEY", "kkey")
        with caplog.at_level(logging.WARNING, logger="mullet.core.router.resolver"):
            with patch.dict(sys.modules, {"mullet.core.providers.kimi": None}):
                router = Router(providers={
                    "kimi": ProviderEntry(name="kimi", tier="cheap", models={"default": "kimi-k2.7-code"}),
                })
                resolver = ProviderResolver(router)
                decision = RoutingDecision(provider="kimi", model="default", tier=Tier.STANDARD)
                assert resolver.resolve(decision) is None
        assert any("Kimi provider requested but not importable" in r.message for r in caplog.records)

    # Lines 182-202 — openrouter
    def test_openrouter_with_key_success(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "orkey")
        mock_reg = MagicMock()
        mock_reg.get_best_model.return_value = "deepseek/deepseek-v4-flash"
        with patch("mullet.core.providers.model_registry.get_registry", return_value=mock_reg):
            with patch("mullet.core.providers.openrouter.OpenRouterProvider") as MockProv:
                router = Router(providers={
                    "openrouter": ProviderEntry(name="openrouter", tier="cheap", models={"default": "deepseek/deepseek-v4-flash"}),
                })
                resolver = ProviderResolver(router)
                decision = RoutingDecision(provider="openrouter", model="default", tier=Tier.LOW)
                provider = resolver.resolve(decision)
        assert provider is not None
        assert MockProv.call_args.kwargs.get("model") == "deepseek/deepseek-v4-flash"

    def test_openrouter_with_key_exception_fallback_openai_compat(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "orkey")
        mock_reg = MagicMock()
        mock_reg.get_best_model.return_value = "deepseek/deepseek-v4-flash"
        with patch("mullet.core.providers.model_registry.get_registry", return_value=mock_reg):
            with patch("mullet.core.providers.openrouter.OpenRouterProvider", side_effect=Exception("boom")):
                router = Router(providers={
                    "openrouter": ProviderEntry(name="openrouter", tier="cheap", models={"default": "deepseek/deepseek-v4-flash"}),
                })
                resolver = ProviderResolver(router)
                decision = RoutingDecision(provider="openrouter", model="default", tier=Tier.LOW)
                provider = resolver.resolve(decision)
        assert provider is not None
        assert type(provider).__name__ == "OpenAICompatibleProvider"
        assert provider.model == "deepseek/deepseek-v4-flash"

    def test_openrouter_without_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        router = Router(providers={
            "openrouter": ProviderEntry(name="openrouter", tier="cheap", models={"default": "deepseek/deepseek-v4-flash"}),
        })
        resolver = ProviderResolver(router)
        decision = RoutingDecision(provider="openrouter", model="default", tier=Tier.LOW)
        assert resolver.resolve(decision) is None

    # Lines 211-213 — _pricing
    def test_pricing_model_in_registry(self):
        with patch("mullet.core.providers.pricing.get_known_pricing", return_value={"model-a": (0.1, 0.2)}):
            assert _pricing("model-a", 0.5, 0.6) == (0.1, 0.2)

    def test_pricing_model_not_in_registry(self):
        with patch("mullet.core.providers.pricing.get_known_pricing", return_value={"model-b": (0.1, 0.2)}):
            assert _pricing("model-a", 0.5, 0.6) == (0.5, 0.6)

    def test_pricing_exception_returns_fallback(self):
        with patch("mullet.core.providers.pricing.get_known_pricing", side_effect=ValueError("boom")):
            assert _pricing("model-a", 0.5, 0.6) == (0.5, 0.6)

    # Lines 240-241 — oauth read exception (also covers line 286 kimi guard)
    def test_build_default_router_anthropic_oauth_read_fails(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch("mullet.core.providers.claude_max._read_oauth_token", side_effect=Exception("boom")):
            with patch("mullet.cli.main.detect_local_provider", return_value=None):
                router, _ = build_default_router()
        assert "premium_cloud" not in router.providers

    def test_build_default_router_kimi_sets_cheap_cloud(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "kkey")
        with patch("mullet.cli.main.detect_local_provider", return_value=None):
            router, _ = build_default_router()
        assert router.providers["cheap_cloud"].name == "kimi"
        assert "kimi" in router.providers

    def test_build_default_router_gemini_does_not_set_cheap_cloud(self):
        # Phase E: gemini is last-resort and no longer fills the cheap_cloud alias.
        # Mirror test_router_resolver's isolation (clear env + patched pricing) so the
        # real .env's KIMI/OPENROUTER keys don't leak in and mask the gemini behaviour.
        import os
        with patch("mullet.core.providers.pricing.get_known_pricing", return_value=(0.0003, 0.0025)), \
             patch("mullet.cli.main.detect_local_provider", return_value=None), \
             patch.dict(os.environ, {"GOOGLE_API_KEY": "gkey"}, clear=True):
            router, _ = build_default_router()
        assert "gemini" in router.providers
        assert "cheap_cloud" not in router.providers

    def test_build_default_router_both_keys_kimi_wins_cheap_cloud(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "kkey")
        monkeypatch.setenv("GOOGLE_API_KEY", "gkey")
        with patch("mullet.cli.main.detect_local_provider", return_value=None):
            router, _ = build_default_router()
        assert router.providers["cheap_cloud"].name == "kimi"
        assert "gemini" in router.providers

    def test_build_default_router_openrouter_sets_cheap_cloud(self, monkeypatch):
        monkeypatch.delenv("KIMI_API_KEY", raising=False)
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "orkey")
        with patch("mullet.cli.main.detect_local_provider", return_value=None):
            router, _ = build_default_router()
        assert router.providers["cheap_cloud"].name == "openrouter"
        assert "openrouter" in router.providers

    def test_build_default_router_kimi_and_openrouter_kimi_wins(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "kkey")
        monkeypatch.setenv("OPENROUTER_API_KEY", "orkey")
        with patch("mullet.cli.main.detect_local_provider", return_value=None):
            router, _ = build_default_router()
        assert router.providers["cheap_cloud"].name == "kimi"
        assert "openrouter" in router.providers

    # Line 332 — local_only=True
    def test_build_default_router_local_only_true(self, monkeypatch):
        with patch("mullet.cli.main.detect_local_provider", return_value=None):
            router, _ = build_default_router(local_only=True)
        assert router.local_only is True

    def test_build_default_router_anthropic_key_creates_entries(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "akey")
        with patch("mullet.cli.main.detect_local_provider", return_value=None):
            router, _ = build_default_router()
        assert "premium_cloud" in router.providers
        assert "haiku" in router.providers
        assert "sonnet" in router.providers
        assert "opus" in router.providers

    # Lines 195-200 — generic OpenAI-compatible
    def test_generic_openai_compatible_with_base_url(self):
        entry = ProviderEntry(name="custom", tier="cheap", models={"default": "m"})
        entry.base_url = "http://localhost:1234/v1"
        router = Router(providers={"custom": entry})
        resolver = ProviderResolver(router)
        decision = RoutingDecision(provider="custom", model="m", tier=Tier.LOW)
        provider = resolver.resolve(decision)
        assert provider is not None
        assert type(provider).__name__ == "OpenAICompatibleProvider"

    def test_generic_openai_compatible_no_base_url_returns_none(self):
        entry = ProviderEntry(name="custom", tier="cheap", models={"default": "m"})
        router = Router(providers={"custom": entry})
        resolver = ProviderResolver(router)
        decision = RoutingDecision(provider="custom", model="m", tier=Tier.LOW)
        assert resolver.resolve(decision) is None

    # Extra edge cases
    def test_local_with_model_default(self):
        router = Router(providers={
            "local": ProviderEntry(name="local", tier="free", models={"default": "m"}),
        })
        resolver = ProviderResolver(router)
        decision = RoutingDecision(provider="local", model="default", tier=Tier.LOW)
        provider = resolver.resolve(decision)
        assert provider is not None

    def test_anthropic_with_key_uses_dynamic_model(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "akey")
        mock_reg = MagicMock()
        mock_reg.get_best_model.return_value = "claude-dynamic"
        with patch("mullet.core.providers.model_registry.get_registry", return_value=mock_reg):
            with patch("mullet.core.providers.anthropic.AnthropicProvider") as MockProv:
                router = Router(providers={
                    "anthropic": ProviderEntry(name="anthropic", tier="premium", models={"default": "claude-sonnet-4-6"}),
                })
                resolver = ProviderResolver(router)
                decision = RoutingDecision(provider="anthropic", model="default", tier=Tier.STANDARD)
                resolver.resolve(decision)
        assert MockProv.call_args.kwargs.get("model") == "claude-dynamic"

    def test_anthropic_with_key_explicit_model(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "akey")
        with patch("mullet.core.providers.anthropic.AnthropicProvider") as MockProv:
            router = Router(providers={
                "anthropic": ProviderEntry(name="anthropic", tier="premium", models={"default": "claude-sonnet-4-6"}),
            })
            resolver = ProviderResolver(router)
            decision = RoutingDecision(provider="anthropic", model="claude-explicit", tier=Tier.STANDARD)
            resolver.resolve(decision)
        assert MockProv.call_args.kwargs.get("model") == "claude-explicit"

    def test_sonnet_with_key_success(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "akey")
        with patch("mullet.core.providers.anthropic.AnthropicProvider") as MockProv:
            router = Router(providers={
                "sonnet": ProviderEntry(name="sonnet", tier="premium", models={"default": "claude-sonnet"}),
            })
            resolver = ProviderResolver(router)
            decision = RoutingDecision(provider="sonnet", model="claude-sonnet", tier=Tier.STANDARD)
            provider = resolver.resolve(decision)
        assert provider is not None

    def test_opus_no_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        router = Router(providers={
            "opus": ProviderEntry(name="opus", tier="premium", models={"default": "claude-opus"}),
        })
        resolver = ProviderResolver(router)
        decision = RoutingDecision(provider="opus", model="claude-opus", tier=Tier.STANDARD)
        assert resolver.resolve(decision) is None

    def test_gemini_no_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        router = Router(providers={
            "gemini": ProviderEntry(name="gemini", tier="cheap", models={"default": "gemini-2.5-flash"}),
        })
        resolver = ProviderResolver(router)
        decision = RoutingDecision(provider="gemini", model="default", tier=Tier.LOW)
        assert resolver.resolve(decision) is None

    def test_kimi_no_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("KIMI_API_KEY", raising=False)
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        router = Router(providers={
            "kimi": ProviderEntry(name="kimi", tier="cheap", models={"default": "kimi-k2.7-code"}),
        })
        resolver = ProviderResolver(router)
        decision = RoutingDecision(provider="kimi", model="default", tier=Tier.STANDARD)
        assert resolver.resolve(decision) is None

    def test_kimi_with_key_success(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "kkey")
        with patch("mullet.core.providers.kimi.KimiProvider") as MockProv:
            router = Router(providers={
                "kimi": ProviderEntry(name="kimi", tier="cheap", models={"default": "kimi-k2.7-code"}),
            })
            resolver = ProviderResolver(router)
            decision = RoutingDecision(provider="kimi", model="default", tier=Tier.STANDARD)
            provider = resolver.resolve(decision)
        assert provider is not None
        assert MockProv.call_args.kwargs.get("api_key") == "kkey"

    def test_openrouter_with_explicit_model(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "orkey")
        with patch("mullet.core.providers.openrouter.OpenRouterProvider") as MockProv:
            router = Router(providers={
                "openrouter": ProviderEntry(name="openrouter", tier="cheap", models={"default": "deepseek/deepseek-v4-flash"}),
            })
            resolver = ProviderResolver(router)
            decision = RoutingDecision(provider="openrouter", model="explicit-model", tier=Tier.LOW)
            resolver.resolve(decision)
        assert MockProv.call_args.kwargs.get("model") == "explicit-model"
