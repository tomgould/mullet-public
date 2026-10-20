"""Tests for Router and routing decisions."""
import pytest
from mullet.core.router.complexity import Tier
from mullet.core.router.decision import Router, RoutingDecision, ProviderEntry, AGENT_TIER_OVERRIDES


@pytest.fixture
def router():
    providers = {
        "local": ProviderEntry(name="local", tier="free", models={"default": "local-model"}),
        "kimi": ProviderEntry(name="kimi", tier="cheap", models={"default": "moonshot-v1"}, cost_per_1k_input=0.6),
        "gemini": ProviderEntry(name="gemini", tier="premium", models={"default": "gemini-flash"}, cost_per_1k_input=0.15),
    }
    return Router(providers=providers)


def test_route_returns_decision(router):
    d = router.route("hello")
    assert isinstance(d, RoutingDecision)
    assert d.provider != ""
    assert d.tier is not None


def test_local_only_mode():
    r = Router(
        providers={"local": ProviderEntry(name="local", tier="free", models={"default": "m"})},
        local_only=True,
    )
    d = r.route("complex task needing architecture")
    assert d.provider == "local"
    assert d.tier == Tier.LOW


def test_force_provider(router):
    d = router.route("task", force_provider="kimi")
    assert d.provider == "kimi"


def test_force_tier(router):
    d = router.route("task", force_tier="STANDARD")
    assert d.tier == Tier.STANDARD


def test_agent_tier_overrides():
    assert AGENT_TIER_OVERRIDES["architect"] == Tier.ADVANCED
    assert AGENT_TIER_OVERRIDES["explore"] == Tier.LOW
    assert AGENT_TIER_OVERRIDES["debugger"] is None


def test_agent_override_in_routing(router):
    d = router.route("review code", agent_type="architect")
    # Router may downgrade if no HIGH-tier provider available
    assert d.tier in (Tier.ADVANCED, Tier.STANDARD)


def test_routing_decision_fields():
    d = RoutingDecision(provider="test", model="m", tier=Tier.LOW)
    assert d.confidence == 1.0
    assert d.reasons == []
    assert d.fallback_chain == []


def test_provider_entry_defaults():
    p = ProviderEntry(name="x", tier="free")
    assert p.healthy is True
    assert p.cost_per_1k_input == 0.0
    assert p.models == {}


def test_empty_router():
    r = Router(providers={})
    d = r.route("task")
    assert d.provider is not None  # Should handle gracefully
