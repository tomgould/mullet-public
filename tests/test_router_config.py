"""Tests for routing config, budget, and health together."""

from mullet.core.router.config import RoutingConfig, load_routing_config, load_provider_configs
from mullet.core.router.budget import BudgetTracker, BudgetLimits
from mullet.core.providers.health import HealthTracker, classify_error
from mullet.core.router.decision import Router, ProviderEntry


def test_default_routing_config():
    config = RoutingConfig()
    assert config.enabled is True
    assert config.default_tier == "STANDARD"
    assert "LOW" in config.tier_map


def test_load_routing_config_from_yaml():
    yaml_data = {
        "routing": {
            "default_tier": "ADVANCED",
            "local_only": True,
            "budget": {"daily_max_dollars": 5.0},
        }
    }
    config = load_routing_config(yaml_data)
    assert config.default_tier == "ADVANCED"
    assert config.local_only is True
    assert config.budget["daily_max_dollars"] == 5.0


def test_load_provider_configs():
    yaml_data = {
        "providers": {
            "local": {
                "type": "openai_compatible",
                "base_url": "http://localhost:1234/v1",
                "models": {"default": "mistral"},
                "tier": "free",
            }
        }
    }
    configs = load_provider_configs(yaml_data)
    assert "local" in configs
    assert configs["local"].base_url == "http://localhost:1234/v1"


def test_router_with_health_and_budget():
    """Full integration: router uses health tracker and budget."""
    health = HealthTracker()
    budget = BudgetTracker(limits=BudgetLimits(daily_max_dollars=0.01))

    providers = {
        "local": ProviderEntry(name="local", tier="free", models={"default": "m"}),
        "cloud": ProviderEntry(name="cloud", tier="premium", models={"default": "s"},
                               cost_per_1k_input=0.003, cost_per_1k_output=0.015),
    }

    router = Router(providers=providers, health_tracker=health, budget_tracker=budget)

    # Route simple task — should go to local
    decision = router.route("list files")
    assert decision.provider == "local"

    # Exhaust budget
    budget.record_usage("cloud", "opus", 50000, 20000)

    # Now cloud should be budget-blocked
    decision = router.route(
        "refactor the entire auth system for OAuth2 with JWT refresh in production",
        force_tier="ADVANCED",
    )
    # Should still find a provider (fallback to what's affordable)
    assert decision.provider is not None


def test_classify_various_errors():
    assert classify_error(Exception("429 rate limit")) == "rate_limit"
    assert classify_error(Exception("503 overloaded")) == "overloaded"
    assert classify_error(ConnectionError("refused")) == "connection"
    assert classify_error(TimeoutError("timed out")) == "connection"
    assert classify_error(Exception("401 unauthorized")) == "auth"
    assert classify_error(Exception("something random")) == "unknown"


# --- Coverage gap tests ---

def test_load_routing_config_none_returns_default():
    config = load_routing_config(None)
    assert config.enabled is True
    assert config.default_tier == "STANDARD"


def test_load_routing_config_empty_dict_returns_default():
    config = load_routing_config({})
    assert config.enabled is True


def test_load_routing_config_sets_enabled():
    yaml_data = {"routing": {"enabled": False}}
    config = load_routing_config(yaml_data)
    assert config.enabled is False


def test_load_routing_config_sets_thresholds():
    yaml_data = {"routing": {"thresholds": {"local_max": 20}}}
    config = load_routing_config(yaml_data)
    assert config.thresholds["local_max"] == 20


def test_load_routing_config_sets_tier_map():
    yaml_data = {"routing": {"tier_map": {"LOW": ["local", "kimi"]}}}
    config = load_routing_config(yaml_data)
    assert config.tier_map["LOW"] == ["local", "kimi"]


def test_load_routing_config_sets_agent_overrides():
    yaml_data = {"routing": {"agent_overrides": {"planner": "LOW"}}}
    config = load_routing_config(yaml_data)
    assert config.agent_overrides["planner"] == "LOW"


def test_load_provider_configs_none_returns_empty():
    result = load_provider_configs(None)
    assert result == {}


def test_load_provider_configs_no_providers_key_returns_empty():
    result = load_provider_configs({"routing": {}})
    assert result == {}
