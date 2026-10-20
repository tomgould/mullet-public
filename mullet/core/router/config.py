"""Routing configuration schema and defaults.

Loaded from mullet.yaml `routing:` and `providers:` sections.
"""

from dataclasses import dataclass, field


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""
    name: str
    type: str = "openai_compatible"  # openai_compatible | anthropic | bedrock
    base_url: str = ""
    api_key_env: str = ""
    models: dict[str, str] = field(default_factory=dict)
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    tier: str = "cheap"  # free | cheap | premium
    auto_detect: bool = False


@dataclass
class RoutingConfig:
    """Configuration for the intelligent router."""
    enabled: bool = True
    default_tier: str = "STANDARD"
    local_only: bool = False

    # Complexity scoring thresholds
    thresholds: dict[str, int] = field(default_factory=lambda: {
        "local_max": 30,
        "standard_max": 76,
        "advanced_max": 90,
    })

    # Tier → provider priority list
    tier_map: dict[str, list[str]] = field(default_factory=lambda: {
        "LOW": ["local"],
        "STANDARD": ["cheap_cloud", "local"],
        "ADVANCED": ["premium_cloud", "cheap_cloud"],
        "CRITICAL": ["premium_cloud"],
    })

    # Agent tier overrides
    agent_overrides: dict[str, str] = field(default_factory=lambda: {
        "explore": "LOW",
        "writer": "LOW",
        "executor": "STANDARD",
        "test-engineer": "STANDARD",
        "planner": "ADVANCED",
        "architect": "ADVANCED",
        "code-reviewer": "CRITICAL",
    })

    # Budget limits
    budget: dict[str, float] = field(default_factory=lambda: {
        "per_task_max_tokens": 100_000,
        "session_max_tokens": 1_000_000,
        "daily_max_dollars": 10.0,
    })


def load_routing_config(yaml_data: dict | None = None) -> RoutingConfig:
    """Load routing config from parsed YAML data."""
    if not yaml_data:
        return RoutingConfig()

    routing_data = yaml_data.get("routing", {})
    config = RoutingConfig()

    if "enabled" in routing_data:
        config.enabled = routing_data["enabled"]
    if "default_tier" in routing_data:
        config.default_tier = routing_data["default_tier"]
    if "local_only" in routing_data:
        config.local_only = routing_data["local_only"]
    if "thresholds" in routing_data:
        config.thresholds.update(routing_data["thresholds"])
    if "tier_map" in routing_data:
        config.tier_map.update(routing_data["tier_map"])
    if "agent_overrides" in routing_data:
        config.agent_overrides.update(routing_data["agent_overrides"])
    if "budget" in routing_data:
        config.budget.update(routing_data["budget"])

    return config


def load_provider_configs(yaml_data: dict | None = None) -> dict[str, ProviderConfig]:
    """Load provider configs from parsed YAML data."""
    if not yaml_data or "providers" not in yaml_data:
        return {}

    configs = {}
    for name, pdata in yaml_data["providers"].items():
        configs[name] = ProviderConfig(
            name=name,
            type=pdata.get("type", "openai_compatible"),
            base_url=pdata.get("base_url", ""),
            api_key_env=pdata.get("api_key_env", ""),
            models=pdata.get("models", {}),
            cost_per_1k_input=pdata.get("cost_per_1k_input", 0.0),
            cost_per_1k_output=pdata.get("cost_per_1k_output", 0.0),
            tier=pdata.get("tier", "cheap"),
            auto_detect=pdata.get("auto_detect", False),
        )

    return configs
