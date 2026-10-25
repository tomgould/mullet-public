"""Config-driven setup — load mullet.yaml and build the full stack.

This is the single entry point for building a configured Mullet instance
from a YAML config file. It resolves providers, creates the router,
loads skills, and wires everything together.
"""

import logging
import os
from pathlib import Path

import yaml

from mullet.config.defaults import load_config
from mullet.core.paths import config_file
from mullet.core.providers.health import HealthTracker
from mullet.core.router.budget import BudgetTracker, BudgetLimits
from mullet.core.router.config import load_routing_config, load_provider_configs, ProviderConfig
from mullet.core.router.decision import Router, ProviderEntry

logger = logging.getLogger(__name__)


def load_from_yaml(path: str | None = None) -> dict:
    """Load raw YAML config from file."""
    if path is not None:
        # Explicit path — only try that file, no fallbacks.
        if Path(path).exists():
            try:
                with open(path) as f:
                    return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                logger.warning("Failed to parse config %s: %s", path, e)
        return {}

    # No explicit path — search well-known locations.
    candidates = [
        "mullet.yaml",
        str(config_file("config.yaml")),
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                with open(candidate) as f:
                    return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                logger.warning("Failed to parse config %s: %s", candidate, e)
                return {}
    return {}


def build_stack(config_path: str | None = None) -> dict:
    """Build the full Mullet stack from config.

    Returns a dict with:
    - router: Router instance
    - health: HealthTracker
    - budget: BudgetTracker
    - providers: dict of Provider instances
    - config: MulletConfig
    """
    yaml_data = load_from_yaml(config_path)

    # Validate config
    if yaml_data:
        from mullet.config.validate import validate_config
        validation = validate_config(yaml_data)
        if not validation.valid:
            logger.warning("Config validation issues:\n%s", validation.format())
        elif validation.issues:
            logger.debug("Config warnings:\n%s", validation.format())
    config = load_config(config_path)
    routing_config = load_routing_config(yaml_data)
    provider_configs = load_provider_configs(yaml_data)

    # Build health tracker
    health = HealthTracker()

    # Build budget tracker
    budget_limits = BudgetLimits(
        per_task_max_tokens=int(routing_config.budget.get("per_task_max_tokens", 100_000)),
        session_max_tokens=int(routing_config.budget.get("session_max_tokens", 1_000_000)),
        daily_max_dollars=float(routing_config.budget.get("daily_max_dollars", 10.0)),
    )
    budget = BudgetTracker(limits=budget_limits)

    # Build provider entries
    provider_entries = _build_provider_entries(provider_configs)

    # Auto-detect local if not configured
    if "local" not in provider_entries:
        local = _detect_local()
        if local:
            provider_entries["local"] = local

    # Auto-detect cloud providers from env
    def _p(model_id: str, fallback_inp: float, fallback_out: float) -> tuple[float, float]:
        try:
            from mullet.core.providers.pricing import get_known_pricing
            pr = get_known_pricing()
            if model_id in pr:
                return pr[model_id]
        except Exception:
            pass
        return (fallback_inp, fallback_out)

    if "premium_cloud" not in provider_entries and os.environ.get("ANTHROPIC_API_KEY"):
        _sonnet_p = _p("claude-sonnet-4-6", 0.003, 0.015)
        provider_entries["premium_cloud"] = ProviderEntry(
            name="anthropic", tier="premium",
            models={"default": "claude-sonnet-4-6", "high": "claude-opus-4-8"},
            cost_per_1k_input=_sonnet_p[0], cost_per_1k_output=_sonnet_p[1],
        )

    gemini_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if "gemini" not in provider_entries and "cheap_cloud" not in provider_entries and gemini_key:
        _gemini_p = _p("gemini-2.5-flash", 0.0003, 0.0025)
        provider_entries["cheap_cloud"] = ProviderEntry(
            name="gemini", tier="cheap",
            models={"default": "gemini-2.5-flash", "high": "gemini-2.5-pro"},
            cost_per_1k_input=_gemini_p[0], cost_per_1k_output=_gemini_p[1],
        )

    if "cheap_cloud" not in provider_entries and os.environ.get("OPENROUTER_API_KEY"):
        _or_p = _p("deepseek/deepseek-v4-flash", 0.000098, 0.000196)
        provider_entries["cheap_cloud"] = ProviderEntry(
            name="openrouter", tier="cheap",
            models={"default": "deepseek/deepseek-v4-flash"},
            cost_per_1k_input=_or_p[0], cost_per_1k_output=_or_p[1],
        )

    # Build router
    router = Router(
        providers=provider_entries,
        tier_map=routing_config.tier_map,
        thresholds=routing_config.thresholds,
        local_only=routing_config.local_only,
        health_tracker=health,
        budget_tracker=budget,
    )

    return {
        "router": router,
        "health": health,
        "budget": budget,
        "provider_entries": provider_entries,
        "config": config,
        "routing_config": routing_config,
    }


def _build_provider_entries(configs: dict[str, ProviderConfig]) -> dict[str, ProviderEntry]:
    """Convert ProviderConfigs to ProviderEntries for the Router."""
    entries = {}
    for name, pc in configs.items():
        entries[name] = ProviderEntry(
            name=pc.name,
            tier=pc.tier,
            models=pc.models,
            cost_per_1k_input=pc.cost_per_1k_input,
            cost_per_1k_output=pc.cost_per_1k_output,
        )
    return entries


def _detect_local() -> ProviderEntry | None:
    """Try to detect a local LLM server (LOCAL_LLM_URL, OLLAMA_URL, then port 8000)."""
    import urllib.request
    from mullet.core.env import local_llm_url, ollama_url
    for base_url in (local_llm_url(), f"{ollama_url()}/v1", "http://localhost:8000/v1"):
        try:
            url = f"{base_url}/models"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return ProviderEntry(
                        name="local", tier="free",
                        models={"default": "default", "low": "default"},
                    )
        except Exception:
            continue
    return None
