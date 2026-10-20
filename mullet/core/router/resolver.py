"""Provider resolver — bridges the Router with actual Provider instances.

Creates Provider objects from ProviderEntry configs, resolving
credentials from environment variables.
"""

import logging
import os

from mullet.core.providers.base import Provider
from mullet.core.providers.openai_compat import OpenAICompatibleProvider
from mullet.core.router.decision import ProviderEntry, Router, RoutingDecision

logger = logging.getLogger(__name__)


class ProviderResolver:
    """Resolves routing decisions to actual Provider instances."""

    def __init__(self, router: Router):
        self.router = router
        self._provider_cache: dict[str, Provider] = {}

    def resolve(self, decision: RoutingDecision) -> Provider | None:
        """Get a Provider instance for a routing decision."""
        tier_str = decision.tier.value if hasattr(decision.tier, "value") else str(decision.tier)
        cache_key = f"{decision.provider}:{tier_str}"
        if cache_key in self._provider_cache:
            return self._provider_cache[cache_key]

        entry = self.router.providers.get(decision.provider)
        if not entry:
            return None

        provider = self._create_provider(entry, decision.model, tier=tier_str, allowed_providers=getattr(decision, "allowed_providers", None))
        if provider:
            self._provider_cache[cache_key] = provider
        return provider

    def route_and_resolve(
        self,
        task: str,
        agent_type: str | None = None,
        force_tier: str | None = None,
        force_provider: str | None = None,
        allowed_providers: frozenset[str] | None = None,
    ) -> tuple[Provider | None, RoutingDecision]:
        """Route a task and resolve to a Provider in one step, honouring allowed_providers if set."""
        decision = self.router.route(
            task, agent_type, force_tier, force_provider,
            allowed_providers=allowed_providers,
        )
        provider = self.resolve(decision)
        tier_str = decision.tier.value if hasattr(decision.tier, "value") else str(decision.tier)

        # If primary fails, try fallback chain
        if not provider:
            for fallback_name in decision.fallback_chain:
                fb_entry = self.router.providers.get(fallback_name)
                if fb_entry:
                    fb_provider = self._create_provider(fb_entry, "", tier=tier_str, allowed_providers=getattr(decision, "allowed_providers", None))
                    if fb_provider:
                        decision.provider = fallback_name
                        decision.reasons.append(f"fallback: {fallback_name}")
                        return fb_provider, decision

        # Log the routing decision so users can see why a provider was chosen.
        try:
            from mullet.core.routing_history import append_decision_log
            model_name = getattr(provider, "model", decision.model) if provider else decision.model
            append_decision_log(
                provider=decision.provider,
                model=model_name or "",
                tier=tier_str,
                reasons=list(decision.reasons),
                task_preview=task,
            )
        except Exception as _exc:
            logger.debug("resolver: decision log failed: %s", _exc)

        return provider, decision

    def _create_provider(
        self,
        entry: ProviderEntry,
        model: str,
        tier: str = "STANDARD",
        context_needed: int = 4096,
        *,
        allowed_providers: frozenset[str] | set[str] | None = None,
        context_tokens: int | None = None,
    ) -> Provider | None:
        """Create a Provider instance from a ProviderEntry.

        For API providers (kimi, gemini, anthropic) the model is selected
        dynamically via ModelRegistry.get_best_model() — picking the best fit
        for the given tier and context window. Named aliases (haiku/sonnet/opus)
        are always honoured as explicit user choices.

        ``context_tokens`` is threaded to ``get_best_model`` so the D2 post-pick
        assertion can reject candidates whose max_context cannot hold the prompt.
        """
        def _dynamic_model(provider_name: str, fallback: str) -> str:
            try:
                from mullet.core.providers.model_registry import get_registry
                return get_registry().get_best_model(
                    provider_name, tier, context_needed,
                    context_tokens=context_tokens,
                )
            except Exception:
                return fallback or model

        if entry.name in ("local", "lmstudio", "ollama"):
            from mullet.core.providers.openai_compat import is_escalation_recommended
            if is_escalation_recommended("local", allowed=allowed_providers):
                logger.warning("Local provider has hit failure threshold — skipping, will use cloud fallback")
                return None
            base_url = getattr(entry, "base_url", "") or "http://localhost:1234/v1"
            # Use context-aware local model selection if no explicit model given
            if not model or model == "default":
                try:
                    from mullet.core.providers.lmstudio_manager import LMStudioManager
                    mgr = LMStudioManager()
                    best = mgr.pick_best_model(context_needed=context_needed, complexity=tier)
                    if best:
                        model = best.get("id", model)
                except Exception:
                    pass
            return OpenAICompatibleProvider(base_url=base_url, model=model)

        if entry.name in ("anthropic",):
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                from mullet.core.providers.anthropic import AnthropicProvider
                # Use adaptive model selection unless explicit model provided
                chosen = model if model and model != "default" else _dynamic_model("anthropic", "claude-sonnet-5")
                return AnthropicProvider(api_key=api_key, model=chosen)

        if entry.name in ("haiku", "sonnet", "opus", "opus-4-8", "opus-4-7"):
            # Explicit user-chosen alias — honour the static model, no dynamic selection
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                from mullet.core.providers.anthropic import AnthropicProvider
                return AnthropicProvider(api_key=api_key, model=model)

        if entry.name == "gemini":
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if api_key:
                try:
                    from mullet.core.providers.gemini import GeminiProvider
                    # Use adaptive model selection unless explicit model provided
                    chosen = model if model and model != "default" else _dynamic_model("gemini", "gemini-2.5-flash")
                    return GeminiProvider(api_key=api_key, model=chosen)
                except ImportError:
                    logger.warning("Gemini provider requested but 'mullet.core.providers.gemini' not found.")

        if entry.name == "kimi":
            api_key = os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
            if api_key:
                try:
                    from mullet.core.providers.kimi import KimiProvider
                    # Use adaptive model selection unless explicit model provided
                    chosen = model if model and model != "default" else _dynamic_model("kimi", "kimi-k2.7-code")
                    return KimiProvider(api_key=api_key, model=chosen)
                except ImportError:
                    logger.warning("Kimi provider requested but not importable.")
            return None

        if entry.name == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                return None
            chosen = model if model and model != "default" else _dynamic_model("openrouter", "deepseek/deepseek-v4-flash")
            try:
                from mullet.core.providers.openrouter import OpenRouterProvider
                return OpenRouterProvider(model=chosen, api_key=api_key)
            except Exception:
                return OpenAICompatibleProvider(
                    base_url="https://openrouter.ai/api/v1", api_key=api_key, model=chosen
                )

        # Generic OpenAI-compatible
        api_key_env = getattr(entry, "api_key_env", "")
        api_key = os.environ.get(api_key_env, "no-key-required") if api_key_env else "no-key-required"
        base_url = getattr(entry, "base_url", "")
        if base_url:
            return OpenAICompatibleProvider(base_url=base_url, api_key=api_key, model=model)

        return None


def _pricing(model_id: str, fallback_inp: float, fallback_out: float) -> tuple[float, float]:
    try:
        from mullet.core.providers.pricing import get_known_pricing
        p = get_known_pricing()
        if model_id in p:
            return p[model_id]
    except Exception:
        pass
    return (fallback_inp, fallback_out)


def build_default_router(local_only: bool = False) -> tuple[Router, ProviderResolver]:
    """Build a Router with providers auto-detected from environment.

    Returns (router, resolver) ready to use.
    """
    from mullet.cli.main import detect_local_provider

    providers: dict[str, ProviderEntry] = {}

    # Local provider
    local = detect_local_provider()
    if local:
        providers["local"] = ProviderEntry(
            name="local", tier="free",
            models={"default": local.model, "low": local.model},
        )

    # Anthropic — register all tiers as named providers so delegation can target them directly.
    if os.environ.get("ANTHROPIC_API_KEY"):
        _premium_cloud_p = _pricing("claude-sonnet-5", 0.002, 0.01)
        providers["premium_cloud"] = ProviderEntry(
            name="anthropic", tier="premium",
            models={"default": "claude-sonnet-5", "high": "claude-opus-4-8"},
            cost_per_1k_input=_premium_cloud_p[0], cost_per_1k_output=_premium_cloud_p[1],
        )
        # Named aliases — usable via force_provider="haiku" / "sonnet" / "opus" / "opus-4-8" / "opus-4-7"
        _haiku_p = _pricing("claude-haiku-4-5-20251001", 0.001, 0.005)
        providers["haiku"] = ProviderEntry(
            name="haiku", tier="cheap",
            models={"default": "claude-haiku-4-5-20251001"},
            cost_per_1k_input=_haiku_p[0], cost_per_1k_output=_haiku_p[1],
        )
        _sonnet_p = _pricing("claude-sonnet-5", 0.002, 0.01)
        providers["sonnet"] = ProviderEntry(
            name="sonnet", tier="premium",
            models={"default": "claude-sonnet-5"},
            cost_per_1k_input=_sonnet_p[0], cost_per_1k_output=_sonnet_p[1],
        )
        _opus_p = _pricing("claude-opus-4-8", 0.005, 0.025)
        providers["opus"] = ProviderEntry(
            name="opus", tier="premium",
            models={"default": "claude-opus-4-8"},
            cost_per_1k_input=_opus_p[0], cost_per_1k_output=_opus_p[1],
        )
        _opus48_p = _pricing("claude-opus-4-8", 0.005, 0.025)
        providers["opus-4-8"] = ProviderEntry(
            name="opus-4-8", tier="premium",
            models={"default": "claude-opus-4-8"},
            cost_per_1k_input=_opus48_p[0], cost_per_1k_output=_opus48_p[1],
        )
        _opus47_p = _pricing("claude-opus-4-7", 0.005, 0.025)
        providers["opus-4-7"] = ProviderEntry(
            name="opus-4-7", tier="premium",
            models={"default": "claude-opus-4-7"},
            cost_per_1k_input=_opus47_p[0], cost_per_1k_output=_opus47_p[1],
        )

    # Kimi K2 (preferred cheap_cloud — 3M TPM, 100 concurrent connections)
    kimi_key = os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
    if kimi_key:
        _kimi_p = _pricing("kimi-k2.7-code", 0.00095, 0.004)
        if "cheap_cloud" not in providers:
            providers["cheap_cloud"] = ProviderEntry(
                name="kimi", tier="cheap",
                models={"default": "kimi-k2.7-code"},
                cost_per_1k_input=_kimi_p[0], cost_per_1k_output=_kimi_p[1],
            )
        # Always register named alias for direct targeting via force_provider="kimi"
        providers["kimi"] = ProviderEntry(
            name="kimi", tier="cheap",
            models={"default": "kimi-k2.7-code"},
            cost_per_1k_input=_kimi_p[0], cost_per_1k_output=_kimi_p[1],
        )

    # Gemini — last-resort provider (NOT cheap; pay-per-token and not cost-competitive).
    # Phase E: register named alias for direct targeting, but do NOT fill cheap_cloud alias.
    # cheap_cloud is filled by kimi or openrouter only (see blocks above and below).
    gemini_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        _gemini_p = _pricing("gemini-2.5-flash", 0.0003, 0.0025)
        providers["gemini"] = ProviderEntry(
            name="gemini", tier="cheap",
            models={"default": "gemini-2.5-flash", "medium": "gemini-2.5-flash", "high": "gemini-3.1-pro"},
            cost_per_1k_input=_gemini_p[0], cost_per_1k_output=_gemini_p[1],
        )
        # Phase E: cheap_cloud is NOT filled by gemini — only kimi or openrouter fill it.

    # OpenRouter — always register named alias for direct targeting via force_provider="openrouter".
    # Also fills cheap_cloud if Kimi is unavailable (Phase E: gemini no longer fills it).
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        _openrouter_p = _pricing("deepseek/deepseek-v4-flash", 0.000098, 0.000196)
        providers["openrouter"] = ProviderEntry(
            name="openrouter", tier="cheap",
            models={
                "default": "deepseek/deepseek-v4-flash",
                "fast":    "deepseek/deepseek-v4-flash",
                "high":    "deepseek/deepseek-v4-pro",
                "code":    "qwen/qwen3-coder",
                "think":   "deepseek/deepseek-v4-pro",
            },
            cost_per_1k_input=_openrouter_p[0], cost_per_1k_output=_openrouter_p[1],
        )
        if "cheap_cloud" not in providers:
            providers["cheap_cloud"] = ProviderEntry(
                name="openrouter", tier="cheap",
                models={"default": "deepseek/deepseek-v4-flash", "fast": "deepseek/deepseek-v4-flash"},
                cost_per_1k_input=_openrouter_p[0], cost_per_1k_output=_openrouter_p[1],
            )

    router = Router(providers=providers, local_only=local_only, use_pressure_routing=True)
    resolver = ProviderResolver(router)
    return router, resolver
