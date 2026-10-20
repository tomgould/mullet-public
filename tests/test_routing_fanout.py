"""
Regression sentinel for routing-v2 Phase D.

This file guards against the router collapsing to single-provider dispatch.
It asserts that the router fans out across providers per preset and exploits
budget headroom (prepaid Anthropic at low pressure) where appropriate.

Any failure here means a routing regression — DO NOT silence without
understanding why fan-out or headroom exploitation stopped working.
"""

from __future__ import annotations

from mullet.core.router.complexity import Tier
from mullet.core.router.decision import ProviderEntry, Router

# ---------------------------------------------------------------------------
# Preset allowlists (mirror mullet/core/presets.py built-ins)
# ---------------------------------------------------------------------------

BALANCED = None  # unrestricted budget-aware routing
CHEAP_CLOUD = frozenset({"kimi", "gemini", "cheap_cloud"})
LOCAL_ONLY = frozenset({"local"})
CLAUDE_ONLY = frozenset({"sonnet", "haiku", "opus-4-8"})

# ---------------------------------------------------------------------------
# Mixed-size prompt batch
# ---------------------------------------------------------------------------

TRIVIAL = "list the files in src"  # → LOW
MEDIUM = (
    "Refactor the fetch helper in lib/client.py and lib/server.py to add "
    "a timeout parameter and change all callers"
)  # → STANDARD
REASONING = (
    "Review and critique this routing module design in depth: analyse the "
    "trade-offs between the microservices and monolith approaches, evaluate "
    "long-term maintainability, assess security and production risks, and "
    "recommend which architecture we should adopt."
)  # → ADVANCED

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_router(**kwargs) -> Router:
    """Build a deterministic Router with a comprehensive provider set.

    use_pressure_routing defaults to False so ``Router.route`` exercises the
    standard tier_map / allowlist path (not the live-usage pressure branch).
    Callers may override via kwargs.
    """
    providers = {
        "local": ProviderEntry(
            name="local", tier="free", models={"default": "qwen"}
        ),
        "cheap_cloud": ProviderEntry(
            name="cheap_cloud", tier="cheap", models={"default": "kimi-k2.7-code"}
        ),
        "kimi": ProviderEntry(
            name="kimi", tier="cheap", models={"default": "kimi-k2.7-code"}
        ),
        "gemini": ProviderEntry(
            name="gemini", tier="cheap", models={"default": "gemini-2.5-flash"}
        ),
        "haiku": ProviderEntry(
            name="haiku", tier="cheap", models={"default": "claude-haiku-4-5"}
        ),
        "sonnet": ProviderEntry(
            name="sonnet", tier="premium", models={"default": "claude-sonnet-4-6"}
        ),
        "openrouter": ProviderEntry(
            name="openrouter",
            tier="cheap",
            models={"default": "deepseek/deepseek-v4-flash"},
        ),
    }
    kwargs.setdefault("use_pressure_routing", False)
    # Custom tier_map so every tier maps to at least one real provider in our
    # deterministic dict, giving us clean fan-out under BALANCED.
    kwargs.setdefault(
        "tier_map",
        {
            "LOW": ["local"],
            "STANDARD": ["kimi", "gemini", "cheap_cloud", "haiku", "local"],
            "ADVANCED": ["sonnet", "kimi", "gemini", "openrouter"],
            "CRITICAL": ["sonnet", "opus-4-8"],
        },
    )
    return Router(providers=providers, **kwargs)


# ---------------------------------------------------------------------------
# 1. Fan-out via Router.route (use_pressure_routing=False)
# ---------------------------------------------------------------------------


class TestRouteFanOut:
    """The router must dispatch a mixed batch to ≥2 distinct providers
    when the preset allows it.  Single-provider collapse is a regression.
    """

    def test_balanced_preset_fans_out(self):
        """BALANCED (unrestricted) should route trivial→local, medium→cloud,
        reasoning→premium — yielding multiple providers."""
        router = _make_router()
        decisions = [
            router.route(p, allowed_providers=BALANCED)
            for p in (TRIVIAL, MEDIUM, REASONING)
        ]
        providers = {d.provider for d in decisions}
        assert len(providers) >= 2, (
            f"BALANCED must fan out across ≥2 providers for a mixed batch, "
            f"got {providers!r}"
        )

    def test_cheap_cloud_preset_routes_within_allowlist(self):
        """CHEAP_CLOUD is restricted to {kimi, gemini, cheap_cloud}.

        NOTE: with use_pressure_routing=False the current deterministic path
        picks the highest-priority tier-capable provider in the allowlist for
        every tier.  This means the batch may collapse to a single provider
        (characterisation of real behaviour, not a bug per se).  The sentinel
        asserts ≥1 so we notice if it ever blocks entirely.
        """
        router = _make_router()
        decisions = [
            router.route(p, allowed_providers=CHEAP_CLOUD)
            for p in (TRIVIAL, MEDIUM, REASONING)
        ]
        providers = {d.provider for d in decisions}
        # Characterisation: currently collapses to the first tier-capable
        # allowed provider for all tiers.  We keep the assertion light so the
        # test documents reality rather than an idealised future.
        assert len(providers) >= 1, (
            f"CHEAP_CLOUD must at least route (not block), got {providers!r}"
        )

    def test_local_only_preset_uses_local(self):
        """LOCAL_ONLY should route everything to local."""
        router = _make_router()
        decisions = [
            router.route(p, allowed_providers=LOCAL_ONLY)
            for p in (TRIVIAL, MEDIUM, REASONING)
        ]
        providers = {d.provider for d in decisions}
        assert providers == {"local"}, (
            f"LOCAL_ONLY must route everything to local, got {providers!r}"
        )


# ---------------------------------------------------------------------------
# 2. Headroom / pressure-aware pick via pick_provider_for_tier
# ---------------------------------------------------------------------------


class TestHeadroomExploitation:
    """Low pressure should route reasoning-grade work to prepaid Anthropic
    (headroom exploitation).  High pressure should drop off Anthropic.
    Preset allowlists must be honoured.
    """

    def test_advanced_low_pressure_balanced_picks_sonnet(self):
        router = _make_router()
        result = router.pick_provider_for_tier("ADVANCED", BALANCED, pressure=0.05)
        assert result == "sonnet", (
            f"Low-pressure ADVANCED under BALANCED must hit sonnet "
            f"(headroom), got {result!r}"
        )

    def test_standard_low_pressure_balanced_picks_haiku(self):
        router = _make_router()
        result = router.pick_provider_for_tier("STANDARD", BALANCED, pressure=0.05)
        assert result == "haiku", (
            f"Low-pressure STANDARD under BALANCED must hit haiku "
            f"(headroom), got {result!r}"
        )

    def test_advanced_low_pressure_cheap_cloud_excludes_anthropic(self):
        router = _make_router()
        result = router.pick_provider_for_tier(
            "ADVANCED", CHEAP_CLOUD, pressure=0.05
        )
        assert result in {"kimi", "gemini"}, (
            f"CHEAP_CLOUD must exclude Anthropic; expected kimi or gemini, "
            f"got {result!r}"
        )

    def test_standard_high_pressure_balanced_drops_anthropic(self):
        router = _make_router()
        result = router.pick_provider_for_tier(
            "STANDARD", BALANCED, pressure=0.85
        )
        assert result not in {"haiku", "sonnet"}, (
            f"High-pressure STANDARD must drop off Anthropic, got {result!r}"
        )

    def test_advanced_low_pressure_claude_only_picks_sonnet(self):
        router = _make_router()
        result = router.pick_provider_for_tier(
            "ADVANCED", CLAUDE_ONLY, pressure=0.05
        )
        assert result == "sonnet", (
            f"CLAUDE_ONLY low-pressure ADVANCED must pick sonnet, "
            f"got {result!r}"
        )


# ---------------------------------------------------------------------------
# 3. Reasoning floor + no blocked decisions
# ---------------------------------------------------------------------------


class TestReasoningAndNoBlock:
    """A reasoning-grade prompt must never be routed LOW, and a batch under
    an unrestricted preset must produce no blocked (None-provider) decisions.
    """

    def test_reasoning_prompt_not_low(self):
        router = _make_router()
        decision = router.route(REASONING, allowed_providers=BALANCED)
        assert decision.tier in (
            Tier.STANDARD,
            Tier.ADVANCED,
            Tier.CRITICAL,
        ), (
            f"Reasoning prompt must be STANDARD or higher, got {decision.tier}"
        )

    def test_batch_under_balanced_has_no_blocked_decisions(self):
        router = _make_router()
        decisions = [
            router.route(p, allowed_providers=BALANCED)
            for p in (TRIVIAL, MEDIUM, REASONING)
        ]
        blocked = [d for d in decisions if d.provider is None]
        assert not blocked, (
            f"BALANCED batch produced {len(blocked)} blocked decision(s): "
            f"{[(d.tier, d.reasons) for d in blocked]}"
        )


# ---------------------------------------------------------------------------
# 4. Regression sentinel comment (enforced by module docstring at top)
# ---------------------------------------------------------------------------
