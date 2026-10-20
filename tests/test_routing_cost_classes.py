"""Routing-v2 Phase E — cost-class corrections (TDD).

RED phase: all tests must FAIL before the code changes (phase E) are applied.
GREEN phase: all tests pass after the four source changes.

Corrections being validated:
1. Local LLM is a candidate at every tier (free, always reachable).
2. Gemini is NOT cheap — demoted to last-resort in both ordering lists.
3. Anthropic (prepaid via Claude Max) free-first selection preserved.
4. cheap_cloud alias is NOT filled by gemini (only kimi or openrouter).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mullet.core.router.complexity import PROVIDER_TIER_CAPABILITIES, Tier
from mullet.core.router.decision import ProviderEntry, Router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_router(provider_names: list[str] | None = None, **kwargs) -> Router:
    """Build a Router with the given providers (or a sane default set)."""
    _all = {
        "local":         ProviderEntry(name="local",         tier="free",    models={"default": "llama-8b"}),
        "kimi":          ProviderEntry(name="kimi",          tier="cheap",   models={"default": "kimi-k2.7-code"}),
        "gemini":        ProviderEntry(name="gemini",        tier="cheap",   models={"default": "gemini-2.5-flash"}),
        "haiku":         ProviderEntry(name="haiku",         tier="cheap",   models={"default": "claude-haiku-4-5"}),
        "sonnet":        ProviderEntry(name="sonnet",        tier="premium", models={"default": "claude-sonnet-4-6"}),
        "opus":          ProviderEntry(name="opus",          tier="premium", models={"default": "claude-opus-4-8"}),
        "opus-4-8":      ProviderEntry(name="opus-4-8",      tier="premium", models={"default": "claude-opus-4-8"}),
        "cheap_cloud":   ProviderEntry(name="cheap_cloud",   tier="cheap",   models={"default": "kimi-k2.7-code"}),
        "premium_cloud": ProviderEntry(name="premium_cloud", tier="premium", models={"default": "claude-sonnet-4-6"}),
        "openrouter":    ProviderEntry(name="openrouter",    tier="cheap",   models={"default": "deepseek/deepseek-v4-flash"}),
    }
    if provider_names is not None:
        providers = {n: _all[n] for n in provider_names if n in _all}
    else:
        providers = _all
    kwargs.setdefault("use_pressure_routing", False)
    return Router(providers=providers, **kwargs)


# ---------------------------------------------------------------------------
# 1. local is capable at ALL tiers (ADVANCED and CRITICAL included)
# ---------------------------------------------------------------------------


class TestLocalAllTierCapability:
    """Change 1: PROVIDER_TIER_CAPABILITIES["local"] must include ADVANCED and CRITICAL."""

    def test_local_serves_advanced(self):
        caps = PROVIDER_TIER_CAPABILITIES.get("local", set())
        assert "ADVANCED" in caps, (
            f"local must be capable of ADVANCED after Phase E, got {caps}"
        )

    def test_local_serves_critical(self):
        caps = PROVIDER_TIER_CAPABILITIES.get("local", set())
        assert "CRITICAL" in caps, (
            f"local must be capable of CRITICAL after Phase E, got {caps}"
        )

    def test_local_still_serves_low_and_standard(self):
        caps = PROVIDER_TIER_CAPABILITIES.get("local", set())
        assert "LOW" in caps, "local must retain LOW capability"
        assert "STANDARD" in caps, "local must retain STANDARD capability"

    def test_pick_provider_for_tier_critical_local_only(self):
        """pick_provider_for_tier('CRITICAL', {local}, 0.05) must return 'local', not None.

        Before Phase E, local was not CRITICAL-capable, so this returned None.
        After Phase E, local serves every tier, so it must return 'local'.
        """
        router = _make_router(["local", "kimi"])
        result = router.pick_provider_for_tier(
            "CRITICAL",
            frozenset({"local"}),
            pressure=0.05,
        )
        assert result == "local", (
            f"local must now serve CRITICAL (Phase E), expected 'local', got {result!r}"
        )

    def test_pick_provider_for_tier_advanced_local_only(self):
        """pick_provider_for_tier('ADVANCED', {local}, 0.85) must return 'local', not None."""
        router = _make_router(["local"])
        result = router.pick_provider_for_tier(
            "ADVANCED",
            frozenset({"local"}),
            pressure=0.85,
        )
        assert result == "local", (
            f"local must serve ADVANCED after Phase E, got {result!r}"
        )

    def test_tiers_for_local_includes_advanced_and_critical(self):
        """tiers_for_providers({'local'}) must now return all 4 tiers."""
        from mullet.core.router.complexity import tiers_for_providers
        tiers = tiers_for_providers({"local"})
        assert "ADVANCED" in tiers, f"ADVANCED must be in local's tiers: {tiers}"
        assert "CRITICAL" in tiers, f"CRITICAL must be in local's tiers: {tiers}"


# ---------------------------------------------------------------------------
# 2. Gemini is demoted — last in both ordering lists
# ---------------------------------------------------------------------------


class TestGeminiDemotion:
    """Change 2: gemini must be LAST in _ALLOWLIST_FALLBACK_ORDER and _ALLOWLIST_TIER_PICK_ORDER."""

    def test_gemini_last_in_fallback_order(self):
        router = _make_router()
        order = router._ALLOWLIST_FALLBACK_ORDER
        assert "gemini" in order, "gemini must still appear in _ALLOWLIST_FALLBACK_ORDER"
        assert order[-1] == "gemini", (
            f"gemini must be LAST in _ALLOWLIST_FALLBACK_ORDER, "
            f"but got {order!r} — gemini is at index {order.index('gemini')}"
        )

    def test_gemini_last_in_tier_pick_order(self):
        router = _make_router()
        order = router._ALLOWLIST_TIER_PICK_ORDER
        assert "gemini" in order, "gemini must still appear in _ALLOWLIST_TIER_PICK_ORDER"
        assert order[-1] == "gemini", (
            f"gemini must be LAST in _ALLOWLIST_TIER_PICK_ORDER, "
            f"but got {order!r} — gemini is at index {order.index('gemini')}"
        )

    def test_openrouter_in_fallback_order(self):
        """openrouter must appear in _ALLOWLIST_FALLBACK_ORDER."""
        router = _make_router()
        order = router._ALLOWLIST_FALLBACK_ORDER
        assert "openrouter" in order, (
            f"openrouter must be in _ALLOWLIST_FALLBACK_ORDER, got {order!r}"
        )

    def test_openrouter_in_tier_pick_order(self):
        """openrouter must appear in _ALLOWLIST_TIER_PICK_ORDER."""
        router = _make_router()
        order = router._ALLOWLIST_TIER_PICK_ORDER
        assert "openrouter" in order, (
            f"openrouter must be in _ALLOWLIST_TIER_PICK_ORDER, got {order!r}"
        )

    def test_pick_allowlisted_prefers_kimi_over_gemini(self):
        """When kimi and gemini both in allowlist for STANDARD, kimi is preferred, not gemini."""
        router = _make_router(["kimi", "gemini", "openrouter"])
        with patch("mullet.hud.guard.recommend_provider", return_value="haiku"):
            result = router.pick_provider_for_tier(
                "STANDARD",
                frozenset({"kimi", "gemini"}),
                pressure=0.05,
            )
        # kimi must be preferred over gemini (gemini is last-resort)
        assert result == "kimi", (
            f"kimi must be preferred over gemini for STANDARD allowlist, got {result!r}"
        )

    def test_pick_allowlisted_prefers_openrouter_over_gemini(self):
        """When openrouter and gemini both in allowlist for STANDARD, openrouter is preferred."""
        router = _make_router(["openrouter", "gemini"])
        with patch("mullet.hud.guard.recommend_provider", return_value="haiku"):
            result = router.pick_provider_for_tier(
                "STANDARD",
                frozenset({"openrouter", "gemini"}),
                pressure=0.05,
            )
        # openrouter must be preferred over gemini
        assert result == "openrouter", (
            f"openrouter must be preferred over gemini for STANDARD allowlist, got {result!r}"
        )

    def test_gemini_picked_only_when_sole_option(self):
        """Gemini is used only as last resort when it's the only option in the allowlist."""
        router = _make_router(["gemini"])
        with patch("mullet.hud.guard.recommend_provider", return_value="haiku"):
            result = router.pick_provider_for_tier(
                "STANDARD",
                frozenset({"gemini"}),
                pressure=0.05,
            )
        # gemini is STANDARD-capable, so it must be returned when it's the only option
        assert result == "gemini", (
            f"gemini must be returned when it's the sole option, got {result!r}"
        )


# ---------------------------------------------------------------------------
# 3. Anthropic free-first: recommend_provider never returns gemini
# ---------------------------------------------------------------------------


class TestRecommendProviderNoGemini:
    """Verify guard.recommend_provider never recommends gemini at any tier/pressure."""

    def test_recommend_never_returns_gemini(self):
        """recommend_provider must never return 'gemini' for any tier/pressure combo."""
        from mullet.hud.guard import recommend_provider

        tiers = ["LOW", "STANDARD", "ADVANCED", "CRITICAL", "PREMIUM"]
        pressures = [0.0, 0.05, 0.30, 0.55, 0.60, 0.80, 0.85, 0.90, 1.0]

        for tier in tiers:
            for pressure in pressures:
                result = recommend_provider(pressure, tier)
                assert result != "gemini", (
                    f"recommend_provider({pressure}, {tier!r}) returned 'gemini' — "
                    f"gemini is not a valid recommendation (it's last-resort only)"
                )


# ---------------------------------------------------------------------------
# 4. Free-first selection: verify the correct tier ordering
# ---------------------------------------------------------------------------


class TestFreeFirstSelection:
    """Change 3 verification: Anthropic free-first selection for prepaid budget tiers.

    LOW → local (free, don't waste prepaid on trivial tasks)
    STANDARD → haiku (Anthropic prepaid at low pressure)
    ADVANCED → sonnet (Anthropic prepaid at low pressure)
    """

    def test_low_pressure_low_tier_picks_local(self):
        """LOW tier + low pressure → local (free, don't burn prepaid)."""
        router = _make_router(["local", "haiku", "kimi", "sonnet"])
        with patch("mullet.hud.guard.recommend_provider", return_value="local"):
            result = router.pick_provider_for_tier("LOW", None, pressure=0.05)
        assert result == "local", (
            f"LOW tier must route to local (free), not waste prepaid, got {result!r}"
        )

    def test_low_pressure_standard_tier_picks_haiku(self):
        """STANDARD + low pressure + no allowlist → haiku (Anthropic prepaid)."""
        router = _make_router(["local", "kimi", "haiku", "sonnet"])
        with patch("mullet.hud.guard.recommend_provider", return_value="haiku"):
            result = router.pick_provider_for_tier("STANDARD", None, pressure=0.05)
        assert result == "haiku", (
            f"STANDARD + low pressure → haiku (prepaid), got {result!r}"
        )

    def test_low_pressure_advanced_tier_picks_sonnet(self):
        """ADVANCED + low pressure + no allowlist → sonnet (Anthropic prepaid)."""
        router = _make_router(["local", "kimi", "haiku", "sonnet"])
        with patch("mullet.hud.guard.recommend_provider", return_value="sonnet"):
            result = router.pick_provider_for_tier("ADVANCED", None, pressure=0.05)
        assert result == "sonnet", (
            f"ADVANCED + low pressure → sonnet (prepaid), got {result!r}"
        )


# ---------------------------------------------------------------------------
# 5. Swarm dispatcher — gemini demoted in _PREFERENCE
# ---------------------------------------------------------------------------


class TestSwarmDispatcherGeminiDemotion:
    """Change 4: gemini must be LAST in SwarmDispatcher._PREFERENCE."""

    def test_gemini_last_in_swarm_preference(self):
        from mullet.core.orchestration.swarm_dispatcher import SwarmDispatcher
        pref = SwarmDispatcher._PREFERENCE if hasattr(SwarmDispatcher, "_PREFERENCE") else None
        # _PREFERENCE is defined inside _pick_task_provider; we need to inspect it
        # by looking at the source or by checking the actual pick behaviour.
        # If _PREFERENCE is a class attribute, use it directly:
        import inspect
        src = inspect.getsource(SwarmDispatcher._pick_task_provider)
        assert 'gemini' in src, "_PREFERENCE must still list gemini"
        # The last item before local must NOT be gemini (gemini must be after local or at end)
        # We verify by checking that kimi comes before gemini in the source
        kimi_pos = src.index('"kimi"')
        gemini_pos = src.index('"gemini"')
        assert kimi_pos < gemini_pos, (
            "In _pick_task_provider, 'kimi' must appear before 'gemini' in the preference list"
        )
        # Also verify local comes before gemini (gemini is last resort, even after local)
        local_pos = src.index('"local"')
        assert local_pos < gemini_pos, (
            "In _pick_task_provider, 'local' must appear before 'gemini' — gemini is last resort"
        )

    def test_swarm_prefers_kimi_over_gemini(self):
        """With kimi+gemini allowlist, _pick_task_provider must return kimi, not gemini."""
        from mullet.core.orchestration.swarm_dispatcher import SwarmDispatcher

        # Patch build_default_router to return None so it falls through to static preference
        with patch(
            "mullet.core.router.resolver.build_default_router",
            side_effect=RuntimeError("no router"),
        ):
            d = SwarmDispatcher(allowed_providers=frozenset({"kimi", "gemini"}))
            result = d._pick_task_provider("STANDARD")

        assert result == "kimi", (
            f"With kimi+gemini, swarm must prefer kimi over gemini, got {result!r}"
        )

    def test_swarm_prefers_openrouter_over_gemini(self):
        """With openrouter+gemini allowlist, _pick_task_provider must return openrouter."""
        from mullet.core.orchestration.swarm_dispatcher import SwarmDispatcher

        with patch(
            "mullet.core.router.resolver.build_default_router",
            side_effect=RuntimeError("no router"),
        ):
            d = SwarmDispatcher(allowed_providers=frozenset({"openrouter", "gemini"}))
            result = d._pick_task_provider("STANDARD")

        assert result == "openrouter", (
            f"With openrouter+gemini, swarm must prefer openrouter over gemini, got {result!r}"
        )


# ---------------------------------------------------------------------------
# 6. resolver.py — gemini must NOT fill cheap_cloud alias
# ---------------------------------------------------------------------------


class TestGeminiNotCheapCloud:
    """Change 3: gemini must not fill the cheap_cloud alias in build_default_router.

    cheap_cloud should only be filled by kimi or openrouter.
    """

    def test_cheap_cloud_not_gemini_when_kimi_available(self):
        """When kimi key is set, cheap_cloud must be kimi, not gemini."""
        from mullet.core.router.resolver import build_default_router

        with patch.dict(
            "os.environ",
            {
                "KIMI_API_KEY": "test-kimi-key",
                "GOOGLE_API_KEY": "test-gemini-key",
                "ANTHROPIC_API_KEY": "",
                "OPENROUTER_API_KEY": "",
            },
            clear=False,
        ):
            # Patch detect_local_provider to return None so local doesn't interfere
            with patch("mullet.cli.main.detect_local_provider", return_value=None):
                # Patch Claude Max to avoid reading real OAuth token
                with patch(
                    "mullet.core.providers.claude_max._read_oauth_token",
                    return_value=None,
                ):
                    router, _ = build_default_router()

        cheap = router.providers.get("cheap_cloud")
        assert cheap is not None, "cheap_cloud must be registered when kimi key is set"
        assert cheap.name == "kimi", (
            f"cheap_cloud must be kimi when KIMI_API_KEY is set, got {cheap.name!r}"
        )

    def test_cheap_cloud_not_gemini_when_only_gemini_available(self):
        """When only gemini key is set (no kimi, no openrouter), cheap_cloud must NOT be gemini.

        After Phase E, gemini no longer fills cheap_cloud — so cheap_cloud is absent.
        """
        from mullet.core.router.resolver import build_default_router

        with patch.dict(
            "os.environ",
            {
                "KIMI_API_KEY": "",
                "MOONSHOT_API_KEY": "",
                "GOOGLE_API_KEY": "test-gemini-key",
                "ANTHROPIC_API_KEY": "",
                "OPENROUTER_API_KEY": "",
            },
            clear=False,
        ):
            with patch("mullet.cli.main.detect_local_provider", return_value=None):
                with patch(
                    "mullet.core.providers.claude_max._read_oauth_token",
                    return_value=None,
                ):
                    router, _ = build_default_router()

        cheap = router.providers.get("cheap_cloud")
        # After Phase E: gemini does NOT fill cheap_cloud
        assert cheap is None or cheap.name != "gemini", (
            f"After Phase E, cheap_cloud must NOT be gemini, got {cheap!r}"
        )
        # gemini named alias must still be registered
        gemini = router.providers.get("gemini")
        assert gemini is not None, "gemini named provider must still be registered"
        assert gemini.name == "gemini", f"gemini entry must have name='gemini', got {gemini.name!r}"

    def test_cheap_cloud_filled_by_openrouter_when_no_kimi(self):
        """When openrouter key is set and kimi is absent, cheap_cloud → openrouter (not gemini)."""
        from mullet.core.router.resolver import build_default_router

        with patch.dict(
            "os.environ",
            {
                "KIMI_API_KEY": "",
                "MOONSHOT_API_KEY": "",
                "GOOGLE_API_KEY": "test-gemini-key",
                "ANTHROPIC_API_KEY": "",
                "OPENROUTER_API_KEY": "test-openrouter-key",
            },
            clear=False,
        ):
            with patch("mullet.cli.main.detect_local_provider", return_value=None):
                with patch(
                    "mullet.core.providers.claude_max._read_oauth_token",
                    return_value=None,
                ):
                    router, _ = build_default_router()

        cheap = router.providers.get("cheap_cloud")
        assert cheap is not None, "cheap_cloud must be registered when openrouter key is set"
        assert cheap.name == "openrouter", (
            f"cheap_cloud must be openrouter when OPENROUTER_API_KEY is set (not gemini), "
            f"got {cheap.name!r}"
        )
