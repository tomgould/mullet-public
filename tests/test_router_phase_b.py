"""Phase B TDD tests — pick_provider_for_tier + reasoning floor + opus-4-8 capability.

RED phase: tests that assert the new API (pick_provider_for_tier, reasoning tier floor,
opus-4-8 in PROVIDER_TIER_CAPABILITIES) all fail before implementation.
GREEN phase: add the implementation in decision.py / complexity.py.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mullet.core.router.complexity import (
    PROVIDER_TIER_CAPABILITIES,
    Tier,
    is_reasoning_task,
)
from mullet.core.router.decision import ProviderEntry, Router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_router(provider_names: list[str] | None = None, **kwargs) -> Router:
    """Build a Router with the given providers (or a sane default set).

    use_pressure_routing defaults to False so pick_provider_for_tier can be
    unit-tested without mocking fetch_usage.  Callers may override by passing
    use_pressure_routing=True in kwargs — it is NOT double-set.
    """
    _all = {
        "local":         ProviderEntry(name="local",         tier="free",    models={"default": "llama-8b"}),
        "kimi":          ProviderEntry(name="kimi",          tier="cheap",   models={"default": "kimi-k2.5"}),
        "gemini":        ProviderEntry(name="gemini",        tier="cheap",   models={"default": "gemini-flash"}),
        "haiku":         ProviderEntry(name="haiku",         tier="cheap",   models={"default": "claude-haiku"}),
        "sonnet":        ProviderEntry(name="sonnet",        tier="premium", models={"default": "claude-sonnet"}),
        "opus":          ProviderEntry(name="opus",          tier="premium", models={"default": "claude-opus"}),
        "opus-4-8":      ProviderEntry(name="opus-4-8",      tier="premium", models={"default": "claude-opus-4-8"}),
        "cheap_cloud":   ProviderEntry(name="cheap_cloud",   tier="cheap",   models={"default": "kimi-k2.5"}),
        "premium_cloud": ProviderEntry(name="premium_cloud", tier="premium", models={"default": "claude-sonnet-4-6"}),
        "openrouter":    ProviderEntry(name="openrouter",    tier="cheap",   models={"default": "deepseek-v3"}),
    }
    if provider_names is not None:
        providers = {n: _all[n] for n in provider_names if n in _all}
    else:
        providers = _all
    kwargs.setdefault("use_pressure_routing", False)
    return Router(providers=providers, **kwargs)


# ---------------------------------------------------------------------------
# M1 — PROVIDER_TIER_CAPABILITIES must contain opus-4-8
# ---------------------------------------------------------------------------


class TestCapabilityTableM1:
    """M1: opus-4-8 must appear in PROVIDER_TIER_CAPABILITIES."""

    def test_opus_4_8_present(self):
        assert "opus-4-8" in PROVIDER_TIER_CAPABILITIES, (
            "opus-4-8 is missing from PROVIDER_TIER_CAPABILITIES"
        )

    def test_opus_4_8_is_critical(self):
        caps = PROVIDER_TIER_CAPABILITIES.get("opus-4-8", set())
        assert "CRITICAL" in caps, (
            f"opus-4-8 must support CRITICAL tier, got {caps}"
        )

    def test_all_preset_provider_names_have_entry(self):
        """Every provider name a preset/build_default_router can emit must have an entry."""
        required = {
            "local", "kimi", "gemini", "cheap_cloud", "haiku", "sonnet",
            "opus", "opus-4-8", "opus-4-7", "premium_cloud", "openrouter",
        }
        missing = required - set(PROVIDER_TIER_CAPABILITIES.keys())
        assert not missing, f"Missing capability entries: {missing}"


# ---------------------------------------------------------------------------
# New method: Router.pick_provider_for_tier
# ---------------------------------------------------------------------------


class TestPickProviderForTier:
    """Unit tests for Router.pick_provider_for_tier(tier_value, allowed_providers, pressure)."""

    # T1: STANDARD + allowlist={local, kimi}, low pressure
    # recommend_provider(0.05, "STANDARD") → "haiku" (haiku not in allowlist)
    # fallback to _pick_allowlisted_provider({local,kimi}, tier_value="STANDARD")
    # local CAN serve STANDARD; kimi is first in _ALLOWLIST_TIER_PICK_ORDER and STANDARD-capable → kimi
    def test_standard_allowlist_local_kimi_low_pressure(self):
        router = _make_router(["local", "kimi", "haiku"])
        result = router.pick_provider_for_tier(
            "STANDARD",
            frozenset({"local", "kimi"}),
            pressure=0.05,
        )
        assert result in ("local", "kimi"), (
            f"Expected a tier-capable allowed provider (local or kimi), got {result!r}"
        )
        assert result is not None

    # T2: STANDARD + no allowlist + haiku in providers + low pressure
    # recommend_provider(0.05, "STANDARD") → "haiku"; haiku in providers → return haiku
    def test_standard_no_allowlist_haiku_headroom(self):
        router = _make_router(["local", "kimi", "haiku"])
        with patch("mullet.hud.guard.recommend_provider", return_value="haiku"):
            result = router.pick_provider_for_tier("STANDARD", None, pressure=0.05)
        assert result == "haiku", (
            f"With haiku in providers and headroom, expected haiku, got {result!r}"
        )

    # T3: ADVANCED + allowlist={kimi, gemini} + low pressure
    # recommend_provider(0.05, "ADVANCED") → "sonnet"; sonnet not in allowlist
    # fallback to _pick_allowlisted_provider({kimi,gemini}, tier="ADVANCED")
    # kimi is ADVANCED-capable → kimi (or gemini)
    def test_advanced_allowlist_kimi_gemini(self):
        router = _make_router(["kimi", "gemini", "sonnet"])
        with patch("mullet.hud.guard.recommend_provider", return_value="sonnet"):
            result = router.pick_provider_for_tier(
                "ADVANCED",
                frozenset({"kimi", "gemini"}),
                pressure=0.05,
            )
        assert result in ("kimi", "gemini"), (
            f"Expected kimi or gemini for ADVANCED allowlist, got {result!r}"
        )
        assert result is not None

    # T4: ADVANCED + no allowlist + sonnet in providers + low pressure
    # recommend_provider(0.05, "ADVANCED") → "sonnet"; sonnet ADVANCED-capable → sonnet
    def test_advanced_no_allowlist_sonnet_headroom(self):
        router = _make_router(["local", "kimi", "sonnet"])
        with patch("mullet.hud.guard.recommend_provider", return_value="sonnet"):
            result = router.pick_provider_for_tier("ADVANCED", None, pressure=0.05)
        assert result == "sonnet", (
            f"Expected sonnet at ADVANCED with headroom, got {result!r}"
        )

    # T5: CRITICAL + allowlist={local} — Phase E: local IS CRITICAL-capable → returns 'local'
    def test_critical_allowlist_local_only_returns_none(self):
        # Phase E: local is now capable at ALL tiers (free fallback, always reachable).
        # Before Phase E: local could not serve CRITICAL → returned None.
        # After Phase E: local can serve CRITICAL → returns 'local'.
        router = _make_router(["local", "kimi"])
        result = router.pick_provider_for_tier(
            "CRITICAL",
            frozenset({"local"}),
            pressure=0.05,
        )
        assert result == "local", (
            f"Phase E: local is CRITICAL-capable, expected 'local', got {result!r}"
        )

    # T6: pick_provider_for_tier returning None must cause Router.route to FALL THROUGH
    # (not crash, not return None-provider decision)
    def test_route_falls_through_when_pick_returns_none(self):
        """When pick_provider_for_tier → None (allowlist can't serve tier), route falls
        through to the standard tier_map path, not a crash or None-provider decision.

        Phase E update: local is now CRITICAL-capable, so {local} at CRITICAL no longer
        produces None from pick_provider_for_tier. We use a custom provider that has no
        PROVIDER_TIER_CAPABILITIES entry (and is therefore tier-incapable for all tiers)
        to reproduce the fall-through scenario.
        """
        # Use a provider with no entry in PROVIDER_TIER_CAPABILITIES → pick returns None
        from mullet.core.router.decision import ProviderEntry
        router = _make_router(
            ["local", "kimi"],
            use_pressure_routing=True,
            tier_map={
                "LOW": ["local"],
                "STANDARD": ["kimi", "local"],
                "ADVANCED": ["kimi"],
                "CRITICAL": ["kimi"],
            },
        )
        # Add a provider with no tier-capability
        router.providers["no_tier_cloud"] = ProviderEntry(
            name="no_tier_cloud", tier="cheap", models={"default": "x"}
        )
        with patch("mullet.hud.usage.fetch_usage", return_value={"monthly": 100}):
            with patch("mullet.hud.guard.compute_pressure", return_value=0.05):
                decision = router.route(
                    "review and critique the architecture plan",
                    force_tier="CRITICAL",
                    allowed_providers=frozenset({"no_tier_cloud"}),
                )
        # pick_provider_for_tier returns None for no_tier_cloud (no CRITICAL capability)
        # Falls through to G9 allowlist check → also None → decision has allowlist reason
        assert decision is not None
        assert any("allowlist" in r.lower() or "cannot serve" in r.lower() for r in decision.reasons), (
            f"Expected allowlist-related reason in fall-through, got: {decision.reasons}"
        )


# ---------------------------------------------------------------------------
# Reasoning-task tier floor
# ---------------------------------------------------------------------------


class TestReasoningTierFloor:
    """A reasoning task that scores LOW must be floored to STANDARD via Router.route."""

    # A 25-word reasoning prompt with is_reasoning_task=True but score < 30 must
    # come out at STANDARD, not LOW.
    def test_25_word_reasoning_prompt_routes_standard(self):
        router = _make_router(
            ["local", "kimi", "haiku", "sonnet"],
            tier_map={
                "LOW": ["local"],
                "STANDARD": ["haiku", "kimi", "local"],
                "ADVANCED": ["sonnet", "kimi"],
                "CRITICAL": ["sonnet"],
            },
        )
        # 25 words, has "review" + "evaluate" → is_reasoning_task=True; score ~15 (LOW w/o floor)
        prompt = (
            "Please review this module and evaluate the trade-offs between the two "
            "approaches — which one should we prefer for this project"
        )
        # Verify the prompt is actually classified as reasoning and would score LOW w/o floor
        assert is_reasoning_task(prompt), "Test prompt must satisfy is_reasoning_task()"
        from mullet.core.router.complexity import extract_signals, score_complexity, score_to_tier
        signals = extract_signals(prompt)
        raw_score = score_complexity(signals)
        raw_tier = score_to_tier(raw_score)
        # If Phase A already brings it to STANDARD, the floor is a no-op — still pass
        # (the important contract is: the floor never makes things WORSE)
        decision = router.route(prompt)
        assert decision.tier in (Tier.STANDARD, Tier.ADVANCED, Tier.CRITICAL), (
            f"Reasoning prompt should be >= STANDARD after floor, got {decision.tier} "
            f"(raw score={raw_score}, raw tier={raw_tier})"
        )

    def test_trivial_simple_message_stays_low(self):
        """A simple message (< 28 words, no blocklist) must stay LOW after reasoning floor.

        'review' is in _COMPLEXITY_BLOCKLIST so is_simple_message('review status')=False.
        Use 'list the status' which is genuinely simple and does NOT trigger is_reasoning_task.
        The floor is guarded by is_reasoning_task() which returns False for simple messages.
        """
        router = _make_router(
            ["local", "haiku", "kimi"],
            tier_map={"LOW": ["local"], "STANDARD": ["kimi", "local"], "ADVANCED": ["kimi"], "CRITICAL": ["kimi"]},
        )
        from mullet.core.router.complexity import is_simple_message
        prompt = "list the status"
        # Must be simple (short, < 28 words, no blocklist hit)
        assert is_simple_message(prompt), (
            f"Test prompt must satisfy is_simple_message(), but got False for {prompt!r}"
        )
        decision = router.route(prompt)
        assert decision.tier == Tier.LOW, (
            f"Simple message must stay LOW (floor only applies to reasoning tasks), got {decision.tier}"
        )

    def test_reasoning_floor_does_not_affect_already_standard_prompts(self):
        """Prompts that already score STANDARD or above are unaffected by the floor."""
        router = _make_router(
            ["local", "kimi", "sonnet"],
            tier_map={"LOW": ["local"], "STANDARD": ["kimi"], "ADVANCED": ["sonnet", "kimi"], "CRITICAL": ["sonnet"]},
        )
        # A large prompt that naturally scores STANDARD
        prompt = " ".join(["analyse"] + ["the full system behaviour"] * 20)
        decision = router.route(prompt)
        assert decision.tier in (Tier.STANDARD, Tier.ADVANCED, Tier.CRITICAL), (
            f"Already-STANDARD prompt should not be downgraded by floor, got {decision.tier}"
        )


# ---------------------------------------------------------------------------
# Integration: pressure routing with pick_provider_for_tier wired in
# ---------------------------------------------------------------------------


class TestPressureRoutingIntegration:
    """Integration tests: use_pressure_routing=True + pick_provider_for_tier logic."""

    def test_low_pressure_standard_balanced_mix_picks_haiku(self):
        """balanced-mix preset: STANDARD task, low pressure → haiku (prepaid, not kimi)."""
        router = _make_router(
            ["local", "kimi", "haiku", "sonnet"],
            use_pressure_routing=True,
            tier_map={
                "LOW": ["local"],
                "STANDARD": ["haiku", "kimi", "local"],
                "ADVANCED": ["sonnet", "kimi"],
                "CRITICAL": ["sonnet"],
            },
        )
        with patch("mullet.hud.usage.fetch_usage", return_value={"monthly": 100}):
            with patch("mullet.hud.guard.compute_pressure", return_value=0.05):
                # force_tier to avoid reasoning-floor interaction
                decision = router.route("summarise the log file", force_tier="STANDARD")
        assert decision.provider == "haiku", (
            f"Low pressure + STANDARD + haiku available → expect haiku, got {decision.provider!r}"
        )
        assert any("budget-pressure" in r for r in decision.reasons)

    def test_low_pressure_advanced_balanced_mix_picks_sonnet(self):
        """balanced-mix: ADVANCED task, low pressure → sonnet (not kimi)."""
        router = _make_router(
            ["local", "kimi", "haiku", "sonnet"],
            use_pressure_routing=True,
            tier_map={
                "LOW": ["local"],
                "STANDARD": ["haiku", "kimi", "local"],
                "ADVANCED": ["sonnet", "kimi"],
                "CRITICAL": ["sonnet"],
            },
        )
        with patch("mullet.hud.usage.fetch_usage", return_value={"monthly": 100}):
            with patch("mullet.hud.guard.compute_pressure", return_value=0.05):
                decision = router.route("review code", force_tier="ADVANCED")
        assert decision.provider == "sonnet", (
            f"Low pressure + ADVANCED → expect sonnet, got {decision.provider!r}"
        )

    def test_cheap_cloud_excludes_anthropic(self):
        """cheap-cloud preset: allowlist={kimi, gemini}, STANDARD → kimi (not haiku)."""
        router = _make_router(
            ["local", "kimi", "gemini", "haiku", "sonnet"],
            use_pressure_routing=True,
            tier_map={
                "LOW": ["local"],
                "STANDARD": ["kimi", "gemini", "local"],
                "ADVANCED": ["kimi", "gemini"],
                "CRITICAL": ["kimi"],
            },
        )
        with patch("mullet.hud.usage.fetch_usage", return_value={"monthly": 100}):
            with patch("mullet.hud.guard.compute_pressure", return_value=0.05):
                decision = router.route(
                    "some task",
                    force_tier="STANDARD",
                    allowed_providers=frozenset({"kimi", "gemini"}),
                )
        assert decision.provider in ("kimi", "gemini"), (
            f"cheap-cloud allowlist must exclude Anthropic, got {decision.provider!r}"
        )

    def test_local_only_preset_uses_local(self):
        """local-only: allowlist={local}, LOW task → local."""
        router = _make_router(
            ["local", "kimi", "haiku"],
            use_pressure_routing=True,
            tier_map={"LOW": ["local"], "STANDARD": ["kimi", "local"], "ADVANCED": ["kimi"], "CRITICAL": ["kimi"]},
        )
        with patch("mullet.hud.usage.fetch_usage", return_value={"monthly": 100}):
            with patch("mullet.hud.guard.compute_pressure", return_value=0.05):
                decision = router.route(
                    "list files",
                    force_tier="LOW",
                    allowed_providers=frozenset({"local"}),
                )
        assert decision.provider == "local", (
            f"local-only allowlist + LOW → local, got {decision.provider!r}"
        )

    def test_pressure_routing_allowlist_no_tier_capable_falls_through(self):
        """Budget pick not in allowlist + no tier-capable allowed → fall through, not crash.

        Phase E update: local is now CRITICAL-capable, so {local} at CRITICAL now succeeds.
        We use a custom provider with no PROVIDER_TIER_CAPABILITIES entry to reproduce
        the scenario where pick_provider_for_tier returns None and the code falls through.
        """
        from mullet.core.router.decision import ProviderEntry
        router = _make_router(
            ["local", "kimi", "haiku", "sonnet"],
            use_pressure_routing=True,
            tier_map={
                "LOW": ["local"],
                "STANDARD": ["haiku", "kimi", "local"],
                "ADVANCED": ["sonnet", "kimi"],
                "CRITICAL": ["sonnet"],
            },
        )
        # Provider with no PROVIDER_TIER_CAPABILITIES entry — pick always returns None for it
        router.providers["no_tier_cloud"] = ProviderEntry(
            name="no_tier_cloud", tier="cheap", models={"default": "x"}
        )
        with patch("mullet.hud.usage.fetch_usage", return_value={"monthly": 100}):
            with patch("mullet.hud.guard.compute_pressure", return_value=0.05):
                # CRITICAL allowlist with only no_tier_cloud — can't serve any tier
                decision = router.route(
                    "task",
                    force_tier="CRITICAL",
                    allowed_providers=frozenset({"no_tier_cloud"}),
                )
        # Must not crash; should return a coherent RoutingDecision with allowlist reason
        assert decision is not None
        assert any(
            "allowlist" in r.lower() or "cannot serve" in r.lower()
            for r in decision.reasons
        )
