"""Tests for mullet.core.agent.profiles — specialized agent profiles."""

from mullet.core.agent.profiles import AGENT_PROFILES, get_profile, list_profiles
from mullet.core.router.complexity import Tier


def test_profiles_defined():
    assert len(AGENT_PROFILES) >= 5


def test_get_profile_exists():
    p = get_profile("analyst")
    assert p is not None
    assert p.name == "analyst"
    assert p.default_tier == Tier.CRITICAL


def test_get_profile_not_found():
    assert get_profile("nonexistent") is None


def test_list_profiles():
    profiles = list_profiles()
    assert len(profiles) >= 5
    names = {p["name"] for p in profiles}
    assert "analyst" in names
    assert "writer" in names
    assert "designer" in names


def test_profile_has_system_prompt():
    for name, profile in AGENT_PROFILES.items():
        assert profile.system_prompt, f"{name} has empty system_prompt"
        assert len(profile.system_prompt) > 20, f"{name} system_prompt too short"


def test_profile_has_toolsets():
    for name, profile in AGENT_PROFILES.items():
        assert profile.default_toolsets, f"{name} has no default_toolsets"


def test_tier_values():
    assert get_profile("analyst").default_tier == Tier.CRITICAL
    assert get_profile("writer").default_tier == Tier.LOW
    assert get_profile("designer").default_tier == Tier.STANDARD
