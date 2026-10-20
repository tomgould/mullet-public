"""Tests for mullet.core.agent.profiles — AgentProfile and lookup functions."""

from mullet.core.agent.profiles import (
    AgentProfile,
    AGENT_PROFILES,
    get_profile,
    list_profiles,
    load_all_profiles,
)
from mullet.core.router.complexity import Tier


class TestAgentProfileDataclass:
    def test_required_fields(self):
        p = AgentProfile(
            name="test",
            description="a test profile",
            system_prompt="You are a tester.",
            default_tier=Tier.LOW,
        )
        assert p.name == "test"
        assert p.description == "a test profile"
        assert p.default_tier == Tier.LOW

    def test_default_toolsets_empty(self):
        p = AgentProfile(
            name="test",
            description="",
            system_prompt="",
            default_tier=Tier.LOW,
        )
        assert p.default_toolsets == []

    def test_toolsets_set(self):
        p = AgentProfile(
            name="dev",
            description="",
            system_prompt="",
            default_tier=Tier.STANDARD,
            default_toolsets=["terminal", "file"],
        )
        assert "terminal" in p.default_toolsets
        assert "file" in p.default_toolsets


class TestBuiltinProfiles:
    def test_profiles_dict_not_empty(self):
        assert len(AGENT_PROFILES) >= 5

    def test_all_profiles_have_names(self):
        for key, profile in AGENT_PROFILES.items():
            assert profile.name, f"Profile {key!r} has no name"

    def test_all_profiles_have_system_prompts(self):
        for key, profile in AGENT_PROFILES.items():
            assert profile.system_prompt, f"Profile {key!r} has no system_prompt"

    def test_all_profiles_have_valid_tiers(self):
        valid_tiers = set(Tier)
        for key, profile in AGENT_PROFILES.items():
            assert profile.default_tier in valid_tiers, f"Profile {key!r} has invalid tier"

    def test_expected_profiles_present(self):
        expected = {"analyst", "writer", "executor", "code-reviewer"}
        missing = expected - set(AGENT_PROFILES.keys())
        assert not missing, f"Missing expected profiles: {missing}"


class TestGetProfile:
    def test_existing_profile_returned(self):
        profile = get_profile("analyst")
        assert profile is not None
        assert profile.name == "analyst"

    def test_nonexistent_profile_returns_none(self):
        assert get_profile("nonexistent-xyz-profile") is None

    def test_executor_profile(self):
        p = get_profile("executor")
        assert p is not None
        assert p.default_tier in (Tier.STANDARD, Tier.LOW, Tier.ADVANCED, Tier.CRITICAL, Tier.MEDIUM)

    def test_writer_profile_is_low_tier(self):
        p = get_profile("writer")
        assert p is not None
        assert p.default_tier == Tier.LOW


class TestListProfiles:
    def test_returns_list(self):
        result = list_profiles()
        assert isinstance(result, list)
        assert len(result) >= 5

    def test_each_entry_has_name_description_tier(self):
        for entry in list_profiles():
            assert "name" in entry
            assert "description" in entry
            assert "tier" in entry

    def test_tier_is_string(self):
        for entry in list_profiles():
            assert isinstance(entry["tier"], str)

    def test_analyst_in_list(self):
        names = [e["name"] for e in list_profiles()]
        assert "analyst" in names


class TestLoadAllProfiles:
    def test_returns_dict(self):
        result = load_all_profiles()
        assert isinstance(result, dict)

    def test_includes_builtin_profiles(self):
        result = load_all_profiles()
        for key in AGENT_PROFILES:
            assert key in result

    def test_all_values_are_agent_profiles(self):
        result = load_all_profiles()
        for key, val in result.items():
            assert isinstance(val, AgentProfile), f"{key} is not an AgentProfile"
