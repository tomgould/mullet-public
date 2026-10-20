"""Tests for YAML agent template loading."""

import tempfile
from pathlib import Path
from unittest import mock

import yaml

from mullet.core.agent.profiles import (
    _load_yaml_profiles,
    get_profile,
    list_profiles,
    load_all_profiles,
)
from mullet.core.router.complexity import Tier


def test_builtin_yaml_templates_load():
    """Builtin YAML templates in mullet/agents/ should load."""
    profiles = _load_yaml_profiles()
    assert len(profiles) >= 6  # 6 shipped templates
    assert "security-reviewer" in profiles
    assert "code-reviewer" in profiles
    assert "test-generator" in profiles
    assert "performance-analyzer" in profiles
    assert "architecture-analyst" in profiles
    assert "documentation-writer" in profiles


def test_yaml_profile_has_correct_fields():
    profiles = _load_yaml_profiles()
    sr = profiles["security-reviewer"]
    assert sr.name == "Security Reviewer"
    assert "OWASP" in sr.description
    assert sr.default_tier == Tier.ADVANCED
    assert "terminal" in sr.default_toolsets
    assert "injection" in sr.system_prompt.lower()


def test_yaml_profile_tier_mapping():
    profiles = _load_yaml_profiles()
    assert profiles["test-generator"].default_tier == Tier.STANDARD
    assert profiles["documentation-writer"].default_tier == Tier.LOW
    assert profiles["architecture-analyst"].default_tier == Tier.ADVANCED


def test_load_all_profiles_merges_builtin_and_yaml():
    all_profiles = load_all_profiles()
    # Should have built-in profiles
    assert "analyst" in all_profiles
    assert "executor" in all_profiles
    # Should also have YAML profiles
    assert "security-reviewer" in all_profiles
    assert "test-generator" in all_profiles


def test_get_profile_finds_yaml_template():
    profile = get_profile("security-reviewer")
    assert profile is not None
    assert profile.name == "Security Reviewer"


def test_get_profile_finds_builtin():
    profile = get_profile("analyst")
    assert profile is not None
    assert profile.name == "analyst"


def test_get_profile_returns_none_for_missing():
    assert get_profile("nonexistent-profile-xyz") is None


def test_list_profiles_includes_yaml():
    profiles = list_profiles()
    names = [p["name"] for p in profiles]
    assert "Security Reviewer" in names
    assert "analyst" in names


def test_user_yaml_overrides_builtin():
    """User YAML in ~/.mullet/agents/ should override built-in YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agents_dir = Path(tmpdir) / ".mullet" / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "security-reviewer.yaml").write_text(yaml.dump({
            "name": "Custom Security",
            "description": "My custom security reviewer",
            "tier": "STANDARD",
            "system_prompt": "Custom prompt.",
        }))
        with mock.patch("pathlib.Path.home", return_value=Path(tmpdir)):
            profiles = _load_yaml_profiles()
            assert profiles["security-reviewer"].name == "Custom Security"
            assert profiles["security-reviewer"].default_tier == Tier.STANDARD


def test_invalid_yaml_file_skipped():
    """Malformed YAML files should be silently skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agents_dir = Path(tmpdir) / ".mullet" / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "bad.yaml").write_text("not: valid: yaml: [[[")
        with mock.patch("pathlib.Path.home", return_value=Path(tmpdir)):
            # Should not raise
            profiles = _load_yaml_profiles()
            assert "bad" not in profiles


def test_yaml_missing_name_skipped():
    """YAML without a 'name' field should be skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agents_dir = Path(tmpdir) / ".mullet" / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "noname.yaml").write_text(yaml.dump({
            "description": "Missing name field",
        }))
        with mock.patch("pathlib.Path.home", return_value=Path(tmpdir)):
            profiles = _load_yaml_profiles()
            assert "noname" not in profiles


def test_yaml_default_tier_is_medium():
    """YAML without tier field should default to MEDIUM."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agents_dir = Path(tmpdir) / ".mullet" / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "notier.yaml").write_text(yaml.dump({
            "name": "No Tier Agent",
        }))
        with mock.patch("pathlib.Path.home", return_value=Path(tmpdir)):
            profiles = _load_yaml_profiles()
            assert profiles["notier"].default_tier == Tier.STANDARD
