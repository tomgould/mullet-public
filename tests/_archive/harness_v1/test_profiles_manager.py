"""Manager tests for ``mullet.harness.profiles.manager``."""

from __future__ import annotations

from pathlib import Path

import pytest

from mullet.harness.profiles.builtins import BUILTIN_PROFILES, BUILTIN_PROFILE_NAMES
from mullet.harness.profiles.manager import ProfileManager
from mullet.harness.profiles.schema import ProfilesConfig, ProviderProfile


pytestmark = pytest.mark.harness


def _custom_profile(name: str = "custom-kimi") -> ProviderProfile:
    return ProviderProfile(
        name=name,
        label="Custom Kimi",
        provider_type="kimi",
        api_format="openai",
        auth_source="KIMI_API_KEY",
        model="kimi-k2.6",
    )


def test_load_returns_builtins_when_no_yaml(tmp_path: Path) -> None:
    config_dir = tmp_path / "mullet_config_empty"
    mgr = ProfileManager(config_dir=config_dir)
    # Remove the yaml so we hit the default-path.
    if mgr.profiles_path.exists():
        mgr.profiles_path.unlink()
    config = mgr.load()
    assert config.active == "kimi"
    assert len(config.profiles) == len(BUILTIN_PROFILES)
    assert {p.name for p in config.profiles} == set(BUILTIN_PROFILE_NAMES)


def test_save_then_load_roundtrip(isolated_profiles_dir: Path) -> None:
    mgr = ProfileManager(config_dir=isolated_profiles_dir)
    profile = _custom_profile()
    config = ProfilesConfig(active="custom-kimi", profiles=[profile])
    mgr.save(config)
    reloaded = mgr.load()
    assert reloaded.active == "custom-kimi"
    assert reloaded.profiles == [profile]


def test_add_rejects_duplicate_name(isolated_profiles_dir: Path) -> None:
    mgr = ProfileManager(config_dir=isolated_profiles_dir)
    mgr.save(ProfilesConfig(active=None, profiles=[]))
    profile = _custom_profile()
    mgr.add(profile)
    with pytest.raises(ValueError, match="already exists"):
        mgr.add(profile)


def test_add_appends_and_persists(isolated_profiles_dir: Path) -> None:
    mgr = ProfileManager(config_dir=isolated_profiles_dir)
    mgr.save(ProfilesConfig(active=None, profiles=[]))
    profile_a = _custom_profile("alpha")
    profile_b = _custom_profile("beta")
    mgr.add(profile_a)
    mgr.add(profile_b)
    listed = mgr.list()
    assert [p.name for p in listed] == ["alpha", "beta"]


def test_use_sets_active_persists(isolated_profiles_dir: Path) -> None:
    mgr = ProfileManager(config_dir=isolated_profiles_dir)
    mgr.save(ProfilesConfig(active=None, profiles=[_custom_profile("alpha"), _custom_profile("beta")]))
    mgr.use("beta")
    assert mgr.load().active == "beta"


def test_use_unknown_raises(isolated_profiles_dir: Path) -> None:
    mgr = ProfileManager(config_dir=isolated_profiles_dir)
    mgr.save(ProfilesConfig(active=None, profiles=[_custom_profile("alpha")]))
    with pytest.raises(ValueError, match="unknown profile"):
        mgr.use("missing")


def test_remove_active_without_force_raises(isolated_profiles_dir: Path) -> None:
    mgr = ProfileManager(config_dir=isolated_profiles_dir)
    profile = _custom_profile("alpha")
    mgr.save(ProfilesConfig(active="alpha", profiles=[profile]))
    with pytest.raises(ValueError, match="active profile"):
        mgr.remove("alpha")


def test_remove_builtin_without_force_raises(isolated_profiles_dir: Path) -> None:
    mgr = ProfileManager(config_dir=isolated_profiles_dir)
    # Seed config with built-ins but a non-built-in active so we isolate the
    # built-in guard from the active guard.
    seed_profiles = [p.model_copy(deep=True) for p in BUILTIN_PROFILES]
    seed_profiles.append(_custom_profile("custom-active"))
    mgr.save(ProfilesConfig(active="custom-active", profiles=seed_profiles))
    with pytest.raises(ValueError, match="built-in profile"):
        mgr.remove("kimi")


def test_remove_with_force_succeeds(isolated_profiles_dir: Path) -> None:
    mgr = ProfileManager(config_dir=isolated_profiles_dir)
    seed_profiles = [p.model_copy(deep=True) for p in BUILTIN_PROFILES]
    mgr.save(ProfilesConfig(active="kimi", profiles=seed_profiles))
    mgr.remove("kimi", force=True)
    config = mgr.load()
    assert "kimi" not in {p.name for p in config.profiles}
    assert config.active is None


def test_resolve_returns_provider_instance(
    isolated_profiles_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr = ProfileManager(config_dir=isolated_profiles_dir)
    profile = _custom_profile("kimi")
    mgr.save(ProfilesConfig(active="kimi", profiles=[profile]))
    monkeypatch.setenv("KIMI_API_KEY", "fake-test-key")

    captured: dict = {}

    class FakeKimi:
        def __init__(self, api_key=None, model=None, base_url=None):
            captured["api_key"] = api_key
            captured["model"] = model
            captured["base_url"] = base_url

    import mullet.core.providers.kimi as kimi_mod

    monkeypatch.setattr(kimi_mod, "KimiProvider", FakeKimi)

    instance = mgr.resolve("kimi")
    assert isinstance(instance, FakeKimi)
    assert captured["api_key"] == "fake-test-key"
    assert captured["model"] == "kimi-k2.6"


def test_resolve_unknown_provider_type_raises(
    isolated_profiles_dir: Path,
) -> None:
    mgr = ProfileManager(config_dir=isolated_profiles_dir)
    profile = ProviderProfile(
        name="weirdo",
        label="Weirdo",
        provider_type="not-a-real-provider",
        api_format="openai",
        auth_source="",
        model="x",
    )
    mgr.save(ProfilesConfig(active="weirdo", profiles=[profile]))
    with pytest.raises(ValueError, match="unknown provider_type"):
        mgr.resolve("weirdo")
