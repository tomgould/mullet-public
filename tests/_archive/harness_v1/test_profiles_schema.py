"""Schema tests for ``mullet.harness.profiles.schema``."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from mullet.harness.profiles.schema import ProfilesConfig, ProviderProfile


pytestmark = pytest.mark.harness


def _valid_kwargs(**overrides) -> dict:
    base = {
        "name": "kimi",
        "label": "Kimi K2",
        "provider_type": "kimi",
        "api_format": "openai",
        "auth_source": "KIMI_API_KEY",
        "model": "kimi-k2.6",
    }
    base.update(overrides)
    return base


def test_provider_profile_required_fields_validate() -> None:
    profile = ProviderProfile(**_valid_kwargs())
    assert profile.name == "kimi"
    assert profile.label == "Kimi K2"
    assert profile.provider_type == "kimi"
    assert profile.api_format == "openai"
    assert profile.auth_source == "KIMI_API_KEY"
    assert profile.model == "kimi-k2.6"
    assert profile.base_url is None
    assert profile.extra_headers == {}


def test_provider_profile_rejects_empty_name_label() -> None:
    with pytest.raises(ValidationError):
        ProviderProfile(**_valid_kwargs(name=""))
    with pytest.raises(ValidationError):
        ProviderProfile(**_valid_kwargs(name="   "))
    with pytest.raises(ValidationError):
        ProviderProfile(**_valid_kwargs(label=""))
    with pytest.raises(ValidationError):
        ProviderProfile(**_valid_kwargs(label="   "))


def test_provider_profile_rejects_bad_api_format() -> None:
    with pytest.raises(ValidationError):
        ProviderProfile(**_valid_kwargs(api_format="bogus"))
    with pytest.raises(ValidationError):
        ProviderProfile(**_valid_kwargs(api_format=""))


def test_provider_profile_rejects_lowercase_auth_source() -> None:
    with pytest.raises(ValidationError):
        ProviderProfile(**_valid_kwargs(auth_source="kimi_api_key"))
    with pytest.raises(ValidationError):
        ProviderProfile(**_valid_kwargs(auth_source="1KIMI"))
    # Empty is allowed (local providers, claude CLI).
    profile = ProviderProfile(**_valid_kwargs(auth_source=""))
    assert profile.auth_source == ""


def test_profiles_config_defaults() -> None:
    config = ProfilesConfig()
    assert config.active is None
    assert config.profiles == []


def test_profiles_config_serializes_yaml_round_trip() -> None:
    profile = ProviderProfile(
        **_valid_kwargs(
            base_url="https://api.moonshot.ai/v1",
            extra_headers={"X-Title": "Mullet"},
        )
    )
    config = ProfilesConfig(active="kimi", profiles=[profile])
    payload = config.model_dump(mode="python")
    text = yaml.safe_dump(payload, sort_keys=False)
    reloaded = yaml.safe_load(text)
    rebuilt = ProfilesConfig.model_validate(reloaded)
    assert rebuilt.active == "kimi"
    assert len(rebuilt.profiles) == 1
    assert rebuilt.profiles[0] == profile
