"""Tests for mullet.config.loader."""

import contextlib
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mullet.config.loader import load_from_yaml, build_stack, _build_provider_entries, _detect_local
from mullet.core.router.config import ProviderConfig
from mullet.core.router.decision import ProviderEntry


class FakeValidationResult:
    valid = True
    issues = []

    def format(self):
        return ""


class FakeValidationResultInvalid:
    valid = False
    issues = []

    def format(self):
        return "invalid"


class FakeValidationResultWithWarning:
    valid = True
    issues = [MagicMock()]

    def format(self):
        return "warning"


def test_load_from_yaml_explicit_path_valid(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("foo: bar\n")
    result = load_from_yaml(str(p))
    assert result == {"foo": "bar"}


def test_load_from_yaml_explicit_path_missing(tmp_path):
    result = load_from_yaml(str(tmp_path / "missing.yaml"))
    assert result == {}


def test_load_from_yaml_explicit_path_parse_error(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("{{bad\n")
    result = load_from_yaml(str(p))
    assert result == {}


def test_load_from_yaml_candidate_found(tmp_path, monkeypatch):
    p = tmp_path / "mullet.yaml"
    p.write_text("foo: bar\n")
    monkeypatch.chdir(tmp_path)
    result = load_from_yaml()
    assert result == {"foo": "bar"}


def test_load_from_yaml_candidate_parse_error(tmp_path, monkeypatch):
    p = tmp_path / "mullet.yaml"
    p.write_text("{{bad\n")
    monkeypatch.chdir(tmp_path)
    result = load_from_yaml()
    assert result == {}


def test_load_from_yaml_no_candidate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    result = load_from_yaml()
    assert result == {}


def test_build_provider_entries():
    configs = {
        "p1": ProviderConfig(name="p1", tier="cheap", models={"default": "m1"}, cost_per_1k_input=0.1, cost_per_1k_output=0.2),
    }
    entries = _build_provider_entries(configs)
    assert "p1" in entries
    assert entries["p1"].name == "p1"
    assert entries["p1"].tier == "cheap"


def test_detect_local_found():
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_resp)
    mock_cm.__exit__ = MagicMock(return_value=False)
    with patch("urllib.request.urlopen", return_value=mock_cm):
        result = _detect_local()
    assert result is not None
    assert result.name == "local"
    assert result.tier == "free"


def test_detect_local_not_found():
    with patch("urllib.request.urlopen", side_effect=Exception("no server")):
        result = _detect_local()
    assert result is None


def _build_stack_patches(local_return=None, yaml_data=None, env=None):
    """Return an ExitStack with common build_stack patches applied."""
    stack = contextlib.ExitStack()
    if env:
        # Clear competing API keys from the host environment so each test
        # exercises exactly the branch it intends to cover.
        clean_env = {
            "ANTHROPIC_API_KEY": "",
            "GOOGLE_API_KEY": "",
            "GEMINI_API_KEY": "",
            "OPENROUTER_API_KEY": "",
        }
        clean_env.update(env)
        stack.enter_context(patch.dict(os.environ, clean_env, clear=False))
    stack.enter_context(patch("mullet.config.loader.load_from_yaml", return_value=yaml_data or {}))
    stack.enter_context(patch("mullet.config.loader.load_config", return_value=MagicMock()))
    stack.enter_context(patch("mullet.config.loader.load_routing_config", return_value=MagicMock(budget={}, tier_map={}, thresholds={}, local_only=False)))
    stack.enter_context(patch("mullet.config.loader.load_provider_configs", return_value={}))
    stack.enter_context(patch("mullet.config.validate.validate_config", return_value=FakeValidationResult()))
    stack.enter_context(patch("mullet.config.loader._detect_local", return_value=local_return))
    stack.enter_context(patch("mullet.config.loader.Router"))
    return stack


def test_build_stack_basic():
    with _build_stack_patches():
        result = build_stack()
    assert "router" in result
    assert "health" in result
    assert "budget" in result


def test_build_stack_local_detected():
    local_entry = ProviderEntry(name="local", tier="free", models={"default": "default"})
    with _build_stack_patches(local_return=local_entry):
        result = build_stack()
    assert "local" in result["provider_entries"]


def test_build_stack_anthropic_key():
    with _build_stack_patches(env={"ANTHROPIC_API_KEY": "secret"}):
        result = build_stack()
    assert "premium_cloud" in result["provider_entries"]


def test_build_stack_google_key():
    with _build_stack_patches(env={"GOOGLE_API_KEY": "secret"}):
        result = build_stack()
    assert "cheap_cloud" in result["provider_entries"]


def test_build_stack_gemini_key():
    with _build_stack_patches(env={"GEMINI_API_KEY": "secret"}):
        result = build_stack()
    assert "cheap_cloud" in result["provider_entries"]


def test_build_stack_openrouter_key():
    with _build_stack_patches(env={"OPENROUTER_API_KEY": "secret"}):
        result = build_stack()
    assert "cheap_cloud" in result["provider_entries"]


def test_build_stack_validation_warnings():
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("mullet.config.validate.validate_config", return_value=FakeValidationResultWithWarning()))
        stack.enter_context(patch("mullet.config.loader.load_from_yaml", return_value={"routing": {}}))
        stack.enter_context(patch("mullet.config.loader.load_config", return_value=MagicMock()))
        stack.enter_context(patch("mullet.config.loader.load_routing_config", return_value=MagicMock(budget={}, tier_map={}, thresholds={}, local_only=False)))
        stack.enter_context(patch("mullet.config.loader.load_provider_configs", return_value={}))
        stack.enter_context(patch("mullet.config.loader._detect_local", return_value=None))
        stack.enter_context(patch("mullet.config.loader.Router"))
        result = build_stack()
    assert "router" in result


def test_build_stack_validation_invalid():
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("mullet.config.validate.validate_config", return_value=FakeValidationResultInvalid()))
        stack.enter_context(patch("mullet.config.loader.load_from_yaml", return_value={"routing": {}}))
        stack.enter_context(patch("mullet.config.loader.load_config", return_value=MagicMock()))
        stack.enter_context(patch("mullet.config.loader.load_routing_config", return_value=MagicMock(budget={}, tier_map={}, thresholds={}, local_only=False)))
        stack.enter_context(patch("mullet.config.loader.load_provider_configs", return_value={}))
        stack.enter_context(patch("mullet.config.loader._detect_local", return_value=None))
        stack.enter_context(patch("mullet.config.loader.Router"))
        result = build_stack()
    assert "router" in result


def test_build_stack_pricing_exception():
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("mullet.core.providers.pricing.get_known_pricing", side_effect=Exception("boom")))
        stack.enter_context(_build_stack_patches(env={"ANTHROPIC_API_KEY": "secret"}))
        result = build_stack()
    assert "premium_cloud" in result["provider_entries"]
