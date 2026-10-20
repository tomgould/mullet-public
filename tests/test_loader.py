"""Tests for config loader."""

import yaml
from mullet.config.loader import load_from_yaml, build_stack


def test_load_missing_yaml(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # No mullet.yaml in temp dir
    data = load_from_yaml("/nonexistent/path.yaml")
    assert data == {}


def test_load_real_yaml(tmp_path):
    config = tmp_path / "mullet.yaml"
    config.write_text(yaml.dump({
        "providers": {"local": {"type": "openai_compatible"}},
        "routing": {"default_tier": "LOW"},
    }))
    data = load_from_yaml(str(config))
    assert data["routing"]["default_tier"] == "LOW"


def test_build_stack_no_crash():
    """build_stack should work even without config."""
    stack = build_stack()
    assert "router" in stack
    assert "health" in stack
    assert "budget" in stack
    assert "config" in stack


def test_build_stack_has_router():
    stack = build_stack()
    from mullet.core.router.decision import Router
    assert isinstance(stack["router"], Router)
