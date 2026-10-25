"""Tests for default configuration."""

from pathlib import Path
from mullet.config.defaults import (
    MulletConfig,
    ProviderConfig,
    RoutingConfig,
    load_config,
)


def test_mullet_config_defaults():
    cfg = MulletConfig()
    assert cfg.max_iterations == 90
    assert cfg.providers == {}
    assert cfg.routing.enabled is True
    assert cfg.routing.default_tier == "STANDARD"


def test_routing_config_defaults():
    rc = RoutingConfig()
    assert rc.thresholds == {"low_max": 20, "medium_max": 50}
    assert rc.budget["daily_max_dollars"] == 10.0
    assert rc.local_only is False


def test_provider_config():
    pc = ProviderConfig(name="test", type="anthropic")
    assert pc.name == "test"
    assert pc.type == "anthropic"
    assert pc.base_url == ""


def test_user_config_dir():
    path = MulletConfig.user_config_dir()
    assert path == Path.home() / ".config" / "mullet"


def test_user_skills_dir():
    path = MulletConfig.user_skills_dir()
    assert "skills" in str(path)


def test_load_config_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)  # Isolate from user's ~/.mullet/config.yaml
    cfg = load_config()
    assert cfg.max_iterations == 90


def test_load_config_from_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "mullet.yaml"
    import yaml
    config_file.write_text(yaml.dump({
        "max_iterations": 50,
        "routing": {"default_tier": "LOW", "local_only": True},
        "providers": {
            "test": {"type": "openai_compatible", "base_url": "http://localhost:1234/v1"},
        },
    }))
    cfg = load_config(str(config_file))
    assert cfg.max_iterations == 50
    assert cfg.routing.default_tier == "LOW"
    assert cfg.routing.local_only is True
    assert "test" in cfg.providers
    assert cfg.providers["test"].type == "openai_compatible"


def test_load_config_nonexistent_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)  # Isolate from user's ~/.mullet/config.yaml
    cfg = load_config("/nonexistent/path/mullet.yaml")
    assert cfg.max_iterations == 90  # Falls back to defaults


def test_load_config_finds_local_mullet_yaml(tmp_path, monkeypatch):
    """Lines 73-75: load_config() discovers mullet.yaml in cwd via candidate loop."""
    import yaml
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mullet.config.defaults.MulletConfig.user_config_dir",
                        staticmethod(lambda: tmp_path / "no-such-dir"))
    (tmp_path / "mullet.yaml").write_text(yaml.dump({"max_iterations": 77}))
    cfg = load_config()
    assert cfg.max_iterations == 77


def test_load_config_skills_section(tmp_path, monkeypatch):
    """Lines 101-104: YAML with skills key → skills_dirs and disabled_skills populated."""
    import yaml
    monkeypatch.chdir(tmp_path)
    cfg_file = tmp_path / "mullet.yaml"
    cfg_file.write_text(yaml.dump({
        "skills": {
            "external_dirs": ["/my/skills"],
            "disabled": ["legacy_skill"],
        },
    }))
    cfg = load_config(str(cfg_file))
    assert cfg.skills_dirs == ["/my/skills"]
    assert cfg.disabled_skills == ["legacy_skill"]
