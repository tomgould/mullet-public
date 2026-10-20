"""Tests for configuration system."""

import yaml
from mullet.config.defaults import MulletConfig, load_config
from mullet.config.loader import load_from_yaml
from mullet.config.mode import MulletMode
from mullet.config.logging import setup_logging, StructuredFormatter, HumanFormatter


def test_default_config():
    config = MulletConfig()
    assert config.max_iterations == 90
    assert "terminal" in config.default_toolsets


def test_load_config_defaults():
    config = load_config("/nonexistent/path.yaml")
    assert isinstance(config, MulletConfig)


def test_load_from_yaml_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # No mullet.yaml here
    data = load_from_yaml("/nonexistent/mullet.yaml")
    assert data == {}


def test_load_from_yaml_real(tmp_path):
    config_file = tmp_path / "mullet.yaml"
    config_file.write_text(yaml.dump({
        "providers": {
            "local": {
                "type": "openai_compatible",
                "base_url": "http://localhost:1234/v1",
                "models": {"default": "mistral"},
            }
        },
        "routing": {
            "default_tier": "LOW",
        },
        "max_iterations": 50,
    }))

    data = load_from_yaml(str(config_file))
    assert "providers" in data
    assert data["routing"]["default_tier"] == "LOW"


def test_mullet_mode_default(monkeypatch):
    monkeypatch.delenv("MULLET_MODE", raising=False)
    assert MulletMode.is_solo() is True
    assert MulletMode.is_enterprise() is False


def test_mullet_mode_enterprise(monkeypatch):
    monkeypatch.setenv("MULLET_MODE", "enterprise")
    assert MulletMode.is_enterprise() is True
    assert MulletMode.is_solo() is False


def test_structured_formatter():
    import logging
    formatter = StructuredFormatter()
    record = logging.LogRecord("test", logging.INFO, "", 0, "test message", (), None)
    output = formatter.format(record)
    assert '"msg": "test message"' in output
    assert '"level": "INFO"' in output


def test_human_formatter():
    import logging
    formatter = HumanFormatter()
    record = logging.LogRecord("mullet.test", logging.INFO, "", 0, "hello", (), None)
    output = formatter.format(record)
    assert "INFO" in output
    assert "hello" in output


def test_setup_logging():
    setup_logging(verbose=False)
    setup_logging(verbose=True)
    # Should not crash


def test_user_config_dir():
    path = MulletConfig.user_config_dir()
    assert ".mullet" in str(path)


def test_user_skills_dir():
    path = MulletConfig.user_skills_dir()
    assert "skills" in str(path)
