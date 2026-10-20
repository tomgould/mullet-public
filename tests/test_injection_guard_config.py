"""Tests for injection guard YAML config loading."""

import yaml
from mullet.core.agent.injection_guard import _load_threat_patterns, _HARDCODED_THREAT_PATTERNS


def test_patterns_loaded_from_yaml(tmp_path):
    """Custom patterns in YAML file are detected."""
    yaml_content = {
        "threat_patterns": [
            {"pattern": "test_custom_pattern", "category": "test_category"}
        ],
        "invisible_chars": []
    }
    config = tmp_path / "threat_patterns.yaml"
    config.write_text(yaml.dump(yaml_content))

    patterns, chars = _load_threat_patterns(config_path=config)
    assert len(patterns) == 1
    assert patterns[0] == ("test_custom_pattern", "test_category")


def test_fallback_when_yaml_missing(tmp_path):
    """Falls back to hardcoded patterns when config file is missing."""
    missing_path = tmp_path / "nonexistent.yaml"
    patterns, chars = _load_threat_patterns(config_path=missing_path)
    assert patterns == _HARDCODED_THREAT_PATTERNS
    assert len(chars) > 0


def test_yaml_pattern_update_detected():
    """Default YAML config has all 9 hardcoded patterns."""
    from mullet.core.agent.injection_guard import _CONFIG_PATH
    patterns, chars = _load_threat_patterns(config_path=_CONFIG_PATH)
    assert len(patterns) == len(_HARDCODED_THREAT_PATTERNS)
    assert len(chars) == 10  # 10 invisible char codepoints
