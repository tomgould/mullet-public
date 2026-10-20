"""Tests for mullet.knowledge.engine.logging_config.AuditConfig."""


from mullet.knowledge.engine.logging_config import (
    AuditConfig,
    get_default_audit_config,
    load_audit_config,
)


# --- dataclass defaults ----------------------------------------------------


def test_default_values():
    cfg = AuditConfig()
    assert cfg.enabled is True
    assert cfg.log_level == "full"
    assert cfg.include_chunks is True
    assert cfg.include_scores is True
    assert cfg.include_timing is True
    assert cfg.retention_days is None


def test_get_default_audit_config_returns_defaults():
    cfg = get_default_audit_config()
    assert cfg.enabled is True
    assert cfg.log_level == "full"


# --- should_log / is_minimal / is_full ------------------------------------


def test_should_log_enabled_full():
    assert AuditConfig(enabled=True, log_level="full").should_log() is True


def test_should_log_enabled_minimal():
    assert AuditConfig(enabled=True, log_level="minimal").should_log() is True


def test_should_log_disabled_false():
    assert AuditConfig(enabled=False, log_level="full").should_log() is False


def test_should_log_level_none_false():
    assert AuditConfig(enabled=True, log_level="none").should_log() is False


def test_is_minimal():
    assert AuditConfig(log_level="minimal").is_minimal() is True
    assert AuditConfig(log_level="full").is_minimal() is False
    assert AuditConfig(log_level="none").is_minimal() is False


def test_is_full():
    assert AuditConfig(log_level="full").is_full() is True
    assert AuditConfig(log_level="minimal").is_full() is False
    assert AuditConfig(log_level="none").is_full() is False


# --- load_audit_config -----------------------------------------------------


def test_load_audit_config_empty_config_returns_defaults():
    cfg = load_audit_config({})
    assert cfg.enabled is True
    assert cfg.log_level == "full"


def test_load_audit_config_missing_audit_section():
    cfg = load_audit_config({"other_section": {}})
    assert cfg.enabled is True


def test_load_audit_config_full_custom():
    cfg = load_audit_config({
        "audit": {
            "enabled": False,
            "log_level": "minimal",
            "include_chunks": False,
            "include_scores": False,
            "include_timing": False,
            "retention_days": 30,
        }
    })
    assert cfg.enabled is False
    assert cfg.log_level == "minimal"
    assert cfg.include_chunks is False
    assert cfg.include_scores is False
    assert cfg.include_timing is False
    assert cfg.retention_days == 30


def test_load_audit_config_partial():
    """Only some keys set → others use defaults."""
    cfg = load_audit_config({
        "audit": {
            "log_level": "minimal",
            "retention_days": 90,
        }
    })
    assert cfg.enabled is True  # default
    assert cfg.log_level == "minimal"
    assert cfg.include_chunks is True  # default
    assert cfg.retention_days == 90


def test_load_audit_config_retention_explicit_none():
    """retention_days: null → stays None."""
    cfg = load_audit_config({"audit": {"retention_days": None}})
    assert cfg.retention_days is None
