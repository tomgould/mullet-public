
from mullet.knowledge.engine.logging_config import AuditConfig, load_audit_config, get_default_audit_config


def test_should_log_enabled_full():
    cfg = AuditConfig(enabled=True, log_level="full")
    assert cfg.should_log() is True


def test_should_log_enabled_none():
    cfg = AuditConfig(enabled=True, log_level="none")
    assert cfg.should_log() is False


def test_should_log_disabled():
    cfg = AuditConfig(enabled=False, log_level="full")
    assert cfg.should_log() is False


def test_is_minimal():
    assert AuditConfig(log_level="minimal").is_minimal() is True
    assert AuditConfig(log_level="full").is_minimal() is False


def test_is_full():
    assert AuditConfig(log_level="full").is_full() is True
    assert AuditConfig(log_level="minimal").is_full() is False


def test_load_audit_config_defaults():
    cfg = load_audit_config({})
    assert cfg.enabled is True
    assert cfg.log_level == "full"
    assert cfg.include_chunks is True
    assert cfg.include_scores is True
    assert cfg.include_timing is True
    assert cfg.retention_days is None


def test_load_audit_config_overrides():
    cfg = load_audit_config({"audit": {"enabled": False, "log_level": "minimal", "include_chunks": False, "retention_days": 30}})
    assert cfg.enabled is False
    assert cfg.log_level == "minimal"
    assert cfg.include_chunks is False
    assert cfg.include_scores is True
    assert cfg.retention_days == 30


def test_get_default_audit_config():
    cfg = get_default_audit_config()
    assert cfg.enabled is True
    assert cfg.log_level == "full"
