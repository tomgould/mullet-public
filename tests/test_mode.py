"""Tests for MULLET_MODE configuration."""

from mullet.config.mode import MulletMode


def test_default_is_solo(monkeypatch):
    monkeypatch.delenv("MULLET_MODE", raising=False)
    assert MulletMode.current() == "solo"
    assert MulletMode.is_solo() is True
    assert MulletMode.is_enterprise() is False


def test_enterprise_mode(monkeypatch):
    monkeypatch.setenv("MULLET_MODE", "enterprise")
    assert MulletMode.current() == "enterprise"
    assert MulletMode.is_enterprise() is True
    assert MulletMode.is_solo() is False


def test_case_insensitive(monkeypatch):
    monkeypatch.setenv("MULLET_MODE", "ENTERPRISE")
    assert MulletMode.is_enterprise() is True

    monkeypatch.setenv("MULLET_MODE", "Solo")
    assert MulletMode.is_solo() is True
