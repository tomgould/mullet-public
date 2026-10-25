"""Tests for mullet/core/env.py."""

from pathlib import Path

from mullet.core.env import get_pools_dir


def test_get_pools_dir_default(monkeypatch):
    monkeypatch.delenv("TRAWL_POOLS_DIR", raising=False)
    assert get_pools_dir() == Path.home() / ".local" / "share" / "mullet" / "pools"


def test_get_pools_dir_default_on_an_older_install(monkeypatch):
    monkeypatch.delenv("TRAWL_POOLS_DIR", raising=False)
    (Path.home() / ".mullet" / "pools").mkdir(parents=True)
    assert get_pools_dir() == Path.home() / ".mullet" / "pools"


def test_get_pools_dir_override(monkeypatch):
    monkeypatch.setenv("TRAWL_POOLS_DIR", "/custom/pools")
    assert get_pools_dir() == Path("/custom/pools")


def test_get_pools_dir_empty_string(monkeypatch):
    monkeypatch.setenv("TRAWL_POOLS_DIR", "")
    assert get_pools_dir() == Path.home() / ".local" / "share" / "mullet" / "pools"
