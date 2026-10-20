"""Tests for the mullet init command enhancements."""

import os
from unittest import mock

import pytest


@pytest.fixture
def clean_tmpdir(monkeypatch, tmp_path):
    """Change to a clean temp directory and restore after."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_init_creates_config_file(clean_tmpdir):
    """mullet init should create mullet.yaml in the current directory."""
    with mock.patch("mullet.cli.main.detect_local_provider", return_value=None), \
         mock.patch.dict(os.environ, {}, clear=True):
        from mullet.cli.main import _run_init
        _run_init()
        assert (clean_tmpdir / "mullet.yaml").exists()


def test_init_skips_existing_config(clean_tmpdir, capsys):
    """mullet init should not overwrite existing mullet.yaml."""
    config = clean_tmpdir / "mullet.yaml"
    config.write_text("existing: true\n")
    with mock.patch("mullet.cli.main.detect_local_provider", return_value=None):
        from mullet.cli.main import _run_init
        _run_init()
        assert config.read_text() == "existing: true\n"
        captured = capsys.readouterr()
        assert "exists, skipping" in captured.out


def test_init_detects_anthropic_key(clean_tmpdir):
    """mullet init should detect ANTHROPIC_API_KEY."""
    with mock.patch("mullet.cli.main.detect_local_provider", return_value=None), \
         mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True):
        from mullet.cli.main import _run_init
        _run_init()
        content = (clean_tmpdir / "mullet.yaml").read_text()
        assert "anthropic" in content.lower()


def test_init_detects_gemini_key(clean_tmpdir):
    """mullet init should detect GOOGLE_API_KEY."""
    with mock.patch("mullet.cli.main.detect_local_provider", return_value=None), \
         mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "AIza-test"}, clear=True):
        from mullet.cli.main import _run_init
        _run_init()
        content = (clean_tmpdir / "mullet.yaml").read_text()
        assert "gemini" in content.lower()


def test_init_detects_openrouter_key(clean_tmpdir):
    """mullet init should detect OPENROUTER_API_KEY."""
    with mock.patch("mullet.cli.main.detect_local_provider", return_value=None), \
         mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test"}, clear=True):
        from mullet.cli.main import _run_init
        _run_init()
        content = (clean_tmpdir / "mullet.yaml").read_text()
        assert "openrouter" in content.lower()


def test_init_config_has_routing_section(clean_tmpdir):
    """Generated config should have routing section."""
    with mock.patch("mullet.cli.main.detect_local_provider", return_value=None), \
         mock.patch.dict(os.environ, {}, clear=True):
        from mullet.cli.main import _run_init
        _run_init()
        content = (clean_tmpdir / "mullet.yaml").read_text()
        assert "routing:" in content
