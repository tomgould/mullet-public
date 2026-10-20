"""Tests for mullet.cli.config_cli."""

from __future__ import annotations

import json

import pytest

from mullet.cli.config_cli import cmd_config

pytestmark = pytest.mark.harness


class TestConfig:
    def test_config_set_then_get(self, tmp_path: pytest.TempPathFactory, monkeypatch, capsys):
        config_path = tmp_path / "config.json"
        monkeypatch.setattr("mullet.cli.config_cli._CONFIG_PATH", config_path)
        monkeypatch.setattr("mullet.cli.config_cli._CONFIG_DIR", config_path.parent)

        assert cmd_config(["set", "theme", "light"]) == 0
        captured = capsys.readouterr()
        assert "Set theme = 'light'" in captured.out

        assert cmd_config(["get", "theme"]) == 0
        captured = capsys.readouterr()
        assert captured.out.strip() == "light"

    def test_config_list_outputs_json(self, tmp_path: pytest.TempPathFactory, monkeypatch, capsys):
        config_path = tmp_path / "config.json"
        monkeypatch.setattr("mullet.cli.config_cli._CONFIG_PATH", config_path)
        monkeypatch.setattr("mullet.cli.config_cli._CONFIG_DIR", config_path.parent)

        cmd_config(["set", "default_provider", "fast"])
        capsys.readouterr()  # clear buffer
        assert cmd_config(["list"]) == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["default_provider"] == "fast"
        assert "theme" in data

    def test_config_set_parses_boolean(self, tmp_path: pytest.TempPathFactory, monkeypatch, capsys):
        config_path = tmp_path / "config.json"
        monkeypatch.setattr("mullet.cli.config_cli._CONFIG_PATH", config_path)
        monkeypatch.setattr("mullet.cli.config_cli._CONFIG_DIR", config_path.parent)

        cmd_config(["set", "telemetry", "false"])
        captured = capsys.readouterr()
        assert "Set telemetry = False" in captured.out

        cmd_config(["get", "telemetry"])
        captured = capsys.readouterr()
        assert captured.out.strip() == "False"

    def test_config_set_parses_integer(self, tmp_path: pytest.TempPathFactory, monkeypatch, capsys):
        config_path = tmp_path / "config.json"
        monkeypatch.setattr("mullet.cli.config_cli._CONFIG_PATH", config_path)
        monkeypatch.setattr("mullet.cli.config_cli._CONFIG_DIR", config_path.parent)

        cmd_config(["set", "max_iterations", "42"])
        captured = capsys.readouterr()
        assert "Set max_iterations = 42" in captured.out

        cmd_config(["get", "max_iterations"])
        captured = capsys.readouterr()
        assert captured.out.strip() == "42"

    def test_config_get_unknown_returns_friendly(self, tmp_path: pytest.TempPathFactory, monkeypatch, capsys):
        config_path = tmp_path / "config.json"
        monkeypatch.setattr("mullet.cli.config_cli._CONFIG_PATH", config_path)
        monkeypatch.setattr("mullet.cli.config_cli._CONFIG_DIR", config_path.parent)

        rc = cmd_config(["get", "nonexistent_key"])
        captured = capsys.readouterr()
        assert rc == 1
        assert "not set" in captured.err

    def test_config_reset_restores_defaults(self, tmp_path: pytest.TempPathFactory, monkeypatch, capsys):
        config_path = tmp_path / "config.json"
        monkeypatch.setattr("mullet.cli.config_cli._CONFIG_PATH", config_path)
        monkeypatch.setattr("mullet.cli.config_cli._CONFIG_DIR", config_path.parent)

        cmd_config(["set", "theme", "neon"])
        assert cmd_config(["reset"]) == 0
        captured = capsys.readouterr()
        assert "reset to defaults" in captured.out

        cmd_config(["get", "theme"])
        captured = capsys.readouterr()
        assert captured.out.strip() == "dark"
