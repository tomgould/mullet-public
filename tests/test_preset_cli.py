"""Tests for mullet.cli.preset_cli."""
from __future__ import annotations

import pytest

from mullet.cli.preset_cli import cmd_preset
from mullet.core.presets import BUILTIN_PRESETS, BUILTIN_PRESET_NAMES, PresetManager


class TestCmdPresetList:
    def test_list_prints_builtins(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
        rc = cmd_preset(["list"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "balanced-mix" in captured.out
        assert "local-only" in captured.out
        assert "claude-only" in captured.out
        assert "cheap-cloud" in captured.out

    def test_no_args_prints_builtins(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
        rc = cmd_preset([])
        assert rc == 0
        captured = capsys.readouterr()
        assert "balanced-mix" in captured.out

    def test_active_marked(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
        mgr = PresetManager(config_dir=tmp_path)
        mgr.load()
        mgr.use("local-only")
        rc = cmd_preset(["list"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "* local-only" in captured.out


class TestCmdPresetUse:
    def test_use_sets_active(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
        rc = cmd_preset(["use", "local-only"])
        assert rc == 0
        mgr = PresetManager(config_dir=tmp_path)
        assert mgr.active().name == "local-only"

    def test_use_unknown_returns_1(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
        rc = cmd_preset(["use", "bogus"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "unknown preset" in captured.err


class TestCmdPresetShow:
    def test_show_builtin(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
        rc = cmd_preset(["show", "balanced-mix"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "balanced-mix" in captured.out
        assert "label:" in captured.out

    def test_show_unknown_returns_1(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
        rc = cmd_preset(["show", "missing"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "unknown preset" in captured.err


class TestCmdPresetAdd:
    def test_add_custom(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
        rc = cmd_preset(
            [
                "add",
                "--name",
                "custom",
                "--label",
                "Custom",
                "--providers",
                "local,kimi",
                "--speed",
                "fast",
                "--opencode-model",
                "openrouter/deepseek-v3",
            ]
        )
        assert rc == 0
        mgr = PresetManager(config_dir=tmp_path)
        preset = mgr.get("custom")
        assert preset is not None
        assert preset.label == "Custom"
        assert preset.allowed_providers == ["local", "kimi"]
        assert preset.speed_mode == "fast"
        assert preset.opencode_model == "openrouter/deepseek-v3"

    def test_add_duplicate_returns_1(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
        cmd_preset(["add", "--name", "custom", "--label", "Custom"])
        rc = cmd_preset(["add", "--name", "custom", "--label", "Custom2"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "already exists" in captured.err


class TestCmdPresetRemove:
    def test_remove_custom(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
        cmd_preset(["add", "--name", "custom", "--label", "Custom"])
        rc = cmd_preset(["remove", "custom"])
        assert rc == 0
        mgr = PresetManager(config_dir=tmp_path)
        assert mgr.get("custom") is None

    def test_remove_builtin_without_force_returns_1(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
        # Switch active away so the built-in check fires first
        cmd_preset(["use", "local-only"])
        rc = cmd_preset(["remove", "balanced-mix"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "refusing to remove built-in" in captured.err

    def test_remove_builtin_with_force(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
        rc = cmd_preset(["remove", "balanced-mix", "--force"])
        assert rc == 0

    def test_remove_unknown_returns_1(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
        rc = cmd_preset(["remove", "missing"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "unknown preset" in captured.err
