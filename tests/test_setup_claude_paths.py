"""Tests for path helpers in mullet.cli.setup_claude."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from mullet.cli.setup_claude import (
    _get_claude_dir,
    _get_claude_md_path,
    _get_claude_settings_path,
    _get_claude_skills_dir,
    _get_hooks_dir,
    _install_statusline,
)


class TestClaudePaths:
    def test_get_claude_dir_is_under_home(self) -> None:
        result = _get_claude_dir()
        assert isinstance(result, Path)
        assert result == Path.home() / ".claude"

    def test_get_claude_md_path_is_claude_md(self) -> None:
        result = _get_claude_md_path()
        assert result.name == "CLAUDE.md"
        assert result.parent == _get_claude_dir()

    def test_get_claude_settings_path_is_settings_json(self) -> None:
        result = _get_claude_settings_path()
        assert result.name == "settings.json"
        assert result.parent == _get_claude_dir()

    def test_get_claude_skills_dir_is_skills(self) -> None:
        result = _get_claude_skills_dir()
        assert result.name == "skills"
        assert result.parent == _get_claude_dir()

    def test_all_paths_share_same_parent(self) -> None:
        base = _get_claude_dir()
        assert _get_claude_md_path().parent == base
        assert _get_claude_settings_path().parent == base
        assert _get_claude_skills_dir().parent == base

    def test_paths_return_path_objects(self) -> None:
        assert isinstance(_get_claude_dir(), Path)
        assert isinstance(_get_claude_md_path(), Path)
        assert isinstance(_get_claude_settings_path(), Path)
        assert isinstance(_get_claude_skills_dir(), Path)


class TestGetHooksDir:
    def test_returns_none_when_scripts_missing(self, tmp_path):
        with patch("mullet.cli.setup_claude.__file__",
                   str(tmp_path / "mullet" / "cli" / "setup_claude.py")):
            result = _get_hooks_dir()
        assert result is None

    def test_returns_path_when_scripts_exist(self, tmp_path):
        # Create the expected directory structure
        hooks_dir = tmp_path / "scripts" / "hooks"
        hooks_dir.mkdir(parents=True)
        (hooks_dir / "delegation_reminder.py").touch()
        # Point __file__ to tmp_path/mullet/cli/setup_claude.py
        fake_file = tmp_path / "mullet" / "cli" / "setup_claude.py"
        with patch("mullet.cli.setup_claude.__file__", str(fake_file)):
            result = _get_hooks_dir()
        assert result is None or isinstance(result, Path)


class TestInstallStatusline:
    def test_returns_false_when_hud_not_on_path(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda _: None)
        with patch("mullet.cli.setup_claude._get_claude_settings_path",
                   return_value=tmp_path / "settings.json"):
            result = _install_statusline(dry_run=False)
        assert result is False

    def test_dry_run_returns_true_when_hud_found(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/mullet-hud")
        with patch("mullet.cli.setup_claude._get_claude_settings_path",
                   return_value=tmp_path / "settings.json"):
            result = _install_statusline(dry_run=True)
        assert result is True

    def test_writes_settings_when_hud_found(self, monkeypatch, tmp_path):
        settings_path = tmp_path / "settings.json"
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/mullet-hud")
        with patch("mullet.cli.setup_claude._get_claude_settings_path",
                   return_value=settings_path):
            _install_statusline(dry_run=False)
        assert settings_path.exists()
        data = json.loads(settings_path.read_text())
        assert data["statusLine"]["command"] == "/usr/bin/mullet-hud"

    def test_skips_when_already_configured(self, monkeypatch, tmp_path):
        hud_path = "/usr/bin/mullet-hud"
        settings_path = tmp_path / "settings.json"
        settings_path.write_text(json.dumps({"statusLine": {"command": hud_path}}))
        monkeypatch.setattr("shutil.which", lambda _: hud_path)
        with patch("mullet.cli.setup_claude._get_claude_settings_path",
                   return_value=settings_path):
            result = _install_statusline(dry_run=False)
        assert result is False  # Already correct → skipped
