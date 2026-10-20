"""Tests for mullet.cli.new_cli."""

from __future__ import annotations

import subprocess

import pytest

from mullet.cli.new_cli import cmd_new

pytestmark = pytest.mark.harness


class TestNew:
    def test_new_skill_delegates_to_skill_author(self, tmp_path: pytest.TempPathFactory, monkeypatch, capsys):
        project = tmp_path / "project"
        project.mkdir()
        monkeypatch.setattr("mullet.cli.skill_author.Path.cwd", lambda: project)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        rc = cmd_new(["skill", "my-skill"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "Created skill 'my-skill'" in captured.out
        skill_file = project / ".mullet" / "skills" / "my-skill" / "SKILL.md"
        assert skill_file.exists()

    def test_new_plugin_creates_manifest(self, tmp_path: pytest.TempPathFactory, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        rc = cmd_new(["plugin", "my-plugin"])
        captured = capsys.readouterr()
        assert rc == 0
        manifest = tmp_path / "my-plugin" / ".claude-plugin" / "plugin.json"
        assert manifest.exists()
        import json
        data = json.loads(manifest.read_text())
        assert data["name"] == "my-plugin"

    def test_new_plugin_creates_subdirs(self, tmp_path: pytest.TempPathFactory, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        rc = cmd_new(["plugin", "my-plugin"])
        captured = capsys.readouterr()
        assert rc == 0
        base = tmp_path / "my-plugin" / ".claude-plugin"
        assert (base / "skills").is_dir()
        assert (base / "commands").is_dir()
        assert (base / "agents").is_dir()
        assert (tmp_path / "my-plugin" / "README.md").exists()

    def test_new_project_runs_harness_init(self, tmp_path: pytest.TempPathFactory, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        # Ensure git is available; skip test if not
        git_check = subprocess.run(["git", "--version"], capture_output=True)
        if git_check.returncode != 0:
            pytest.skip("git not available")

        rc = cmd_new(["project", "my-project"])
        captured = capsys.readouterr()
        assert rc == 0
        dot_mullet = tmp_path / "my-project" / ".mullet"
        assert dot_mullet.is_dir()
        assert (dot_mullet / "permissions.yaml").exists()
        assert (dot_mullet / "MEMORY.md").exists()
        assert (tmp_path / "my-project" / ".git").exists()
