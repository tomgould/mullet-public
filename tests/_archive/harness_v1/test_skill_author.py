"""Tests for mullet.cli.skill_author."""

from __future__ import annotations

import pytest

from mullet.cli.skill_author import cmd_skill_new

pytestmark = pytest.mark.harness


class TestSkillNew:
    def test_new_creates_project_scope_by_default(self, tmp_path: pytest.TempPathFactory, monkeypatch, capsys):
        project = tmp_path / "project"
        project.mkdir()
        monkeypatch.setattr("mullet.cli.skill_author.Path.cwd", lambda: project)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        rc = cmd_skill_new(["test-skill"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "Created skill 'test-skill'" in captured.out
        skill_file = project / ".mullet" / "skills" / "test-skill" / "SKILL.md"
        assert skill_file.exists()
        content = skill_file.read_text()
        assert "name: test-skill" in content
        assert "tools: [Bash, Read, Edit]" in content

    def test_new_user_flag_creates_user_scope(self, tmp_path, monkeypatch, capsys):
        user_dir = tmp_path / "user_skills"
        monkeypatch.setattr(
            "mullet.cli.skill_author.Path.expanduser",
            lambda self: user_dir if str(self).startswith("~") else self,
        )
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        rc = cmd_skill_new(["user-skill", "--user"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "Created skill 'user-skill'" in captured.out
        skill_file = user_dir / "user-skill" / "SKILL.md"
        assert skill_file.exists()

    def test_new_global_flag_rejected_for_safety(self, capsys):
        rc = cmd_skill_new(["global-skill", "--global"])
        captured = capsys.readouterr()
        assert rc == 1
        assert "--global scope is rejected for safety" in captured.err

    def test_new_existing_skill_refused_unless_force(self, tmp_path, monkeypatch, capsys):
        project = tmp_path / "project"
        project.mkdir()
        monkeypatch.setattr("mullet.cli.skill_author.Path.cwd", lambda: project)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        cmd_skill_new(["existing-skill"])
        rc = cmd_skill_new(["existing-skill"])
        captured = capsys.readouterr()
        assert rc == 1
        assert "already exists" in captured.err

    def test_new_force_overwrites(self, tmp_path, monkeypatch, capsys):
        project = tmp_path / "project"
        project.mkdir()
        monkeypatch.setattr("mullet.cli.skill_author.Path.cwd", lambda: project)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        cmd_skill_new(["overwrite-skill"])
        skill_file = project / ".mullet" / "skills" / "overwrite-skill" / "SKILL.md"
        skill_file.write_text("old content")
        rc = cmd_skill_new(["overwrite-skill", "--force"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "Created skill 'overwrite-skill'" in captured.out
        content = skill_file.read_text()
        assert "name: overwrite-skill" in content
