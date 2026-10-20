"""Tests for `mullet harness-init` CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from mullet.cli.init_cli import (
    _build_memory_md,
    _build_mcp_template,
    _build_permissions,
    _detect_project_type,
    _read_git_log,
    _read_readme_snippet,
    _suggest_skills,
    cmd_init,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_python_project(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "myproj"\n')
    (tmp_path / "README.md").write_text("# MyProj\n\nA python project.\n")
    return tmp_path


@pytest.fixture
def fake_js_project(tmp_path: Path) -> Path:
    (tmp_path / "package.json").write_text('{"name": "myapp"}\n')
    return tmp_path


@pytest.fixture
def fake_rust_project(tmp_path: Path) -> Path:
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "mylib"\n')
    return tmp_path


# ---------------------------------------------------------------------------
# _detect_project_type
# ---------------------------------------------------------------------------

class TestDetectProjectType:
    def test_python_via_pyproject_toml(self, fake_python_project: Path):
        assert _detect_project_type(fake_python_project) == "python"

    def test_python_via_requirements_txt(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("pytest\n")
        assert _detect_project_type(tmp_path) == "python"

    def test_js_via_package_json(self, fake_js_project: Path):
        assert _detect_project_type(fake_js_project) == "js"

    def test_rust_via_cargo_toml(self, fake_rust_project: Path):
        assert _detect_project_type(fake_rust_project) == "rust"

    def test_go_via_go_mod(self, tmp_path: Path):
        (tmp_path / "go.mod").write_text("module example.com/foo\n")
        assert _detect_project_type(tmp_path) == "go"

    def test_other_empty_dir(self, tmp_path: Path):
        assert _detect_project_type(tmp_path) == "other"


# ---------------------------------------------------------------------------
# _build_permissions
# ---------------------------------------------------------------------------

class TestBuildPermissions:
    def test_python_has_deny_rm_rf(self, tmp_path: Path):
        perms = _build_permissions("python", tmp_path)
        assert "rm -rf /" in perms["denied_commands"]
        assert any("pip uninstall" in c for c in perms["denied_commands"])

    def test_js_has_deny_npm_unpublish(self, tmp_path: Path):
        perms = _build_permissions("js", tmp_path)
        assert any("npm unpublish" in c for c in perms["denied_commands"])

    def test_default_mode(self, tmp_path: Path):
        perms = _build_permissions("other", tmp_path)
        assert perms["mode"] == "default"
        assert perms["denied_tools"] == []

    def test_path_rules_allow_project(self, tmp_path: Path):
        perms = _build_permissions("go", tmp_path)
        patterns = [r["pattern"] for r in perms["path_rules"] if r.get("allow")]
        assert any(str(tmp_path) in p for p in patterns)


# ---------------------------------------------------------------------------
# _build_mcp_template
# ---------------------------------------------------------------------------

class TestBuildMcpTemplate:
    def test_always_has_filesystem(self, tmp_path: Path):
        mcp = _build_mcp_template("python", tmp_path)
        assert "filesystem" in mcp

    def test_always_has_github(self, tmp_path: Path):
        mcp = _build_mcp_template("python", tmp_path)
        assert "github" in mcp

    def test_python_sqlalchemy_gets_sqlite(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("sqlalchemy>=2.0\n")
        mcp = _build_mcp_template("python", tmp_path)
        assert "sqlite" in mcp

    def test_python_django_gets_sqlite(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text('[project]\ndependencies = ["django"]\n')
        mcp = _build_mcp_template("python", tmp_path)
        assert "sqlite" in mcp

    def test_python_no_sqlalchemy_no_sqlite(self, tmp_path: Path):
        mcp = _build_mcp_template("python", tmp_path)
        assert "sqlite" not in mcp

    def test_js_playwright_gets_playwright(self, tmp_path: Path):
        (tmp_path / "package.json").write_text('{"devDependencies": {"@playwright/test": "^1.0"}}')
        mcp = _build_mcp_template("js", tmp_path)
        assert "playwright" in mcp

    def test_js_no_playwright(self, tmp_path: Path):
        mcp = _build_mcp_template("js", tmp_path)
        assert "playwright" not in mcp


# ---------------------------------------------------------------------------
# _suggest_skills
# ---------------------------------------------------------------------------

class TestSuggestSkills:
    def test_python_returns_code_review(self):
        skills = _suggest_skills("python")
        assert "code-review" in skills
        assert "plan" in skills

    def test_js_returns_ultrawork(self):
        skills = _suggest_skills("js")
        assert "ultrawork" in skills

    def test_other_returns_brainstorm(self):
        skills = _suggest_skills("other")
        assert "brainstorm" in skills


# ---------------------------------------------------------------------------
# _read_readme_snippet
# ---------------------------------------------------------------------------

class TestReadReadmeSnippet:
    def test_reads_first_500_chars(self, tmp_path: Path):
        readme = tmp_path / "README.md"
        readme.write_text("A" * 1000)
        assert len(_read_readme_snippet(tmp_path)) == 500

    def test_fallback_to_readme_txt(self, tmp_path: Path):
        (tmp_path / "README.txt").write_text("Hello\n")
        assert "Hello" in _read_readme_snippet(tmp_path)

    def test_empty_when_no_readme(self, tmp_path: Path):
        assert _read_readme_snippet(tmp_path) == ""


# ---------------------------------------------------------------------------
# _read_git_log
# ---------------------------------------------------------------------------

class TestReadGitLog:
    def test_returns_empty_outside_git(self, tmp_path: Path):
        assert _read_git_log(tmp_path) == ""

    def test_returns_commits_inside_git(self, tmp_path: Path):
        import subprocess
        subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(tmp_path), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=str(tmp_path), check=True, capture_output=True)
        (tmp_path / "a.txt").write_text("a")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "first"], cwd=str(tmp_path), check=True, capture_output=True)
        log = _read_git_log(tmp_path, lines=5)
        assert "first" in log


# ---------------------------------------------------------------------------
# _build_memory_md
# ---------------------------------------------------------------------------

class TestBuildMemoryMd:
    def test_includes_project_name(self, tmp_path: Path):
        md = _build_memory_md(tmp_path, "myproj")
        assert "# myproj" in md

    def test_includes_readme(self, tmp_path: Path):
        (tmp_path / "README.md").write_text("# Hello")
        md = _build_memory_md(tmp_path, "p")
        assert "Hello" in md

    def test_escapes_backticks(self, tmp_path: Path):
        (tmp_path / "README.md").write_text("```python\nprint(1)\n```")
        md = _build_memory_md(tmp_path, "p")
        assert "```" not in md or "` ` `" in md


# ---------------------------------------------------------------------------
# cmd_init integration tests
# ---------------------------------------------------------------------------

class TestCmdInitIntegration:
    def test_creates_dot_mullet(self, fake_python_project: Path, monkeypatch, capsys):
        monkeypatch.chdir(fake_python_project)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        with patch("mullet.cli.init_cli._run_pool_create") as mock_pool:
            rc = cmd_init(["--yes"])
        captured = capsys.readouterr()
        assert rc == 0
        assert (fake_python_project / ".mullet").exists()
        assert (fake_python_project / ".mullet" / "skills").is_dir()
        assert (fake_python_project / ".mullet" / "plugins").is_dir()
        assert (fake_python_project / ".mullet" / "memory").is_dir()
        assert (fake_python_project / ".mullet" / "permissions.yaml").exists()
        assert (fake_python_project / ".mullet" / "MEMORY.md").exists()
        assert (fake_python_project / ".mcp.json").exists()
        mock_pool.assert_called_once()

    def test_overwrites_with_force(self, fake_python_project: Path, monkeypatch, capsys):
        monkeypatch.chdir(fake_python_project)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        with patch("mullet.cli.init_cli._run_pool_create"):
            cmd_init(["--yes"])
        old_perms = (fake_python_project / ".mullet" / "permissions.yaml").read_text()
        with patch("mullet.cli.init_cli._run_pool_create"):
            rc = cmd_init(["--yes", "--force"])
        captured = capsys.readouterr()
        assert rc == 0
        # Should have overwritten
        assert "✓ Wrote" in captured.out

    def test_dry_run_does_not_create(self, fake_python_project: Path, monkeypatch, capsys):
        monkeypatch.chdir(fake_python_project)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        with patch("mullet.cli.init_cli._run_pool_create") as mock_pool:
            rc = cmd_init(["--yes", "--dry-run"])
        captured = capsys.readouterr()
        assert rc == 0
        assert not (fake_python_project / ".mullet").exists()
        assert "[dry-run]" in captured.out
        mock_pool.assert_called_once()

    def test_no_pool_skips_pool(self, fake_python_project: Path, monkeypatch, capsys):
        monkeypatch.chdir(fake_python_project)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        with patch("mullet.cli.init_cli._run_pool_create") as mock_pool:
            rc = cmd_init(["--yes", "--no-pool"])
        captured = capsys.readouterr()
        assert rc == 0
        mock_pool.assert_not_called()
        assert "Pool creation skipped" in captured.out

    def test_type_override(self, tmp_path: Path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        with patch("mullet.cli.init_cli._run_pool_create"):
            rc = cmd_init(["--yes", "--type=rust"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "rust" in captured.out
        perms = yaml.safe_load((tmp_path / ".mullet" / "permissions.yaml").read_text())
        assert any("cargo clean" in c for c in perms["denied_commands"])

    def test_mcp_json_content(self, fake_python_project: Path, monkeypatch, capsys):
        monkeypatch.chdir(fake_python_project)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        with patch("mullet.cli.init_cli._run_pool_create"):
            cmd_init(["--yes"])
        mcp = json.loads((fake_python_project / ".mcp.json").read_text())
        assert "filesystem" in mcp
        assert "github" in mcp

    def test_memory_md_populated(self, fake_python_project: Path, monkeypatch, capsys):
        monkeypatch.chdir(fake_python_project)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        with patch("mullet.cli.init_cli._run_pool_create"):
            cmd_init(["--yes"])
        mem = (fake_python_project / ".mullet" / "MEMORY.md").read_text()
        assert "MyProj" in mem
        assert "A python project" in mem
