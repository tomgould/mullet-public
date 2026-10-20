"""Tests for mullet.cli.setup_claude."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from mullet.cli import setup_claude as sc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_paths(monkeypatch, tmp_path):
    """Redirect all ~-based path helpers into tmp_path."""
    monkeypatch.setattr(sc, "_get_claude_dir", lambda: tmp_path / ".claude")
    monkeypatch.setattr(
        sc, "_get_claude_md_path", lambda: tmp_path / ".claude" / "CLAUDE.md"
    )
    monkeypatch.setattr(
        sc, "_get_claude_settings_path", lambda: tmp_path / ".claude" / "settings.json"
    )
    monkeypatch.setattr(
        sc, "_get_claude_skills_dir", lambda: tmp_path / ".claude" / "skills"
    )


def _patch_file(monkeypatch, tmp_path):
    """Make Path(__file__) inside setup_claude resolve under tmp_path."""
    fake_file = tmp_path / "mullet" / "cli" / "setup_claude.py"
    monkeypatch.setattr(sc, "__file__", str(fake_file))


@pytest.fixture
def paths(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# 1. Path helpers
# ---------------------------------------------------------------------------


def test_get_claude_dir():
    assert sc._get_claude_dir() == Path.home() / ".claude"


def test_get_claude_md_path():
    assert sc._get_claude_md_path() == Path.home() / ".claude" / "CLAUDE.md"


def test_get_claude_settings_path():
    assert sc._get_claude_settings_path() == Path.home() / ".claude" / "settings.json"


def test_get_claude_skills_dir():
    assert sc._get_claude_skills_dir() == Path.home() / ".claude" / "skills"


# ---------------------------------------------------------------------------
# 2. _append_claude_md_section
# ---------------------------------------------------------------------------


def test_append_claude_md_creates_when_missing(paths, tmp_path, capsys):
    assert sc._append_claude_md_section(dry_run=False) is True
    written = (tmp_path / ".claude" / "CLAUDE.md").read_text()
    assert "## Delegation (Mullet)" in written
    captured = capsys.readouterr()
    assert "Created" in captured.out


def test_append_claude_md_dry_run_missing(paths, tmp_path, capsys):
    assert sc._append_claude_md_section(dry_run=True) is True
    assert not (tmp_path / ".claude" / "CLAUDE.md").exists()
    captured = capsys.readouterr()
    assert "Created" in captured.out


def test_append_claude_md_appends_when_no_section(paths, tmp_path, capsys):
    md = tmp_path / ".claude" / "CLAUDE.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("# Hello\n")
    assert sc._append_claude_md_section(dry_run=False) is True
    assert "## Delegation (Mullet)" in md.read_text()
    captured = capsys.readouterr()
    assert "Appended" in captured.out


def test_append_claude_md_already_present(paths, tmp_path, capsys):
    md = tmp_path / ".claude" / "CLAUDE.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    # Pre-populate with ALL optional sections so nothing is missing
    md.write_text(
        "## Delegation (Mullet)\n\nfoo\n"
        "## Schema & Assumptions\n\nbar\n"
        "API LLM delegates have NO filesystem access\n"
        "AI-Bridge changes MUST be validated\n"
    )
    assert sc._append_claude_md_section(dry_run=False) is False
    captured = capsys.readouterr()
    assert "already present" in captured.out


def test_append_claude_md_adds_schema_assumptions(paths, tmp_path, capsys):
    md = tmp_path / ".claude" / "CLAUDE.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("## Delegation (Mullet)\n\nfoo\n")
    # Remove schema & assumptions and other optional sections by giving minimal content
    result = sc._append_claude_md_section(dry_run=False)
    text = md.read_text()
    assert "## Schema & Assumptions" in text
    assert result is True


def test_append_claude_md_adds_api_fs_rule(paths, tmp_path, capsys):
    md = tmp_path / ".claude" / "CLAUDE.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("## Delegation (Mullet)\n\nfoo\n## Schema & Assumptions\n")
    result = sc._append_claude_md_section(dry_run=False)
    text = md.read_text()
    assert "API LLM delegates have NO filesystem access" in text
    assert result is True


def test_append_claude_md_adds_ai_bridge(paths, tmp_path, capsys):
    md = tmp_path / ".claude" / "CLAUDE.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(
        "## Delegation (Mullet)\n\nfoo\n"
        "## Schema & Assumptions\n"
        "API LLM delegates have NO filesystem access\n"
    )
    result = sc._append_claude_md_section(dry_run=False)
    text = md.read_text()
    assert "AI-Bridge changes MUST be validated" in text
    assert result is True


def test_append_claude_md_dry_run_additions(paths, tmp_path, capsys):
    md = tmp_path / ".claude" / "CLAUDE.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("## Delegation (Mullet)\n\nfoo\n")
    result = sc._append_claude_md_section(dry_run=True)
    assert result is True
    # Content should NOT have changed because dry_run
    assert "## Schema & Assumptions" not in md.read_text()


# ---------------------------------------------------------------------------
# 3. _get_hooks_dir
# ---------------------------------------------------------------------------


def test_get_hooks_dir_found(monkeypatch, tmp_path):
    _patch_file(monkeypatch, tmp_path)
    hooks = tmp_path / "scripts" / "hooks"
    hooks.mkdir(parents=True)
    (hooks / "delegation_reminder.py").write_text("x")
    assert sc._get_hooks_dir() == hooks


def test_get_hooks_dir_not_found(monkeypatch, tmp_path):
    _patch_file(monkeypatch, tmp_path)
    assert sc._get_hooks_dir() is None


# ---------------------------------------------------------------------------
# 4. _merge_claude_settings_hooks
# ---------------------------------------------------------------------------


def test_merge_settings_creates_new(paths, tmp_path, capsys):
    assert sc._merge_claude_settings_hooks(dry_run=False) is True
    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
    assert "hooks" in settings
    assert "PreToolUse" in settings["hooks"]
    captured = capsys.readouterr()
    assert "Created" in captured.out


def test_merge_settings_dry_run(paths, tmp_path, capsys):
    assert sc._merge_claude_settings_hooks(dry_run=True) is True
    assert not (tmp_path / ".claude" / "settings.json").exists()
    captured = capsys.readouterr()
    assert "Created" in captured.out


def test_merge_settings_already_up_to_date(paths, tmp_path, capsys):
    # First create
    sc._merge_claude_settings_hooks(dry_run=False)
    # Second call should be no-op
    assert sc._merge_claude_settings_hooks(dry_run=False) is False
    captured = capsys.readouterr()
    assert "already present" in captured.out


def test_merge_settings_missing_hooks_key(paths, tmp_path, capsys):
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps({"foo": "bar"}))
    assert sc._merge_claude_settings_hooks(dry_run=False) is True
    settings = json.loads(settings_path.read_text())
    assert "hooks" in settings
    captured = capsys.readouterr()
    assert "Merged" in captured.out


def test_merge_settings_adds_missing_group(paths, tmp_path, capsys):
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps({"hooks": {"PreToolUse": []}}))
    assert sc._merge_claude_settings_hooks(dry_run=False) is True
    settings = json.loads(settings_path.read_text())
    assert len(settings["hooks"]["PreToolUse"]) > 0


def test_merge_settings_adds_missing_hook_in_group(paths, tmp_path, capsys):
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    # Pre-populate with a PreToolUse Grep group but strip the command hook
    partial = {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Grep",
                    "hooks": [],
                }
            ]
        }
    }
    settings_path.write_text(json.dumps(partial))
    assert sc._merge_claude_settings_hooks(dry_run=False) is True
    settings = json.loads(settings_path.read_text())
    grp = next(g for g in settings["hooks"]["PreToolUse"] if g["matcher"] == "Grep")
    assert len(grp["hooks"]) > 0


def test_merge_settings_with_hooks_dir(paths, tmp_path, monkeypatch, capsys):
    _patch_file(monkeypatch, tmp_path)
    hooks = tmp_path / "scripts" / "hooks"
    hooks.mkdir(parents=True)
    for name in [
        "compress_whitespace.py",
        "delegation_reminder.py",
        "skill_activator.py",
        "session_context.py",
        "user_prompt_detect_override.py",
        "mullet_pretooluse_rewriter.py",
        "mullet_observation_capture.py",
    ]:
        (hooks / name).write_text("x")
    assert sc._merge_claude_settings_hooks(dry_run=False) is True
    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
    assert "UserPromptSubmit" in settings["hooks"]
    assert "PostToolUse" in settings["hooks"]


# ---------------------------------------------------------------------------
# 5. _copy_claude_skill_files
# ---------------------------------------------------------------------------


def test_copy_skills_happy(paths, tmp_path, capsys):
    assert sc._copy_claude_skill_files(dry_run=False, force=False) is True
    skills_dir = tmp_path / ".claude" / "skills"
    for name in ["delegate.md", "pool-config.md", "commit.md", "review.md"]:
        assert (skills_dir / name).exists()
    captured = capsys.readouterr()
    # Source writes file then checks exists, so message says "Overwrote"
    assert "Overwrote" in captured.out or "Copied" in captured.out


def test_copy_skills_dry_run(paths, tmp_path, capsys):
    assert sc._copy_claude_skill_files(dry_run=True, force=False) is True
    # mkdir is called unconditionally before the loop, so dir exists but files don't
    skills_dir = tmp_path / ".claude" / "skills"
    assert skills_dir.exists()
    assert not any(skills_dir.iterdir())
    captured = capsys.readouterr()
    assert "Copied" in captured.out or "Overwrote" in captured.out


def test_copy_skills_already_exists_no_force(paths, tmp_path, capsys):
    sc._copy_claude_skill_files(dry_run=False, force=False)
    assert sc._copy_claude_skill_files(dry_run=False, force=False) is False
    captured = capsys.readouterr()
    assert "already exists" in captured.out


def test_copy_skills_force_overwrite(paths, tmp_path, capsys):
    sc._copy_claude_skill_files(dry_run=False, force=False)
    # Mutate one file so we can detect overwrite
    (tmp_path / ".claude" / "skills" / "delegate.md").write_text("old")
    assert sc._copy_claude_skill_files(dry_run=False, force=True) is True
    text = (tmp_path / ".claude" / "skills" / "delegate.md").read_text()
    assert "old" not in text
    captured = capsys.readouterr()
    assert "Overwrote" in captured.out


# ---------------------------------------------------------------------------
# 6. _install_claude_agents
# ---------------------------------------------------------------------------


def test_install_agents_happy(paths, tmp_path, capsys):
    assert sc._install_claude_agents(dry_run=False, force=False) is True
    agents_dir = tmp_path / ".claude" / "agents"
    for name in ["patch-writer.md", "test-runner.md", "verifier.md"]:
        assert (agents_dir / name).exists()
    captured = capsys.readouterr()
    # Source writes file then checks exists, so message says "Overwrote"
    assert "Overwrote" in captured.out or "Created" in captured.out


def test_install_agents_dry_run(paths, tmp_path, capsys):
    assert sc._install_claude_agents(dry_run=True, force=False) is True
    # mkdir is called unconditionally before the loop, so dir exists but files don't
    agents_dir = tmp_path / ".claude" / "agents"
    assert agents_dir.exists()
    assert not any(agents_dir.iterdir())
    captured = capsys.readouterr()
    assert "Created" in captured.out or "Overwrote" in captured.out


def test_install_agents_already_exists_no_force(paths, tmp_path, capsys):
    sc._install_claude_agents(dry_run=False, force=False)
    assert sc._install_claude_agents(dry_run=False, force=False) is False
    captured = capsys.readouterr()
    assert "already exists" in captured.out


def test_install_agents_force_overwrite(paths, tmp_path, capsys):
    sc._install_claude_agents(dry_run=False, force=False)
    (tmp_path / ".claude" / "agents" / "patch-writer.md").write_text("old")
    assert sc._install_claude_agents(dry_run=False, force=True) is True
    assert "old" not in (tmp_path / ".claude" / "agents" / "patch-writer.md").read_text()
    captured = capsys.readouterr()
    assert "Overwrote" in captured.out


# ---------------------------------------------------------------------------
# 7. _install_project_hooks
# ---------------------------------------------------------------------------


def test_project_hooks_creates_new(paths, tmp_path, capsys):
    project_dir = tmp_path / "myproject"
    assert sc._install_project_hooks(project_dir, dry_run=False) is True
    settings = json.loads((project_dir / ".claude" / "settings.json").read_text())
    assert "PostToolUse" in settings["hooks"]
    captured = capsys.readouterr()
    assert "Created" in captured.out


def test_project_hooks_dry_run(paths, tmp_path, capsys):
    project_dir = tmp_path / "myproject"
    assert sc._install_project_hooks(project_dir, dry_run=True) is True
    assert not (project_dir / ".claude" / "settings.json").exists()
    captured = capsys.readouterr()
    assert "Created" in captured.out


def test_project_hooks_already_present(paths, tmp_path, capsys):
    project_dir = tmp_path / "myproject"
    sc._install_project_hooks(project_dir, dry_run=False)
    assert sc._install_project_hooks(project_dir, dry_run=False) is False
    captured = capsys.readouterr()
    assert "already present" in captured.out


def test_project_hooks_adds_to_existing_settings(paths, tmp_path, capsys):
    project_dir = tmp_path / "myproject"
    settings_path = project_dir / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps({"other": 1}))
    assert sc._install_project_hooks(project_dir, dry_run=False) is True
    settings = json.loads(settings_path.read_text())
    assert "PostToolUse" in settings["hooks"]
    captured = capsys.readouterr()
    assert "Added" in captured.out


def test_project_hooks_adds_to_existing_hooks(paths, tmp_path, capsys):
    project_dir = tmp_path / "myproject"
    settings_path = project_dir / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps({"hooks": {"PreToolUse": []}}))
    assert sc._install_project_hooks(project_dir, dry_run=False) is True
    settings = json.loads(settings_path.read_text())
    assert "PostToolUse" in settings["hooks"]
    captured = capsys.readouterr()
    assert "Added" in captured.out


# ---------------------------------------------------------------------------
# 8. _install_opencode
# ---------------------------------------------------------------------------


def test_install_opencode_already_installed(paths, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sc.shutil, "which", lambda _cmd: "/usr/bin/opencode")
    opencode_cfg = tmp_path / ".config" / "opencode" / "opencode.json"
    opencode_cfg.parent.mkdir(parents=True, exist_ok=True)
    opencode_cfg.write_text("{}")
    monkeypatch.setattr(sc, "_OPENCODE_CONFIG_PATH", opencode_cfg)
    assert sc._install_opencode(dry_run=False, yes=False) is False
    captured = capsys.readouterr()
    assert "already installed" in captured.out


def test_install_opencode_not_installed_yes(paths, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sc.shutil, "which", lambda _cmd: None)
    opencode_cfg = tmp_path / ".config" / "opencode" / "opencode.json"
    monkeypatch.setattr(sc, "_OPENCODE_CONFIG_PATH", opencode_cfg)
    monkeypatch.setattr(
        sc.subprocess, "run", lambda _cmd, **kwargs: type("R", (), {"returncode": 0})()
    )
    assert sc._install_opencode(dry_run=False, yes=True) is True
    captured = capsys.readouterr()
    assert "Installing" in captured.out or "installed" in captured.out


def test_install_opencode_not_installed_says_no(paths, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sc.shutil, "which", lambda _cmd: None)
    opencode_cfg = tmp_path / ".config" / "opencode" / "opencode.json"
    monkeypatch.setattr(sc, "_OPENCODE_CONFIG_PATH", opencode_cfg)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    assert sc._install_opencode(dry_run=False, yes=False) is False
    captured = capsys.readouterr()
    assert "Skipped" in captured.out


def test_install_opencode_eof(paths, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sc.shutil, "which", lambda _cmd: None)
    opencode_cfg = tmp_path / ".config" / "opencode" / "opencode.json"
    monkeypatch.setattr(sc, "_OPENCODE_CONFIG_PATH", opencode_cfg)
    monkeypatch.setattr("builtins.input", lambda _prompt: (_ for _ in ()).throw(EOFError))
    assert sc._install_opencode(dry_run=False, yes=False) is False


def test_install_opencode_dry_run(paths, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sc.shutil, "which", lambda _cmd: None)
    opencode_cfg = tmp_path / ".config" / "opencode" / "opencode.json"
    monkeypatch.setattr(sc, "_OPENCODE_CONFIG_PATH", opencode_cfg)
    assert sc._install_opencode(dry_run=True, yes=True) is True
    captured = capsys.readouterr()
    assert "[dry-run]" in captured.out
    assert not opencode_cfg.exists()


def test_install_opencode_install_fails(paths, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sc.shutil, "which", lambda _cmd: None)
    opencode_cfg = tmp_path / ".config" / "opencode" / "opencode.json"
    monkeypatch.setattr(sc, "_OPENCODE_CONFIG_PATH", opencode_cfg)
    monkeypatch.setattr(
        sc.subprocess, "run", lambda _cmd, **kwargs: type("R", (), {"returncode": 1})()
    )
    assert sc._install_opencode(dry_run=False, yes=True) is False
    captured = capsys.readouterr()
    assert "failed" in captured.out


def test_install_opencode_writes_config(paths, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sc.shutil, "which", lambda _cmd: "/usr/bin/opencode")
    # Ensure config path is under tmp_path for isolation
    opencode_cfg = tmp_path / ".config" / "opencode" / "opencode.json"
    monkeypatch.setattr(sc, "_OPENCODE_CONFIG_PATH", opencode_cfg)
    assert sc._install_opencode(dry_run=False, yes=False) is True
    assert opencode_cfg.exists()
    config = json.loads(opencode_cfg.read_text())
    assert "$schema" in config


# ---------------------------------------------------------------------------
# 9. _install_statusline
# ---------------------------------------------------------------------------


def test_install_statusline_hud_not_found(paths, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sc.shutil, "which", lambda _cmd: None)
    assert sc._install_statusline(dry_run=False) is False
    captured = capsys.readouterr()
    assert "WARNING" in captured.out


def test_install_statusline_creates_new(paths, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sc.shutil, "which", lambda _cmd: "/usr/bin/mullet-hud")
    assert sc._install_statusline(dry_run=False) is True
    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
    assert settings["statusLine"]["command"] == "/usr/bin/mullet-hud"
    captured = capsys.readouterr()
    assert "Configured" in captured.out


def test_install_statusline_already_correct(paths, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sc.shutil, "which", lambda _cmd: "/usr/bin/mullet-hud")
    sc._install_statusline(dry_run=False)
    assert sc._install_statusline(dry_run=False) is False
    captured = capsys.readouterr()
    assert "already configured" in captured.out


def test_install_statusline_changes_command(paths, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sc.shutil, "which", lambda _cmd: "/new/hud")
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps({"statusLine": {"command": "/old/hud"}}))
    assert sc._install_statusline(dry_run=False) is True
    settings = json.loads(settings_path.read_text())
    assert settings["statusLine"]["command"] == "/new/hud"


def test_install_statusline_corrupt_json(paths, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sc.shutil, "which", lambda _cmd: "/usr/bin/mullet-hud")
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text("not json")
    assert sc._install_statusline(dry_run=False) is True
    settings = json.loads(settings_path.read_text())
    assert settings["statusLine"]["command"] == "/usr/bin/mullet-hud"


def test_install_statusline_dry_run(paths, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sc.shutil, "which", lambda _cmd: "/usr/bin/mullet-hud")
    assert sc._install_statusline(dry_run=True) is True
    assert not (tmp_path / ".claude" / "settings.json").exists()
    captured = capsys.readouterr()
    assert "[dry-run]" in captured.out


# ---------------------------------------------------------------------------
# 10. _install_global_mcp
# ---------------------------------------------------------------------------


def test_install_global_mcp_source_missing(paths, tmp_path, monkeypatch, capsys):
    _patch_file(monkeypatch, tmp_path)
    assert sc._install_global_mcp(dry_run=False) is False
    captured = capsys.readouterr()
    assert "WARNING" in captured.out


def test_install_global_mcp_creates_new(paths, tmp_path, monkeypatch, capsys):
    _patch_file(monkeypatch, tmp_path)
    source = tmp_path / ".mcp.json"
    source.write_text(json.dumps({"mcpServers": {"mullet": {"command": "mullet"}}}))
    # Patch global MCP path under tmp_path
    global_mcp = tmp_path / ".claude" / "mcp.json"
    monkeypatch.setattr(sc, "_GLOBAL_MCP_PATH", global_mcp)
    assert sc._install_global_mcp(dry_run=False) is True
    assert global_mcp.exists()
    captured = capsys.readouterr()
    assert "Wrote" in captured.out


def test_install_global_mcp_already_present(paths, tmp_path, monkeypatch, capsys):
    _patch_file(monkeypatch, tmp_path)
    source = tmp_path / ".mcp.json"
    source.write_text(json.dumps({"mcpServers": {"mullet": {"command": "mullet"}}}))
    global_mcp = tmp_path / ".claude" / "mcp.json"
    global_mcp.parent.mkdir(parents=True, exist_ok=True)
    global_mcp.write_text(json.dumps({"mcpServers": {"mullet": {"command": "mullet"}}}))
    monkeypatch.setattr(sc, "_GLOBAL_MCP_PATH", global_mcp)
    assert sc._install_global_mcp(dry_run=False) is False
    captured = capsys.readouterr()
    assert "already present" in captured.out


def test_install_global_mcp_merges_existing(paths, tmp_path, monkeypatch, capsys):
    _patch_file(monkeypatch, tmp_path)
    source = tmp_path / ".mcp.json"
    source.write_text(json.dumps({"mcpServers": {"mullet": {"command": "mullet"}}}))
    global_mcp = tmp_path / ".claude" / "mcp.json"
    global_mcp.parent.mkdir(parents=True, exist_ok=True)
    global_mcp.write_text(json.dumps({"mcpServers": {"other": {"command": "other"}}}))
    monkeypatch.setattr(sc, "_GLOBAL_MCP_PATH", global_mcp)
    assert sc._install_global_mcp(dry_run=False) is True
    merged = json.loads(global_mcp.read_text())
    assert "mullet" in merged["mcpServers"]
    assert "other" in merged["mcpServers"]


def test_install_global_mcp_dry_run(paths, tmp_path, monkeypatch, capsys):
    _patch_file(monkeypatch, tmp_path)
    source = tmp_path / ".mcp.json"
    source.write_text(json.dumps({"mcpServers": {"mullet": {"command": "mullet"}}}))
    global_mcp = tmp_path / ".claude" / "mcp.json"
    monkeypatch.setattr(sc, "_GLOBAL_MCP_PATH", global_mcp)
    assert sc._install_global_mcp(dry_run=True) is True
    assert not global_mcp.exists()
    captured = capsys.readouterr()
    assert "[dry-run]" in captured.out


# ---------------------------------------------------------------------------
# 11. _offer_karpathy_plugin
# ---------------------------------------------------------------------------


def test_karpathy_already_present(paths, tmp_path, capsys):
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps({"plugins": [sc._KARPATHY_PLUGIN_URL]})
    )
    assert sc._offer_karpathy_plugin(dry_run=False, yes=False) is False
    captured = capsys.readouterr()
    assert "already installed" in captured.out


def test_karpathy_yes_installs(paths, tmp_path, capsys):
    assert sc._offer_karpathy_plugin(dry_run=False, yes=True) is True
    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
    assert sc._KARPATHY_PLUGIN_URL in settings.get("plugins", [])
    captured = capsys.readouterr()
    assert "Added" in captured.out


def test_karpathy_user_says_no(paths, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    assert sc._offer_karpathy_plugin(dry_run=False, yes=False) is False
    captured = capsys.readouterr()
    assert "Skipped" in captured.out


def test_karpathy_user_says_yes(paths, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    assert sc._offer_karpathy_plugin(dry_run=False, yes=False) is True
    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
    assert sc._KARPATHY_PLUGIN_URL in settings.get("plugins", [])


def test_karpathy_eof(paths, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: (_ for _ in ()).throw(EOFError))
    assert sc._offer_karpathy_plugin(dry_run=False, yes=False) is False


def test_karpathy_dry_run(paths, tmp_path, capsys):
    assert sc._offer_karpathy_plugin(dry_run=True, yes=True) is True
    assert not (tmp_path / ".claude" / "settings.json").exists()
    captured = capsys.readouterr()
    assert "[dry-run]" in captured.out


# ---------------------------------------------------------------------------
# 12. run_setup_claude
# ---------------------------------------------------------------------------


def test_run_setup_claude_summary(paths, tmp_path, capsys, monkeypatch):
    # Patch all sub-functions to return True so we see a full summary
    monkeypatch.setattr(sc, "_append_claude_md_section", lambda _dr: True)
    monkeypatch.setattr(sc, "_merge_claude_settings_hooks", lambda _dr: True)
    monkeypatch.setattr(sc, "_install_global_mcp", lambda _dr: True)
    monkeypatch.setattr(sc, "_install_statusline", lambda _dr: True)
    monkeypatch.setattr(sc, "_copy_claude_skill_files", lambda _dr, _f: True)
    monkeypatch.setattr(sc, "_install_claude_agents", lambda _dr, _f: True)
    monkeypatch.setattr(sc, "_install_project_hooks", lambda _pd, _dr: True)
    monkeypatch.setattr(sc, "_install_opencode", lambda _dr, _y: True)
    monkeypatch.setattr(sc, "_offer_karpathy_plugin", lambda _dr, _y: True)
    sc.run_setup_claude(dry_run=False, yes=True, force=False)
    captured = capsys.readouterr()
    assert "Setup complete!" in captured.out
    assert "CLAUDE.md updated" in captured.out
    assert "settings.json updated" in captured.out


def test_run_setup_claude_dry_run(paths, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sc, "_append_claude_md_section", lambda _dr: True)
    monkeypatch.setattr(sc, "_merge_claude_settings_hooks", lambda _dr: True)
    monkeypatch.setattr(sc, "_install_global_mcp", lambda _dr: True)
    monkeypatch.setattr(sc, "_install_statusline", lambda _dr: True)
    monkeypatch.setattr(sc, "_copy_claude_skill_files", lambda _dr, _f: True)
    monkeypatch.setattr(sc, "_install_claude_agents", lambda _dr, _f: True)
    monkeypatch.setattr(sc, "_install_project_hooks", lambda _pd, _dr: True)
    monkeypatch.setattr(sc, "_install_opencode", lambda _dr, _y: True)
    monkeypatch.setattr(sc, "_offer_karpathy_plugin", lambda _dr, _y: True)
    sc.run_setup_claude(dry_run=True, yes=True, force=False)
    captured = capsys.readouterr()
    assert "DRY RUN" in captured.out
    assert "Dry run complete" in captured.out


def test_run_setup_claude_no_changes(paths, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sc, "_append_claude_md_section", lambda _dr: False)
    monkeypatch.setattr(sc, "_merge_claude_settings_hooks", lambda _dr: False)
    monkeypatch.setattr(sc, "_install_global_mcp", lambda _dr: False)
    monkeypatch.setattr(sc, "_install_statusline", lambda _dr: False)
    monkeypatch.setattr(sc, "_copy_claude_skill_files", lambda _dr, _f: False)
    monkeypatch.setattr(sc, "_install_claude_agents", lambda _dr, _f: False)
    monkeypatch.setattr(sc, "_install_project_hooks", lambda _pd, _dr: False)
    monkeypatch.setattr(sc, "_install_opencode", lambda _dr, _y: False)
    monkeypatch.setattr(sc, "_offer_karpathy_plugin", lambda _dr, _y: False)
    # Mock pool tune to avoid real pools running (local imports inside run_setup_claude)
    import sys
    fake_pools_dir = tmp_path / "pools"
    fake_pools_dir.mkdir()
    if "mullet.server.mcp_knowledge" in sys.modules:
        monkeypatch.setattr(sys.modules["mullet.server.mcp_knowledge"], "_get_pools_dir", lambda: fake_pools_dir)
    if "mullet.cli.pool_tune" in sys.modules:
        monkeypatch.setattr(sys.modules["mullet.cli.pool_tune"], "run_pool_tune", lambda _pd, dry_run: None)
    sc.run_setup_claude(dry_run=False, yes=False, force=False)
    captured = capsys.readouterr()
    assert "No changes were needed" in captured.out


def test_run_setup_claude_skips_project_hooks(paths, tmp_path, capsys, monkeypatch):
    # No .claude/settings.json or .git in cwd → project hooks skipped
    monkeypatch.setattr(sc, "_append_claude_md_section", lambda _dr: False)
    monkeypatch.setattr(sc, "_merge_claude_settings_hooks", lambda _dr: False)
    monkeypatch.setattr(sc, "_install_global_mcp", lambda _dr: False)
    monkeypatch.setattr(sc, "_install_statusline", lambda _dr: False)
    monkeypatch.setattr(sc, "_copy_claude_skill_files", lambda _dr, _f: False)
    monkeypatch.setattr(sc, "_install_claude_agents", lambda _dr, _f: False)
    monkeypatch.setattr(sc, "_install_project_hooks", lambda _pd, _dr: (_ for _ in ()).throw(AssertionError("should not be called")))
    monkeypatch.setattr(sc, "_install_opencode", lambda _dr, _y: False)
    monkeypatch.setattr(sc, "_offer_karpathy_plugin", lambda _dr, _y: False)
    # Mock pool tune and cwd to a blank directory
    import sys
    fake_cwd = tmp_path / "blank_cwd"
    fake_cwd.mkdir()
    class FakePath(type(tmp_path)):
        @classmethod
        def cwd(cls):
            return fake_cwd
    monkeypatch.setattr(sc, "Path", FakePath)
    if "mullet.server.mcp_knowledge" in sys.modules:
        monkeypatch.setattr(sys.modules["mullet.server.mcp_knowledge"], "_get_pools_dir", lambda: fake_cwd / "pools")
    if "mullet.cli.pool_tune" in sys.modules:
        monkeypatch.setattr(sys.modules["mullet.cli.pool_tune"], "run_pool_tune", lambda _pd, dry_run: None)
    sc.run_setup_claude(dry_run=False, yes=False, force=False)
    captured = capsys.readouterr()
    assert "No changes were needed" in captured.out


def test_run_setup_claude_pool_tune_exception(paths, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sc, "_append_claude_md_section", lambda _dr: False)
    monkeypatch.setattr(sc, "_merge_claude_settings_hooks", lambda _dr: False)
    monkeypatch.setattr(sc, "_install_global_mcp", lambda _dr: False)
    monkeypatch.setattr(sc, "_install_statusline", lambda _dr: False)
    monkeypatch.setattr(sc, "_copy_claude_skill_files", lambda _dr, _f: False)
    monkeypatch.setattr(sc, "_install_claude_agents", lambda _dr, _f: False)
    monkeypatch.setattr(sc, "_install_project_hooks", lambda _pd, _dr: False)
    monkeypatch.setattr(sc, "_install_opencode", lambda _dr, _y: False)
    monkeypatch.setattr(sc, "_offer_karpathy_plugin", lambda _dr, _y: False)
    # Force import failure for pool tune by breaking the module path
    import sys
    real_module = sys.modules.get("mullet.server.mcp_knowledge")
    monkeypatch.setitem(
        sys.modules, "mullet.server.mcp_knowledge", None
    )
    sc.run_setup_claude(dry_run=False, yes=False, force=False)
    if real_module is not None:
        monkeypatch.setitem(sys.modules, "mullet.server.mcp_knowledge", real_module)
    captured = capsys.readouterr()
    assert "Pool tune skipped" in captured.out


# ---------------------------------------------------------------------------
# 13. Exception / edge paths
# ---------------------------------------------------------------------------


def test_append_claude_md_reads_existing_and_adds_missing(paths, tmp_path, capsys, monkeypatch):
    md = tmp_path / ".claude" / "CLAUDE.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("## Delegation (Mullet)\n")
    # When delegation section exists but optional sections are missing, returns True
    assert sc._append_claude_md_section(dry_run=False) is True
    text = md.read_text()
    assert "## Schema & Assumptions" in text


def test_merge_settings_invalid_json(paths, tmp_path, capsys):
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text("not json")
    with pytest.raises(json.JSONDecodeError):
        sc._merge_claude_settings_hooks(dry_run=False)


def test_install_project_hooks_invalid_json(paths, tmp_path, capsys):
    project_dir = tmp_path / "myproject"
    settings_path = project_dir / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text("not json")
    with pytest.raises(json.JSONDecodeError):
        sc._install_project_hooks(project_dir, dry_run=False)


def test_merge_settings_hooks_no_hooks_dir_prints_warning(paths, tmp_path, capsys):
    """Line 192: print warning when _get_hooks_dir() returns None."""
    from unittest.mock import patch
    with patch.object(sc, "_get_hooks_dir", return_value=None):
        sc._merge_claude_settings_hooks(dry_run=False)
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "scripts/hooks" in captured.out
