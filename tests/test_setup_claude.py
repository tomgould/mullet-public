"""`mullet setup-claude`: what it writes, that it's idempotent, and that --uninstall
removes exactly what it added.

Every test runs with the throwaway HOME from conftest and a fake `claude` on
PATH, so nothing touches the real Claude Code setup.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from mullet.cli import setup_claude as sc
from mullet.hooks import HOOKS

MULLET = "/opt/mullet/bin/mullet"


@pytest.fixture
def env(fake_claude, monkeypatch):
    """A fake claude, a fixed mullet command, and the paths the tests look at."""
    monkeypatch.setattr(sc, "mullet_command", lambda: [MULLET])
    home = Path.home()

    class Env:
        claude_log = fake_claude
        registered = Path(f"{fake_claude}.registered")
        claude_dir = home / ".claude"
        settings = home / ".claude" / "settings.json"
        manifest = home / ".config" / "mullet" / "setup-manifest.json"

        def calls(self) -> list[str]:
            return self.claude_log.read_text().splitlines() if self.claude_log.exists() else []

    return Env()


def _snapshot(root: Path) -> dict[str, bytes | None]:
    return {str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
            for p in sorted(root.rglob("*"))}


def _hook_commands(settings: dict) -> list[str]:
    return [h["command"] for groups in settings.get("hooks", {}).values()
            for g in groups for h in g.get("hooks", [])]


USER_SETTINGS = {
    "model": "opus",
    "permissions": {"allow": ["Bash(git status)"]},
    "statusLine": {"type": "command", "command": "my-statusline"},
    "hooks": {
        "UserPromptSubmit": [{"matcher": "", "hooks": [{"type": "command", "command": "echo mine"}]}],
        "PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "guard.sh"}]}],
    },
}


# --- Fresh install -------------------------------------------------------------


def test_fresh_install_registers_mcp_and_skills_and_adds_no_hooks(env, capsys):
    assert sc.run_setup_claude() == 0

    assert f"mcp add --scope user mullet -- {MULLET} mcp" in env.calls()
    for name in sc.CLAUDE_SKILLS:
        installed = env.claude_dir / "skills" / name / "SKILL.md"
        packaged = Path(sc.__file__).parent.parent / "core" / "skills" / "builtin" / name / "SKILL.md"
        assert installed.read_text() == packaged.read_text()
    assert (env.claude_dir / "skills" / "mullet-job" / "SKILL.md").exists()
    # No hooks by default, so settings.json is never written.
    assert not env.settings.exists()
    assert not (env.claude_dir / "CLAUDE.md").exists()
    assert not (env.claude_dir / "mcp.json").exists()
    manifest = json.loads(env.manifest.read_text())
    assert manifest["mcp"]["command"] == [MULLET]
    assert len(manifest["files"]) == len(sc.CLAUDE_SKILLS)
    assert "undo with: mullet setup-claude --uninstall" in capsys.readouterr().out


def test_hooks_are_opt_in_and_run_through_mullet_hook(env):
    specs = [HOOKS["delegation-reminder"], HOOKS["agent-hud-tracker"]]
    assert sc.run_setup_claude(specs) == 0

    settings = json.loads(env.settings.read_text())
    hooks = settings["hooks"]
    assert [h["command"] for h in hooks["UserPromptSubmit"][0]["hooks"]] == [
        f"{MULLET} hook delegation-reminder"]
    for event in ("PreToolUse", "PostToolUse"):
        (group,) = hooks[event]
        assert group["matcher"] == "Agent"
        assert group["hooks"] == [{"type": "command", "command": f"{MULLET} hook agent-hud-tracker"}]
    assert not any("scripts/hooks" in c for c in _hook_commands(settings))


def test_never_writes_into_the_current_repo(env, tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.chdir(repo)
    sc.run_setup_claude(list(HOOKS.values()), statusline=True)
    assert sorted(p.name for p in repo.iterdir()) == [".git"]


def test_never_touches_claude_md(env):
    env.claude_dir.mkdir()
    (env.claude_dir / "CLAUDE.md").write_text("# my rules\n")
    sc.run_setup_claude(list(HOOKS.values()))
    sc.run_uninstall()
    assert (env.claude_dir / "CLAUDE.md").read_text() == "# my rules\n"


# --- Idempotency ----------------------------------------------------------------


def test_running_again_changes_nothing(env, capsys):
    specs = [HOOKS["skill-activator"]]
    sc.run_setup_claude(specs)
    before = _snapshot(Path.home())
    capsys.readouterr()

    assert sc.run_setup_claude(specs) == 0

    assert _snapshot(Path.home()) == before
    assert "Nothing to change." in capsys.readouterr().out
    assert sum(c.startswith("mcp add") for c in env.calls()) == 1


def test_hooks_are_kept_when_run_again_without_hooks(env):
    sc.run_setup_claude([HOOKS["skill-activator"]])
    sc.run_setup_claude()
    assert f"{MULLET} hook skill-activator" in _hook_commands(json.loads(env.settings.read_text()))


def test_a_hook_removed_by_hand_is_added_back_once(env):
    sc.run_setup_claude([HOOKS["skill-activator"]])
    env.settings.write_text(json.dumps({"hooks": {"UserPromptSubmit": [{"matcher": "", "hooks": []}]}}))
    sc.run_setup_claude()
    sc.run_setup_claude()
    assert _hook_commands(json.loads(env.settings.read_text())) == [f"{MULLET} hook skill-activator"]
    assert len(json.loads(env.manifest.read_text())["settings"]["hooks"]) == 1


def test_commands_follow_mullet_when_it_moves(env, monkeypatch):
    sc.run_setup_claude([HOOKS["skill-activator"]])
    monkeypatch.setattr(sc, "mullet_command", lambda: ["/new/bin/mullet"])

    assert sc.run_setup_claude() == 0

    assert _hook_commands(json.loads(env.settings.read_text())) == ["/new/bin/mullet hook skill-activator"]
    assert "mcp remove --scope user mullet" in env.calls()
    assert env.calls()[-1] == "mcp add --scope user mullet -- /new/bin/mullet mcp"


# --- Uninstall ---------------------------------------------------------------------


def test_uninstall_removes_exactly_what_was_added(env, fake_claude):
    env.claude_dir.mkdir()
    env.settings.write_text(json.dumps(USER_SETTINGS, indent=4))
    user_skill = env.claude_dir / "skills" / "my-skill" / "SKILL.md"
    user_skill.parent.mkdir(parents=True)
    user_skill.write_text("mine")
    before = _snapshot(env.claude_dir)

    assert sc.run_setup_claude(list(HOOKS.values()), statusline=True) == 0
    settings = json.loads(env.settings.read_text())
    assert settings["statusLine"] == USER_SETTINGS["statusLine"], "a user's status line is left alone"
    assert "echo mine" in _hook_commands(settings)

    assert sc.run_uninstall() == 0

    assert json.loads(env.settings.read_text()) == USER_SETTINGS
    after = _snapshot(env.claude_dir)
    after.pop("settings.json")
    before.pop("settings.json")
    assert after == before
    assert not env.manifest.exists()
    assert "mcp remove --scope user mullet" in env.calls()
    assert not env.registered.exists()


def test_uninstall_removes_the_settings_file_and_folders_it_created(env, monkeypatch, tmp_path):
    hud = tmp_path / "bin" / "mullet-hud"
    hud.parent.mkdir()
    hud.write_text("#!/bin/sh\n")
    hud.chmod(0o755)
    monkeypatch.setenv("PATH", f"{hud.parent}{os.pathsep}{os.environ['PATH']}")

    sc.run_setup_claude([HOOKS["preset-prompt"]], statusline=True)
    settings = json.loads(env.settings.read_text())
    assert settings["statusLine"] == {"type": "command", "command": str(hud)}
    assert settings["hooks"]["SessionStart"] == [
        {"hooks": [{"type": "command", "command": f"{MULLET} hook preset-prompt"}]}]

    sc.run_uninstall()

    assert not env.settings.exists()
    assert not (env.claude_dir / "skills").exists()


def test_a_skill_edited_after_install_is_left_on_uninstall(env, capsys):
    sc.run_setup_claude()
    skill = env.claude_dir / "skills" / "mullet-job" / "SKILL.md"
    skill.write_text("my notes")

    assert sc.run_uninstall() == 0

    assert skill.read_text() == "my notes"
    assert "left in place" in capsys.readouterr().out
    assert [f["path"] for f in json.loads(env.manifest.read_text())["files"]] == [str(skill)]


def test_uninstall_with_nothing_recorded(env, capsys):
    assert sc.run_uninstall() == 0
    assert "Nothing to remove" in capsys.readouterr().out
    assert env.calls() == []


# --- Dry run -------------------------------------------------------------------------


def test_dry_run_writes_nothing(env, capsys):
    env.claude_dir.mkdir()
    env.settings.write_text(json.dumps(USER_SETTINGS))
    before = _snapshot(Path.home())

    assert sc.run_setup_claude(list(HOOKS.values()), statusline=True, dry_run=True) == 0

    assert _snapshot(Path.home()) == before
    out = capsys.readouterr().out
    assert "would run: claude mcp add --scope user mullet" in out
    assert "would create ~/.claude/skills/mullet-job/SKILL.md" in out
    assert f"would add hook delegation-reminder for UserPromptSubmit: {MULLET} hook delegation-reminder" in out
    assert not any(c.startswith("mcp add") for c in env.calls())


def test_uninstall_dry_run_writes_nothing(env, capsys):
    sc.run_setup_claude([HOOKS["skill-activator"]])
    before = _snapshot(Path.home())
    capsys.readouterr()

    assert sc.run_uninstall(dry_run=True) == 0

    assert _snapshot(Path.home()) == before
    out = capsys.readouterr().out
    assert "would run: claude mcp remove --scope user mullet" in out
    assert "would remove hook skill-activator" in out
    assert not any(c.startswith("mcp remove") for c in env.calls())


# --- Things that belong to the user ---------------------------------------------------------


def test_an_existing_mcp_registration_is_left_alone(env, capsys):
    env.registered.touch()
    sc.run_setup_claude()
    assert not any(c.startswith("mcp add") for c in env.calls())
    assert "not by setup-claude; left as it is" in capsys.readouterr().out

    sc.run_uninstall()
    assert env.registered.exists()
    assert not any(c.startswith("mcp remove") for c in env.calls())


def test_a_users_skill_with_the_same_name_is_not_replaced(env):
    skill = env.claude_dir / "skills" / "mullet-job" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("mine")
    sc.run_setup_claude(force=True)
    sc.run_uninstall()
    assert skill.read_text() == "mine"


def test_invalid_settings_json_is_not_touched(env, capsys):
    env.claude_dir.mkdir()
    env.settings.write_text("{not json")
    assert sc.run_setup_claude([HOOKS["skill-activator"]]) == 1
    assert env.settings.read_text() == "{not json"
    assert "isn't valid JSON" in capsys.readouterr().err


def test_without_claude_the_rest_still_installs(env, monkeypatch, capsys):
    monkeypatch.setattr(sc.shutil, "which", lambda name: None)
    assert sc.run_setup_claude() == 1
    assert (env.claude_dir / "skills" / "mullet-job" / "SKILL.md").exists()
    err = capsys.readouterr().err
    assert "claude CLI is not on PATH" in err
    assert f"claude mcp add --scope user mullet -- {MULLET} mcp" in err


def test_legacy_checkout_hooks_are_pointed_out(env, capsys):
    env.claude_dir.mkdir()
    env.settings.write_text(json.dumps({"hooks": {"UserPromptSubmit": [{"matcher": "", "hooks": [
        {"type": "command", "command": "python3 /src/Mullet/scripts/hooks/delegation_reminder.py"}]}]}}))
    sc.run_setup_claude()
    assert "run scripts from a Mullet checkout" in capsys.readouterr().out


def test_dry_run_without_claude_warns_and_exits_0(env, monkeypatch, capsys):
    """--dry-run with no claude CLI previews skills and the MCP command, then exits 0."""
    monkeypatch.setattr(sc.shutil, "which", lambda name: None if name == "claude" else "/fake/bin")
    assert sc.run_setup_claude(dry_run=True) == 0
    out = capsys.readouterr()
    assert "claude is not on PATH" in out.err
    assert "would run: claude mcp add --scope user mullet" in out.out
    for name in sc.CLAUDE_SKILLS:
        assert f"would create ~/.claude/skills/{name}/SKILL.md" in out.out
    # Nothing should be written.
    assert not env.registered.exists()
    assert not env.manifest.exists()


def test_real_run_without_claude_still_exits_1(env, monkeypatch, capsys):
    """A non-dry run without claude keeps exiting 1 with the existing message."""
    monkeypatch.setattr(sc.shutil, "which", lambda name: None if name == "claude" else "/fake/bin")
    assert sc.run_setup_claude() == 1
    err = capsys.readouterr().err
    assert "claude CLI is not on PATH" in err
    assert "claude mcp add --scope user mullet" in err
    # Skills still install.
    for name in sc.CLAUDE_SKILLS:
        assert (env.claude_dir / "skills" / name / "SKILL.md").exists()


# --- Command line and the mullet command ---------------------------------------------------


def test_mullet_command_prefers_mullet_on_path(monkeypatch):
    monkeypatch.setattr(sc.shutil, "which", lambda name: "/usr/local/bin/mullet" if name == "mullet" else None)
    assert sc.mullet_command() == ["/usr/local/bin/mullet"]


def test_mullet_command_falls_back_to_this_python(monkeypatch):
    monkeypatch.setattr(sc.shutil, "which", lambda name: None)
    assert sc.mullet_command() == [sys.executable, "-m", "mullet"]


def test_hooks_list_describes_every_hook_and_changes_nothing(env, capsys):
    assert sc.cmd_setup_claude(["--hooks", "list"]) == 0
    out = capsys.readouterr().out
    for name in HOOKS:
        assert name in out
    assert env.calls() == []
    assert not env.manifest.exists()


def test_unknown_hook_is_refused(env, capsys):
    assert sc.cmd_setup_claude(["--hooks", "delegation-reminder,nope"]) == 2
    assert "unknown hook(s): nope" in capsys.readouterr().err
    assert not env.manifest.exists()


def test_cli_hooks_all_and_uninstall(env):
    assert sc.cmd_setup_claude(["--hooks", "all"]) == 0
    commands = _hook_commands(json.loads(env.settings.read_text()))
    assert {c.rsplit(" ", 1)[1] for c in commands} == set(HOOKS)
    assert sc.cmd_setup_claude(["--uninstall"]) == 0
    assert not env.settings.exists()


def test_config_dir_follows_mullet_config_dir(env, tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path / "cfg"))
    sc.run_setup_claude()
    assert (tmp_path / "cfg" / "setup-manifest.json").exists()
    assert not env.manifest.exists()


@pytest.mark.parametrize("argv", [
    ["mullet", "setup-claude", "--dry-run"],
    ["mullet", "setup-claude", "--check"],
    ["mullet", "--verbose", "setup-claude", "--dry-run"],
])
def test_mullet_setup_claude_previews_with_dry_run_or_check(argv, monkeypatch):
    from unittest.mock import patch

    from mullet.cli import main as cli_main

    monkeypatch.setattr("sys.argv", argv)
    with patch("mullet.cli.setup_claude.run_setup_claude", return_value=0) as run, \
            pytest.raises(SystemExit) as exit_info:
        cli_main._main_inner()
    assert exit_info.value.code == 0
    run.assert_called_once_with([], statusline=False, dry_run=True, force=False)
