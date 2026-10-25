"""`mullet hook <name>`: dispatch to the packaged hooks, and failing loudly."""

from __future__ import annotations

import importlib
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from mullet import hooks
from mullet.cli.hook_cli import cmd_hook

ROOT = Path(__file__).resolve().parent.parent
# Events Claude Code accepts hooks for.
CLAUDE_EVENTS = {"PreToolUse", "PostToolUse", "UserPromptSubmit", "SessionStart", "SessionEnd",
                 "Stop", "SubagentStop", "Notification", "PreCompact"}


def _run_mullet(*args: str, stdin: str = "{}", extra_env: dict | None = None):
    env = {**os.environ, "PYTHONPATH": str(ROOT), **(extra_env or {})}
    return subprocess.run([sys.executable, "-m", "mullet", *args], input=stdin,
                          capture_output=True, text=True, env=env, timeout=120)


@pytest.mark.parametrize("spec", list(hooks.HOOKS.values()), ids=list(hooks.HOOKS))
def test_every_hook_loads_and_registers_for_real_events(spec):
    module = importlib.import_module(f"mullet.hooks.{spec.module}")
    assert callable(module.main)
    assert spec.registrations
    for event, matcher in spec.registrations:
        assert event in CLAUDE_EVENTS
        assert matcher is None or isinstance(matcher, str)


@pytest.mark.parametrize("name,expected", [
    ("delegation-reminder", "delegation-reminder"),
    ("delegation_reminder", "delegation-reminder"),
    ("delegation_reminder.py", "delegation-reminder"),
    ("mullet_pretooluse_rewriter", "pretooluse-rewriter"),
    ("mullet_observation_capture.py", "observation-capture"),
    ("user_prompt_detect_override", "opus-override"),
])
def test_find_accepts_old_script_names(name, expected):
    assert hooks.find(name).name == expected


def test_unknown_hook_fails_loudly(capsys):
    assert hooks.run("no-such-hook") == 1
    assert "no hook called 'no-such-hook'" in capsys.readouterr().err


def test_a_hook_that_cannot_be_imported_fails_loudly(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "mullet.hooks.compress_whitespace", None)
    assert hooks.run("compress-whitespace") == 1
    err = capsys.readouterr().err
    assert "can't load mullet.hooks.compress_whitespace" in err
    assert "ModuleNotFoundError" in err


def test_cmd_hook_usage_and_list(capsys):
    assert cmd_hook([]) == 2
    assert "Usage: mullet hook <name>" in capsys.readouterr().err
    assert cmd_hook(["list"]) == 0
    out = capsys.readouterr().out
    for name in hooks.HOOKS:
        assert name in out
    assert cmd_hook(["a", "b"]) == 2


def test_cmd_hook_runs_the_hook(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"prompt": "a   b"})))
    with pytest.raises(SystemExit) as exit_info:
        cmd_hook(["compress-whitespace"])
    assert exit_info.value.code == 0
    assert json.loads(capsys.readouterr().out) == {"prompt": "a b"}


def test_mullet_hook_subcommand_end_to_end():
    result = _run_mullet("hook", "compress-whitespace", stdin=json.dumps({"prompt": "x    y"}))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"prompt": "x y"}


def test_mullet_hook_unknown_name_exits_1_with_a_message():
    result = _run_mullet("hook", "nope")
    assert result.returncode == 1
    assert "no hook called 'nope'" in result.stderr


def test_a_hook_missing_mullet_modules_says_so_instead_of_exiting_quietly():
    # skill-activator imports Mullet's skill registry; make that import fail.
    code = ("import sys; sys.modules['mullet.core.skills.loader'] = None; "
            "from mullet.hooks import run; sys.exit(run('skill-activator'))")
    result = subprocess.run([sys.executable, "-c", code], input=json.dumps({"prompt": "brainstorm it"}),
                            capture_output=True, text=True, timeout=60,
                            env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert result.returncode == 1
    assert "can't import Mullet's skill registry" in result.stderr


@pytest.mark.parametrize("shim", sorted((ROOT / "scripts" / "hooks").glob("*.py")),
                         ids=lambda p: p.name)
def test_checkout_shims_name_real_hooks(shim):
    match = re.search(r'run\("([a-z-]+)"\)', shim.read_text())
    assert match, f"{shim.name} doesn't call mullet.hooks.run"
    assert hooks.find(match.group(1)) is not None
    assert hooks.find(shim.name) is not None, "the shim's file name should still find its hook"


def test_checkout_shim_runs_the_package_hook():
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "hooks" / "compress_whitespace.py")],
                            input=json.dumps({"prompt": "p    q"}), capture_output=True, text=True,
                            timeout=60, cwd="/")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"prompt": "p q"}
