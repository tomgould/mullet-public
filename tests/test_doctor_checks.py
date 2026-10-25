"""`mullet doctor`'s toolchain and Claude Code setup checks.

All in-process, with the throwaway HOME from conftest and a fake `claude`, so
they are quick and never run the real Claude Code or reach the network.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from mullet.cli import doctor_checks as dc
from mullet.cli import setup_claude
from mullet.core import env
from mullet.hooks import HOOKS
from mullet.jobs import machine, sandbox


def _row(rows, name):
    matches = [r for r in rows if r[0] == name]
    assert matches, f"no {name!r} row in {rows}"
    return matches[0]


# --- Toolchain ------------------------------------------------------------------


def test_python_version_ok_here_and_warns_when_too_old(monkeypatch):
    assert dc.python_check()[2] == "ok"
    monkeypatch.setattr(dc, "MIN_PYTHON", (99, 0))
    name, detail, status = dc.python_check()
    assert status == "warn" and "needs 99.0 or newer" in detail


def test_claude_cli_found_reports_its_version(fake_claude):
    assert dc.claude_check() == ("Claude Code CLI", "9.9.9 (Claude Code)", "ok")
    assert fake_claude.read_text().splitlines() == ["--version"]


def test_claude_cli_missing_warns(monkeypatch):
    monkeypatch.setattr(dc.shutil, "which", lambda name: None)
    name, detail, status = dc.claude_check()
    assert status == "warn" and "claude not on PATH" in detail


def test_claude_cli_that_fails_warns(tmp_path, monkeypatch):
    broken = tmp_path / "claude"
    broken.write_text("#!/bin/sh\nexit 3\n")
    broken.chmod(0o755)
    monkeypatch.setattr(dc.shutil, "which", lambda name: str(broken) if name == "claude" else None)
    assert dc.claude_check()[2] == "warn"


def test_git_check():
    name, detail, status = dc.git_check()
    assert status == "ok" and detail and "git version" not in detail


def test_linux_sandbox_rows_when_everything_is_there(monkeypatch):
    monkeypatch.setattr(dc.sys, "platform", "linux")
    monkeypatch.setattr(dc.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(dc, "_version", lambda argv: "bubblewrap 0.9.0")
    monkeypatch.setattr(sandbox, "check", lambda: sandbox.SandboxCheck(True, "bwrap"))
    rows = dc.sandbox_checks()
    assert _row(rows, "Bubblewrap") == ("Bubblewrap", "bubblewrap 0.9.0", "ok")
    assert _row(rows, "Job sandbox")[2] == "ok"
    assert _row(rows, "socat")[2] == "ok"


def test_linux_sandbox_rows_warn_without_bwrap_or_socat(monkeypatch):
    monkeypatch.setattr(dc.sys, "platform", "linux")
    monkeypatch.setattr(dc.shutil, "which", lambda name: None)
    monkeypatch.setattr(sandbox, "check",
                        lambda: sandbox.SandboxCheck(False, "bwrap", "bubblewrap (bwrap) is not installed"))
    rows = dc.sandbox_checks()
    assert _row(rows, "Bubblewrap")[2] == "warn"
    assert _row(rows, "socat")[2] == "warn"
    assert "apt install socat" in _row(rows, "socat")[1]


def test_linux_sandbox_self_test_failure_warns(monkeypatch):
    monkeypatch.setattr(dc.sys, "platform", "linux")
    monkeypatch.setattr(dc.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(dc, "_version", lambda argv: "bubblewrap 0.9.0")
    monkeypatch.setattr(sandbox, "check",
                        lambda: sandbox.SandboxCheck(False, "bwrap", "no user namespaces"))
    assert _row(dc.sandbox_checks(), "Job sandbox") == (
        "Job sandbox", "Self-test failed: no user namespaces", "warn")


def test_linux_sandbox_self_test_failure_hints_for_container_restriction(monkeypatch):
    monkeypatch.setattr(dc.sys, "platform", "linux")
    monkeypatch.setattr(dc.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(dc, "_version", lambda argv: "bubblewrap 0.9.0")
    for reason in (
        "bubblewrap can't create a sandbox: bwrap: Can't mount proc on /proc",
        "bubblewrap can't create a sandbox: mount: Operation not permitted",
    ):
        monkeypatch.setattr(sandbox, "check",
                            lambda reason=reason: sandbox.SandboxCheck(False, "bwrap", reason))
        name, detail, status = _row(dc.sandbox_checks(), "Job sandbox")
        assert status == "warn"
        assert detail.startswith("Self-test failed:")
        assert "--security-opt seccomp=unconfined" in detail
        assert "--privileged" in detail


def test_linux_sandbox_self_test_failure_does_not_hint_for_other_errors(monkeypatch):
    monkeypatch.setattr(dc.sys, "platform", "linux")
    monkeypatch.setattr(dc.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(dc, "_version", lambda argv: "bubblewrap 0.9.0")
    monkeypatch.setattr(sandbox, "check",
                        lambda: sandbox.SandboxCheck(False, "bwrap", "no user namespaces"))
    name, detail, status = _row(dc.sandbox_checks(), "Job sandbox")
    assert status == "warn"
    assert "--security-opt" not in detail
    assert "--privileged" not in detail


def test_macos_reports_the_sandbox_without_bwrap_or_socat(monkeypatch):
    monkeypatch.setattr(dc.sys, "platform", "darwin")
    monkeypatch.setattr(sandbox, "check",
                        lambda: sandbox.SandboxCheck(False, "none", "no sandbox backend for darwin yet"))
    assert dc.sandbox_checks() == [("Job sandbox", "Unavailable: no sandbox backend for darwin yet", "info")]


def test_anthropic_key_warns_only_when_set(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert dc.anthropic_key_check() is None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    name, detail, status = dc.anthropic_key_check()
    assert status == "warn"
    assert "claude -p" in detail and "subscription" in detail
    assert "sk-ant-test" not in detail


# --- The GPU --------------------------------------------------------------------------

_GPU = "NVIDIA GeForce RTX 4070 Laptop GPU, {free}, 12282, {temp}, 23, {power}\n"
_APPS = ("21145, /usr/bin/sunshine, 186\n"
         "1954453, /home/u/.cache/lm-studio/backends/llama-server, 6752\n")


@pytest.fixture
def gpu(monkeypatch):
    """Fake nvidia-smi answers; set gpu["free"], gpu["temp"], gpu["apps"] and gpu["power"].

    gpu["calls"] lists the timeout of each call.
    """
    answers = {"free": 8000, "temp": 60, "apps": "", "power": "100.00, 100.00, 150.00, 5.00, 175.00",
               "calls": []}

    def nvidia_smi(args, timeout=None):
        answers["calls"].append(timeout)
        if args[0] == f"--query-gpu={machine.GPU_QUERY}":
            return _GPU.format(**answers)
        if args[0] == f"--query-compute-apps={machine.APPS_QUERY}":
            return answers["apps"]
        raise AssertionError(f"unexpected nvidia-smi call {args}")

    for name in ("MULLET_GPU_MIN_FREE_MB", "MULLET_GPU_MAX_TEMP_C"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(env, "dotenv_files", lambda: [])
    monkeypatch.setattr(machine, "_nvidia_smi", nvidia_smi)
    return answers


def test_no_gpu_is_one_info_row():
    # conftest's nvidia-smi stand-in fails, as it does on a machine without an NVIDIA GPU.
    assert dc.gpu_checks() == [("GPU", "No NVIDIA GPU found (nvidia-smi missing or not answering): "
                                "GPU checks are skipped", "info")]


def test_a_quiet_gpu_is_ok(gpu):
    rows = dc.gpu_checks()
    assert rows == [
        ("GPU", "NVIDIA GeForce RTX 4070 Laptop GPU: 8000 of 12282 MiB free, 60°C, 23% busy", "ok"),
        ("GPU lease", "Free", "ok"),
        ("GPU power limit", "100 W (default 150 W)", "ok"),
    ]
    assert gpu["calls"] == [3.0, 3.0], "two nvidia-smi calls, 3 s each"


def test_a_busy_gpu_shows_the_lease_holder_lm_studio_and_a_power_suggestion(gpu):
    gpu.update(free=514, apps=_APPS, power="[N/A], 150.00, 150.00, 5.00, 175.00")
    with machine.gpu_lease("reindex of pool docs"):
        rows = dc.gpu_checks()
    assert _row(rows, "GPU") == ("GPU", "NVIDIA GeForce RTX 4070 Laptop GPU: 514 of 12282 MiB free, 60°C, "
                                 "23% busy (GPU work waits for 2048 MiB)", "info")
    name, detail, status = _row(rows, "GPU lease")
    assert status == "info"
    assert detail.startswith(f"Held by reindex of pool docs (pid {os.getpid()}, since ")
    assert _row(rows, "LM Studio on GPU") == ("LM Studio on GPU", "llama-server (pid 1954453, 6752 MiB)", "info")
    assert _row(rows, "GPU power limit") == (
        "GPU power limit", "150 W, the default. For all-day local use try: sudo nvidia-smi -pl 100 "
        "(lasts until reboot)", "info")


def test_a_hot_gpu_warns(gpu, monkeypatch):
    gpu.update(temp=88)
    monkeypatch.setenv("MULLET_GPU_MAX_TEMP_C", "85")
    name, detail, status = _row(dc.gpu_checks(), "GPU")
    assert status == "warn" and detail.endswith("(over 85°C, so GPU work waits)")


def test_the_power_suggestion_stays_within_the_gpus_range(gpu):
    gpu.update(power="150.00, 150.00, 150.00, 120.00, 175.00")
    assert "sudo nvidia-smi -pl 120 " in _row(dc.gpu_checks(), "GPU power limit")[1]


def test_no_power_row_when_nvidia_smi_reports_no_limits(gpu):
    gpu.update(power="[N/A], [N/A], [N/A], [N/A], [N/A]")
    assert [r[0] for r in dc.gpu_checks()] == ["GPU", "GPU lease"]


# --- Claude Code setup ----------------------------------------------------------------


def test_setup_not_done_is_info():
    assert dc.claude_setup_checks() == [("Claude Code setup", "Not set up: run mullet setup-claude", "info")]


def test_setup_done_is_summarised(fake_claude, monkeypatch):
    monkeypatch.setattr(setup_claude, "mullet_command", lambda: ["/bin/sh"])
    setup_claude.run_setup_claude([HOOKS["skill-activator"]])
    (row,) = dc.claude_setup_checks()
    assert row[2] == "ok"
    assert "MCP server registered" in row[1] and "hooks: skill-activator" in row[1]


def test_setup_with_a_vanished_mullet_warns(fake_claude, monkeypatch):
    monkeypatch.setattr(setup_claude, "mullet_command", lambda: ["/nowhere/mullet"])
    setup_claude.run_setup_claude()
    rows = dc.claude_setup_checks()
    assert any(r[2] == "warn" and "/nowhere/mullet no longer exists" in r[1] for r in rows)


def test_legacy_checkout_hooks_warn():
    settings = Path.home() / ".claude" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"hooks": {"PreToolUse": [{"matcher": "mcp__mullet__.*", "hooks": [
        {"type": "command", "command": "python3 /src/mullet/scripts/hooks/mullet_pretooluse_rewriter.py"}]}]}}))
    row = _row(dc.claude_setup_checks(), "Claude hooks")
    assert row[2] == "warn" and "1 hook(s)" in row[1]


def test_unreadable_settings_warn():
    settings = Path.home() / ".claude" / "settings.json"
    settings.parent.mkdir()
    settings.write_text("{nope")
    assert _row(dc.claude_setup_checks(), "Claude settings")[2] == "warn"


# --- Paths ---------------------------------------------------------------------------


def test_config_and_database_paths_prefer_xdg_and_fall_back_to_an_older_install(monkeypatch):
    home = Path.home()
    assert dc.config_file("config.yaml") == home / ".config" / "mullet" / "config.yaml"
    assert dc.database_path() == home / ".local" / "share" / "mullet" / "mullet.db"
    (home / ".mullet").mkdir()
    (home / ".mullet" / "mullet.db").write_text("")
    assert dc.database_path() == home / ".mullet" / "mullet.db"
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(home / "cfg"))
    assert dc.config_file("config.yaml") == home / "cfg" / "config.yaml"


# --- The whole doctor -------------------------------------------------------------------


@pytest.fixture
def quiet_doctor(monkeypatch, fake_claude):
    """Doctor with no local LLM, no GPU, and the update check recorded rather than run."""
    from mullet.cli import main as cli_main
    import mullet.cli.update_checker as uc

    calls = []
    monkeypatch.setattr(cli_main, "detect_local_provider", lambda: None)
    monkeypatch.setattr(machine, "_nvidia_smi", lambda args, timeout=None: None)
    monkeypatch.setattr(uc, "check_for_update", lambda timeout=0: calls.append(timeout) or {
        "update_available": False})
    return cli_main, calls


def test_doctor_is_fast_offline_and_has_the_new_checks(quiet_doctor, capsys):
    cli_main, update_calls = quiet_doctor
    started = time.monotonic()
    cli_main._run_doctor()
    elapsed = time.monotonic() - started
    out = capsys.readouterr().out
    for row in ("Python", "Claude Code CLI", "Git", "GPU", "Claude Code setup", "checks passed"):
        assert row in out
    for gone in ("Version drift", "kimi-docs", "Claude CLAUDE.md"):
        assert gone not in out
    assert update_calls == [], "the update check needs the network, so only --full runs it"
    assert elapsed < 20


def test_doctor_full_checks_for_updates(quiet_doctor, capsys):
    cli_main, update_calls = quiet_doctor
    cli_main._run_doctor(full=True)
    assert update_calls == [3.0]


def test_doctor_warns_about_anthropic_api_key(quiet_doctor, monkeypatch, capsys):
    cli_main, _ = quiet_doctor
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    cli_main._run_doctor()
    line = next(ln for ln in capsys.readouterr().out.splitlines() if "ANTHROPIC_API_KEY" in ln)
    assert "!" in line and "sk-ant-test" not in line
