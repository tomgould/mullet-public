"""Tests for ``mullet local`` (mullet/cli/local_cli.py).

The watchdog functions, shutil.which and subprocess.run are all faked: no
network, no real systemctl, no real LM Studio.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from mullet.cli import local_cli
from mullet.local import watchdog
from mullet.local.watchdog import Probe, RestartResult, Verdict


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Keep any stray state or log writes inside tmp_path, and fake the defaults."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(watchdog, "_default_base_url", lambda: "http://localhost:1234/v1")
    monkeypatch.setattr(local_cli.shutil, "which", lambda name: None)

    def no_run(*args, **kwargs):
        raise AssertionError(f"unexpected subprocess.run: {args}")

    monkeypatch.setattr(local_cli.subprocess, "run", no_run)
    monkeypatch.setattr(watchdog, "probe", lambda *a, **k: Probe(reachable=False, error="fake"))
    monkeypatch.setattr(watchdog, "last_restart", lambda *a, **k: None)

    def no_restart(**kwargs):
        raise AssertionError("restart should not be called")

    monkeypatch.setattr(watchdog, "restart", no_restart)


def _probe_ok() -> Probe:
    return Probe(
        reachable=True,
        latency_s=0.12,
        models=[("qwen3-coder", "loaded"), ("gemma-3", "not-loaded")],
    )


# ---- status -----------------------------------------------------------------


def test_status_without_nvidia_smi(monkeypatch, capsys):
    monkeypatch.setattr(watchdog, "probe", lambda base, **k: _probe_ok())

    assert local_cli.cmd_local(["status"]) == 0
    out = capsys.readouterr().out
    assert "Server: http://localhost:1234/v1" in out
    assert "Reachable: yes" in out
    assert "Latency: 0.12 s" in out
    assert "qwen3-coder: loaded" in out
    assert "gemma-3: not-loaded" in out
    assert "GPU" not in out
    assert "Last restart: none" in out


def test_status_with_nvidia_smi(monkeypatch, capsys):
    monkeypatch.setattr(watchdog, "probe", lambda base, **k: _probe_ok())
    monkeypatch.setattr(local_cli.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    seen: list[list[str]] = []

    def fake_run(argv, **kwargs):
        seen.append(argv)
        return SimpleNamespace(returncode=0, stdout="20480, 24576\n")

    monkeypatch.setattr(local_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(
        watchdog, "last_restart",
        lambda *a, **k: {"time": 1_700_000_000.0, "reason": "not answering", "ok": True},
    )

    assert local_cli.cmd_local(["status"]) == 0
    out = capsys.readouterr().out
    assert seen == [["/usr/bin/nvidia-smi", *local_cli.NVIDIA_SMI_QUERY]]
    assert "GPU memory: 20480 / 24576 MiB" in out
    assert "Last restart:" in out
    assert "not answering (ok)" in out


def test_status_nvidia_smi_failing_is_skipped_quietly(monkeypatch, capsys):
    monkeypatch.setattr(local_cli.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def failing_run(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 10)

    monkeypatch.setattr(local_cli.subprocess, "run", failing_run)

    assert local_cli.cmd_local(["status"]) == 0
    out = capsys.readouterr().out
    assert "Reachable: no" in out
    assert "GPU" not in out


def test_gpu_memory_bad_exit_code_is_none():
    def run(argv, **kwargs):
        return SimpleNamespace(returncode=9, stdout="")

    assert local_cli.gpu_memory(which=lambda n: "/x/nvidia-smi", run=run) is None


# ---- check ------------------------------------------------------------------


def _verdict(action: str, reason: str = "answering") -> Verdict:
    return Verdict(action=action, reason=reason, probe=_probe_ok())


@pytest.mark.parametrize(
    "action, code",
    [("ok", 0), ("restart", 1), ("busy", 2)],
)
def test_check_exit_codes_without_fix(monkeypatch, capsys, action, code):
    monkeypatch.setattr(watchdog, "check", lambda *a, **k: _verdict(action, "not answering"))

    assert local_cli.cmd_local(["check"]) == code
    assert f"Verdict: {action}" in capsys.readouterr().out


def test_check_fix_restarts_with_verdict_reason(monkeypatch, capsys):
    monkeypatch.setattr(watchdog, "check", lambda *a, **k: _verdict("restart", "stuck loading m1"))
    calls: list[dict] = []

    def fake_restart(**kwargs):
        calls.append(kwargs)
        return RestartResult(ok=True, steps=[["systemctl", "--user", "restart", "lmstudio.service"]])

    monkeypatch.setattr(watchdog, "restart", fake_restart)

    assert local_cli.cmd_local(["check", "--fix"]) == 0
    assert calls == [{"reason": "stuck loading m1"}]
    out = capsys.readouterr().out
    assert "systemctl --user restart lmstudio.service" in out


def test_check_fix_failed_restart_exits_1(monkeypatch):
    monkeypatch.setattr(watchdog, "check", lambda *a, **k: _verdict("restart", "not answering"))
    monkeypatch.setattr(
        watchdog, "restart", lambda **k: RestartResult(ok=False, steps=[], error="boom"),
    )

    assert local_cli.cmd_local(["check", "--fix"]) == 1


@pytest.mark.parametrize("action, code", [("ok", 0), ("busy", 2)])
def test_check_fix_does_not_restart_unless_verdict_says_so(monkeypatch, action, code):
    # The autouse fixture makes restart() fail the test if it is called.
    monkeypatch.setattr(watchdog, "check", lambda *a, **k: _verdict(action))

    assert local_cli.cmd_local(["check", "--fix"]) == code


# ---- restart ----------------------------------------------------------------


@pytest.mark.parametrize("argv, hard", [(["restart"], False), (["restart", "--hard"], True)])
def test_restart_passes_hard(monkeypatch, capsys, argv, hard):
    calls: list[dict] = []

    def fake_restart(**kwargs):
        calls.append(kwargs)
        return RestartResult(ok=True, steps=[["lms", "server", "stop"]])

    monkeypatch.setattr(watchdog, "restart", fake_restart)

    assert local_cli.cmd_local(argv) == 0
    assert calls == [{"hard": hard, "reason": "manual"}]
    assert "lms server stop" in capsys.readouterr().out


def test_restart_failure_exits_1(monkeypatch, capsys):
    monkeypatch.setattr(
        watchdog, "restart", lambda **k: RestartResult(ok=False, steps=[], error="no lms"),
    )

    assert local_cli.cmd_local(["restart"]) == 1
    assert "Restart failed: no lms" in capsys.readouterr().out


def test_no_subcommand_is_an_error(capsys):
    assert local_cli.cmd_local([]) == 2


# ---- watchdog ---------------------------------------------------------------


def test_watchdog_requires_install_or_uninstall():
    assert local_cli.cmd_local(["watchdog"]) == 2


@pytest.mark.parametrize("flag", ["--install", "--uninstall"])
def test_watchdog_with_systemd_unavailable_prints_crontab(monkeypatch, tmp_path, capsys, flag):
    unit_dir = tmp_path / "systemd"
    monkeypatch.setenv("MULLET_SYSTEMD_USER_DIR", str(unit_dir))
    mullet_exe = tmp_path / "mullet"
    mullet_exe.write_text("")
    monkeypatch.setattr(sys, "argv", [str(mullet_exe)])

    assert local_cli.cmd_local(["watchdog", flag]) == 0
    out = capsys.readouterr().out
    assert f"*/2 * * * * {mullet_exe.resolve()} local check --fix" in out
    assert not unit_dir.exists()


def test_watchdog_install_writes_service_and_timer(monkeypatch, tmp_path, capsys):
    unit_dir = tmp_path / "systemd"
    monkeypatch.setenv("MULLET_SYSTEMD_USER_DIR", str(unit_dir))
    mullet_exe = tmp_path / "mullet"
    mullet_exe.write_text("")
    monkeypatch.setattr(sys, "argv", [str(mullet_exe)])
    monkeypatch.setattr(
        local_cli.shutil, "which", lambda name: "/usr/bin/systemctl" if name == "systemctl" else None,
    )
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(local_cli.subprocess, "run", fake_run)

    assert local_cli.cmd_local(["watchdog", "--install"]) == 0
    service = (unit_dir / "mullet-local-watchdog.service").read_text()
    timer = (unit_dir / "mullet-local-watchdog.timer").read_text()
    assert "Type=oneshot" in service
    assert f"ExecStart={mullet_exe.resolve()} local check --fix" in service
    assert "Nice=10" in service
    assert "OnBootSec=2min" in timer
    assert "OnUnitActiveSec=2min" in timer
    assert "Unit=mullet-local-watchdog.service" in timer
    assert "WantedBy=timers.target" in timer
    assert ["systemctl", "--user", "daemon-reload"] in calls
    assert ["systemctl", "--user", "enable", "--now", "mullet-local-watchdog.timer"] in calls


def test_watchdog_uninstall_removes_files_and_disables(monkeypatch, tmp_path, capsys):
    unit_dir = tmp_path / "systemd"
    unit_dir.mkdir()
    (unit_dir / "mullet-local-watchdog.service").write_text("service")
    (unit_dir / "mullet-local-watchdog.timer").write_text("timer")
    monkeypatch.setenv("MULLET_SYSTEMD_USER_DIR", str(unit_dir))
    monkeypatch.setattr(
        local_cli.shutil, "which", lambda name: "/usr/bin/systemctl" if name == "systemctl" else None,
    )
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(local_cli.subprocess, "run", fake_run)

    assert local_cli.cmd_local(["watchdog", "--uninstall"]) == 0
    assert not (unit_dir / "mullet-local-watchdog.service").exists()
    assert not (unit_dir / "mullet-local-watchdog.timer").exists()
    assert ["systemctl", "--user", "disable", "--now", "mullet-local-watchdog.timer"] in calls
    assert ["systemctl", "--user", "daemon-reload"] in calls


# ---- dispatch from main -----------------------------------------------------


def test_main_dispatches_local(monkeypatch):
    from mullet.cli import main as main_mod

    seen: list[list[str]] = []

    def fake_cmd_local(argv):
        seen.append(argv)
        return 2

    monkeypatch.setattr(local_cli, "cmd_local", fake_cmd_local)
    monkeypatch.setattr(sys, "argv", ["mullet", "local", "check", "--fix"])

    with pytest.raises(SystemExit) as exc:
        main_mod._main_inner()
    assert exc.value.code == 2
    assert seen == [["check", "--fix"]]
