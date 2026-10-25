"""CLI dispatcher for ``mullet local``: keep the local LLM server usable.

Wired into ``mullet/cli/main.py`` through the same ``sys.argv[1]`` early
branch as ``mullet job``. The logic lives in ``mullet.local.watchdog``; this
module only parses arguments and prints. The watchdog functions are looked up
on the module at call time, so tests can monkeypatch them.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import shutil
import subprocess
import sys
from pathlib import Path

from mullet.local import watchdog

NVIDIA_SMI_QUERY = [
    "--query-gpu=memory.used,memory.total",
    "--format=csv,noheader,nounits",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mullet local",
        description="Check and restart the local LLM server (LM Studio by default).",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    sub.add_parser("status", help="show the server, its models, GPU memory and the last restart")

    p_restart = sub.add_parser("restart", help="restart the local LLM server now")
    p_restart.add_argument("--hard", action="store_true",
                           help="also take the lms daemon down and up again first")

    p_check = sub.add_parser("check", help="decide whether the server needs a restart")
    p_check.add_argument("--fix", action="store_true",
                         help="restart the server when the check says so")

    p_watchdog = sub.add_parser("watchdog", help="install or uninstall the systemd user timer")
    g = p_watchdog.add_mutually_exclusive_group(required=True)
    g.add_argument("--install", action="store_true", help="install the watchdog timer")
    g.add_argument("--uninstall", action="store_true", help="remove the watchdog timer")
    return parser


def gpu_memory(
    *,
    which=None,
    run=None,
) -> list[tuple[int, int]] | None:
    """Return (used MiB, total MiB) for each GPU, or None when nvidia-smi is absent or fails."""
    which = which or shutil.which
    run = run or subprocess.run
    exe = which("nvidia-smi")
    if not exe:
        return None
    try:
        proc = run([exe, *NVIDIA_SMI_QUERY], capture_output=True, text=True, timeout=10, check=False)
    except Exception:  # noqa: BLE001
        return None
    if getattr(proc, "returncode", 1) != 0:
        return None
    gpus: list[tuple[int, int]] = []
    for line in str(getattr(proc, "stdout", "") or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            gpus.append((int(float(parts[0])), int(float(parts[1]))))
        except ValueError:
            continue
    return gpus or None


def _systemd_user_dir(environ: dict[str, str] | None = None) -> Path:
    """Return the systemd user unit directory."""
    environ = environ or os.environ
    if "MULLET_SYSTEMD_USER_DIR" in environ:
        return Path(environ["MULLET_SYSTEMD_USER_DIR"])
    if "XDG_CONFIG_HOME" in environ:
        return Path(environ["XDG_CONFIG_HOME"]) / "systemd" / "user"
    return Path.home() / ".config" / "systemd" / "user"


def _mullet_executable(argv0: str | None = None, which=None) -> str:
    """Return the absolute path of the mullet executable for timer units."""
    argv0 = argv0 if argv0 is not None else sys.argv[0]
    which = which or shutil.which
    path = Path(argv0)
    if path.exists():
        return str(path.resolve())
    found = which("mullet")
    if found:
        return str(Path(found).resolve())
    return str(path.resolve())


def _systemd_available(run=None, which=None) -> bool:
    """True when ``systemctl --user`` can be used."""
    run = run or subprocess.run
    which = which or shutil.which
    if which("systemctl") is None:
        return False
    try:
        result = run(
            ["systemctl", "--user", "show-environment"],
            timeout=10,
            check=False,
            capture_output=True,
        )
    except Exception:  # noqa: BLE001
        return False
    return getattr(result, "returncode", -1) == 0


def _install_watchdog(unit_dir: Path, mullet: str, run) -> int:
    unit_dir.mkdir(parents=True, exist_ok=True)
    service_path = unit_dir / "mullet-local-watchdog.service"
    timer_path = unit_dir / "mullet-local-watchdog.timer"
    service_path.write_text(
        "[Unit]\n"
        "Description=Mullet local LLM watchdog\n\n"
        "[Service]\n"
        "Type=oneshot\n"
        f"ExecStart={mullet} local check --fix\n"
        "Nice=10\n",
        encoding="utf-8",
    )
    timer_path.write_text(
        "[Unit]\n"
        "Description=Run Mullet local LLM watchdog every 2 minutes\n\n"
        "[Timer]\n"
        "OnBootSec=2min\n"
        "OnUnitActiveSec=2min\n"
        "Unit=mullet-local-watchdog.service\n\n"
        "[Install]\n"
        "WantedBy=timers.target\n",
        encoding="utf-8",
    )
    run(["systemctl", "--user", "daemon-reload"], timeout=30, check=False, capture_output=True)
    result = run(
        ["systemctl", "--user", "enable", "--now", "mullet-local-watchdog.timer"],
        timeout=30,
        check=False,
        capture_output=True,
    )
    return 0 if getattr(result, "returncode", -1) == 0 else 1


def _uninstall_watchdog(unit_dir: Path, run) -> int:
    try:
        run(
            ["systemctl", "--user", "disable", "--now", "mullet-local-watchdog.timer"],
            timeout=30,
            check=False,
            capture_output=True,
        )
    except Exception:  # noqa: BLE001
        pass
    (unit_dir / "mullet-local-watchdog.service").unlink(missing_ok=True)
    (unit_dir / "mullet-local-watchdog.timer").unlink(missing_ok=True)
    run(["systemctl", "--user", "daemon-reload"], timeout=30, check=False, capture_output=True)
    return 0


def _cmd_watchdog(
    install: bool,
    uninstall: bool,
    *,
    run=None,
    environ: dict[str, str] | None = None,
    argv0: str | None = None,
) -> int:
    run = run or subprocess.run
    unit_dir = _systemd_user_dir(environ)
    mullet = _mullet_executable(argv0)
    if not _systemd_available(run):
        print(f"*/2 * * * * {mullet} local check --fix")
        return 0
    if install:
        return _install_watchdog(unit_dir, mullet, run)
    return _uninstall_watchdog(unit_dir, run)


def _format_time(when: object) -> str:
    if not isinstance(when, (int, float)):
        return "unknown time"
    return _dt.datetime.fromtimestamp(when).strftime("%Y-%m-%d %H:%M:%S")


def _print_probe(result: watchdog.Probe) -> None:
    print(f"Reachable: {'yes' if result.reachable else 'no'}")
    if result.latency_s is not None:
        print(f"Latency: {result.latency_s:.2f} s")
    if result.error and not result.reachable:
        print(f"Error: {result.error}")
    if result.reachable:
        if result.models:
            print("Models:")
            for model_id, state in result.models:
                print(f"  {model_id}: {state}")
        else:
            print("Models: none")


def _cmd_status() -> int:
    base = watchdog._default_base_url()
    print(f"Server: {base}")
    _print_probe(watchdog.probe(base))

    gpus = gpu_memory()
    if gpus:
        for i, (used, total) in enumerate(gpus):
            label = "GPU memory" if len(gpus) == 1 else f"GPU {i} memory"
            print(f"{label}: {used} / {total} MiB")

    last = watchdog.last_restart()
    if last is None:
        print("Last restart: none")
    else:
        outcome = "ok" if last.get("ok") else "failed"
        print(f"Last restart: {_format_time(last.get('time'))}, {last.get('reason', 'unknown')} ({outcome})")
    return 0


def _print_restart(result: watchdog.RestartResult) -> None:
    for argv in result.steps:
        print(f"  ran: {' '.join(argv)}")
    if result.ok:
        print("Local LLM is answering again.")
    else:
        print(f"Restart failed: {result.error or 'unknown error'}")


def _cmd_restart(hard: bool) -> int:
    print("Restarting the local LLM server" + (" (hard)" if hard else "") + "...")
    result = watchdog.restart(hard=hard, reason="manual")
    _print_restart(result)
    return 0 if result.ok else 1


def _cmd_check(fix: bool) -> int:
    verdict = watchdog.check()
    print(f"Verdict: {verdict.action} ({verdict.reason})")
    if verdict.action == "ok":
        return 0
    if verdict.action == "busy":
        print("Restarted under 5 minutes ago, so not restarting again yet.")
        return 2
    if not fix:
        print("Run `mullet local check --fix` to restart it.")
        return 1
    print("Restarting the local LLM server...")
    result = watchdog.restart(reason=verdict.reason)
    _print_restart(result)
    return 0 if result.ok else 1


def cmd_local(argv: list[str]) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    if args.command == "status":
        return _cmd_status()
    if args.command == "restart":
        return _cmd_restart(args.hard)
    if args.command == "check":
        return _cmd_check(args.fix)
    if args.command == "watchdog":
        return _cmd_watchdog(args.install, args.uninstall)
    parser.print_help()
    return 1
