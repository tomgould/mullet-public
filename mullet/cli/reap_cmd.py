"""CLI for ``mullet reap``: find and stop idle or orphaned Mullet work.

Wired into ``mullet.cli.main`` through the same ``sys.argv[1]`` early branch
as ``mullet migrate-state``. The safety core lives in ``mullet.jobs.reaper``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mullet.jobs.ledger import Ledger
from mullet.jobs import reaper


_DESCRIPTION = (
    "Find and stop idle or orphaned Mullet work: pytest-xdist workers, "
    "opencode runs of finished jobs, idle LM Studio models, mullet mcp servers, "
    "and stale /tmp/mullet-* directories."
)


class _LmStudioHttp:
    """Minimal HTTP client for the reaper's LM Studio helpers."""

    def get(self, url: str) -> dict[str, Any]:
        req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())

    def post(self, url: str, data: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())


def _default_config_home() -> Path:
    """The user's systemd configuration directory, respecting XDG_CONFIG_HOME."""
    override = os.environ.get("XDG_CONFIG_HOME", "")
    if override and os.path.isabs(override):
        return Path(override)
    return Path.home() / ".config"


def _default_systemctl_runner(args: list[str]) -> subprocess.CompletedProcess:
    """Run ``systemctl --user`` with the given arguments."""
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True)


def _resolve_mullet_path() -> str:
    """Best-effort path to the ``mullet`` executable for unit files."""
    found = shutil.which("mullet")
    if found:
        return found
    if sys.argv[0] and os.path.isabs(sys.argv[0]):
        return sys.argv[0]
    return sys.executable


def _install_timer(
    *,
    dry_run: bool,
    now: float,
    config_home: Path | None,
    systemctl: Callable[[list[str]], subprocess.CompletedProcess] | None,
) -> int:
    """Write the systemd user units, record the install time and enable the timer."""
    cfg_home = config_home if config_home is not None else _default_config_home()
    unit_dir = cfg_home / "systemd" / "user"
    runner = systemctl if systemctl is not None else _default_systemctl_runner
    mullet_path = _resolve_mullet_path()
    service_path = unit_dir / "mullet-reap.service"
    timer_path = unit_dir / "mullet-reap.timer"

    service_text = (
        "[Unit]\n"
        "Description=Mullet reaper pass\n"
        "\n"
        "[Service]\n"
        "Type=oneshot\n"
        f"ExecStart={mullet_path} reap --from-timer\n"
    )
    timer_text = (
        "[Unit]\n"
        "Description=Run mullet reap every 3 minutes\n"
        "\n"
        "[Timer]\n"
        "OnBootSec=3min\n"
        "OnUnitActiveSec=3min\n"
        "Persistent=true\n"
        "\n"
        "[Install]\n"
        "WantedBy=timers.target\n"
    )

    if dry_run:
        print(f"Would write {service_path}")
        print(f"Would write {timer_path}")
        print("Would record install time in state directory")
        print("Would run: systemctl --user daemon-reload")
        print("Would run: systemctl --user enable --now mullet-reap.timer")
        return 0

    unit_dir.mkdir(parents=True, exist_ok=True)
    service_path.write_text(service_text, encoding="utf-8")
    timer_path.write_text(timer_text, encoding="utf-8")
    reaper.record_timer_install(now)
    print(f"Wrote {service_path}")
    print(f"Wrote {timer_path}")
    print("Recorded install time")

    for cmd_args in (["daemon-reload"], ["enable", "--now", "mullet-reap.timer"]):
        print(f"Running: systemctl --user {' '.join(cmd_args)}")
        result = runner(list(cmd_args))
        if result.returncode != 0:
            msg = (result.stderr or result.stdout or "unknown error").strip()
            print(f"systemctl --user {cmd_args[0]} failed: {msg}", file=sys.stderr)
            return 1
    return 0


def _remove_timer(
    *,
    dry_run: bool,
    config_home: Path | None,
    systemctl: Callable[[list[str]], subprocess.CompletedProcess] | None,
) -> int:
    """Disable the timer and remove the unit files."""
    cfg_home = config_home if config_home is not None else _default_config_home()
    unit_dir = cfg_home / "systemd" / "user"
    runner = systemctl if systemctl is not None else _default_systemctl_runner
    service_path = unit_dir / "mullet-reap.service"
    timer_path = unit_dir / "mullet-reap.timer"

    if dry_run:
        print("Would run: systemctl --user disable mullet-reap.timer")
        if service_path.exists():
            print(f"Would remove {service_path}")
        if timer_path.exists():
            print(f"Would remove {timer_path}")
        print("Would run: systemctl --user daemon-reload")
        return 0

    print("Running: systemctl --user disable mullet-reap.timer")
    result = runner(["disable", "mullet-reap.timer"])
    if result.returncode != 0:
        msg = (result.stderr or result.stdout or "unknown error").strip()
        print(f"systemctl --user disable failed: {msg}", file=sys.stderr)
        # Keep going; the unit files may still exist.

    for path in (service_path, timer_path):
        if path.exists():
            path.unlink()
            print(f"Removed {path}")

    print("Running: systemctl --user daemon-reload")
    result = runner(["daemon-reload"])
    if result.returncode != 0:
        msg = (result.stderr or result.stdout or "unknown error").strip()
        print(f"systemctl --user daemon-reload failed: {msg}", file=sys.stderr)
        return 1
    return 0


def _run_pass(dry_run: bool) -> list[dict[str, Any]]:
    """Run one reaper pass with real process and LM Studio probes."""
    ledger = Ledger()
    try:
        http = _LmStudioHttp()
        return reaper.run_pass(
            dry_run=dry_run,
            now=time.time(),
            ledger=ledger,
            http=http,
        )
    finally:
        ledger.close()


def cmd_reap(
    argv: list[str],
    *,
    config_home: Path | None = None,
    systemctl: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
) -> int:
    """Run ``mullet reap [OPTIONS]`` and return the exit code.

    Args:
        argv: Command-line arguments after ``mullet reap``.
        config_home: Directory that holds ``systemd/user``; defaults to
            ``$XDG_CONFIG_HOME`` or ``~/.config``.
        systemctl: Callable that runs ``systemctl --user`` with a list of
            arguments; defaults to a subprocess runner.
    """
    parser = argparse.ArgumentParser(prog="mullet reap", description=_DESCRIPTION)
    parser.add_argument("--dry-run", action="store_true", help="list what would be done and exit 0")
    parser.add_argument("--json", action="store_true", help="output pass records as JSON")
    parser.add_argument("--install-timer", action="store_true", help="install the systemd user timer")
    parser.add_argument("--remove-timer", action="store_true", help="remove the systemd user timer")
    parser.add_argument("--from-timer", action="store_true",
                        help="used by the timer: dry run for the first 24 hours")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 1 if exc.code not in (0, None) else 0

    now = time.time()

    if args.install_timer:
        return _install_timer(
            dry_run=args.dry_run,
            now=now,
            config_home=config_home,
            systemctl=systemctl,
        )

    if args.remove_timer:
        return _remove_timer(
            dry_run=args.dry_run,
            config_home=config_home,
            systemctl=systemctl,
        )

    dry_run = args.dry_run or (args.from_timer and reaper.timer_dry_run_active(now))

    try:
        records = _run_pass(dry_run=dry_run)
    except Exception as exc:
        print(f"Reaper pass failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(records))
        return 0

    if dry_run:
        print("Dry run: no changes made.")
    if not records:
        print("Nothing to reap.")
        return 0

    for record in records:
        kind = record.get("kind", "unknown")
        action = record.get("action", "skipped")
        if "path" in record:
            target = record["path"]
        elif "model" in record:
            target = f"model {record['model']}"
        elif "pid" in record:
            target = f"pid {record['pid']}"
        else:
            target = "unknown target"
        print(f"{action}: {kind} {target} ({record.get('result', '')})")
    return 0
