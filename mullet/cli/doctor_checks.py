"""Checks for `mullet doctor`: the toolchain Mullet jobs need, and the Claude Code setup.

Every check is read-only and quick: it looks at PATH, runs `--version`, probes
the sandbox once, queries nvidia-smi, and reads files. None of them touches
the network or runs `claude mcp`. Each returns rows of (name, detail, status), status being
"ok", "warn" or "info".
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from mullet.core import paths

Row = tuple[str, str, str]
Snapshot = Mapping[str, Mapping[str, Any]]

DEFAULT_WARN_BELOW_USD = 5.0

MIN_PYTHON = (3, 10)
_VERSION_TIMEOUT = 10
_CONTAINER_SANDBOX_HINT = (
    "inside a container, run it with --security-opt seccomp=unconfined "
    "--security-opt apparmor=unconfined --security-opt systempaths=unconfined, "
    "or --privileged"
)


# --- Where Mullet keeps things ------------------------------------------------
def _config_dir() -> Path:
    return paths.config_dir()


def config_file(name: str) -> Path:
    """A file in Mullet's config directory, e.g. config_file("config.yaml")."""
    return paths.config_file(name)


def database_path() -> Path:
    """Mullet's own SQLite database."""
    return paths.data_dir() / "mullet.db"


def _version(argv: list[str]) -> str | None:
    """First line of `<tool> --version`, or None if it won't run."""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=_VERSION_TIMEOUT,
                              stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    lines = (proc.stdout or proc.stderr).strip().splitlines()
    return lines[0].strip() if lines else ""


def python_check() -> Row:
    version = ".".join(str(n) for n in sys.version_info[:3])
    if sys.version_info[:2] < MIN_PYTHON:
        return ("Python", f"{version}: Mullet needs {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer", "warn")
    return ("Python", version, "ok")


def claude_check() -> Row:
    path = shutil.which("claude")
    if not path:
        return ("Claude Code CLI", "claude not on PATH: install Claude Code; jobs and "
                "setup-claude need it", "warn")
    version = _version([path, "--version"])
    if version is None:
        return ("Claude Code CLI", f"{path} found, but `claude --version` failed", "warn")
    return ("Claude Code CLI", version or path, "ok")


def git_check() -> Row:
    path = shutil.which("git")
    version = _version([path, "--version"]) if path else None
    if not version:
        return ("Git", "Not found: jobs and the git tool need it", "warn")
    return ("Git", version.replace("git version ", ""), "ok")


def sandbox_checks() -> list[Row]:
    """Linux: bubblewrap and Mullet's sandbox self-test, plus socat for Claude Code's own sandbox."""
    from mullet.jobs import sandbox

    if not sys.platform.startswith("linux"):
        status = sandbox.check()
        return [("Job sandbox", "ok" if status.ok else f"Unavailable: {status.reason}",
                 "ok" if status.ok else "info")]
    rows: list[Row] = []
    bwrap = shutil.which("bwrap")
    if bwrap:
        rows.append(("Bubblewrap", _version([bwrap, "--version"]) or bwrap, "ok"))
    else:
        rows.append(("Bubblewrap", "Not found: jobs need it (for example: sudo apt install bubblewrap)",
                     "warn"))
    status = sandbox.check()
    if status.ok:
        rows.append(("Job sandbox", "Self-test passed", "ok"))
    elif bwrap:
        detail = f"Self-test failed: {status.reason}"
        if "Can't mount proc" in status.reason or "Operation not permitted" in status.reason:
            detail += f"\n      Hint: {_CONTAINER_SANDBOX_HINT}"
        rows.append(("Job sandbox", detail, "warn"))
    if shutil.which("socat"):
        rows.append(("socat", "Found (Claude Code's sandbox needs it on Linux)", "ok"))
    else:
        rows.append(("socat", "Not found: Claude Code's sandbox needs it on Linux "
                     "(for example: sudo apt install socat)", "warn"))
    return rows


def anthropic_key_check() -> Row | None:
    """A warning when ANTHROPIC_API_KEY is set, since `claude -p` then bills the API."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    return ("ANTHROPIC_API_KEY", "Set: `claude -p` uses it instead of your subscription login "
            "and bills the API (Mullet jobs leave it out unless planned with --billing api)", "warn")


def gpu_checks() -> list[Row]:
    """The GPU: free VRAM and temperature, who holds Mullet's GPU lease, LM Studio
    on the GPU, and a lower power limit to try when it is at the default.

    Reads the lease's record and asks nvidia-smi twice at most, 3 seconds each;
    never runs sudo.
    """
    from mullet.jobs import machine

    room = machine.headroom(timeout=3.0)
    if not room.has_gpu:
        return [("GPU", "No NVIDIA GPU found (nvidia-smi missing or not answering): "
                 "GPU checks are skipped", "info")]
    want = machine.limits()
    parts = []
    if room.vram_free_mb is not None:
        total = f" of {room.vram_total_mb}" if room.vram_total_mb is not None else ""
        parts.append(f"{room.vram_free_mb}{total} MiB free")
    if room.gpu_temp_c is not None:
        parts.append(f"{room.gpu_temp_c:.0f}°C")
    if room.gpu_util_pct is not None:
        parts.append(f"{room.gpu_util_pct:.0f}% busy")
    detail, status = f"{room.gpu_name or 'GPU'}: {', '.join(parts)}", "ok"
    if room.gpu_temp_c is not None and room.gpu_temp_c > want.gpu_max_temp_c:
        detail += f" (over {want.gpu_max_temp_c:.0f}°C, so GPU work waits)"
        status = "warn"
    elif room.vram_free_mb is not None and room.vram_free_mb < want.gpu_min_free_mb:
        detail += f" (GPU work waits for {want.gpu_min_free_mb:.0f} MiB)"
        status = "info"
    rows: list[Row] = [("GPU", detail, status)]

    holder = machine.current_holder()
    rows.append(("GPU lease", f"Held by {holder.describe()}", "info") if holder else
                ("GPU lease", "Free", "ok"))

    lm_studio = room.lm_studio_processes
    if lm_studio:
        rows.append(("LM Studio on GPU", ", ".join(
            f"{os.path.basename(p.name)} (pid {p.pid}"
            + (f", {p.used_mb} MiB)" if p.used_mb is not None else ")") for p in lm_studio), "info"))

    power = room.power
    if power is not None and power.current_w is not None:
        lower = power.suggestion() if power.at_default else None
        if lower is not None:
            rows.append(("GPU power limit", f"{power.current_w:g} W, the default. For all-day "
                         f"local use try: sudo nvidia-smi -pl {lower:g} (lasts until reboot)", "info"))
        else:
            default = f" (default {power.default_w:g} W)" if power.default_w is not None else ""
            rows.append(("GPU power limit", f"{power.current_w:g} W{default}", "ok"))
    return rows


def toolchain_checks() -> list[Row]:
    """Python, Claude Code, git, the sandbox and the auth mode."""
    rows = [python_check(), claude_check(), git_check(), *sandbox_checks()]
    key = anthropic_key_check()
    if key:
        rows.append(key)
    return rows


def claude_setup_checks() -> list[Row]:
    """What `mullet setup-claude` recorded, and hooks an older version left behind."""
    from mullet.cli import setup_claude

    rows: list[Row] = []
    try:
        manifest = setup_claude.load_manifest()
    except setup_claude.SetupError as exc:
        manifest = None
        rows.append(("Claude Code setup", f"Can't read the setup record: {exc}", "warn"))
    if manifest is not None:
        hooks = sorted({h["name"] for h in manifest["settings"]["hooks"]})
        mcp = manifest["mcp"]
        if not (mcp or manifest["files"] or hooks):
            rows.append(("Claude Code setup", "Not set up: run mullet setup-claude", "info"))
        else:
            parts = ["MCP server registered" if mcp else "no MCP server",
                     f"{len(manifest['files'])} skill(s)", f"hooks: {', '.join(hooks) or 'none'}"]
            rows.append(("Claude Code setup", "; ".join(parts), "ok"))
            command = (mcp or {}).get("command") or []
            if command and not (Path(command[0]).exists() or shutil.which(command[0])):
                rows.append(("Claude Code setup", f"{command[0]} no longer exists: run mullet "
                             "setup-claude again", "warn"))

    path = setup_claude.settings_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "{}") if path.exists() else {}
    except (OSError, json.JSONDecodeError) as exc:
        rows.append(("Claude settings", f"Can't read {path}: {exc}", "warn"))
        return rows
    if not isinstance(data, dict):
        return rows
    legacy = setup_claude.legacy_hook_commands(data)
    if legacy:
        rows.append(("Claude hooks", f"{len(legacy)} hook(s) in {path} run scripts from a Mullet "
                     "checkout (an older setup-claude); remove them and use "
                     "mullet setup-claude --hooks", "warn"))
    status_line = data.get("statusLine")
    command = status_line.get("command", "") if isinstance(status_line, dict) else ""
    if "mullet" in command and not (Path(command).is_file() or shutil.which(command)):
        rows.append(("Claude statusLine", f"Not found: {command}", "warn"))
    return rows


# --- Spending limits and provider credit ----------------------------------------------
def warn_below_usd(path: Path | None = None) -> float:
    """The `budget.warn_below_usd` setting from the Mullet config file: the credit, in dollars,
    under which a paid provider is flagged. 5.0 when unset or not a number of 0 or more."""
    import yaml

    from mullet.jobs.ladder import config_path

    try:
        data = yaml.safe_load((path or config_path()).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return DEFAULT_WARN_BELOW_USD
    budget = data.get("budget") if isinstance(data, dict) else None
    value = budget.get("warn_below_usd") if isinstance(budget, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return DEFAULT_WARN_BELOW_USD
    return float(value)


def low_credit(snapshot: Snapshot | None, threshold: float) -> list[tuple[str, float]]:
    """Providers whose remaining credit is under `threshold` dollars, as (provider, remaining).

    A provider that reports only usage, or nothing, can't be judged and is left out.
    """
    low = []
    for name, figures in sorted((snapshot or {}).items()):
        remaining = figures.get("remaining_usd") if isinstance(figures, Mapping) else None
        if isinstance(remaining, (int, float)) and not isinstance(remaining, bool) and remaining < threshold:
            low.append((name, float(remaining)))
    return low


def low_credit_text(name: str, remaining: float, threshold: float) -> str:
    """One line saying a provider is under the warning threshold, and how to change it."""
    return (f"{name} has ${remaining:.2f} left, under ${threshold:g} (budget.warn_below_usd): "
            "top it up, or unattended jobs on that rung will stop")


def spending_limit_check(load: Callable[[], Any] | None = None) -> Row | None:
    """A warning when no spending envelope or per-rung limit is set."""
    from mullet.jobs.budget import load_budget

    if (load or load_budget)().active:
        return None
    return ("Spending limits", "No spending envelope is set: unattended jobs are limited only per job. "
            "Add budget.envelope.daily/weekly to your config (see `mullet config show budget`).", "warn")


def toolset_checks(policy: Any = None) -> list[Row]:
    """A warning naming each non-Claude provider the toolset policy lets run commands or touch files."""
    from mullet.core.toolset_policy import KNOWN_TOOLSETS, load_policy, risky_grants

    grants = risky_grants(policy or load_policy(), KNOWN_TOOLSETS)
    if not grants:
        return []
    who = "; ".join(f"{name} ({', '.join(tools)})" for name, tools in grants)
    return [("Delegation toolsets",
             f"These providers may act on this machine: {who}. A model that reads a hostile page or file "
             "could misuse them. Limit them with delegation.toolsets (see `mullet config show delegation`).",
             "warn")]


def credit_checks(probe: Callable[[], Snapshot] | None = None, threshold: float | None = None,
                  providers: list[str] | None = None) -> list[Row]:
    """Each paid provider's credit, with a warning under the threshold.

    `providers` defaults to the paid rungs of the ladder config that have a key;
    with none, nothing is probed. `probe` (default: the providers' own balance
    APIs, so it uses the network) is for tests.
    """
    from mullet.jobs import reconcile

    if providers is None:
        from mullet.jobs.ladder import PAID_RUNGS, load_config, rung_has_key

        try:
            providers = sorted({r.name for r in load_config().rungs() if r.name in PAID_RUNGS and rung_has_key(r)})
        except Exception:
            providers = []
    if not providers:
        return []
    limit = warn_below_usd() if threshold is None else threshold
    snapshot = reconcile.take(probe)
    rows: list[Row] = []
    for name in providers:
        figures = snapshot.get(name) or {}
        remaining, used = figures.get("remaining_usd"), figures.get("used_usd")
        if remaining is not None:
            below = low_credit({name: figures}, limit)
            if below:
                rows.append((f"{name} credit", low_credit_text(name, remaining, limit), "warn"))
            else:
                rows.append((f"{name} credit", f"${remaining:.2f} left", "ok"))
        elif used is not None:
            rows.append((f"{name} credit", f"${used:.2f} used (no balance reported)", "info"))
        else:
            rows.append((f"{name} credit", "Couldn't read the balance", "info"))
    return rows


def spending_checks(full: bool = False, probe: Callable[[], Snapshot] | None = None) -> list[Row]:
    """The spending-limit warning, plus provider credit when `full` (that one calls the providers)."""
    rows: list[Row] = []
    limits = spending_limit_check()
    if limits:
        rows.append(limits)
    if full:
        rows.extend(credit_checks(probe))
    return rows
