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
from pathlib import Path

from mullet.core import paths

Row = tuple[str, str, str]

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
