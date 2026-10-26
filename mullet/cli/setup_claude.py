"""`mullet setup-claude`: connect Mullet to Claude Code, and undo it.

What it changes:

- registers the MCP server: `claude mcp add --scope user mullet -- <mullet> mcp`;
- installs Mullet's Claude Code skills as `~/.claude/skills/<name>/SKILL.md`;
- adds the `session-strategy` hook to `~/.claude/settings.json` as
  `<mullet> hook session-strategy`, so every Claude Code session in every
  directory opens with the budget and delegation question; with `--hooks a,b`
  it adds those hooks instead (`--hooks none` adds none, `--hooks list`
  describes them);
- with `--statusline`, sets Claude Code's status line to `mullet-hud` when
  none is set.

Every change is recorded in `<config dir>/setup-manifest.json`, so
`--uninstall` removes exactly what was added and nothing else, and running it
again changes nothing. `--dry-run` prints the changes without making them. It
never edits CLAUDE.md and never writes into the current repository.
`<mullet>` is the `mullet` found on PATH, or `python -m mullet` with this
interpreter when there isn't one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from mullet.core.paths import config_dir
from mullet.hooks import HOOKS, HookSpec, describe

MCP_NAME = "mullet"
# Mullet's own skills for Claude Code, copied from the packaged built-ins.
CLAUDE_SKILLS = ("mullet-job", "pool-improve")
# Hooks registered when --hooks isn't given. The hook lives in the user-level
# settings, so it covers every directory, new ones included.
DEFAULT_HOOKS = ("session-strategy",)
MANIFEST_VERSION = 1
_BUILTIN_SKILLS = Path(__file__).resolve().parent.parent / "core" / "skills" / "builtin"
_CLAUDE_TIMEOUT = 120


class SetupError(Exception):
    """A step can't go ahead; the message says why."""


# --- Locations ---------------------------------------------------------------


def _config_dir() -> Path:
    """Mullet's config directory (see mullet.core.paths), computed at call time."""
    return config_dir()


def manifest_path() -> Path:
    """Where setup-claude records what it changed."""
    return _config_dir() / "setup-manifest.json"


def claude_dir() -> Path:
    """Claude Code's user directory: $CLAUDE_CONFIG_DIR, else ~/.claude."""
    override = os.environ.get("CLAUDE_CONFIG_DIR")
    return Path(override).expanduser() if override else Path.home() / ".claude"


def settings_path() -> Path:
    return claude_dir() / "settings.json"


def mullet_command() -> list[str]:
    """How Claude Code should start Mullet: `mullet` on PATH, else this interpreter's `-m mullet`."""
    found = shutil.which("mullet")
    return [found] if found else [sys.executable, "-m", "mullet"]


def hook_command(cmd: list[str], spec: HookSpec) -> str:
    return shlex.join([*cmd, "hook", spec.name])


# --- Manifest ----------------------------------------------------------------


def _empty_manifest() -> dict:
    return {
        "version": MANIFEST_VERSION,
        "mcp": None,
        "files": [],
        "dirs": [],
        "settings": {
            "path": None, "created_file": False, "keys_added": [],
            "events_added": [], "groups_added": [], "hooks": [], "statusline": None,
        },
    }


def load_manifest() -> dict:
    """The recorded changes, or an empty record when setup-claude hasn't run."""
    path = manifest_path()
    if not path.exists():
        return _empty_manifest()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SetupError(f"can't read {path}: {exc}") from exc
    manifest = _empty_manifest()
    manifest.update({k: v for k, v in data.items() if k != "settings"})
    manifest["settings"].update(data.get("settings") or {})
    return manifest


def _manifest_is_empty(manifest: dict) -> bool:
    s = manifest["settings"]
    return not (manifest["mcp"] or manifest["files"] or manifest["dirs"] or s["created_file"]
                or s["keys_added"] or s["events_added"] or s["groups_added"] or s["hooks"]
                or s["statusline"])


def _save_manifest(manifest: dict) -> None:
    path = manifest_path()
    if _manifest_is_empty(manifest):
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, json.dumps(manifest, indent=2) + "\n")


# --- Small helpers -----------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        if path.exists():
            os.chmod(tmp, path.stat().st_mode & 0o777)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _home_relative(path: Path) -> str:
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


class _Out:
    """Prints each change as done, or as planned under --dry-run."""

    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.changes = 0
        self.failures = 0

    def change(self, done: str, planned: str) -> None:
        self.changes += 1
        print(f"  {'would ' + planned if self.dry_run else done}")

    def same(self, message: str) -> None:
        print(f"  {message}")

    def fail(self, message: str) -> None:
        self.failures += 1
        print(f"  ! {message}", file=sys.stderr)


# --- Claude Code CLI (MCP registration) --------------------------------------


def _claude(*args: str) -> subprocess.CompletedProcess:
    """Run the claude CLI from an empty directory, so only user-scope MCP config is seen."""
    claude = shutil.which("claude")
    if not claude:
        raise SetupError("the claude CLI is not on PATH")
    with tempfile.TemporaryDirectory(prefix="mullet-setup-") as cwd:
        try:
            return subprocess.run([claude, *args], cwd=cwd, capture_output=True, text=True,
                                  stdin=subprocess.DEVNULL, timeout=_CLAUDE_TIMEOUT)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SetupError(f"claude {' '.join(args)} failed: {exc}") from exc


def _mcp_add_args(cmd: list[str]) -> list[str]:
    return ["mcp", "add", "--scope", "user", MCP_NAME, "--", *cmd, "mcp"]


def _register_mcp(manifest: dict, cmd: list[str], out: _Out) -> None:
    shown = "claude " + shlex.join(_mcp_add_args(cmd))
    try:
        _register_mcp_steps(manifest, cmd, out, shown)
    except SetupError as exc:
        out.fail(f"MCP server not registered: {exc}. Install Claude Code, then run: {shown}")


def _register_mcp_steps(manifest: dict, cmd: list[str], out: _Out, shown: str) -> None:
    registered = _claude("mcp", "get", MCP_NAME).returncode == 0
    recorded = manifest["mcp"]
    if registered and not recorded:
        out.same(f"MCP server '{MCP_NAME}' is already registered, not by setup-claude; left as it is")
        return
    if registered and recorded.get("command") == cmd:
        out.same(f"MCP server '{MCP_NAME}' already registered (user scope)")
        return
    if out.dry_run:
        out.change("", f"run: {shown}")
        return
    if registered:
        removed = _claude("mcp", "remove", "--scope", "user", MCP_NAME)
        if removed.returncode != 0:
            out.fail(f"couldn't replace the MCP server registration: {removed.stderr.strip()}")
            return
    result = _claude(*_mcp_add_args(cmd))
    if result.returncode != 0:
        out.fail(f"{shown} failed: {(result.stderr or result.stdout).strip()}")
        return
    manifest["mcp"] = {"name": MCP_NAME, "scope": "user", "command": cmd}
    out.change(f"registered MCP server '{MCP_NAME}' (user scope): {shlex.join(cmd)} mcp", "")


def _unregister_mcp(manifest: dict, out: _Out) -> None:
    recorded = manifest["mcp"]
    if not recorded:
        return
    if out.dry_run:
        out.change("", f"run: claude mcp remove --scope user {recorded['name']}")
        return
    try:
        result = _claude("mcp", "remove", "--scope", "user", recorded["name"])
    except SetupError as exc:
        out.fail(f"MCP server '{recorded['name']}' not removed: {exc}")
        return
    if result.returncode != 0:
        text = (result.stderr or result.stdout).strip()
        # Already gone ("No ... MCP server found ..."): nothing left to remove.
        if not re.search(r"\bno\b.*\bfound\b|not found", text, re.IGNORECASE):
            out.fail(f"claude mcp remove failed: {text}")
            return
    manifest["mcp"] = None
    out.change(f"removed MCP server '{recorded['name']}' (user scope)", "")


# --- Skills ------------------------------------------------------------------


def _record_dirs(manifest: dict, created: list[Path]) -> None:
    for d in created:
        if str(d) not in manifest["dirs"]:
            manifest["dirs"].append(str(d))


def _missing_parents(path: Path) -> list[Path]:
    """Directories that would have to be created for `path`, outermost first."""
    missing = []
    for parent in reversed(path.parents):
        if not parent.exists():
            missing.append(parent)
    return missing


def _install_skills(manifest: dict, out: _Out, force: bool) -> None:
    recorded = {f["path"]: f for f in manifest["files"]}
    for name in CLAUDE_SKILLS:
        source = _BUILTIN_SKILLS / name / "SKILL.md"
        text = source.read_text(encoding="utf-8")
        dest = claude_dir() / "skills" / name / "SKILL.md"
        entry = recorded.get(str(dest))
        shown = _home_relative(dest)
        if dest.exists():
            if entry is None:
                out.same(f"skill {name}: {shown} exists and isn't Mullet's; left alone")
                continue
            if dest.read_text(encoding="utf-8") == text:
                out.same(f"skill {name} already installed")
                continue
            if _sha256(dest) != entry["sha256"] and not force:
                out.same(f"skill {name}: {shown} was changed after setup-claude installed it; "
                         "left alone (--force replaces it)")
                continue
            out.change(f"updated skill {name}: {shown}", f"update {shown}")
        else:
            out.change(f"installed skill {name}: {shown}", f"create {shown}")
        if out.dry_run:
            continue
        created = _missing_parents(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        _record_dirs(manifest, created)
        dest.write_text(text, encoding="utf-8")
        new_entry = {"path": str(dest), "sha256": _sha256(dest)}
        manifest["files"] = [f for f in manifest["files"] if f["path"] != str(dest)] + [new_entry]


def _remove_files(manifest: dict, out: _Out) -> None:
    kept = []
    for entry in manifest["files"]:
        path = Path(entry["path"])
        shown = _home_relative(path)
        if not path.exists():
            continue
        if _sha256(path) != entry["sha256"]:
            out.same(f"{shown} was changed after setup-claude installed it; left in place")
            kept.append(entry)
            continue
        out.change(f"removed {shown}", f"remove {shown}")
        if not out.dry_run:
            path.unlink()
    if out.dry_run:
        for d in manifest["dirs"]:
            if Path(d).is_dir():
                out.same(f"would remove the folder {_home_relative(Path(d))} if nothing else is in it")
        return
    manifest["files"] = kept
    # Folders setup-claude created go only when empty; one that now holds
    # something else isn't Mullet's alone any more, so it stays.
    for d in sorted(manifest["dirs"], key=lambda p: len(Path(p).parts), reverse=True):
        path = Path(d)
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
            out.change(f"removed the empty folder {_home_relative(path)}", "")
    manifest["dirs"] = []


# --- settings.json (hooks and status line) -----------------------------------


def _read_settings(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
    except json.JSONDecodeError as exc:
        raise SetupError(f"{path} isn't valid JSON ({exc}); fix it first, nothing was changed there")
    if not isinstance(data, dict):
        raise SetupError(f"{path} doesn't hold a JSON object; nothing was changed there")
    return data


def _find_group(groups: list, matcher: str | None) -> dict | None:
    for group in groups:
        if not isinstance(group, dict):
            continue
        if matcher is None and "matcher" not in group:
            return group
        if matcher is not None and group.get("matcher") == matcher:
            return group
    return None


def _same_registration(entry: dict, name: str, event: str, matcher: str | None) -> bool:
    return entry["name"] == name and entry["event"] == event and entry["matcher"] == matcher


def _remove_command(data: dict, event: str, matcher: str | None, command: str) -> bool:
    group = _find_group((data.get("hooks") or {}).get(event) or [], matcher)
    if group is None:
        return False
    before = len(group.get("hooks", []))
    group["hooks"] = [h for h in group.get("hooks", [])
                      if not (isinstance(h, dict) and h.get("command") == command)]
    return len(group["hooks"]) != before


def _prune(data: dict, record: dict) -> None:
    """Drop empty hook groups, events and keys that setup-claude itself created."""
    hooks = data.get("hooks")
    if isinstance(hooks, dict):
        for event, matcher in [tuple(g) for g in record["groups_added"]]:
            groups = hooks.get(event)
            group = _find_group(groups or [], matcher)
            if group is not None and not group.get("hooks"):
                groups.remove(group)
        for event in record["events_added"]:
            if event in hooks and not hooks[event]:
                del hooks[event]
        if "hooks" in record["keys_added"] and not hooks:
            del data["hooks"]


# Hook scripts an older setup-claude registered by their path in a checkout.
_LEGACY_SCRIPTS = (
    "compress_whitespace.py", "delegation_reminder.py", "skill_activator.py",
    "session_context.py", "user_prompt_detect_override.py", "mullet_pretooluse_rewriter.py",
    "mullet_observation_capture.py", "agent_hud_tracker.py", "preset_prompt.py",
    "manual_tool_guard.py", "pre_agent_guard.py",
)


def legacy_hook_commands(data: dict) -> list[str]:
    """Hook commands an older setup-claude added: scripts run from a Mullet checkout."""
    found = []
    for groups in (data.get("hooks") or {}).values():
        for group in groups if isinstance(groups, list) else []:
            for hook in (group.get("hooks") or []) if isinstance(group, dict) else []:
                command = hook.get("command", "") if isinstance(hook, dict) else ""
                if any(f"scripts/hooks/{script}" in command for script in _LEGACY_SCRIPTS):
                    found.append(command)
    return found


def _update_settings(manifest: dict, specs: list[HookSpec], cmd: list[str],
                     statusline: bool, out: _Out) -> None:
    path = settings_path()
    record = manifest["settings"]
    try:
        data = _read_settings(path)
    except SetupError as exc:
        out.fail(str(exc))
        return
    original = json.dumps(data, sort_keys=True)
    shown = _home_relative(path)

    legacy = legacy_hook_commands(data)
    if legacy:
        out.same(f"note: {len(legacy)} hook command(s) in {shown} run scripts from a Mullet checkout "
                 "(added by an older setup-claude); remove them by hand and use --hooks instead")

    wanted = {s.name: s for s in specs}
    for entry in record["hooks"]:
        if entry["name"] in HOOKS:
            wanted.setdefault(entry["name"], HOOKS[entry["name"]])
    if not wanted:
        out.same("no hooks (add some with --hooks; --hooks list shows what each does)")

    for spec in wanted.values():
        command = hook_command(cmd, spec)
        # A registration this version no longer makes (its event or matcher changed).
        for stale in [e for e in record["hooks"] if e["name"] == spec.name
                      and (e["event"], e["matcher"]) not in spec.registrations]:
            _remove_command(data, stale["event"], stale["matcher"], stale["command"])
            record["hooks"].remove(stale)
            out.change(f"removed the old registration of {spec.name} for {stale['event']}",
                       f"remove the old registration of {spec.name} for {stale['event']}")
        for event, matcher in spec.registrations:
            old = next((e for e in record["hooks"]
                        if _same_registration(e, spec.name, event, matcher)), None)
            if old and old["command"] != command:
                # Mullet moved (a reinstall); point the hook at the new command.
                _remove_command(data, event, matcher, old["command"])
                record["hooks"].remove(old)
                old = None
            hooks = data.get("hooks")
            if not isinstance(hooks, dict):
                hooks = data["hooks"] = {}
                if "hooks" not in record["keys_added"]:
                    record["keys_added"].append("hooks")
            groups = hooks.get(event)
            if not isinstance(groups, list):
                groups = hooks[event] = []
                if event not in record["events_added"]:
                    record["events_added"].append(event)
            group = _find_group(groups, matcher)
            if group is None:
                group = {"hooks": []} if matcher is None else {"matcher": matcher, "hooks": []}
                groups.append(group)
                if [event, matcher] not in record["groups_added"]:
                    record["groups_added"].append([event, matcher])
            where = f"{event}" + (f" [{matcher}]" if matcher else "")
            if any(isinstance(h, dict) and h.get("command") == command for h in group.get("hooks", [])):
                out.same(f"hook {spec.name} already registered for {where}")
                continue
            group.setdefault("hooks", []).append({"type": "command", "command": command})
            if old is None:
                record["hooks"].append({"name": spec.name, "event": event, "matcher": matcher,
                                        "command": command})
            out.change(f"added hook {spec.name} for {where}: {command}",
                       f"add hook {spec.name} for {where}: {command}")

    if statusline:
        _set_statusline(data, record, out)

    if json.dumps(data, sort_keys=True) == original:
        return
    if out.dry_run:
        return
    if not path.exists():
        record["created_file"] = True
        path.parent.mkdir(parents=True, exist_ok=True)
    record["path"] = str(path)
    _write_atomic(path, json.dumps(data, indent=2) + "\n")


def _set_statusline(data: dict, record: dict, out: _Out) -> None:
    current = data.get("statusLine")
    ours = record["statusline"]
    if current is not None and not (ours and isinstance(current, dict)
                                    and current.get("command") == ours):
        out.same("status line already set, not by setup-claude; left as it is")
        return
    hud = shutil.which("mullet-hud")
    if not hud:
        out.fail("status line not set: mullet-hud is not on PATH")
        return
    if ours == hud and current is not None:
        out.same("status line already set to mullet-hud")
        return
    data["statusLine"] = {"type": "command", "command": hud}
    if "statusLine" not in record["keys_added"]:
        record["keys_added"].append("statusLine")
    record["statusline"] = hud
    out.change(f"set the status line to {hud}", f"set the status line to {hud}")


def _revert_settings(manifest: dict, out: _Out) -> None:
    record = manifest["settings"]
    path = Path(record["path"]) if record["path"] else settings_path()
    if not (record["hooks"] or record["statusline"] or record["keys_added"] or record["created_file"]):
        return
    if not path.exists():
        manifest["settings"] = _empty_manifest()["settings"]
        return
    try:
        data = _read_settings(path)
    except SetupError as exc:
        out.fail(str(exc))
        return
    shown = _home_relative(path)
    original = json.dumps(data, sort_keys=True)
    for entry in record["hooks"]:
        if _remove_command(data, entry["event"], entry["matcher"], entry["command"]):
            out.change(f"removed hook {entry['name']} ({entry['event']}) from {shown}",
                       f"remove hook {entry['name']} ({entry['event']}) from {shown}")
    if record["statusline"]:
        current = data.get("statusLine")
        if isinstance(current, dict) and current.get("command") == record["statusline"]:
            del data["statusLine"]
            out.change(f"removed the status line from {shown}", f"remove the status line from {shown}")
    _prune(data, record)
    if out.dry_run:
        return
    if record["created_file"] and not data:
        path.unlink()
        out.change(f"removed {shown}, which setup-claude created", "")
    elif json.dumps(data, sort_keys=True) != original:
        _write_atomic(path, json.dumps(data, indent=2) + "\n")
    manifest["settings"] = _empty_manifest()["settings"]


# --- Entry points ------------------------------------------------------------


def run_setup_claude(hooks: list[HookSpec] | None = None, statusline: bool = False,
                     dry_run: bool = False, force: bool = False) -> int:
    """Connect Mullet to Claude Code. Returns 0, or 1 if any step failed."""
    out = _Out(dry_run)
    print("mullet setup-claude" + (" (dry run: nothing is changed)" if dry_run else ""))
    try:
        manifest = load_manifest()
    except SetupError as exc:
        out.fail(str(exc))
        return 1
    cmd = mullet_command()
    claude_missing = shutil.which("claude") is None
    if dry_run and claude_missing:
        print("Warning: claude is not on PATH; install it, then run this again to register the MCP server.",
              file=sys.stderr)
        print(f"  would run: claude {shlex.join(_mcp_add_args(cmd))}")
    else:
        _register_mcp(manifest, cmd, out)
    _install_skills(manifest, out, force)
    _update_settings(manifest, list(hooks or []), cmd, statusline, out)
    if not dry_run:
        _save_manifest(manifest)
    if out.changes == 0 and out.failures == 0:
        print("Nothing to change.")
    elif not dry_run and out.changes:
        print(f"Recorded in {_home_relative(manifest_path())}; undo with: mullet setup-claude --uninstall")
        print("Restart Claude Code to load the changes.")
    return 1 if out.failures else 0


def run_uninstall(dry_run: bool = False) -> int:
    """Remove exactly what setup-claude recorded. Returns 0, or 1 if anything was left."""
    out = _Out(dry_run)
    print("mullet setup-claude --uninstall" + (" (dry run: nothing is changed)" if dry_run else ""))
    try:
        manifest = load_manifest()
    except SetupError as exc:
        out.fail(str(exc))
        return 1
    if _manifest_is_empty(manifest):
        print("Nothing to remove: setup-claude has no record of changes.")
        return 0
    _unregister_mcp(manifest, out)
    _revert_settings(manifest, out)
    _remove_files(manifest, out)
    if not dry_run:
        _save_manifest(manifest)
        if not _manifest_is_empty(manifest):
            print(f"Some changes were left in place; they stay recorded in {_home_relative(manifest_path())}.")
    return 1 if out.failures else 0


def _parse_hooks(value: str) -> list[HookSpec]:
    from mullet.hooks import find

    if value.strip().lower() == "all":
        return list(HOOKS.values())
    if value.strip().lower() == "none":
        return []
    specs, unknown = [], []
    for name in filter(None, (part.strip() for part in value.split(","))):
        spec = find(name)
        (specs.append(spec) if spec else unknown.append(name))
    if unknown:
        raise SetupError(f"unknown hook(s): {', '.join(unknown)}. Choose from: {', '.join(HOOKS)}")
    return specs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mullet setup-claude",
        description="Register Mullet's MCP server and skills with Claude Code, and optionally "
                    "its hooks and status line. Every change is recorded so --uninstall can "
                    "remove exactly those changes.",
    )
    parser.add_argument("--hooks", metavar="NAMES",
                        help="comma-separated hooks to add, 'all', 'none' or 'list' to describe "
                             f"them (default: {', '.join(DEFAULT_HOOKS)})")
    parser.add_argument("--statusline", action="store_true",
                        help="set Claude Code's status line to mullet-hud if none is set")
    parser.add_argument("--dry-run", "--check", dest="dry_run", action="store_true",
                        help="print the changes without making them")
    parser.add_argument("--uninstall", action="store_true",
                        help="remove everything setup-claude recorded")
    parser.add_argument("--force", action="store_true",
                        help="replace an installed skill file even if it was edited")
    parser.add_argument("--yes", action="store_true", help=argparse.SUPPRESS)
    return parser


def cmd_setup_claude(argv: list[str]) -> int:
    """`mullet setup-claude [options]`; returns the exit code."""
    args = build_parser().parse_args(argv)
    if args.hooks is not None and args.hooks.strip().lower() == "list":
        print(f"Hooks for --hooks (added by default: {', '.join(DEFAULT_HOOKS)}):")
        print(describe())
        return 0
    if args.uninstall:
        if args.hooks or args.statusline:
            print("mullet setup-claude: --uninstall takes no --hooks or --statusline", file=sys.stderr)
            return 2
        return run_uninstall(dry_run=args.dry_run)
    try:
        specs = _parse_hooks(args.hooks) if args.hooks else _parse_hooks(",".join(DEFAULT_HOOKS))
    except SetupError as exc:
        print(f"mullet setup-claude: {exc}", file=sys.stderr)
        return 2
    return run_setup_claude(specs, statusline=args.statusline, dry_run=args.dry_run, force=args.force)
