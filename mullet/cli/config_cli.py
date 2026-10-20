"""CLI subcommand: ``mullet config {get|set|list|edit|reset}``.

Reads and writes ``~/.config/mullet/config.json``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from mullet.core.providers.spend_caps import _find_budget_file, load_budget_config


_CONFIG_DIR = Path.home() / ".config" / "mullet"
_CONFIG_PATH = _CONFIG_DIR / "config.json"

_VALID_BUDGET_PROVIDERS = {"openrouter", "kimi", "anthropic", "gemini", "local"}

_DEFAULTS: dict[str, object] = {
    "telemetry": True,
    "default_provider": "economy",
    "theme": "dark",
}


def _ensure_dir() -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _load() -> dict[str, object]:
    if _CONFIG_PATH.exists():
        try:
            data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return dict(_DEFAULTS)


def _save(data: dict[str, object]) -> None:
    _ensure_dir()
    _CONFIG_PATH.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_value(raw: str) -> object:
    lower = raw.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _print_usage() -> None:
    print(
        "Usage: mullet config {get <key>|set <key> <value>|list|edit|reset|budget [show|set|monthly|reset]}",
        file=sys.stderr,
    )


def cmd_config(argv: list[str]) -> int:
    """`mullet config {get|set|list|edit|reset}`"""
    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_usage()
        return 0

    action = argv[0]
    args = argv[1:]

    if action == "get":
        if not args:
            _print_usage()
            return 1
        key = args[0]
        data = _load()
        if key not in data:
            print(f"Config key '{key}' not set.", file=sys.stderr)
            return 1
        print(data[key])
        return 0

    if action == "set":
        if len(args) < 2:
            _print_usage()
            return 1
        key, raw_value = args[0], args[1]
        data = _load()
        data[key] = _parse_value(raw_value)
        _save(data)
        print(f"Set {key} = {data[key]!r}")
        return 0

    if action == "list":
        data = _load()
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0

    if action == "edit":
        data = _load()
        initial = json.dumps(data, indent=2, sort_keys=True) + "\n"
        editor = os.environ.get("EDITOR", "")
        if not editor:
            for fallback in ("vim", "nano", "vi"):
                if subprocess.run(["which", fallback], capture_output=True).returncode == 0:
                    editor = fallback
                    break
        if not editor:
            print("Error: No editor found. Set $EDITOR.", file=sys.stderr)
            return 1

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tf:
            tf.write(initial)
            tf_path = tf.name

        try:
            result = subprocess.run([editor, tf_path])
            if result.returncode != 0:
                print(f"Editor exited with code {result.returncode}.", file=sys.stderr)
                return 1
            edited = Path(tf_path).read_text(encoding="utf-8")
            try:
                parsed = json.loads(edited)
            except json.JSONDecodeError as exc:
                print(f"Invalid JSON: {exc}", file=sys.stderr)
                return 1
            if not isinstance(parsed, dict):
                print("Error: Config must be a JSON object.", file=sys.stderr)
                return 1
            _save(parsed)
            print("Config updated.")
            return 0
        finally:
            Path(tf_path).unlink(missing_ok=True)

    if action == "reset":
        _save(dict(_DEFAULTS))
        print("Config reset to defaults.")
        return 0

    if action == "budget":
        return _cmd_budget(args)

    print(f"Unknown action: {action}", file=sys.stderr)
    _print_usage()
    return 1


def _find_mullet_dir() -> Path:
    """Walk up from cwd looking for .mullet/; create in cwd if not found."""
    start = Path.cwd()
    for parent in [start, *start.parents]:
        mullet_dir = parent / ".mullet"
        if mullet_dir.is_dir():
            return mullet_dir
    return start / ".mullet"


def _write_budget_yaml(path: Path, data: dict[str, object]) -> None:
    """Write budget data to YAML, using PyYAML if available."""
    try:
        import yaml

        path.write_text(yaml.safe_dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8")
    except ImportError:
        lines: list[str] = []
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"  {sub_key}: {sub_value}")
            else:
                lines.append(f"{key}: {value}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cmd_budget(args: list[str]) -> int:
    """Handle ``mullet config budget [show|set|monthly|reset]``."""
    subaction = args[0] if args else "show"

    if subaction in ("-h", "--help", "help"):
        print(
            "Usage: mullet config budget {show|set <provider> <cap_usd>|monthly <total_usd>|reset}",
            file=sys.stderr,
        )
        return 0

    if subaction == "show" or subaction not in ("set", "monthly", "reset"):
        if subaction != "show":
            print(f"Unknown budget action: {subaction}", file=sys.stderr)
            print(
                "Usage: mullet config budget {show|set <provider> <cap_usd>|monthly <total_usd>|reset}",
                file=sys.stderr,
            )
            return 1

        config = load_budget_config()
        if not config.caps and config.monthly_total_cap_usd is None:
            print("No spend caps configured.")
            return 0

        rows: list[tuple[str, str, str, str]] = []
        for provider in sorted(_VALID_BUDGET_PROVIDERS):
            cap = config.caps.get(provider)
            if cap is not None:
                remaining = f"${cap.remaining_cap_usd:.2f}"
                rows.append((provider, f"${cap.cap_usd:.2f}", remaining, "active"))

        try:
            from rich.console import Console
            from rich.table import Table

            table = Table(title="Budget Configuration")
            table.add_column("Provider", style="cyan")
            table.add_column("Cap (USD)", style="green")
            table.add_column("Remaining", style="yellow")
            table.add_column("Status", style="magenta")
            for provider, cap_str, remaining, status in rows:
                table.add_row(provider, cap_str, remaining, status)
            console = Console()
            console.print(table)
            if config.monthly_total_cap_usd is not None:
                console.print(f"Monthly total: ${config.monthly_total_cap_usd:.2f}")
        except ImportError:
            if rows:
                print(f"{'Provider':<14} {'Cap (USD)':<12} {'Remaining':<12} Status")
                print("-" * 52)
                for provider, cap_str, remaining, status in rows:
                    print(f"{provider:<14} {cap_str:<12} {remaining:<12} {status}")
                print("-" * 52)
            if config.monthly_total_cap_usd is not None:
                print(f"Monthly total ${config.monthly_total_cap_usd:.2f}")
        return 0

    if subaction == "set":
        if len(args) < 3:
            print(
                "Usage: mullet config budget set <provider> <cap_usd>",
                file=sys.stderr,
            )
            return 1
        provider = args[1]
        raw_cap = args[2]
        if provider not in _VALID_BUDGET_PROVIDERS:
            print(f"Error: Unknown provider '{provider}'.", file=sys.stderr)
            print(f"Valid providers: {', '.join(sorted(_VALID_BUDGET_PROVIDERS))}", file=sys.stderr)
            return 1
        try:
            cap_usd = float(raw_cap)
        except ValueError:
            print(f"Error: Invalid cap value '{raw_cap}'.", file=sys.stderr)
            return 1

        existing_path = _find_budget_file()
        if existing_path is not None:
            budget_path = existing_path
        else:
            mullet_dir = _find_mullet_dir()
            mullet_dir.mkdir(parents=True, exist_ok=True)
            budget_path = mullet_dir / "budget.yaml"

        if budget_path.exists():
            try:
                import yaml
                raw_data: dict[str, object] = yaml.safe_load(budget_path.read_text(encoding="utf-8")) or {}
            except ImportError:
                raw_data = {}
                for line in budget_path.read_text(encoding="utf-8").splitlines():
                    if line.strip() and not line.startswith(" ") and ":" in line:
                        key, val = line.split(":", 1)
                        raw_data[key.strip()] = val.strip()
        else:
            raw_data = {}

        raw_data[provider] = {"cap_usd": cap_usd}
        _write_budget_yaml(budget_path, raw_data)
        print(f"Set {provider} cap to ${cap_usd:.2f}")
        return 0

    if subaction == "monthly":
        if len(args) < 2:
            print(
                "Usage: mullet config budget monthly <total_usd>",
                file=sys.stderr,
            )
            return 1
        raw_total = args[1]
        try:
            total_usd = float(raw_total)
        except ValueError:
            print(f"Error: Invalid total value '{raw_total}'.", file=sys.stderr)
            return 1

        existing_path = _find_budget_file()
        if existing_path is not None:
            budget_path = existing_path
        else:
            mullet_dir = _find_mullet_dir()
            mullet_dir.mkdir(parents=True, exist_ok=True)
            budget_path = mullet_dir / "budget.yaml"

        if budget_path.exists():
            try:
                import yaml
                raw_data: dict[str, object] = yaml.safe_load(budget_path.read_text(encoding="utf-8")) or {}
            except ImportError:
                raw_data = {}
                for line in budget_path.read_text(encoding="utf-8").splitlines():
                    if line.strip() and not line.startswith(" ") and ":" in line:
                        key, val = line.split(":", 1)
                        raw_data[key.strip()] = val.strip()
        else:
            raw_data = {}

        raw_data["monthly_total_cap_usd"] = total_usd
        _write_budget_yaml(budget_path, raw_data)
        print(f"Set monthly total cap to ${total_usd:.2f}")
        return 0

    if subaction == "reset":
        existing_path = _find_budget_file()
        if existing_path is not None:
            existing_path.unlink()
            print("Budget configuration reset.")
        else:
            print("No budget configuration to reset.")
        return 0

    print(f"Unknown budget action: {subaction}", file=sys.stderr)  # pragma: no cover
    print(  # pragma: no cover
        "Usage: mullet config budget {show|set <provider> <cap_usd>|monthly <total_usd>|reset}",
        file=sys.stderr,
    )
    return 1  # pragma: no cover
