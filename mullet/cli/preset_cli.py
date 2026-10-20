"""CLI dispatcher for ``mullet preset`` subcommands.

Provides ``list``, ``use``, ``show``, ``add`` and ``remove`` actions that
wrap :class:`mullet.core.presets.PresetManager`. The dispatcher is
intentionally a plain function rather than an ``argparse`` subparser so
it can be wired into ``mullet/cli/main.py`` via the same ``sys.argv[1]``
early-branch pattern used by ``mullet provider``.
"""

from __future__ import annotations

import argparse
import sys

import yaml

from mullet.core.presets import Preset, PresetManager


_USAGE = "mullet preset {list|use|show|add|remove} [args]"


def _print_usage() -> None:
    print(_USAGE)


def _fmt_providers(preset: Preset) -> str:
    if preset.allowed_providers is None:
        return "all"
    return ",".join(preset.allowed_providers)


def _fmt_opencode(preset: Preset) -> str:
    if preset.opencode_model is None:
        return "native" if preset.native_only else "-"
    return preset.opencode_model


# ----- subactions ----------------------------------------------------------


def _cmd_list(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="mullet preset list")
    try:
        parser.parse_args(argv)
    except SystemExit as exc:
        return 1 if exc.code not in (0, None) else 0

    manager = PresetManager()
    config = manager.load()
    presets = manager.list()

    for preset in presets:
        marker = "* " if preset.name == config.active else "  "
        native_tag = " [native]" if preset.native_only else ""
        print(
            f"{marker}{preset.name}: {preset.label}  "
            f"speed={preset.speed_mode}  providers={_fmt_providers(preset)}  "
            f"opencode={_fmt_opencode(preset)}{native_tag}"
        )
    return 0


def _cmd_use(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="mullet preset use")
    parser.add_argument("name")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 1 if exc.code not in (0, None) else 0

    manager = PresetManager()
    try:
        manager.use(args.name)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"✓ Active preset: {args.name}")
    return 0


def _cmd_show(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="mullet preset show")
    parser.add_argument("name")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 1 if exc.code not in (0, None) else 0

    manager = PresetManager()
    preset = manager.get(args.name)
    if preset is None:
        print(f"error: unknown preset: {args.name!r}", file=sys.stderr)
        return 1

    payload = {
        "name": preset.name,
        "label": preset.label,
        "allowed_providers": preset.allowed_providers,
        "speed_mode": preset.speed_mode,
        "opencode_model": preset.opencode_model,
        "native_only": preset.native_only,
        "description": preset.description,
    }
    print(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False).rstrip())
    return 0


def _cmd_add(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="mullet preset add")
    parser.add_argument("--name", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--providers", default=None)
    parser.add_argument("--speed", default="balanced")
    parser.add_argument("--opencode-model", default=None)
    parser.add_argument("--native-only", action="store_true")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 1 if exc.code not in (0, None) else 0

    allowed_providers = None
    if args.providers is not None:
        allowed_providers = [p.strip() for p in args.providers.split(",") if p.strip()]

    try:
        preset = Preset(
            name=args.name,
            label=args.label,
            allowed_providers=allowed_providers,
            speed_mode=args.speed,
            opencode_model=args.opencode_model,
            native_only=args.native_only,
        )
    except ValueError as exc:
        print(f"error: invalid preset: {exc}", file=sys.stderr)
        return 1

    manager = PresetManager()
    try:
        manager.add(preset)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved preset: {args.name}")
    return 0


def _cmd_remove(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="mullet preset remove")
    parser.add_argument("name")
    parser.add_argument("--force", action="store_true")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 1 if exc.code not in (0, None) else 0

    manager = PresetManager()
    try:
        manager.remove(args.name, force=args.force)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Removed preset: {args.name}")
    return 0


# ----- dispatcher ----------------------------------------------------------


_SUBCOMMANDS = {
    "list": _cmd_list,
    "use": _cmd_use,
    "show": _cmd_show,
    "add": _cmd_add,
    "remove": _cmd_remove,
}


def cmd_preset(argv: list[str]) -> int:
    """Top-level dispatcher for ``mullet preset <action> ...``."""
    if argv and argv[0] in {"-h", "--help", "help"}:
        _print_usage()
        return 0
    action = argv[0] if argv else "list"
    handler = _SUBCOMMANDS.get(action)
    if handler is None:
        print(f"error: unknown subaction: {action!r}", file=sys.stderr)
        _print_usage()
        return 1
    return handler(argv[1:])
