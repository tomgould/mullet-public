"""`mullet hud` — Claude-style terminal HUD.

Subcommands
-----------
show        Print a one-shot HUD snapshot (default).
watch       Refresh the HUD every N seconds.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time

from mullet.hud.claude_style import render_claude_style_hud
from mullet.hud.render import render_hud
from mullet.hud.state import build_context
from mullet.hud.types import HudConfig

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL = 2.0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mullet hud",
        description="Show or watch the Mullet terminal HUD.",
        usage="mullet hud [COMMAND] [OPTIONS]",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colour codes",
    )
    parser.add_argument(
        "--preset",
        choices=("minimal", "compact", "focused", "full", "dense", "claude"),
        default="claude",
        help="HUD layout preset (default: claude)",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=_DEFAULT_INTERVAL,
        help=f"Refresh interval in seconds for watch mode (default: {_DEFAULT_INTERVAL})",
    )
    parser.add_argument(
        "--width",
        "-w",
        type=int,
        default=None,
        help="Maximum line width (default: terminal width)",
    )

    sub = parser.add_subparsers(dest="command", help="HUD subcommands")
    sub.add_parser("show", help="Print a one-shot HUD snapshot")
    sub.add_parser("watch", help="Refresh the HUD continuously")

    return parser


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from *text*."""
    import re
    return re.sub(r"\033\[[0-9;]*m", "", text)


def _print_hud(text: str, no_color: bool) -> None:
    if no_color:
        text = _strip_ansi(text)
    print(text)


def _cmd_show(args: argparse.Namespace) -> int:
    """Render a single HUD snapshot and print it."""
    if args.preset == "claude":
        text = render_claude_style_hud(no_color=args.no_color)
    else:
        ctx = build_context(cwd=os.getcwd())
        config = HudConfig(preset=args.preset, max_width=args.width)
        text = render_hud(ctx, config)

    _print_hud(text, args.no_color)
    return 0


def _cmd_watch(args: argparse.Namespace) -> int:
    """Continuously refresh the HUD."""
    interval = max(0.5, args.interval)
    preset = args.preset
    width = args.width or shutil.get_terminal_size((120, 24)).columns
    no_color = args.no_color

    try:
        while True:
            # Clear screen and move cursor to top-left
            if not no_color and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            else:
                print("\n" + "─" * width)

            if preset == "claude":
                text = render_claude_style_hud(no_color=no_color)
            else:
                ctx = build_context(cwd=os.getcwd())
                config = HudConfig(preset=preset, max_width=width)
                text = render_hud(ctx, config)

            _print_hud(text, no_color)

            # Budget advice line (only for non-claude presets — claude style
            # already embeds budget info)
            if preset != "claude":
                try:
                    from mullet.hud.budget_monitor import check_headroom, format_advice
                    advice = format_advice(check_headroom())
                    _print_hud(advice, no_color)
                except Exception:
                    pass

            time.sleep(interval)
    except KeyboardInterrupt:
        return 0


def cmd_hud(argv: list[str]) -> int:
    """Entry point for ``mullet hud``.

    Args:
        argv: Raw argument list (everything after ``mullet hud``).

    Returns:
        Exit code (0 on success, 1 on error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Default command is "show" when no subcommand is given.
    command = args.command or "show"

    if command == "show":
        return _cmd_show(args)
    if command == "watch":
        return _cmd_watch(args)

    parser.print_help()
    return 1


def cmd_hud_claude(argv: list[str]) -> int:
    """Back-compat entry point for ``mullet hud-claude``."""
    parser = argparse.ArgumentParser(
        prog="mullet hud-claude",
        description="Show the Claude Code plugin-style HUD.",
        usage="mullet hud-claude [--watch] [--interval SECONDS] [--no-color]",
    )
    parser.add_argument(
        "-w", "--watch",
        action="store_true",
        help="Refresh continuously (Ctrl-C to stop)",
    )
    parser.add_argument(
        "-i", "--interval",
        type=float,
        default=_DEFAULT_INTERVAL,
        help=f"Refresh interval in seconds (default: {_DEFAULT_INTERVAL})",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colours")
    args = parser.parse_args(argv)

    no_color = args.no_color or os.environ.get("NO_COLOR") is not None

    if args.watch:
        try:
            while True:
                print("\033[2J\033[H", end="")
                print(render_claude_style_hud(no_color=no_color))
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print()
            return 0
    else:
        print(render_claude_style_hud(no_color=no_color))

    return 0
