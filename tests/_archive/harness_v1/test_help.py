"""Tests for mullet.harness.commands.help."""

from __future__ import annotations

import pytest

from mullet.harness.commands.help import handle_help_command

pytestmark = pytest.mark.harness


def test_help_all_lists_everything() -> None:
    for args in ("", "all"):
        out = handle_help_command(args)
        assert "Available slash commands:" in out
        assert "Loaded skills:" in out or "No skills loaded." in out or "Skills: unavailable." in out
        assert "TUI shortcuts:" in out
        assert "Top-level CLI commands:" in out or "CLI commands: unavailable." in out


def test_help_specific_command() -> None:
    out = handle_help_command("plan")
    assert "/plan" in out
    assert "plan mode" in out.lower()


def test_help_skills() -> None:
    out = handle_help_command("skills")
    assert "Loaded skills:" in out or "No skills loaded." in out or "Skills: unavailable." in out


def test_help_shortcuts() -> None:
    out = handle_help_command("shortcuts")
    assert "Ctrl+C" in out
    assert "Ctrl+L" in out


def test_help_cli() -> None:
    out = handle_help_command("cli")
    assert "Top-level CLI commands:" in out or "CLI commands: unavailable." in out


def test_help_unknown_returns_friendly_error() -> None:
    out = handle_help_command("definitelynotacommand")
    assert "Unknown command" in out
