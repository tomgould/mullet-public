"""Tests for the TUI theme system."""

from __future__ import annotations

import os

import pytest

from mullet.harness.tui.themes import BUILTIN_THEMES, list_themes, load_theme


def test_load_dark_theme() -> None:
    theme = load_theme("dark")
    assert theme.name == "dark"
    assert theme.bg == "#1e1e2e"


def test_load_unknown_falls_back_to_dark() -> None:
    with pytest.warns(UserWarning, match="Unknown theme"):
        theme = load_theme("nonexistent")
    assert theme.name == "dark"


def test_list_themes_returns_5() -> None:
    names = list_themes()
    assert len(names) == 5
    assert set(names) == set(BUILTIN_THEMES.keys())


def test_theme_dataclass_has_all_colors() -> None:
    theme = BUILTIN_THEMES["dark"]
    assert hasattr(theme, "bg")
    assert hasattr(theme, "fg")
    assert hasattr(theme, "accent")
    assert hasattr(theme, "user_color")
    assert hasattr(theme, "assistant_color")
    assert hasattr(theme, "tool_color")
    assert hasattr(theme, "error_color")
    assert hasattr(theme, "dim_color")


def test_theme_command_use_persists_to_config(tmp_path) -> None:
    from mullet.harness.commands.theme import handle_theme_command
    from mullet.harness.config import HarnessConfig, load_harness_config, save_harness_config

    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        config = HarnessConfig(theme="dark")
        save_harness_config(config)

        result = handle_theme_command("use light")
        assert "Theme set to 'light'" in result

        config = load_harness_config()
        assert config.theme == "light"
    finally:
        os.chdir(old_cwd)


def test_theme_command_list_returns_names() -> None:
    from mullet.harness.commands.theme import handle_theme_command

    result = handle_theme_command("list")
    for name in list_themes():
        assert name in result
