"""Tests for the Mullet harness TUI widgets.

These tests are deliberately limited to construction- and render-level
checks.  Driving a full Textual ``App`` via ``App.run_test()`` is out of
scope here — we only verify the units in isolation.
"""

from __future__ import annotations

import pytest

textual = pytest.importorskip("textual")  # noqa: F841

from mullet.harness.tui.screens.permission_modal import PermissionModal
from mullet.harness.tui.widgets.command_picker import CommandPicker
from mullet.harness.tui.widgets.status_bar import StatusBar

pytestmark = [pytest.mark.harness, pytest.mark.tui]


# ---------------------------------------------------------------- StatusBar ---


def test_status_bar_initial_render() -> None:
    """A fresh StatusBar renders a non-empty placeholder line."""
    bar = StatusBar()
    rendered = str(bar._build_text())
    assert rendered  # non-empty
    # Seven segments, all placeholders em-dash.
    assert rendered.count("—") == 7


def test_status_bar_update_status_renders_all_fields() -> None:
    """update_status with all kwargs surfaces every value in the rendered text."""
    bar = StatusBar()
    bar.update_status(
        profile="kimi",
        model="kimi-k2.6",
        mode="default",
        budget_pct=0.21,
        session_id="sess-20260507-120000-abc123",
    )
    rendered = str(bar._build_text())
    assert "kimi/kimi-k2.6" in rendered
    assert "default" in rendered
    assert "5h:21%" in rendered
    # Last six chars of session id.
    assert "sess-abc123" in rendered


def test_status_bar_handles_missing_fields() -> None:
    """Unset fields render as the em-dash placeholder."""
    bar = StatusBar()
    bar.update_status(profile="kimi", model="kimi-k2.6")
    rendered = str(bar._build_text())
    # Profile/model present.
    assert "kimi/kimi-k2.6" in rendered
    # Mode, budget, tokens, cost, latency, session unset → em-dash.
    assert rendered.count("—") == 6


def test_status_bar_clamps_budget_pct_above_one() -> None:
    """Out-of-range budget percentages are clamped to 0..100."""
    bar = StatusBar()
    bar.update_status(budget_pct=1.5)
    rendered = str(bar._build_text())
    assert "5h:100%" in rendered


def test_status_bar_clamps_budget_pct_below_zero() -> None:
    """Negative budget percentages are clamped to zero."""
    bar = StatusBar()
    bar.update_status(budget_pct=-0.2)
    rendered = str(bar._build_text())
    assert "5h:0%" in rendered


# ----------------------------------------------------------- CommandPicker ---


def test_command_picker_takes_commands_list() -> None:
    """CommandPicker stores its (name, description) tuples verbatim."""
    picker = CommandPicker(
        [("commit", "Make a commit"), ("plan", "Plan it")],
    )
    assert picker.commands == [("commit", "Make a commit"), ("plan", "Plan it")]


def test_command_picker_accepts_empty_list() -> None:
    """An empty command list is allowed and stored."""
    picker = CommandPicker([])
    assert picker.commands == []


def test_command_picker_coerces_non_string_inputs() -> None:
    """Non-string inputs are coerced via str() — defensive against bad data."""
    picker = CommandPicker([(123, 456)])  # type: ignore[list-item]
    assert picker.commands == [("123", "456")]


# ---------------------------------------------------------- PermissionModal ---


def test_permission_modal_constructs() -> None:
    """PermissionModal can be constructed with the documented fields."""
    modal = PermissionModal("Bash", "echo hi", "default")
    assert modal.tool_name == "Bash"
    assert modal.args_preview == "echo hi"
    assert modal.mode == "default"


def test_permission_modal_coerces_non_string_args() -> None:
    """Modal coerces non-string fields without raising."""
    modal = PermissionModal("Bash", 12345, None)  # type: ignore[arg-type]
    assert modal.tool_name == "Bash"
    assert modal.args_preview == "12345"
    assert modal.mode == "None"
