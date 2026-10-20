"""Polish tests for TUI markdown rendering, tool cards, and cost tracking."""

from __future__ import annotations

import pytest

textual = pytest.importorskip("textual")  # noqa: F841

from rich.markdown import Markdown
from rich.panel import Panel

from mullet.harness.tui.bridge import estimate_cost
from mullet.harness.tui.screens.chat import ChatScreen
from mullet.harness.tui.widgets.status_bar import StatusBar


class _StubEngine:
    def run(self, *args, **kwargs):  # pragma: no cover - not exercised here
        return {
            "session_id": "sess-stub",
            "response": "ok",
            "stop_reason": "end_turn",
            "input_tokens": 1,
            "output_tokens": 1,
        }


pytestmark = [pytest.mark.harness, pytest.mark.tui]


# ------------------------------------------------------------------ ChatScreen ---


def test_render_tool_card_returns_panel() -> None:
    screen = ChatScreen(_StubEngine())
    panel = screen.render_tool_card("Read", {"file_path": "auth.py"})
    assert isinstance(panel, Panel)


def test_render_tool_card_truncates_long_output() -> None:
    screen = ChatScreen(_StubEngine())
    long_output = "\n".join(f"line {i}" for i in range(20))
    panel = screen.render_tool_card("Read", {}, output=long_output)
    text = str(panel.renderable)
    assert "[...]" in text
    assert text.count("\n") <= 13  # 12 lines + [...] marker


def test_render_tool_card_with_error_styled_red() -> None:
    screen = ChatScreen(_StubEngine())
    panel = screen.render_tool_card("Bash", {}, error="Permission denied")
    rendered = str(panel.renderable)
    assert "Permission denied" in rendered
    # Border or text should reference red styling.
    assert "red" in str(panel.border_style).lower()


def test_render_assistant_markdown_returns_markdown_object() -> None:
    screen = ChatScreen(_StubEngine())
    md = screen.render_assistant_markdown("# Hello\n\nworld")
    assert isinstance(md, Markdown)


# ------------------------------------------------------------------ StatusBar ---


def test_status_bar_renders_cost_when_set() -> None:
    bar = StatusBar()
    bar.update_usage(tokens_in=1200, tokens_out=800, cost_usd=0.003, latency_ms=245)
    text = str(bar._build_text())
    assert "$0.003" in text
    assert "1.2k" in text
    assert "800" in text
    assert "245ms" in text


def test_status_bar_renders_dash_when_no_cost() -> None:
    bar = StatusBar()
    text = str(bar._build_text())
    assert "—" in text
    assert "$" not in text


# ------------------------------------------------------------------ estimate_cost ---


def test_estimate_cost_returns_correct_value() -> None:
    cost = estimate_cost("kimi-k2.6", 1_000_000, 500_000)
    # kimi-k2.6 pricing: (0.00095, 0.004) per 1K tokens
    expected = (1_000_000 * 0.00095 + 500_000 * 0.004) / 1000.0
    assert cost == pytest.approx(expected)


def test_estimate_cost_handles_unknown_model_safely() -> None:
    cost = estimate_cost("unknown-model-xyz", 1000, 1000)
    assert cost == 0.0
