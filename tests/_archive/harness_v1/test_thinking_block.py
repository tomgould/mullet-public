"""Tests for the ThinkingBlock TUI widget."""

from __future__ import annotations

import pytest

textual = pytest.importorskip("textual")  # noqa: F841

from mullet.harness.tui.widgets.thinking_block import ThinkingBlock

pytestmark = [pytest.mark.harness, pytest.mark.tui]


class TestThinkingBlock:
    def test_thinking_block_constructs(self) -> None:
        block = ThinkingBlock()
        assert block.collapsed is True
        assert block._content == ""

    def test_thinking_block_initial_content(self) -> None:
        block = ThinkingBlock("initial thought", collapsed=False)
        assert block._content == "initial thought"
        assert block.collapsed is False

    def test_thinking_block_append_chunk(self) -> None:
        block = ThinkingBlock()
        block.append_chunk("hello")
        assert block._content == "hello"
        block.append_chunk(" world")
        assert block._content == "hello world"

    def test_thinking_block_multiple_appends(self) -> None:
        block = ThinkingBlock()
        for chunk in ("a", "b", "c"):
            block.append_chunk(chunk)
        assert block._content == "abc"

    def test_thinking_block_toggle_collapses(self) -> None:
        block = ThinkingBlock()
        assert block.collapsed is True
        block.toggle()
        assert block.collapsed is False
        block.toggle()
        assert block.collapsed is True

    def test_thinking_block_collapsed_render(self) -> None:
        block = ThinkingBlock(collapsed=True)
        rendered = str(block._build_text())
        assert "▸" in rendered
        assert "Thinking..." in rendered

    def test_thinking_block_expanded_render(self) -> None:
        block = ThinkingBlock("some reasoning", collapsed=False)
        rendered = str(block._build_text())
        assert "▾" in rendered
        assert "some reasoning" in rendered
