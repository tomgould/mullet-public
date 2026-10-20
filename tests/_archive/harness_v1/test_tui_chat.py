"""Tests for :mod:`mullet.harness.tui.screens.chat` — main chat surface."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mullet.harness.tui.bridge import EngineBridge
from mullet.harness.tui.screens.chat import ChatScreen


class _StubEngine:
    def run(self, *args, **kwargs):  # pragma: no cover - not exercised here
        return {
            "session_id": "sess-stub",
            "response": "ok",
            "stop_reason": "end_turn",
            "input_tokens": 1,
            "output_tokens": 1,
        }


@pytest.mark.harness
@pytest.mark.tui
class TestChatScreen:
    def test_chat_screen_constructs(self) -> None:
        engine = _StubEngine()
        screen = ChatScreen(engine)
        assert screen.engine is engine
        assert screen.session_id is None
        assert isinstance(screen.bridge, EngineBridge)
        assert screen._owns_bridge is True

    def test_chat_screen_constructs_with_session_id(self) -> None:
        screen = ChatScreen(_StubEngine(), session_id="sess-prev")
        assert screen.session_id == "sess-prev"

    def test_chat_screen_accepts_external_bridge(self) -> None:
        engine = _StubEngine()
        bridge = EngineBridge(engine)
        screen = ChatScreen(engine, bridge=bridge)
        assert screen.bridge is bridge
        assert screen._owns_bridge is False

    def test_chat_screen_compose_yields_log_and_input(self) -> None:
        from textual.widgets import Input, RichLog, Static

        screen = ChatScreen(_StubEngine())
        widgets = list(screen.compose())
        # Expect: RichLog, Static (status bar), Input.
        assert any(isinstance(w, RichLog) for w in widgets)
        assert any(isinstance(w, Input) for w in widgets)
        assert any(isinstance(w, Static) for w in widgets)
        ids = {getattr(w, "id", None) for w in widgets}
        assert "chat_log" in ids
        assert "chat_input" in ids

    def test_render_user_appends_to_log(self) -> None:
        screen = ChatScreen(_StubEngine())
        fake_log = MagicMock()
        screen.query_one = MagicMock(return_value=fake_log)
        screen.render_user("hello world")
        assert fake_log.write.called
        # Two writes: header + body.
        assert fake_log.write.call_count == 2
        body_arg = fake_log.write.call_args_list[-1].args[0]
        assert "hello world" in str(body_arg)

    def test_render_assistant_appends_to_log(self) -> None:
        from rich.markdown import Markdown

        screen = ChatScreen(_StubEngine())
        fake_log = MagicMock()
        screen.query_one = MagicMock(return_value=fake_log)
        screen.render_assistant("# hi\n\nhello")
        assert fake_log.write.call_count == 2
        body_arg = fake_log.write.call_args_list[-1].args[0]
        assert isinstance(body_arg, Markdown)

    def test_render_tool_call_dim(self) -> None:
        from rich.panel import Panel

        screen = ChatScreen(_StubEngine())
        fake_log = MagicMock()
        screen.query_one = MagicMock(return_value=fake_log)
        screen.render_tool_call("Bash", {"cmd": "ls"})
        assert fake_log.write.called
        rendered = fake_log.write.call_args.args[0]
        assert isinstance(rendered, Panel)
        assert "Bash" in str(rendered.title)

    def test_render_error_red(self) -> None:
        from rich.text import Text

        screen = ChatScreen(_StubEngine())
        fake_log = MagicMock()
        screen.query_one = MagicMock(return_value=fake_log)
        screen.render_error("something broke")
        assert fake_log.write.called
        rendered = fake_log.write.call_args.args[0]
        assert isinstance(rendered, Text)
        assert "something broke" in str(rendered)
        assert "red" in rendered.style
