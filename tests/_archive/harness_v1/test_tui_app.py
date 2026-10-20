"""Tests for the Mullet harness TUI application shell.

We avoid running the app under ``App.run_test()`` here — instead we
construct ``MulletTUI`` and inspect its declared widgets via ``compose()``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

textual = pytest.importorskip("textual")  # noqa: F841

from textual.widgets import Footer, Header, Input, RichLog

from mullet.harness.tui.app import SLASH_COMMANDS, MulletTUI
from mullet.harness.tui.widgets.status_bar import StatusBar

pytestmark = [pytest.mark.harness, pytest.mark.tui]


# ---------------------------------------------------------------- helpers ---


class _StubEngine:
    """Minimal HarnessEngine stand-in for tests."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def run(self, prompt, *, session_id=None, provider="kimi", model=None, **_):
        self.calls.append((prompt, session_id, provider, model))
        return {
            "session_id": session_id or "sess-test",
            "response": f"echo: {prompt}",
            "stop_reason": "end_turn",
            "input_tokens": 1,
            "output_tokens": 1,
        }

    def list_sessions(self, *_, **__):  # pragma: no cover - covered indirectly
        return []


# ------------------------------------------------------------ construction ---


def test_app_constructs_with_default_engine() -> None:
    """MulletTUI() builds with a stubbed HarnessEngine."""
    sentinel = SimpleNamespace(name="stub-engine")
    target = "mullet.harness.engine.loop.HarnessEngine"
    with patch(target, return_value=sentinel):
        app = MulletTUI()
    assert app.engine is sentinel
    assert app.session_id is None
    assert app.current_provider == "kimi"
    assert app.current_model is None
    assert app.permission_mode == "default"


def test_app_constructs_with_explicit_engine() -> None:
    """An explicit engine arg bypasses the default lookup."""
    engine = _StubEngine()
    app = MulletTUI(engine=engine)
    assert app.engine is engine


# ----------------------------------------------------------------- compose ---


def test_app_compose_yields_widgets() -> None:
    """compose() yields Header, RichLog, StatusBar, Input, Footer in order."""
    engine = _StubEngine()
    app = MulletTUI(engine=engine)

    yielded = list(app.compose())
    type_names = [type(w).__name__ for w in yielded]

    assert "Header" in type_names
    assert "RichLog" in type_names
    assert "StatusBar" in type_names
    assert "Input" in type_names
    assert "Footer" in type_names

    # And concrete instances of the expected classes.
    assert any(isinstance(w, Header) for w in yielded)
    assert any(isinstance(w, RichLog) for w in yielded)
    assert any(isinstance(w, StatusBar) for w in yielded)
    assert any(isinstance(w, Input) for w in yielded)
    assert any(isinstance(w, Footer) for w in yielded)


def test_app_input_has_expected_id() -> None:
    """The prompt input is identified as ``#prompt_input``."""
    engine = _StubEngine()
    app = MulletTUI(engine=engine)
    inputs = [w for w in app.compose() if isinstance(w, Input)]
    assert inputs and inputs[0].id == "prompt_input"


def test_slash_commands_catalogue_non_empty() -> None:
    """The slash-command catalogue is exposed for the picker."""
    assert SLASH_COMMANDS, "SLASH_COMMANDS should not be empty"
    names = [n for n, _ in SLASH_COMMANDS]
    for required in ("quit", "clear", "help", "skill"):
        assert required in names


def test_app_bindings_expose_core_shortcuts() -> None:
    """The bindings list covers quit/clear/resume/permissions/model."""
    engine = _StubEngine()
    app = MulletTUI(engine=engine)
    binding_keys = {b.key for b in app.BINDINGS}
    for key in ("ctrl+c", "ctrl+l", "ctrl+r", "ctrl+p", "ctrl+m"):
        assert key in binding_keys
