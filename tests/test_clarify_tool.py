"""Tests for the clarify tool."""

import json
from unittest.mock import MagicMock, patch

from mullet.core.tools.clarify import handle_clarify, set_callback, register, MAX_CHOICES


def test_handle_clarify_with_callback():
    callback = lambda q, c: "yes"
    result = json.loads(handle_clarify("Continue?", _callback=callback))
    assert result["response"] == "yes"
    assert result["question"] == "Continue?"


def test_handle_clarify_with_choices():
    callback = lambda q, c: c[1] if c else "none"
    result = json.loads(handle_clarify(
        "Pick a framework",
        choices=["React", "Vue", "Angular"],
        _callback=callback,
    ))
    assert result["response"] == "Vue"


def test_handle_clarify_no_question():
    result = json.loads(handle_clarify(""))
    assert "error" in result


def test_handle_clarify_max_choices():
    callback = lambda q, c: str(len(c))
    result = json.loads(handle_clarify(
        "Pick",
        choices=["a", "b", "c", "d", "e", "f", "g"],  # 7 choices
        _callback=callback,
    ))
    # Should be trimmed to MAX_CHOICES
    assert result["response"] == str(MAX_CHOICES)


def test_handle_clarify_callback_error():
    def bad_callback(q, c):
        raise RuntimeError("input failed")

    result = json.loads(handle_clarify("test?", _callback=bad_callback))
    assert "error" in result


def test_handle_clarify_no_callback_eof():
    """Non-interactive mode returns graceful fallback."""
    with patch("builtins.input", side_effect=EOFError):
        result = json.loads(handle_clarify("test?"))
        assert "no response" in result["response"]


def test_handle_clarify_cli_choice_selection():
    """CLI mode: typing a number selects that choice."""
    with patch("builtins.input", return_value="2"):
        result = json.loads(handle_clarify("Pick", choices=["A", "B", "C"]))
        assert result["response"] == "B"


def test_handle_clarify_cli_custom_answer():
    """CLI mode: typing text uses it directly."""
    with patch("builtins.input", return_value="custom answer"):
        result = json.loads(handle_clarify("What?", choices=["A", "B"]))
        assert result["response"] == "custom answer"


def test_set_callback():
    """Module-level callback can be set."""
    cb = lambda q, c: "global"
    set_callback(cb)
    result = json.loads(handle_clarify("test?"))
    assert result["response"] == "global"
    set_callback(None)  # Clean up


def test_register():
    registry = MagicMock()
    register(registry)
    registry.register.assert_called_once()
    call_args = registry.register.call_args
    assert call_args[1]["name"] == "clarify" or call_args[0][0] == "clarify"
