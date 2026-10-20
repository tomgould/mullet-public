"""Gap coverage tests for mullet/core/agent/approval.py."""

from unittest.mock import patch
from mullet.core.agent.approval import cli_approval_callback, _normalize


def test_cli_approval_yes():
    with patch("builtins.input", return_value="y"):
        assert cli_approval_callback("rm -rf /", ["recursive delete"]) is True


def test_cli_approval_yes_full():
    with patch("builtins.input", return_value="yes"):
        assert cli_approval_callback("rm -rf /", ["recursive delete"]) is True


def test_cli_approval_no():
    with patch("builtins.input", return_value="n"):
        assert cli_approval_callback("rm -rf /", ["recursive delete"]) is False


def test_cli_approval_empty():
    with patch("builtins.input", return_value=""):
        assert cli_approval_callback("rm -rf /", ["recursive delete"]) is False


def test_cli_approval_eof():
    with patch("builtins.input", side_effect=EOFError()):
        assert cli_approval_callback("rm -rf /", ["recursive delete"]) is False


def test_cli_approval_keyboard_interrupt():
    with patch("builtins.input", side_effect=KeyboardInterrupt()):
        assert cli_approval_callback("rm -rf /", ["recursive delete"]) is False


def test_normalize_skips_format_chars():
    # Cf category char (LEFT-TO-RIGHT MARK) should be stripped
    command = "rm \u200e-rf /"
    result = _normalize(command)
    assert "\u200e" not in result
    assert "rm" in result
    assert "-rf" in result


def test_normalize_skips_cjk_chars():
    # CJK char (Lo category) should be skipped
    command = "rm \u6c49-rf /"
    result = _normalize(command)
    assert "\u6c49" not in result
    assert "rm" in result
    assert "-rf" in result
