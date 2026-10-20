"""Tests for Python REPL tool."""
import json
from mullet.core.tools.python_repl import python_repl_handler


def test_simple_execution():
    result = json.loads(python_repl_handler("print('hello')"))
    assert result["success"] is True
    assert "hello" in result["stdout"]


def test_syntax_error():
    result = json.loads(python_repl_handler("def f("))
    assert result["success"] is False
    assert result["return_code"] != 0


def test_dangerous_pattern_blocked():
    # Test that dangerous patterns are rejected
    result = json.loads(python_repl_handler("import subprocess; subprocess.call(['ls'])"))
    assert result["success"] is False
    assert "Blocked pattern" in result["stderr"]


def test_timeout():
    result = json.loads(python_repl_handler("import time; time.sleep(10)", timeout=1))
    assert result["success"] is False
    assert "timed out" in result["stderr"]


def test_blocks_import_os():
    result = json.loads(python_repl_handler("import os\nprint(os.getcwd())"))
    assert result["success"] is False
    assert "Blocked" in result["stderr"]


def test_blocks_import_shutil():
    result = json.loads(python_repl_handler("import shutil"))
    assert result["success"] is False
    assert "Blocked" in result["stderr"]


def test_blocks_eval():
    result = json.loads(python_repl_handler("eval('1+1')"))
    assert result["success"] is False
    assert "Blocked" in result["stderr"]


def test_blocks_exec():
    result = json.loads(python_repl_handler("exec('x=1')"))
    assert result["success"] is False
    assert "Blocked" in result["stderr"]


def test_blocks_open_write():
    result = json.loads(python_repl_handler("open('/tmp/test.txt', 'w')"))
    assert result["success"] is False
    assert "Blocked" in result["stderr"]


def test_blocks_dunder_import():
    result = json.loads(python_repl_handler("__import__('sys')"))
    assert result["success"] is False
    assert "Blocked" in result["stderr"]


def test_arithmetic_expression():
    result = json.loads(python_repl_handler("print(2 + 2)"))
    assert result["success"] is True
    assert "4" in result["stdout"]
    assert result["return_code"] == 0


def test_stderr_captured():
    result = json.loads(python_repl_handler("import sys; print('err', file=sys.stderr)"))
    assert result["success"] is True
    assert "err" in result["stderr"]


def test_return_code_nonzero_on_error():
    result = json.loads(python_repl_handler("raise ValueError('bad')"))
    assert result["success"] is False
    assert result["return_code"] != 0


def test_multiline_code():
    code = "x = 0\nfor i in range(5):\n    x += i\nprint(x)"
    result = json.loads(python_repl_handler(code))
    assert result["success"] is True
    assert "10" in result["stdout"]


def test_result_has_required_keys():
    result = json.loads(python_repl_handler("pass"))
    for key in ("success", "stdout", "stderr", "return_code", "duration_s"):
        assert key in result


def test_register_calls_registry():
    from mullet.core.tools.python_repl import register
    from unittest.mock import MagicMock
    registry = MagicMock()
    register(registry)
    registry.register.assert_called_once()
    call_kwargs = registry.register.call_args[1]
    assert call_kwargs.get("name") == "python_repl"
