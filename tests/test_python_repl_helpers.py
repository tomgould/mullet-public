"""Tests for mullet.core.tools.python_repl — blocklist and handler."""

import json
from mullet.core.tools.python_repl import (
    _DANGEROUS_PATTERNS,
    PYTHON_REPL_SCHEMA,
    python_repl_handler,
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestPythonReplSchema:
    def test_is_dict(self):
        assert isinstance(PYTHON_REPL_SCHEMA, dict)

    def test_has_description(self):
        assert "description" in PYTHON_REPL_SCHEMA
        assert isinstance(PYTHON_REPL_SCHEMA["description"], str)

    def test_parameters_has_code(self):
        props = PYTHON_REPL_SCHEMA["parameters"]["properties"]
        assert "code" in props

    def test_code_required(self):
        assert "code" in PYTHON_REPL_SCHEMA["parameters"]["required"]


# ---------------------------------------------------------------------------
# _DANGEROUS_PATTERNS
# ---------------------------------------------------------------------------

class TestDangerousPatterns:
    def test_is_list(self):
        assert isinstance(_DANGEROUS_PATTERNS, list)

    def test_each_entry_is_tuple(self):
        for entry in _DANGEROUS_PATTERNS:
            assert isinstance(entry, tuple)
            assert len(entry) == 2

    def test_labels_are_strings(self):
        for label, _ in _DANGEROUS_PATTERNS:
            assert isinstance(label, str)

    def test_import_os_in_patterns(self):
        labels = [label for label, _ in _DANGEROUS_PATTERNS]
        assert any("import os" in l for l in labels)

    def test_eval_in_patterns(self):
        labels = [label for label, _ in _DANGEROUS_PATTERNS]
        assert any("eval" in l for l in labels)

    def test_exec_in_patterns(self):
        labels = [label for label, _ in _DANGEROUS_PATTERNS]
        assert any("exec" in l for l in labels)


# ---------------------------------------------------------------------------
# python_repl_handler — blocking
# ---------------------------------------------------------------------------

class TestPythonReplHandlerBlocking:
    def _parse(self, result: str) -> dict:
        return json.loads(result)

    def test_import_os_blocked(self):
        result = self._parse(python_repl_handler("import os\nprint(os.getcwd())"))
        assert result["success"] is False
        assert "Blocked" in result["stderr"]

    def test_import_subprocess_blocked(self):
        result = self._parse(python_repl_handler("import subprocess"))
        assert result["success"] is False

    def test_eval_blocked(self):
        result = self._parse(python_repl_handler("eval('1+1')"))
        assert result["success"] is False

    def test_exec_blocked(self):
        result = self._parse(python_repl_handler("exec('x=1')"))
        assert result["success"] is False

    def test_dunder_import_blocked(self):
        result = self._parse(python_repl_handler("__import__('os')"))
        assert result["success"] is False

    def test_subprocess_attr_blocked(self):
        result = self._parse(python_repl_handler("subprocess.run(['ls'])"))
        assert result["success"] is False

    def test_blocked_stderr_names_pattern(self):
        result = self._parse(python_repl_handler("import os"))
        assert "import os" in result["stderr"] or "Blocked" in result["stderr"]

    def test_blocked_return_code_nonzero(self):
        result = self._parse(python_repl_handler("import os"))
        assert result["return_code"] != 0


# ---------------------------------------------------------------------------
# python_repl_handler — execution (safe code)
# ---------------------------------------------------------------------------

class TestPythonReplHandlerExecution:
    def _parse(self, result: str) -> dict:
        return json.loads(result)

    def test_simple_print_succeeds(self):
        result = self._parse(python_repl_handler("print('hello')"))
        assert result["success"] is True
        assert "hello" in result["stdout"]

    def test_returns_json_string(self):
        raw = python_repl_handler("x = 1")
        assert isinstance(raw, str)
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_result_has_required_keys(self):
        result = self._parse(python_repl_handler("x = 1"))
        for key in ("success", "stdout", "stderr", "return_code", "duration_s"):
            assert key in result

    def test_syntax_error_not_success(self):
        result = self._parse(python_repl_handler("def broken(:"))
        assert result["success"] is False

    def test_runtime_error_not_success(self):
        result = self._parse(python_repl_handler("raise ValueError('oops')"))
        assert result["success"] is False

    def test_stderr_captured_on_error(self):
        result = self._parse(python_repl_handler("raise ValueError('test error')"))
        assert "test error" in result["stderr"] or result["return_code"] != 0

    def test_math_output(self):
        result = self._parse(python_repl_handler("print(2 + 2)"))
        assert "4" in result["stdout"]

    def test_empty_code_succeeds(self):
        result = self._parse(python_repl_handler(""))
        assert result["success"] is True
