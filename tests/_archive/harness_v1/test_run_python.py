"""Tests for the RunPython and RunBash harness tools."""

from __future__ import annotations

from unittest.mock import MagicMock


from mullet.harness.tools.run_python import (
    RUN_BASH_SCHEMA,
    RUN_PYTHON_SCHEMA,
    execute_bash,
    execute_python,
    register,
)


class TestSchema:
    def test_schema_valid(self):
        assert RUN_PYTHON_SCHEMA["name"] == "RunPython"
        props = RUN_PYTHON_SCHEMA["input_schema"]["properties"]
        assert "code" in props
        assert "timeout_seconds" in props
        assert RUN_PYTHON_SCHEMA["input_schema"]["required"] == ["code"]

        assert RUN_BASH_SCHEMA["name"] == "RunBash"
        bash_props = RUN_BASH_SCHEMA["input_schema"]["properties"]
        assert "code" in bash_props
        assert "timeout_seconds" in bash_props
        assert RUN_BASH_SCHEMA["input_schema"]["required"] == ["code"]


class TestRunPython:
    def test_run_python_success(self):
        result = execute_python('print("hello")')
        assert result["stdout"] == "hello\n"
        assert result["stderr"] == ""
        assert result["returncode"] == 0
        assert result["timed_out"] is False
        assert isinstance(result["duration_ms"], int)
        assert result["duration_ms"] >= 0

    def test_run_python_stderr(self):
        result = execute_python('import sys; sys.stderr.write("errmsg")')
        assert result["stdout"] == ""
        assert result["stderr"] == "errmsg"
        assert result["returncode"] == 0
        assert result["timed_out"] is False

    def test_run_python_returncode(self):
        result = execute_python('import sys; sys.exit(2)')
        assert result["stdout"] == ""
        assert result["stderr"] == ""
        assert result["returncode"] == 2
        assert result["timed_out"] is False

    def test_run_python_timeout(self):
        result = execute_python("while True: pass", timeout_seconds=1)
        assert result["timed_out"] is True
        assert result["returncode"] == -1
        assert isinstance(result["duration_ms"], int)


class TestRunBash:
    def test_run_bash_success(self):
        result = execute_bash('echo hi')
        assert result["stdout"] == "hi\n"
        assert result["stderr"] == ""
        assert result["returncode"] == 0
        assert result["timed_out"] is False
        assert isinstance(result["duration_ms"], int)
        assert result["duration_ms"] >= 0


class TestRegister:
    def test_register_adds_tools(self):
        registry = MagicMock()
        register(registry)
        assert registry.register.call_count == 2
        calls = registry.register.call_args_list
        names = {call.kwargs["name"] for call in calls}
        assert "RunPython" in names
        assert "RunBash" in names
        for call in calls:
            assert call.kwargs["toolset"] == "execution"
