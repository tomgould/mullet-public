"""Tests for mullet.core.tools.test_runner — pure helpers and schema."""

import json
from mullet.core.tools.test_runner import (
    MAX_OUTPUT_CHARS,
    TEST_RUNNER_SCHEMA,
    _detect_test_framework,
    run_tests_tool,
)


# ---------------------------------------------------------------------------
# Constants and schema
# ---------------------------------------------------------------------------

class TestConstants:
    def test_max_output_chars_is_int(self):
        assert isinstance(MAX_OUTPUT_CHARS, int)
        assert MAX_OUTPUT_CHARS > 0

    def test_schema_is_dict(self):
        assert isinstance(TEST_RUNNER_SCHEMA, dict)

    def test_schema_has_description(self):
        assert "description" in TEST_RUNNER_SCHEMA

    def test_schema_parameters_has_path(self):
        props = TEST_RUNNER_SCHEMA["parameters"]["properties"]
        assert "path" in props

    def test_schema_parameters_has_filter(self):
        props = TEST_RUNNER_SCHEMA["parameters"]["properties"]
        assert "filter" in props

    def test_schema_parameters_has_verbose(self):
        props = TEST_RUNNER_SCHEMA["parameters"]["properties"]
        assert "verbose" in props


# ---------------------------------------------------------------------------
# _detect_test_framework
# ---------------------------------------------------------------------------

class TestDetectTestFramework:
    def test_returns_string(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _detect_test_framework()
        assert isinstance(result, str)

    def test_pyproject_toml_gives_pytest(self, tmp_path, monkeypatch):
        (tmp_path / "pyproject.toml").write_text("[tool.pytest]")
        monkeypatch.chdir(tmp_path)
        assert _detect_test_framework() == "pytest"

    def test_setup_py_gives_pytest(self, tmp_path, monkeypatch):
        (tmp_path / "setup.py").write_text("from setuptools import setup")
        monkeypatch.chdir(tmp_path)
        assert _detect_test_framework() == "pytest"

    def test_package_json_gives_npm(self, tmp_path, monkeypatch):
        # Only package.json without pyproject.toml or setup.py
        (tmp_path / "package.json").write_text("{}")
        monkeypatch.chdir(tmp_path)
        assert _detect_test_framework() == "npm"

    def test_no_markers_defaults_to_pytest(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _detect_test_framework()
        assert result == "pytest"


# ---------------------------------------------------------------------------
# run_tests_tool — argument parsing
# ---------------------------------------------------------------------------

class TestRunTestsToolArgParsing:
    def test_invalid_json_returns_error(self):
        result = json.loads(run_tests_tool("not-valid-json{"))
        assert "error" in result

    def test_empty_string_runs_without_crash(self):
        result = run_tests_tool("")
        assert isinstance(result, str)
        data = json.loads(result)
        assert "framework" in data or "error" in data

    def test_kwargs_accepted(self):
        result = run_tests_tool(path="tests/nonexistent_path_xyz")
        assert isinstance(result, str)
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_result_has_passed_key(self):
        result = json.loads(run_tests_tool(""))
        assert "passed" in result or "error" in result

    def test_result_has_framework_key(self):
        result = json.loads(run_tests_tool(""))
        assert "framework" in result or "error" in result
