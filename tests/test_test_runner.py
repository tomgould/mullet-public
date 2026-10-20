"""Tests for the test runner tool."""

import json
import os
from mullet.core.tools.test_runner import run_tests_tool, _detect_test_framework

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_detect_pytest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    assert _detect_test_framework() == "pytest"


def test_detect_npm(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "package.json").write_text('{"name": "x"}\n')
    assert _detect_test_framework() == "npm"


def test_detect_fallback_to_pytest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _detect_test_framework() == "pytest"


def test_run_pytest_on_mullet(monkeypatch):
    """Run a real pytest against a known passing test file."""
    monkeypatch.chdir(PROJECT_ROOT)
    result = json.loads(run_tests_tool(json.dumps({
        "path": "tests/test_reasoning.py",
    })))
    assert result["framework"] == "pytest"
    assert result["passed"] > 0
    assert result["all_passed"] is True
    assert result["exit_code"] == 0


def test_run_pytest_with_filter(monkeypatch):
    """Run pytest with -k filter."""
    monkeypatch.chdir(PROJECT_ROOT)
    result = json.loads(run_tests_tool(json.dumps({
        "path": "tests/test_reasoning.py",
        "filter": "test_strip_none",
    })))
    assert result["framework"] == "pytest"
    assert result["passed"] >= 1


def test_run_pytest_verbose(monkeypatch):
    """Verbose mode should include test names."""
    monkeypatch.chdir(PROJECT_ROOT)
    result = json.loads(run_tests_tool(json.dumps({
        "path": "tests/test_reasoning.py",
        "filter": "test_strip_none",
        "verbose": True,
    })))
    assert "test_strip_none" in result["output"]


def test_run_pytest_nonexistent_path(monkeypatch):
    """Nonexistent test path should fail gracefully."""
    monkeypatch.chdir(PROJECT_ROOT)
    result = json.loads(run_tests_tool(json.dumps({
        "path": "tests/nonexistent_test_file.py",
    })))
    assert result["exit_code"] != 0


def test_invalid_json():
    result = json.loads(run_tests_tool("not json"))
    assert "error" in result


import subprocess
from unittest.mock import patch, MagicMock
from mullet.core.tools.test_runner import _run_pytest, _run_npm, register, TEST_RUNNER_SCHEMA


def test_detect_test_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tests").mkdir()
    assert _detect_test_framework() == "pytest"


def test_detect_test_dir_alt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "test").mkdir()
    assert _detect_test_framework() == "pytest"


def test_run_pytest_timeout():
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["pytest"], timeout=120)):
        result = _run_pytest()
    assert result["error"] == "Tests timed out (120s)"


def test_run_pytest_truncation():
    big_output = "x" * 25_000
    mock_result = MagicMock()
    mock_result.stdout = big_output
    mock_result.stderr = ""
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result):
        result = _run_pytest()
    assert "[truncated]" in result["output"]


def test_run_pytest_failure_parsing():
    output = "FAILED tests/test_foo.py::test_bar\n1 passed, 1 failed in 1.23s"
    mock_result = MagicMock()
    mock_result.stdout = output
    mock_result.stderr = ""
    mock_result.returncode = 1
    with patch("subprocess.run", return_value=mock_result):
        result = _run_pytest()
    assert result["failed"] == 1
    assert result["failures"] == ["tests/test_foo.py::test_bar"]


def test_run_npm_file_not_found():
    with patch("subprocess.run", side_effect=FileNotFoundError()):
        result = _run_npm()
    assert result["error"] == "npm not found"


def test_run_npm_timeout():
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["npm"], timeout=120)):
        result = _run_npm()
    assert result["error"] == "Tests timed out (120s)"


def test_run_npm_truncation():
    big_output = "x" * 25_000
    mock_result = MagicMock()
    mock_result.stdout = big_output
    mock_result.stderr = ""
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result):
        result = _run_npm()
    assert "[truncated]" in result["output"]


def test_run_npm_parse_results():
    output = "Tests: 2 failed, 5 passed"
    mock_result = MagicMock()
    mock_result.stdout = output
    mock_result.stderr = ""
    mock_result.returncode = 1
    with patch("subprocess.run", return_value=mock_result):
        result = _run_npm()
    assert result["passed"] == 5
    assert result["failed"] == 2


def test_run_npm_no_parse():
    output = "some random output"
    mock_result = MagicMock()
    mock_result.stdout = output
    mock_result.stderr = ""
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result):
        result = _run_npm()
    assert result["passed"] == 0
    assert result["failed"] == 0


def test_run_npm_with_filter():
    output = "Tests: 5 passed"
    mock_result = MagicMock()
    mock_result.stdout = output
    mock_result.stderr = ""
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result = _run_npm(filter_expr="foo")
    cmd = mock_run.call_args[0][0]
    assert "--testPathPattern" in cmd
    assert "foo" in cmd


def test_run_npm_verbose():
    output = "Tests: 5 passed"
    mock_result = MagicMock()
    mock_result.stdout = output
    mock_result.stderr = ""
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result = _run_npm(verbose=True)
    cmd = mock_run.call_args[0][0]
    assert "--verbose" in cmd


def test_run_tests_tool_unknown_framework():
    with patch("mullet.core.tools.test_runner._detect_test_framework", return_value="unknown"):
        result = json.loads(run_tests_tool(json.dumps({})))
    assert "error" in result
    assert "Unknown framework" in result["error"]


def test_run_tests_tool_kwargs():
    with patch("mullet.core.tools.test_runner._run_pytest", return_value={"framework": "pytest", "passed": 1}):
        result = json.loads(run_tests_tool(path="tests/test_foo.py"))
    assert result["framework"] == "pytest"
    assert result["passed"] == 1


def test_run_tests_tool_npm_framework(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "package.json").write_text('{"name": "x"}\n')
    with patch("mullet.core.tools.test_runner._run_npm", return_value={"framework": "npm", "passed": 1}):
        result = json.loads(run_tests_tool(json.dumps({})))
    assert result["framework"] == "npm"


def test_register():
    registry = MagicMock()
    register(registry)
    registry.register.assert_called_once_with(
        name="run_tests",
        func=run_tests_tool,
        schema=TEST_RUNNER_SCHEMA,
        toolset="testing",
    )
