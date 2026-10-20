"""Tests for the Docker tool."""

import json
import subprocess
from unittest.mock import patch, MagicMock

from mullet.core.tools.docker import docker_tool, _run_docker, _check_blocked_flags, register


def _mock_run(stdout="", stderr="", returncode=0):
    """Create a mock subprocess result."""
    result = MagicMock()
    result.stdout = stdout
    result.stderr = stderr
    result.returncode = returncode
    return result


def test_ps():
    with patch("subprocess.run", return_value=_mock_run(stdout="CONTAINER ID  IMAGE  STATUS  NAMES\nabc123  nginx  Up  web")):
        result = json.loads(docker_tool(action="ps"))
        assert "containers" in result
        assert "nginx" in result["containers"]


def test_images():
    with patch("subprocess.run", return_value=_mock_run(stdout="REPOSITORY  TAG  SIZE\nnginx  latest  100MB")):
        result = json.loads(docker_tool(action="images"))
        assert "images" in result
        assert "nginx" in result["images"]


def test_run_no_target():
    result = json.loads(docker_tool(action="run"))
    assert "error" in result
    assert "requires" in result["error"]


def test_exec_no_target():
    result = json.loads(docker_tool(action="exec"))
    assert "error" in result


def test_exec_no_command():
    result = json.loads(docker_tool(action="exec", target="container1"))
    assert "error" in result
    assert "command" in result["error"]


def test_exec_success():
    with patch("subprocess.run", return_value=_mock_run(stdout="hello from container")):
        result = json.loads(docker_tool(action="exec", target="web", command="echo hello"))
        assert result["output"] == "hello from container"
        assert result["exit_code"] == 0


def test_logs():
    with patch("subprocess.run", return_value=_mock_run(stdout="2024-01-01 Starting server...")):
        result = json.loads(docker_tool(action="logs", target="web"))
        assert "logs" in result
        assert "Starting server" in result["logs"]


def test_logs_no_target():
    result = json.loads(docker_tool(action="logs"))
    assert "error" in result


def test_stop():
    with patch("subprocess.run", return_value=_mock_run(stdout="web")):
        result = json.loads(docker_tool(action="stop", target="web"))
        assert result["stopped"] is True


def test_rm():
    with patch("subprocess.run", return_value=_mock_run(stdout="web")):
        result = json.loads(docker_tool(action="rm", target="web"))
        assert result["removed"] is True


def test_build_no_target():
    result = json.loads(docker_tool(action="build"))
    assert "error" in result


def test_unknown_action():
    result = json.loads(docker_tool(action="restart"))
    assert "error" in result
    assert "Unknown" in result["error"]


def test_docker_not_found():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        output, code = _run_docker(["ps"])
        assert "not found" in output.lower()
        assert code == 1


class TestDockerToolMissingLines:
    def test_check_blocked_flags_detects_blocked(self):
        result = _check_blocked_flags("--privileged")
        assert result is not None
        assert "Blocked flag" in result
        assert "--privileged" in result

    def test_check_blocked_flags_allows_safe(self):
        result = _check_blocked_flags("-d --rm")
        assert result is None

    def test_run_docker_truncation(self):
        long_output = "a" * 30001
        with patch("subprocess.run", return_value=_mock_run(stdout=long_output)):
            output, code = _run_docker(["ps"])
            assert "[truncated]" in output
            assert code == 0

    def test_run_docker_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="docker", timeout=60)):
            output, code = _run_docker(["ps"])
            assert "timed out" in output
            assert code == 1

    def test_ps_error(self):
        with patch("subprocess.run", return_value=_mock_run(stdout="error output", returncode=1)):
            result = json.loads(docker_tool(action="ps"))
            assert "error" in result
            assert result["error"] == "error output"

    def test_images_error(self):
        with patch("subprocess.run", return_value=_mock_run(stdout="error output", returncode=1)):
            result = json.loads(docker_tool(action="images"))
            assert "error" in result
            assert result["error"] == "error output"

    def test_build_no_target(self):
        result = json.loads(docker_tool(action="build"))
        assert "error" in result
        assert "requires" in result["error"]

    def test_build_blocked_flags(self):
        result = json.loads(docker_tool(action="build", target=".", options="--privileged"))
        assert "error" in result
        assert "Blocked flag" in result["error"]

    def test_build_success_with_options(self):
        with patch("subprocess.run", return_value=_mock_run(stdout="build success")):
            result = json.loads(docker_tool(action="build", target=".", options="--no-cache", command="mytag"))
            assert result["output"] == "build success"
            assert result["exit_code"] == 0
            assert result["success"] is True

    def test_run_no_target(self):
        result = json.loads(docker_tool(action="run"))
        assert "error" in result
        assert "requires" in result["error"]

    def test_run_blocked_flags(self):
        result = json.loads(docker_tool(action="run", target="nginx", options="--privileged"))
        assert "error" in result
        assert "Blocked flag" in result["error"]

    def test_run_success_with_options_and_command(self):
        with patch("subprocess.run", return_value=_mock_run(stdout="run success")):
            result = json.loads(docker_tool(action="run", target="nginx", options="-d", command="echo hello"))
            assert result["output"] == "run success"
            assert result["exit_code"] == 0

    def test_stop_no_target(self):
        result = json.loads(docker_tool(action="stop"))
        assert "error" in result
        assert "requires" in result["error"]

    def test_rm_no_target(self):
        result = json.loads(docker_tool(action="rm"))
        assert "error" in result
        assert "requires" in result["error"]

    def test_register(self):
        mock_registry = MagicMock()
        register(mock_registry)
        mock_registry.register.assert_called_once()
