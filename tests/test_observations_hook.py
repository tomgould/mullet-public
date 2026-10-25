"""Tests for mullet/hooks/observation_capture.py, through its scripts/hooks shim.

Invokes the hook as a subprocess to mirror real PostToolUse execution.
"""
import json
import os
import subprocess
import sys
from pathlib import Path


HOOK = Path(__file__).resolve().parent.parent / "scripts" / "hooks" / "mullet_observation_capture.py"


def _run_hook(payload: dict, tmp_path: Path, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
        "MULLET_OBSERVATIONS_BASE": str(tmp_path),
    }
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
    )


def _make_payload(session_id="test-sess", tool="Read", tool_input=None, tool_response="ok"):
    return {
        "session_id": session_id,
        "tool_name": tool,
        "tool_input": tool_input or {"file": "/tmp/foo.py"},
        "tool_response": tool_response,
    }


def test_hook_exits_zero(tmp_path):
    result = _run_hook(_make_payload(), tmp_path)
    assert result.returncode == 0


def test_hook_writes_jsonl(tmp_path):
    _run_hook(_make_payload(session_id="hook-write"), tmp_path)
    path = tmp_path / ".mullet" / "observations" / "hook-write.jsonl"
    assert path.exists(), f"Expected {path} to exist"
    data = json.loads(path.read_text().strip())
    assert data["tool_name"] == "Read"
    assert data["session_id"] == "hook-write"


def test_hook_dict_response(tmp_path):
    payload = _make_payload(
        session_id="dict-resp",
        tool_response={"result": "ok", "lines": 42},
    )
    result = _run_hook(payload, tmp_path)
    assert result.returncode == 0
    path = tmp_path / ".mullet" / "observations" / "dict-resp.jsonl"
    assert path.exists()
    d = json.loads(path.read_text().strip())
    assert '"result"' in d["tool_output_preview"] or "result" in d["tool_output_preview"]


def test_hook_disable_env(tmp_path):
    result = _run_hook(_make_payload(session_id="disabled"), tmp_path,
                       env_extra={"MULLET_DISABLE_OBSERVATIONS": "1"})
    assert result.returncode == 0
    path = tmp_path / ".mullet" / "observations" / "disabled.jsonl"
    assert not path.exists(), "File should not be written when disabled"


def test_hook_malformed_json_exits_zero(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent)}
    result = subprocess.run(
        [sys.executable, str(HOOK)],
        input="NOT VALID JSON {{{{",
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
    )
    assert result.returncode == 0


def test_hook_empty_stdin_exits_zero(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent)}
    result = subprocess.run(
        [sys.executable, str(HOOK)],
        input="",
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
    )
    assert result.returncode == 0


def test_hook_tool_input_not_dict(tmp_path):
    payload = {
        "session_id": "non-dict-input",
        "tool_name": "Bash",
        "tool_input": "ls -la",
        "tool_response": "file1\nfile2",
    }
    result = _run_hook(payload, tmp_path)
    assert result.returncode == 0
    path = tmp_path / ".mullet" / "observations" / "non-dict-input.jsonl"
    assert path.exists()
