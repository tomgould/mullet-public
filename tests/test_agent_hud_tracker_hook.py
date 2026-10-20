"""Tests for scripts/hooks/agent_hud_tracker.py"""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

_HOOK = Path(__file__).resolve().parent.parent / "scripts" / "hooks" / "agent_hud_tracker.py"


def _load_hook():
    spec = importlib.util.spec_from_file_location("agent_hud_tracker", _HOOK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(payload: dict, event: str, project_dir: str) -> None:
    hook = _load_hook()
    with (
        patch.object(sys, "stdin", __import__("io").StringIO(json.dumps(payload))),
        patch.dict(os.environ, {"CLAUDE_HOOK_EVENT": event, "CLAUDE_PROJECT_DIR": project_dir}),
    ):
        hook.main()


def _read_agents(project_dir: str) -> list[dict]:
    path = Path(project_dir) / ".mullet" / "state" / "agents.json"
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("agents", [])


class TestAgentHudTracker:
    def test_pre_tooluse_writes_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = {
                "tool_name": "Agent",
                "tool_input": {"description": "explore the codebase", "subagent_type": "Explore"},
            }
            _run(payload, "PreToolUse", tmp)
            agents = _read_agents(tmp)
            assert len(agents) == 1
            assert agents[0]["description"] == "explore the codebase"
            assert agents[0]["model"] == "Explore"
            assert agents[0]["status"] == "running"

    def test_post_tooluse_removes_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = {
                "tool_name": "Agent",
                "tool_input": {"description": "explore the codebase", "subagent_type": "Explore"},
            }
            _run(payload, "PreToolUse", tmp)
            assert len(_read_agents(tmp)) == 1
            _run(payload, "PostToolUse", tmp)
            assert len(_read_agents(tmp)) == 0

    def test_non_agent_tool_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = {"tool_name": "Read", "tool_input": {"file_path": "/tmp/foo.py"}}
            _run(payload, "PreToolUse", tmp)
            assert _read_agents(tmp) == []

    def test_duplicate_pre_tooluse_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = {
                "tool_name": "Agent",
                "tool_input": {"description": "same task", "subagent_type": "general-purpose"},
            }
            _run(payload, "PreToolUse", tmp)
            _run(payload, "PreToolUse", tmp)
            assert len(_read_agents(tmp)) == 1

    def test_multiple_agents_tracked(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(3):
                payload = {
                    "tool_name": "Agent",
                    "tool_input": {"description": f"task {i}", "subagent_type": "Explore"},
                }
                _run(payload, "PreToolUse", tmp)
            assert len(_read_agents(tmp)) == 3

    def test_corrupted_agents_json_handled(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / ".mullet" / "state"
            state_dir.mkdir(parents=True)
            (state_dir / "agents.json").write_text("not json")
            payload = {
                "tool_name": "Agent",
                "tool_input": {"description": "recover", "subagent_type": "Explore"},
            }
            _run(payload, "PreToolUse", tmp)
            agents = _read_agents(tmp)
            assert len(agents) == 1

    def test_invalid_json_input_no_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            hook = _load_hook()
            import io
            with (
                patch.object(sys, "stdin", io.StringIO("not json")),
                patch.dict(os.environ, {"CLAUDE_HOOK_EVENT": "PreToolUse", "CLAUDE_PROJECT_DIR": tmp}),
            ):
                hook.main()  # must not raise
