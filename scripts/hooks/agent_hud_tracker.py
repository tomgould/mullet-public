#!/usr/bin/env python3
"""HUD tracker for Claude Code Agent tool invocations.

Wired into PreToolUse (Agent) to write the agent to agents.json,
and PostToolUse (Agent) to remove it. This surfaces subagents spawned
via the Agent tool on the HUD the same way mullet_run delegates do.
"""
import hashlib
import json
import os
import sys
from pathlib import Path


def _agents_path() -> Path:
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", ".")
    return Path(project_dir) / ".mullet" / "state" / "agents.json"


def _read() -> list[dict]:
    path = _agents_path()
    try:
        if path.exists():
            data = json.loads(path.read_text())
            return [a for a in data.get("agents", []) if isinstance(a, dict)]
    except Exception:
        pass
    return []


def _write(agents: list[dict]) -> None:
    path = _agents_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"active_count": len(agents), "agents": agents}))


def main() -> None:
    hook_event = os.environ.get("CLAUDE_HOOK_EVENT", "")
    try:
        payload = json.loads(sys.stdin.read())
    except Exception:
        return

    tool_name = payload.get("tool_name") or payload.get("tool", "")
    if tool_name != "Agent":
        return

    tool_input = payload.get("tool_input") or {}
    description = tool_input.get("description", "subagent")
    subagent_type = tool_input.get("subagent_type", "agent")

    # Stable ID derived from description so PreToolUse and PostToolUse pair up
    agent_id = "cc-" + hashlib.md5(description.encode()).hexdigest()[:10]

    if hook_event == "PreToolUse":
        agents = [a for a in _read() if a.get("id") != agent_id]
        agents.append({
            "id": agent_id,
            "description": description[:160],
            "provider": "claude-code",
            "model": subagent_type,
            "status": "running",
        })
        _write(agents)
    elif hook_event == "PostToolUse":
        agents = [a for a in _read() if a.get("id") != agent_id]
        _write(agents)


if __name__ == "__main__":
    main()
