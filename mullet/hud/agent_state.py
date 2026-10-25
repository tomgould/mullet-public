"""Shared agent state writer for HUD display.

Supports concurrent agents: ``add_agent(id)`` / ``remove_agent(id)`` keep a
merged list of everything currently in flight, so parallel Kimi + local LLM
swarms all surface on the HUD at the same time. The legacy ``write_agent`` /
``write_agents`` / ``clear_agents`` API is preserved for callers that manage
a single slot (the MCP ``mullet_run`` handler writes one agent per invocation).
"""
import json
import logging
import threading
import time
from pathlib import Path

from mullet.core.paths import state_dir

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()


def _state_path() -> Path:
    # User-level absolute path so MCP server (any CWD) and HUD (any session CWD) agree.
    return state_dir() / "state" / "agents.json"


def _read_current() -> list[dict]:
    try:
        path = _state_path()
        if not path.exists():
            return []
        data = json.loads(path.read_text())
        agents = data.get("agents", [])
        return [a for a in agents if isinstance(a, dict)]
    except (json.JSONDecodeError, OSError):
        return []


def _write(agents: list[dict]) -> None:
    try:
        path = _state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "active_count": len(agents),
            "agents": agents,
        }))
    except OSError as _exc:
        logger.warning("agent_state: failed to write state: %s", _exc)


def write_agents(agents: list[dict]) -> None:
    """Atomically replace the full agent list in the HUD state file."""
    with _LOCK:
        _write(agents)


def write_agent(description: str, provider: str, model: str) -> None:
    """Replace the agent list with a single active agent."""
    write_agents([{
        "description": description.replace("\n", " ").strip(),
        "provider": provider,
        "model": model,
        "started_at": time.time(),
    }])


def clear_agents() -> None:
    """Clear all agents from HUD state."""
    write_agents([])


def add_agent(agent_id: str, description: str, provider: str, model: str) -> None:
    """Merge one agent into the currently-running list by id.

    Concurrent calls to multiple providers (Kimi swarm + local LLM + Moonshot)
    all appear at once on the HUD. Pair with :func:`remove_agent` in a
    try/finally at the call site so a crash doesn't leave a stale entry.
    """
    with _LOCK:
        agents = [a for a in _read_current() if a.get("id") != agent_id]
        agents.append({
            "id": agent_id,
            "description": description.replace("\n", " ").strip()[:160],
            "provider": provider,
            "model": model,
            "status": "running",
            "started_at": time.time(),
        })
        _write(agents)


def update_agent_status(agent_id: str, status: str) -> None:
    """Update the status field of a running agent by id. Missing id is a no-op."""
    with _LOCK:
        agents = _read_current()
        for a in agents:
            if a.get("id") == agent_id:
                a["status"] = status
                break
        _write(agents)


def update_agent_model(agent_id: str, provider: str, model: str) -> None:
    """Update the provider/model fields of a running agent by id. Missing id is a no-op.

    Needed when a delegate silently swaps model mid-run (e.g. opencode's
    local -> openrouter -> kimi escalation on failure) — without this, the
    HUD keeps showing the model the run STARTED on, not the one actually
    executing, for the rest of the run's duration.
    """
    with _LOCK:
        agents = _read_current()
        for a in agents:
            if a.get("id") == agent_id:
                a["provider"] = provider
                a["model"] = model
                break
        _write(agents)


def update_agent_tokens(agent_id: str, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
    """Update running token/cost totals for a swarm agent row."""
    with _LOCK:
        agents = _read_current()
        for a in agents:
            if a.get("id") == agent_id:
                a["tokens_in"] = a.get("tokens_in", 0) + input_tokens
                a["tokens_out"] = a.get("tokens_out", 0) + output_tokens
                a["cost_usd"] = a.get("cost_usd", 0.0) + cost_usd
                break
        _write(agents)


def remove_agent(agent_id: str) -> None:
    """Remove one agent from the running list by id. Missing id is a no-op."""
    with _LOCK:
        agents = [a for a in _read_current() if a.get("id") != agent_id]
        _write(agents)


def cleanup_stale_agents(max_age_seconds: int = 1800) -> int:
    """Remove agents older than max_age_seconds. Returns count removed."""
    now = time.time()
    with _LOCK:
        current = _read_current()
        fresh = [
            a for a in current
            if now - float(a.get("started_at", 0)) <= max_age_seconds
        ]
        removed = len(current) - len(fresh)
        if removed:
            _write(fresh)
    return removed
