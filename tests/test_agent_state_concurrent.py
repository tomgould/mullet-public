"""Tests for the concurrent add_agent / remove_agent API in mullet.hud.agent_state.

These tests verify the merged-list semantics: parallel LLM calls (Kimi +
local + Moonshot) all appear at once on the HUD, and each drops off cleanly
when its call finishes.
"""
import json

from mullet.hud import agent_state


def test_add_agent_populates_single_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_state, "_state_path", lambda: tmp_path / "agents.json")
    agent_state.add_agent("abc", "port transcript.ts", "kimi", "kimi-k2.6")
    data = json.loads((tmp_path / "agents.json").read_text())
    assert data["active_count"] == 1
    assert data["agents"][0]["id"] == "abc"
    assert data["agents"][0]["provider"] == "kimi"
    assert data["agents"][0]["model"] == "kimi-k2.6"
    assert data["agents"][0]["description"] == "port transcript.ts"


def test_add_multiple_agents_all_tracked(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_state, "_state_path", lambda: tmp_path / "agents.json")
    agent_state.add_agent("a1", "summarise readme", "local", "qwen3.5-9b")
    agent_state.add_agent("a2", "port transcript", "kimi", "kimi-k2.6")
    agent_state.add_agent("a3", "draft roadmap", "gemini", "gemini-2.5-flash")
    data = json.loads((tmp_path / "agents.json").read_text())
    assert data["active_count"] == 3
    providers = sorted(a["provider"] for a in data["agents"])
    assert providers == ["gemini", "kimi", "local"]


def test_remove_agent_preserves_others(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_state, "_state_path", lambda: tmp_path / "agents.json")
    agent_state.add_agent("a1", "first", "kimi", "kimi-k2.6")
    agent_state.add_agent("a2", "second", "local", "qwen3.5-9b")
    agent_state.remove_agent("a1")
    data = json.loads((tmp_path / "agents.json").read_text())
    assert data["active_count"] == 1
    assert data["agents"][0]["id"] == "a2"


def test_remove_missing_agent_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_state, "_state_path", lambda: tmp_path / "agents.json")
    agent_state.add_agent("a1", "only", "kimi", "kimi-k2.6")
    agent_state.remove_agent("nonexistent")
    data = json.loads((tmp_path / "agents.json").read_text())
    assert data["active_count"] == 1


def test_add_same_id_replaces_entry(tmp_path, monkeypatch):
    """Re-adding the same id updates instead of duplicating — safe for retries."""
    monkeypatch.setattr(agent_state, "_state_path", lambda: tmp_path / "agents.json")
    agent_state.add_agent("same", "first attempt", "kimi", "kimi-k2.6")
    agent_state.add_agent("same", "retried attempt", "kimi", "kimi-k2.6")
    data = json.loads((tmp_path / "agents.json").read_text())
    assert data["active_count"] == 1
    assert data["agents"][0]["description"] == "retried attempt"


def test_description_newlines_collapsed(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_state, "_state_path", lambda: tmp_path / "agents.json")
    agent_state.add_agent("a1", "line one\nline two\nline three", "kimi", "kimi-k2.6")
    data = json.loads((tmp_path / "agents.json").read_text())
    assert "\n" not in data["agents"][0]["description"]


def test_description_truncated_to_160(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_state, "_state_path", lambda: tmp_path / "agents.json")
    long_desc = "x" * 300
    agent_state.add_agent("a1", long_desc, "kimi", "kimi-k2.6")
    data = json.loads((tmp_path / "agents.json").read_text())
    assert len(data["agents"][0]["description"]) == 160


def test_existing_non_id_entries_survive_new_add(tmp_path, monkeypatch):
    """Legacy single-slot write_agent entries (no id) are preserved on add."""
    monkeypatch.setattr(agent_state, "_state_path", lambda: tmp_path / "agents.json")
    agent_state.write_agents([
        {"description": "legacy", "provider": "kimi", "model": "kimi-k2.6"},
    ])
    agent_state.add_agent("new", "new call", "local", "qwen3.5-9b")
    data = json.loads((tmp_path / "agents.json").read_text())
    assert data["active_count"] == 2


def test_cleanup_stale_agents_removes_old_entries(tmp_path, monkeypatch):
    """cleanup_stale_agents removes entries older than max_age_seconds."""
    import time as _time
    monkeypatch.setattr(agent_state, "_state_path", lambda: tmp_path / "agents.json")
    # Add an agent with a very old timestamp
    agent_state.write_agents([
        {"id": "stale-1", "description": "old", "provider": "kimi", "model": "kimi-k2.6",
         "started_at": _time.time() - 9999},
    ])
    removed = agent_state.cleanup_stale_agents(max_age_seconds=60)
    assert removed == 1
    import json
    data = json.loads((tmp_path / "agents.json").read_text())
    assert data["active_count"] == 0


def test_cleanup_stale_agents_keeps_fresh_entries(tmp_path, monkeypatch):
    """cleanup_stale_agents keeps entries within max_age_seconds."""
    import time as _time
    monkeypatch.setattr(agent_state, "_state_path", lambda: tmp_path / "agents.json")
    agent_state.write_agents([
        {"id": "fresh-1", "description": "recent", "provider": "local", "model": "qwen",
         "started_at": _time.time() - 10},
    ])
    removed = agent_state.cleanup_stale_agents(max_age_seconds=60)
    assert removed == 0


def test_cleanup_stale_agents_empty_state(tmp_path, monkeypatch):
    """cleanup_stale_agents on empty state returns 0."""
    monkeypatch.setattr(agent_state, "_state_path", lambda: tmp_path / "agents.json")
    removed = agent_state.cleanup_stale_agents()
    assert removed == 0
