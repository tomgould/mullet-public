"""Tests for persistent agent memory."""

import json
from pathlib import Path
from mullet.core.agent.memory import AgentMemory


def test_save_and_recall(tmp_path, monkeypatch):
    monkeypatch.setattr("mullet.core.agent.memory.AgentMemory._local_path",
                         tmp_path / "memory.json",
                         raising=False)
    mem = AgentMemory.__new__(AgentMemory)
    mem._knowledge = None
    mem._local_path = tmp_path / "memory.json"
    mem._entries = {}

    mem.save("user_pref", "always use early returns")
    mem.save("env", "Python 3.10 on Ubuntu")

    assert mem.count == 2
    assert mem.recall_all()["user_pref"] == "always use early returns"

    # Verify local persistence — new format stores {"content": ..., "expires_at": ...}
    assert mem._local_path.exists()
    data = json.loads(mem._local_path.read_text())
    assert data["user_pref"]["content"] == "always use early returns"


def test_forget(tmp_path):
    mem = AgentMemory.__new__(AgentMemory)
    mem._knowledge = None
    mem._local_path = tmp_path / "memory.json"
    mem._entries = {
        "a": {"content": "1", "expires_at": None},
        "b": {"content": "2", "expires_at": None},
    }

    mem.forget("a")
    assert "a" not in mem.recall_all()
    assert "b" in mem.recall_all()


def test_build_memory_prompt():
    mem = AgentMemory.__new__(AgentMemory)
    mem._knowledge = None
    mem._local_path = Path("/nonexistent")
    mem._entries = {
        "user": {"content": "Senior developer", "expires_at": None},
        "style": {"content": "Prefers concise output", "expires_at": None},
    }

    prompt = mem.build_memory_prompt()
    assert "Agent Memory" in prompt
    assert "Senior developer" in prompt
    assert "concise output" in prompt


def test_empty_memory_prompt():
    mem = AgentMemory.__new__(AgentMemory)
    mem._knowledge = None
    mem._local_path = Path("/nonexistent")
    mem._entries = {}

    assert mem.build_memory_prompt() == ""


def test_memory_char_limit():
    mem = AgentMemory.__new__(AgentMemory)
    mem._knowledge = None
    mem._local_path = Path("/nonexistent")
    # Create many large entries
    mem._entries = {f"key_{i}": {"content": "x" * 500, "expires_at": None} for i in range(20)}

    prompt = mem.build_memory_prompt()
    # Should be truncated, not all 20 entries
    assert len(prompt) < 5000
    assert "more entries" in prompt


def test_ttl_entry_expires():
    import time
    mem = AgentMemory.__new__(AgentMemory)
    mem._knowledge = None
    mem._local_path = Path("/nonexistent")
    expired = time.time() - 1.0  # already expired
    mem._entries = {
        "live": {"content": "present", "expires_at": None},
        "dead": {"content": "gone", "expires_at": expired},
    }
    recalled = mem.recall_all()
    assert "live" in recalled
    assert "dead" not in recalled


def test_ttl_cleanup_removes_expired(tmp_path):
    import time
    mem = AgentMemory.__new__(AgentMemory)
    mem._knowledge = None
    mem._local_path = tmp_path / "memory.json"
    expired = time.time() - 1.0
    mem._entries = {
        "a": {"content": "keep", "expires_at": None},
        "b": {"content": "remove", "expires_at": expired},
        "c": {"content": "also remove", "expires_at": expired},
    }
    removed = mem.cleanup_expired_memories()
    assert removed == 2
    assert "a" in mem._entries
    assert "b" not in mem._entries
    assert "c" not in mem._entries


def test_save_with_ttl(tmp_path):
    import time
    mem = AgentMemory.__new__(AgentMemory)
    mem._knowledge = None
    mem._local_path = tmp_path / "memory.json"
    mem._entries = {}
    mem.save("temporary", "session note", ttl_seconds=3600)
    entry = mem._entries["temporary"]
    assert entry["content"] == "session note"
    assert entry["expires_at"] is not None
    assert entry["expires_at"] > time.time()
