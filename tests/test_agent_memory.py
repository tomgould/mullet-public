"""Tests for AgentMemory knowledge manager and edge-case paths."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mullet.core.agent.memory import AgentMemory


# ---------------------------------------------------------------------------
# KnowledgeManager load paths
# ---------------------------------------------------------------------------

def test_load_from_knowledge_manager_success(monkeypatch, tmp_path):
    """Loading via KnowledgeManager populates entries."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    km = MagicMock()
    km.memory.list.return_value = [
        {"key": "pref", "content": "dark mode"},
        {"key": "env", "content": "Python 3.12"},
    ]

    mem = AgentMemory(knowledge=km)
    assert mem.recall_all() == {"pref": "dark mode", "env": "Python 3.12"}
    km.memory.list.assert_called_once_with(namespace="agent_memory")


def test_load_from_knowledge_manager_empty(monkeypatch, tmp_path):
    """Empty KnowledgeManager results in empty memory."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    km = MagicMock()
    km.memory.list.return_value = []

    mem = AgentMemory(knowledge=km)
    assert mem.recall_all() == {}


def test_load_from_knowledge_manager_exception(monkeypatch, tmp_path):
    """Exception during KnowledgeManager load falls back to local file."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    km = MagicMock()
    km.memory.list.side_effect = RuntimeError("trove down")

    # Pre-seed local file so fallback has data
    local = tmp_path / ".mullet" / "memory.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps({"local_key": {"content": "local_val", "expires_at": None}}))

    mem = AgentMemory(knowledge=km)
    assert mem.recall_all() == {"local_key": "local_val"}


def test_load_local_file_exception(monkeypatch, tmp_path):
    """Exception during local file load leaves entries empty."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    local = tmp_path / ".mullet" / "memory.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text("not json")

    mem = AgentMemory()
    assert mem.recall_all() == {}


# ---------------------------------------------------------------------------
# KnowledgeManager save / forget paths
# ---------------------------------------------------------------------------

def test_save_to_knowledge_manager_success(monkeypatch, tmp_path):
    """Saving forwards to KnowledgeManager.remember."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    km = MagicMock()
    mem = AgentMemory(knowledge=km)
    mem.save("key1", "value1")

    km.remember.assert_called_once_with("key1", "value1", namespace="agent_memory")
    assert mem.recall_all()["key1"] == "value1"


def test_save_to_knowledge_manager_exception(monkeypatch, tmp_path):
    """Exception during KnowledgeManager save still persists locally."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    km = MagicMock()
    km.remember.side_effect = RuntimeError("store full")

    mem = AgentMemory(knowledge=km)
    mem.save("key1", "value1")

    assert mem.recall_all()["key1"] == "value1"
    assert (tmp_path / ".mullet" / "memory.json").exists()


def test_forget_with_knowledge_manager_success(monkeypatch, tmp_path):
    """Forgetting deletes from KnowledgeManager and local file."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    km = MagicMock()
    mem = AgentMemory(knowledge=km)
    mem.save("a", "1")
    mem.forget("a")

    km.memory.delete.assert_called_once_with(namespace="agent_memory", key="a")
    assert "a" not in mem.recall_all()


def test_forget_with_knowledge_manager_exception(monkeypatch, tmp_path):
    """Exception during KnowledgeManager forget still removes locally."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    km = MagicMock()
    mem = AgentMemory(knowledge=km)
    mem.save("a", "1")

    km.memory.delete.side_effect = RuntimeError("network error")
    mem.forget("a")

    assert "a" not in mem.recall_all()


# ---------------------------------------------------------------------------
# build_memory_prompt edge cases
# ---------------------------------------------------------------------------

def test_build_memory_prompt_all_expired():
    """When every entry is expired, prompt is empty string."""
    mem = AgentMemory.__new__(AgentMemory)
    mem._knowledge = None
    mem._local_path = Path("/nonexistent")
    mem._entries = {
        "old": {"content": "gone", "expires_at": time.time() - 10},
    }

    assert mem.build_memory_prompt() == ""


# ---------------------------------------------------------------------------
# Additional coverage for missing lines
# ---------------------------------------------------------------------------

def test_load_local_backward_compat_string(monkeypatch, tmp_path):
    """Local file with old plain-string values loads correctly (line 65)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    local = tmp_path / ".mullet" / "memory.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps({"legacy": "plain string"}))
    mem = AgentMemory()
    assert mem.recall_all() == {"legacy": "plain string"}


def test_save_with_ttl(monkeypatch, tmp_path):
    """Saving with ttl_seconds sets expires_at (line 82)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    mem = AgentMemory()
    mem.save("key", "value", ttl_seconds=3600)
    entry = mem._entries["key"]
    assert entry["expires_at"] is not None
    assert entry["expires_at"] > time.time()


def test_build_memory_prompt_no_entries():
    """When _entries is empty, prompt is empty string (line 122)."""
    mem = AgentMemory.__new__(AgentMemory)
    mem._knowledge = None
    mem._local_path = Path("/nonexistent")
    mem._entries = {}
    assert mem.build_memory_prompt() == ""


def test_build_memory_prompt_truncates(monkeypatch, tmp_path):
    """Prompt truncates when exceeding MAX_MEMORY_CHARS (lines 137-145)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from mullet.core.agent.memory import MAX_MEMORY_CHARS
    mem = AgentMemory()
    for i in range(10):
        mem.save(f"key{i}", "x" * 500)
    prompt = mem.build_memory_prompt()
    assert prompt.startswith("# Agent Memory")
    assert "... (" in prompt
    assert "more entries)" in prompt


def test_cleanup_expired_removes_and_saves(monkeypatch, tmp_path):
    """cleanup_expired_memories removes entries and persists (lines 155, 158)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    mem = AgentMemory()
    mem.save("alive", "yes")
    mem.save("dead", "no", ttl_seconds=-1)
    removed = mem.cleanup_expired_memories()
    assert removed == 1
    assert "dead" not in mem.recall_all()
    data = json.loads((tmp_path / ".mullet" / "memory.json").read_text())
    assert "dead" not in data


def test_save_local_exception_logs_warning(monkeypatch, tmp_path):
    """_save_local logs warning on write failure (lines 167-169)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    mem = AgentMemory()
    mem._local_path = Path("/dev/null") / "foo" / "memory.json"
    warn_mock = MagicMock()
    monkeypatch.setattr("mullet.core.agent.memory.logger.warning", warn_mock)
    mem._save_local()
    warn_mock.assert_called_once()


def test_count_property(monkeypatch, tmp_path):
    """count property returns number of active entries (line 174)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    mem = AgentMemory()
    assert mem.count == 0
    mem.save("a", "1")
    assert mem.count == 1
