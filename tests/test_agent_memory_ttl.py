"""Tests for AgentMemory TTL functionality."""

import json
import time
from pathlib import Path
import pytest
from mullet.core.agent.memory import AgentMemory


@pytest.fixture
def memory(tmp_path, monkeypatch):
    """Create AgentMemory with tmp_path as home directory."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return AgentMemory()


def test_save_and_recall_no_ttl(memory):
    """Test saving without TTL - should be permanent."""
    memory.save("test_key", "test_value")
    assert memory.recall_all() == {"test_key": "test_value"}
    assert memory.count == 1


def test_save_with_ttl_and_expire(memory):
    """Test saving with TTL and automatic expiration."""
    memory.save("temp_key", "temp_value", ttl_seconds=0.01)
    assert memory.recall_all() == {"temp_key": "temp_value"}
    
    time.sleep(0.05)
    # Should be cleaned up on next access
    assert memory.recall_all() == {}
    assert memory.count == 0


def test_cleanup_removes_expired(memory):
    """Test explicit cleanup of expired entries."""
    memory.save("permanent", "value")
    memory.save("expired", "value", ttl_seconds=0.01)
    
    time.sleep(0.05)
    removed = memory.cleanup_expired_memories()
    assert removed == 1
    
    assert "permanent" in memory.recall_all()
    assert "expired" not in memory.recall_all()


def test_backward_compat_plain_string(memory, tmp_path):
    """Test loading old format with plain string values."""
    # Create old format file
    old_data = {"old_key": "old_value"}
    memory._local_path.parent.mkdir(parents=True, exist_ok=True)
    memory._local_path.write_text(json.dumps(old_data))
    
    # Reload memory
    new_memory = AgentMemory()
    assert new_memory.recall_all() == {"old_key": "old_value"}
    assert new_memory._entries["old_key"]["expires_at"] is None


def test_mixed_format_loading(memory, tmp_path):
    """Test loading file with both old and new formats."""
    mixed_data = {
        "old": "plain_string",
        "new": {"content": "dict_format", "expires_at": None},
        "expired": {"content": "expired_value", "expires_at": time.time() - 10}
    }
    memory._local_path.parent.mkdir(parents=True, exist_ok=True)
    memory._local_path.write_text(json.dumps(mixed_data))
    
    new_memory = AgentMemory()
    # Should load old format as permanent
    assert new_memory._entries["old"]["expires_at"] is None
    assert new_memory._entries["old"]["content"] == "plain_string"
    
    # Should load new format correctly
    assert new_memory._entries["new"]["content"] == "dict_format"
    
    # Should load expired entry but filter it out
    assert "expired" not in new_memory.recall_all()
