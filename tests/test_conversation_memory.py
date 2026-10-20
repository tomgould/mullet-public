"""Tests for conversation memory module."""

import time
import pytest
from mullet.core.knowledge.conversation_memory import (
    ConversationSummary,
    ConversationMemory,
    summarize_messages,
    extract_topics,
)


@pytest.fixture
def memory(tmp_path):
    return ConversationMemory(storage_dir=str(tmp_path))


@pytest.fixture
def sample_conv():
    return ConversationSummary(
        id="conv-001",
        summary="Discussed authentication refactoring with OAuth2",
        topics=["auth", "OAuth2", "security"],
        files_touched=["auth/login.py", "tests/test_auth.py"],
        timestamp=time.time(),
        token_count=1500,
    )


# ── ConversationSummary ─────────────────────────────────────────────────────

def test_summary_to_dict(sample_conv):
    d = sample_conv.to_dict()
    assert d["id"] == "conv-001"
    assert "OAuth2" in d["topics"]


def test_summary_from_dict():
    data = {"id": "x", "summary": "test", "topics": ["a"], "files_touched": [], "timestamp": 1.0, "token_count": 100}
    s = ConversationSummary.from_dict(data)
    assert s.id == "x"
    assert s.token_count == 100


def test_summary_roundtrip(sample_conv):
    restored = ConversationSummary.from_dict(sample_conv.to_dict())
    assert restored.id == sample_conv.id
    assert restored.summary == sample_conv.summary


# ── ConversationMemory ───────────────────────────────────────────────────────

def test_save_and_get(memory, sample_conv):
    memory.save_conversation(sample_conv)
    result = memory.get_conversation("conv-001")
    assert result is not None
    assert result.summary == sample_conv.summary


def test_get_nonexistent(memory):
    assert memory.get_conversation("nope") is None


def test_count(memory, sample_conv):
    assert memory.count == 0
    memory.save_conversation(sample_conv)
    assert memory.count == 1


def test_delete(memory, sample_conv):
    memory.save_conversation(sample_conv)
    assert memory.delete("conv-001") is True
    assert memory.count == 0


def test_delete_nonexistent(memory):
    assert memory.delete("nope") is False


def test_clear(memory, sample_conv):
    memory.save_conversation(sample_conv)
    memory.save_conversation(ConversationSummary(id="conv-002", summary="test"))
    assert memory.clear() == 2
    assert memory.count == 0


def test_search_by_summary(memory, sample_conv):
    memory.save_conversation(sample_conv)
    results = memory.search("OAuth2")
    assert len(results) == 1
    assert results[0].id == "conv-001"


def test_search_by_topic(memory, sample_conv):
    memory.save_conversation(sample_conv)
    results = memory.search("security")
    assert len(results) == 1


def test_search_by_file(memory, sample_conv):
    memory.save_conversation(sample_conv)
    results = memory.search("login.py")
    assert len(results) == 1


def test_search_no_match(memory, sample_conv):
    memory.save_conversation(sample_conv)
    results = memory.search("quantum physics")
    assert results == []


def test_recent(memory):
    for i in range(5):
        memory.save_conversation(ConversationSummary(
            id=f"conv-{i}", summary=f"Test {i}", timestamp=time.time() + i,
        ))
    recent = memory.recent(3)
    assert len(recent) == 3
    assert recent[0].id == "conv-4"  # Most recent


def test_persistence(tmp_path):
    mem1 = ConversationMemory(storage_dir=str(tmp_path))
    mem1.save_conversation(ConversationSummary(id="persist", summary="data"))

    mem2 = ConversationMemory(storage_dir=str(tmp_path))
    assert mem2.get_conversation("persist") is not None


def test_overwrite_same_id(memory):
    memory.save_conversation(ConversationSummary(id="x", summary="v1"))
    memory.save_conversation(ConversationSummary(id="x", summary="v2"))
    assert memory.count == 1
    assert memory.get_conversation("x").summary == "v2"


# ── summarize_messages ───────────────────────────────────────────────────────

def test_summarize_basic():
    msgs = [
        {"role": "user", "content": "How do I fix the auth bug?"},
        {"role": "assistant", "content": "Check the token validation logic."},
    ]
    summary = summarize_messages(msgs)
    assert "auth" in summary.lower()
    assert "token" in summary.lower()


def test_summarize_truncates():
    msgs = [{"role": "user", "content": "x" * 1000}] * 20
    summary = summarize_messages(msgs, max_length=100)
    assert len(summary) <= 104  # 100 + "..."


def test_summarize_empty():
    assert summarize_messages([]) == ""


# ── extract_topics ───────────────────────────────────────────────────────────

def test_extract_topics():
    msgs = [
        {"role": "user", "content": "Fix the AuthController in auth/login.py"},
    ]
    topics = extract_topics(msgs)
    assert any("Auth" in t for t in topics) or any("login" in t for t in topics)


def test_extract_topics_empty():
    assert extract_topics([]) == []


# ---------------------------------------------------------------------------
# list_all, _load error, _save error, list content in summarize_messages
# ---------------------------------------------------------------------------

def test_list_all_returns_all(memory):
    """Line 112: list_all() returns copy of all conversations."""
    for i in range(3):
        memory.save_conversation(ConversationSummary(id=f"c{i}", summary=f"s{i}"))
    all_convs = memory.list_all()
    assert len(all_convs) == 3
    assert isinstance(all_convs, list)


def test_load_corrupt_json_falls_back_to_empty(tmp_path):
    """Lines 143-145: corrupt conversations.json → JSONDecodeError → _conversations = []."""
    storage = tmp_path / "conv_store"
    storage.mkdir()
    (storage / "conversations.json").write_text("{ not valid json [")
    mem = ConversationMemory(storage_dir=str(storage))
    assert mem.count == 0


def test_save_oserror_swallowed(tmp_path):
    """Lines 152-153: OSError during _save → exception logged, no crash."""
    storage = tmp_path / "conv_store"
    storage.mkdir()
    mem = ConversationMemory(storage_dir=str(storage))
    # Seed one conversation to ensure the file exists, then make it unwritable
    mem.save_conversation(ConversationSummary(id="seed", summary="seed"))
    mem._file.chmod(0o000)
    try:
        mem.save_conversation(ConversationSummary(id="x", summary="test"))
        # No exception raised — error is swallowed
    finally:
        mem._file.chmod(0o644)


def test_summarize_messages_list_content():
    """Lines 166-170: content is a list of text blocks → joined as string."""
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "world"},
            {"type": "image", "url": "http://example.com/img.png"},
        ]},
    ]
    summary = summarize_messages(msgs)
    assert "Hello" in summary
    assert "world" in summary
