"""Tests for mullet.core.knowledge.conversation_memory — pure helpers and store."""

import time
from mullet.core.knowledge.conversation_memory import (
    ConversationSummary,
    ConversationMemory,
    summarize_messages,
    extract_topics,
)


# ---------------------------------------------------------------------------
# ConversationSummary
# ---------------------------------------------------------------------------

class TestConversationSummary:
    def test_defaults(self):
        s = ConversationSummary(id="c1", summary="Did stuff")
        assert s.topics == []
        assert s.files_touched == []
        assert s.timestamp == 0.0
        assert s.token_count == 0

    def test_to_dict_round_trip(self):
        s = ConversationSummary(id="c1", summary="Did stuff", topics=["auth"], token_count=100)
        d = s.to_dict()
        s2 = ConversationSummary.from_dict(d)
        assert s2.id == "c1"
        assert s2.topics == ["auth"]
        assert s2.token_count == 100

    def test_from_dict_missing_fields_use_defaults(self):
        s = ConversationSummary.from_dict({"id": "x", "summary": "y"})
        assert s.topics == []
        assert s.token_count == 0


# ---------------------------------------------------------------------------
# ConversationMemory — CRUD
# ---------------------------------------------------------------------------

class TestConversationMemoryCRUD:
    def test_empty_on_init(self, tmp_path):
        mem = ConversationMemory(str(tmp_path))
        assert mem.count == 0

    def test_save_then_get(self, tmp_path):
        mem = ConversationMemory(str(tmp_path))
        s = ConversationSummary(id="c1", summary="auth work", timestamp=time.time())
        mem.save_conversation(s)
        result = mem.get_conversation("c1")
        assert result is not None
        assert result.summary == "auth work"

    def test_get_nonexistent_returns_none(self, tmp_path):
        mem = ConversationMemory(str(tmp_path))
        assert mem.get_conversation("does-not-exist") is None

    def test_save_replaces_same_id(self, tmp_path):
        mem = ConversationMemory(str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="first", timestamp=1.0))
        mem.save_conversation(ConversationSummary(id="c1", summary="second", timestamp=2.0))
        assert mem.count == 1
        assert mem.get_conversation("c1").summary == "second"

    def test_delete_removes_entry(self, tmp_path):
        mem = ConversationMemory(str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="s", timestamp=1.0))
        result = mem.delete("c1")
        assert result is True
        assert mem.count == 0

    def test_delete_nonexistent_returns_false(self, tmp_path):
        mem = ConversationMemory(str(tmp_path))
        assert mem.delete("ghost") is False

    def test_clear_returns_count(self, tmp_path):
        mem = ConversationMemory(str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="s", timestamp=1.0))
        mem.save_conversation(ConversationSummary(id="c2", summary="s", timestamp=2.0))
        deleted = mem.clear()
        assert deleted == 2
        assert mem.count == 0

    def test_list_all_returns_all(self, tmp_path):
        mem = ConversationMemory(str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="s", timestamp=1.0))
        mem.save_conversation(ConversationSummary(id="c2", summary="s", timestamp=2.0))
        assert len(mem.list_all()) == 2

    def test_persists_across_instances(self, tmp_path):
        mem1 = ConversationMemory(str(tmp_path))
        mem1.save_conversation(ConversationSummary(id="c1", summary="persistent", timestamp=1.0))
        mem2 = ConversationMemory(str(tmp_path))
        assert mem2.get_conversation("c1") is not None
        assert mem2.get_conversation("c1").summary == "persistent"

    def test_auto_sets_timestamp(self, tmp_path):
        mem = ConversationMemory(str(tmp_path))
        s = ConversationSummary(id="c1", summary="s")  # timestamp=0.0
        mem.save_conversation(s)
        result = mem.get_conversation("c1")
        assert result.timestamp > 0


# ---------------------------------------------------------------------------
# ConversationMemory.recent
# ---------------------------------------------------------------------------

class TestConversationMemoryRecent:
    def test_recent_returns_most_recent(self, tmp_path):
        mem = ConversationMemory(str(tmp_path))
        for i in range(5):
            mem.save_conversation(ConversationSummary(id=f"c{i}", summary="s", timestamp=float(i)))
        recent = mem.recent(3)
        assert len(recent) == 3
        assert recent[0].timestamp > recent[1].timestamp

    def test_recent_empty_returns_empty(self, tmp_path):
        mem = ConversationMemory(str(tmp_path))
        assert mem.recent(5) == []


# ---------------------------------------------------------------------------
# ConversationMemory.search
# ---------------------------------------------------------------------------

class TestConversationMemorySearch:
    def test_search_by_summary_keyword(self, tmp_path):
        mem = ConversationMemory(str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="authentication work done", timestamp=1.0))
        mem.save_conversation(ConversationSummary(id="c2", summary="refactoring code", timestamp=2.0))
        results = mem.search("authentication")
        assert len(results) == 1
        assert results[0].id == "c1"

    def test_search_by_topic(self, tmp_path):
        mem = ConversationMemory(str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="misc", topics=["auth", "login"], timestamp=1.0))
        results = mem.search("auth")
        assert len(results) == 1

    def test_search_by_file(self, tmp_path):
        mem = ConversationMemory(str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="misc", files_touched=["src/auth.py"], timestamp=1.0))
        results = mem.search("auth.py")
        assert len(results) == 1

    def test_search_no_match_empty(self, tmp_path):
        mem = ConversationMemory(str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="unrelated", timestamp=1.0))
        results = mem.search("nonexistent_keyword_xyz")
        assert results == []

    def test_search_respects_max_results(self, tmp_path):
        mem = ConversationMemory(str(tmp_path))
        for i in range(10):
            mem.save_conversation(ConversationSummary(id=f"c{i}", summary="auth work", timestamp=float(i)))
        results = mem.search("auth", max_results=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# summarize_messages
# ---------------------------------------------------------------------------

class TestSummarizeMessages:
    def test_returns_string(self):
        msgs = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}]
        assert isinstance(summarize_messages(msgs), str)

    def test_includes_user_content(self):
        msgs = [{"role": "user", "content": "What is Python?"}]
        result = summarize_messages(msgs)
        assert "What is Python?" in result

    def test_includes_assistant_content(self):
        msgs = [{"role": "assistant", "content": "Python is a language."}]
        result = summarize_messages(msgs)
        assert "Python is a language" in result

    def test_respects_max_length(self):
        long_msg = "A" * 1000
        msgs = [{"role": "user", "content": long_msg}]
        result = summarize_messages(msgs, max_length=100)
        assert len(result) <= 104  # 100 + "..."

    def test_empty_messages_returns_empty(self):
        assert summarize_messages([]) == ""

    def test_list_content_extracted(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "Extract this"}]}]
        result = summarize_messages(msgs)
        assert "Extract this" in result

    def test_tool_messages_skipped(self):
        msgs = [{"role": "tool", "content": "tool output here"}]
        result = summarize_messages(msgs)
        assert "tool output" not in result


# ---------------------------------------------------------------------------
# extract_topics
# ---------------------------------------------------------------------------

class TestExtractTopics:
    def test_returns_list(self):
        msgs = [{"role": "user", "content": "Hello"}]
        assert isinstance(extract_topics(msgs), list)

    def test_extracts_file_references(self):
        msgs = [{"role": "user", "content": "Edit the file src/auth.py please"}]
        topics = extract_topics(msgs)
        assert any("auth.py" in t for t in topics)

    def test_empty_messages_returns_empty_or_short(self):
        result = extract_topics([])
        assert isinstance(result, list)

    def test_only_user_messages_processed(self):
        msgs = [
            {"role": "user", "content": "What about UserService"},
            {"role": "assistant", "content": "AssistantTopic here"},
        ]
        topics = extract_topics(msgs)
        # UserService from user message should appear; AssistantTopic from assistant should not
        assert any("UserService" in t for t in topics)

    def test_max_10_topics_returned(self):
        long_content = "Alpha Beta Gamma Delta Epsilon Zeta Eta Theta Iota Kappa Lambda Mu"
        msgs = [{"role": "user", "content": long_content}]
        topics = extract_topics(msgs)
        assert len(topics) <= 10
