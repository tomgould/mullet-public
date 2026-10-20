"""Tests for conversation_memory pure helpers and observations_cmd pure functions."""


from mullet.core.knowledge.conversation_memory import (
    ConversationSummary,
    ConversationMemory,
    summarize_messages,
    extract_topics,
)


# ---------------------------------------------------------------------------
# ConversationSummary.to_dict / from_dict
# ---------------------------------------------------------------------------

class TestConversationSummaryDict:
    def test_to_dict_fields(self):
        s = ConversationSummary(id="c1", summary="Fixed a bug")
        d = s.to_dict()
        assert d["id"] == "c1"
        assert d["summary"] == "Fixed a bug"

    def test_to_dict_topics(self):
        s = ConversationSummary(id="c1", summary="s", topics=["python", "tests"])
        assert s.to_dict()["topics"] == ["python", "tests"]

    def test_from_dict_round_trips(self):
        s = ConversationSummary(id="c1", summary="s", topics=["t"], files_touched=["f.py"])
        d = s.to_dict()
        restored = ConversationSummary.from_dict(d)
        assert restored.id == "c1"
        assert restored.topics == ["t"]
        assert restored.files_touched == ["f.py"]

    def test_from_dict_defaults(self):
        s = ConversationSummary.from_dict({})
        assert s.id == ""
        assert s.summary == ""
        assert s.topics == []
        assert s.timestamp == 0.0

    def test_token_count_default_zero(self):
        s = ConversationSummary(id="c", summary="s")
        assert s.token_count == 0


# ---------------------------------------------------------------------------
# ConversationMemory — in-memory CRUD via tmp_path
# ---------------------------------------------------------------------------

class TestConversationMemory:
    def test_count_starts_zero(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        assert mem.count == 0

    def test_save_and_count(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="first"))
        assert mem.count == 1

    def test_get_conversation_returns_summary(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="My summary"))
        result = mem.get_conversation("c1")
        assert result is not None
        assert result.summary == "My summary"

    def test_get_nonexistent_returns_none(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        assert mem.get_conversation("nonexistent") is None

    def test_save_replaces_existing_same_id(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="old"))
        mem.save_conversation(ConversationSummary(id="c1", summary="new"))
        assert mem.count == 1
        assert mem.get_conversation("c1").summary == "new"

    def test_delete_removes_conversation(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="s"))
        mem.delete("c1")
        assert mem.count == 0

    def test_clear_removes_all(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="s1"))
        mem.save_conversation(ConversationSummary(id="c2", summary="s2"))
        mem.clear()
        assert mem.count == 0

    def test_search_by_keyword(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="fixed auth bug"))
        mem.save_conversation(ConversationSummary(id="c2", summary="added dashboard"))
        results = mem.search("auth")
        assert len(results) == 1
        assert results[0].id == "c1"

    def test_recent_returns_latest_first(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="first", timestamp=1.0))
        mem.save_conversation(ConversationSummary(id="c2", summary="second", timestamp=2.0))
        results = mem.recent(count=2)
        assert results[0].id == "c2"

    def test_list_all_returns_all(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(ConversationSummary(id="c1", summary="s1"))
        mem.save_conversation(ConversationSummary(id="c2", summary="s2"))
        all_convs = mem.list_all()
        assert len(all_convs) == 2


# ---------------------------------------------------------------------------
# summarize_messages
# ---------------------------------------------------------------------------

class TestSummarizeMessages:
    def test_user_message_included(self):
        messages = [{"role": "user", "content": "What is Python?"}]
        result = summarize_messages(messages)
        assert "Python" in result

    def test_assistant_message_included(self):
        messages = [{"role": "assistant", "content": "Python is a language."}]
        result = summarize_messages(messages)
        assert "Python" in result

    def test_prefix_added_for_user(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = summarize_messages(messages)
        assert "User:" in result

    def test_prefix_added_for_assistant(self):
        messages = [{"role": "assistant", "content": "Hi"}]
        result = summarize_messages(messages)
        assert "Assistant:" in result

    def test_summary_respects_max_length(self):
        messages = [{"role": "user", "content": "A" * 1000}]
        result = summarize_messages(messages, max_length=100)
        assert len(result) <= 103  # max_length + "..."

    def test_empty_messages_returns_empty(self):
        result = summarize_messages([])
        assert result == ""

    def test_multipart_content_handled(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        result = summarize_messages(messages)
        assert "hi" in result

    def test_system_messages_excluded(self):
        messages = [{"role": "system", "content": "System instructions"}]
        result = summarize_messages(messages)
        assert "System" not in result

    def test_returns_string(self):
        assert isinstance(summarize_messages([]), str)


# ---------------------------------------------------------------------------
# extract_topics
# ---------------------------------------------------------------------------

class TestExtractTopics:
    def test_returns_list(self):
        assert isinstance(extract_topics([]), list)

    def test_extracts_capitalized_words(self):
        messages = [{"role": "user", "content": "Fix the AuthMiddleware class"}]
        topics = extract_topics(messages)
        # Should extract CamelCase words
        assert any("Auth" in t for t in topics)

    def test_extracts_file_paths(self):
        messages = [{"role": "user", "content": "update src/utils.py and tests/test_auth.py"}]
        topics = extract_topics(messages)
        # Should extract file paths
        assert any(".py" in t for t in topics)

    def test_empty_messages_returns_empty(self):
        result = extract_topics([])
        assert result == []

    def test_max_ten_topics(self):
        # Generate many capitalized words
        content = " ".join(f"Word{i}" for i in range(50))
        messages = [{"role": "user", "content": content}]
        topics = extract_topics(messages)
        assert len(topics) <= 10
