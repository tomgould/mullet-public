"""Tests for ConversationSummary/ConversationMemory pure helpers and diff_display rendering."""


from mullet.core.knowledge.conversation_memory import (
    ConversationSummary,
    ConversationMemory,
    summarize_messages,
    extract_topics,
)
from mullet.cli.diff_display import (
    ToolDiff,
    format_unified_diff,
    format_tool_diff,
    DiffRenderer,
)


# ---------------------------------------------------------------------------
# ConversationSummary
# ---------------------------------------------------------------------------

class TestConversationSummaryDefaults:
    def test_id_stored(self):
        cs = ConversationSummary(id="conv-1", summary="Fixed the bug")
        assert cs.id == "conv-1"

    def test_summary_stored(self):
        cs = ConversationSummary(id="c", summary="Test summary")
        assert cs.summary == "Test summary"

    def test_topics_defaults_empty(self):
        cs = ConversationSummary(id="c", summary="s")
        assert cs.topics == []

    def test_files_touched_defaults_empty(self):
        cs = ConversationSummary(id="c", summary="s")
        assert cs.files_touched == []

    def test_timestamp_defaults_zero(self):
        cs = ConversationSummary(id="c", summary="s")
        assert cs.timestamp == 0.0

    def test_token_count_defaults_zero(self):
        cs = ConversationSummary(id="c", summary="s")
        assert cs.token_count == 0


class TestConversationSummaryToDict:
    def test_returns_dict(self):
        cs = ConversationSummary(id="c", summary="s")
        assert isinstance(cs.to_dict(), dict)

    def test_id_in_dict(self):
        cs = ConversationSummary(id="conv-1", summary="s")
        assert cs.to_dict()["id"] == "conv-1"

    def test_summary_in_dict(self):
        cs = ConversationSummary(id="c", summary="My summary")
        assert cs.to_dict()["summary"] == "My summary"

    def test_topics_in_dict(self):
        cs = ConversationSummary(id="c", summary="s", topics=["auth", "bug"])
        assert cs.to_dict()["topics"] == ["auth", "bug"]

    def test_all_keys_present(self):
        cs = ConversationSummary(id="c", summary="s")
        d = cs.to_dict()
        for key in ("id", "summary", "topics", "files_touched", "timestamp", "token_count"):
            assert key in d


class TestConversationSummaryFromDict:
    def test_id_set(self):
        cs = ConversationSummary.from_dict({"id": "abc", "summary": "s"})
        assert cs.id == "abc"

    def test_summary_set(self):
        cs = ConversationSummary.from_dict({"id": "c", "summary": "My summary"})
        assert cs.summary == "My summary"

    def test_topics_set(self):
        cs = ConversationSummary.from_dict({"id": "c", "summary": "s", "topics": ["a", "b"]})
        assert cs.topics == ["a", "b"]

    def test_token_count_set(self):
        cs = ConversationSummary.from_dict({"id": "c", "summary": "s", "token_count": 42})
        assert cs.token_count == 42

    def test_empty_dict_uses_defaults(self):
        cs = ConversationSummary.from_dict({})
        assert cs.id == ""
        assert cs.topics == []

    def test_roundtrip(self):
        original = ConversationSummary(id="c1", summary="test", topics=["x"], timestamp=1.5)
        restored = ConversationSummary.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.summary == original.summary
        assert restored.topics == original.topics
        assert restored.timestamp == original.timestamp


# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------

def _make_cs(id_, summary, topics=None, files=None, timestamp=0.0):
    return ConversationSummary(
        id=id_,
        summary=summary,
        topics=topics or [],
        files_touched=files or [],
        timestamp=timestamp,
    )


class TestConversationMemoryConstruction:
    def test_count_zero_initially(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        assert mem.count == 0

    def test_list_all_empty_initially(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        assert mem.list_all() == []

    def test_storage_dir_created(self, tmp_path):
        subdir = tmp_path / "deep" / "path"
        mem = ConversationMemory(storage_dir=str(subdir))
        assert subdir.exists()


class TestConversationMemorySave:
    def test_save_increments_count(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(_make_cs("c1", "summary"))
        assert mem.count == 1

    def test_save_same_id_replaces(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(_make_cs("same", "first"))
        mem.save_conversation(_make_cs("same", "second"))
        assert mem.count == 1
        assert mem.get_conversation("same").summary == "second"

    def test_save_sets_timestamp_if_zero(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        cs = _make_cs("c1", "s", timestamp=0.0)
        mem.save_conversation(cs)
        found = mem.get_conversation("c1")
        assert found.timestamp > 0

    def test_save_preserves_explicit_timestamp(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        cs = _make_cs("c1", "s", timestamp=12345.0)
        mem.save_conversation(cs)
        assert mem.get_conversation("c1").timestamp == 12345.0

    def test_persists_across_instances(self, tmp_path):
        mem1 = ConversationMemory(storage_dir=str(tmp_path))
        mem1.save_conversation(_make_cs("c1", "stored"))
        mem2 = ConversationMemory(storage_dir=str(tmp_path))
        assert mem2.count == 1
        assert mem2.get_conversation("c1").summary == "stored"


class TestConversationMemoryGet:
    def test_get_existing_returns_summary(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(_make_cs("x", "test"))
        assert mem.get_conversation("x").id == "x"

    def test_get_nonexistent_returns_none(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        assert mem.get_conversation("nope") is None


class TestConversationMemorySearch:
    def test_search_finds_in_summary(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(_make_cs("a", "fixed auth bug"))
        results = mem.search("auth")
        assert len(results) == 1

    def test_search_finds_in_topics(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(_make_cs("a", "summary", topics=["deployment"]))
        results = mem.search("deployment")
        assert len(results) == 1

    def test_search_finds_in_files(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(_make_cs("a", "s", files=["src/app.css"]))
        results = mem.search("app.css")
        assert len(results) == 1

    def test_search_no_match_returns_empty(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(_make_cs("a", "fixed auth bug"))
        assert mem.search("xyzzy") == []

    def test_search_case_insensitive(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(_make_cs("a", "Fixed Auth Bug"))
        results = mem.search("auth")
        assert len(results) == 1


class TestConversationMemoryDelete:
    def test_delete_existing_returns_true(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(_make_cs("a", "s"))
        assert mem.delete("a") is True

    def test_delete_decrements_count(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(_make_cs("a", "s"))
        mem.delete("a")
        assert mem.count == 0

    def test_delete_nonexistent_returns_false(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        assert mem.delete("nope") is False


class TestConversationMemoryClear:
    def test_clear_returns_count(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        for i in range(3):
            mem.save_conversation(_make_cs(f"c{i}", "s"))
        assert mem.clear() == 3

    def test_clear_empties_store(self, tmp_path):
        mem = ConversationMemory(storage_dir=str(tmp_path))
        mem.save_conversation(_make_cs("a", "s"))
        mem.clear()
        assert mem.count == 0


# ---------------------------------------------------------------------------
# summarize_messages
# ---------------------------------------------------------------------------

class TestSummarizeMessages:
    def test_returns_string(self):
        msgs = [{"role": "user", "content": "hello"}]
        assert isinstance(summarize_messages(msgs), str)

    def test_includes_user_message(self):
        msgs = [{"role": "user", "content": "what is Python?"}]
        assert "what is Python" in summarize_messages(msgs)

    def test_includes_assistant_message(self):
        msgs = [{"role": "assistant", "content": "Python is a language."}]
        assert "Python is a language" in summarize_messages(msgs)

    def test_truncated_at_max_length(self):
        msgs = [{"role": "user", "content": "x" * 300}] * 5
        result = summarize_messages(msgs, max_length=100)
        assert len(result) <= 103  # max_length + "..." length

    def test_truncated_has_ellipsis(self):
        msgs = [{"role": "user", "content": "x" * 300}] * 5
        result = summarize_messages(msgs, max_length=100)
        assert "..." in result

    def test_empty_messages_returns_empty(self):
        assert summarize_messages([]) == ""

    def test_skips_non_user_assistant(self):
        msgs = [{"role": "system", "content": "system prompt"}]
        result = summarize_messages(msgs)
        assert "system prompt" not in result


# ---------------------------------------------------------------------------
# extract_topics
# ---------------------------------------------------------------------------

class TestExtractTopics:
    def test_returns_list(self):
        msgs = [{"role": "user", "content": "hello"}]
        assert isinstance(extract_topics(msgs), list)

    def test_extracts_capitalized_words(self):
        msgs = [{"role": "user", "content": "Fix Django UserProfile model"}]
        topics = extract_topics(msgs)
        assert any("Django" in t or "UserProfile" in t for t in topics)

    def test_extracts_file_paths(self):
        msgs = [{"role": "user", "content": "update models.py for the fix"}]
        topics = extract_topics(msgs)
        assert "models.py" in topics

    def test_max_10_topics(self):
        msgs = [{"role": "user", "content": " ".join(f"Word{i}" for i in range(50))}]
        topics = extract_topics(msgs)
        assert len(topics) <= 10

    def test_empty_messages_returns_list(self):
        assert extract_topics([]) == []


# ---------------------------------------------------------------------------
# ToolDiff
# ---------------------------------------------------------------------------

class TestToolDiff:
    def test_fields_stored(self):
        td = ToolDiff(tool_name="agent", file_path="a.py", old_content="old", new_content="new")
        assert td.tool_name == "agent"
        assert td.file_path == "a.py"
        assert td.old_content == "old"
        assert td.new_content == "new"

    def test_none_old_content(self):
        td = ToolDiff(tool_name="t", file_path="f.py", old_content=None, new_content="new")
        assert td.old_content is None

    def test_none_new_content(self):
        td = ToolDiff(tool_name="t", file_path="f.py", old_content="old", new_content=None)
        assert td.new_content is None


# ---------------------------------------------------------------------------
# format_unified_diff
# ---------------------------------------------------------------------------

class TestFormatUnifiedDiff:
    def test_returns_string(self):
        assert isinstance(format_unified_diff("old\n", "new\n", "test.py"), str)

    def test_identical_content_returns_empty(self):
        assert format_unified_diff("same\n", "same\n", "test.py") == ""

    def test_additions_shown(self):
        result = format_unified_diff("line1\n", "line1\nnew line\n", "test.py", colorize=False)
        assert "+" in result

    def test_removals_shown(self):
        result = format_unified_diff("line1\nremoved\n", "line1\n", "test.py", colorize=False)
        assert "-" in result

    def test_colorized_has_ansi(self):
        result = format_unified_diff("old\n", "new\n", "test.py", colorize=True)
        assert "\033[" in result or "\x1b[" in result

    def test_no_colorize_no_ansi(self):
        result = format_unified_diff("old\n", "new\n", "test.py", colorize=False)
        assert "\033[" not in result
        assert "\x1b[" not in result

    def test_long_diff_truncated(self):
        old = "\n".join(["line"] * 100)
        new = "\n".join(["line"] * 100 + ["extra"])
        result = format_unified_diff(old, new, "big.py", max_lines=5)
        assert "truncated" in result.lower()

    def test_none_old_treated_as_empty(self):
        result = format_unified_diff(None, "new\n", "test.py", colorize=False)
        assert "+" in result

    def test_filename_in_diff_header(self):
        result = format_unified_diff("old\n", "new\n", "myfile.py", colorize=False)
        assert "myfile.py" in result


# ---------------------------------------------------------------------------
# format_tool_diff
# ---------------------------------------------------------------------------

class TestFormatToolDiff:
    def test_new_file_badge(self):
        result = format_tool_diff("agent", "new.py", None, "content\n", colorize=False)
        assert "[NEW FILE]" in result

    def test_deleted_file_badge(self):
        result = format_tool_diff("agent", "del.py", "old\n", None, colorize=False)
        assert "[DELETED]" in result

    def test_modified_file_badge(self):
        result = format_tool_diff("agent", "mod.py", "old\n", "new\n", colorize=False)
        assert "[MODIFIED]" in result

    def test_no_change_returns_empty(self):
        result = format_tool_diff("agent", "same.py", "same\n", "same\n", colorize=False)
        assert result == ""

    def test_tool_name_in_header(self):
        result = format_tool_diff("my_agent", "f.py", None, "content\n", colorize=False)
        assert "my_agent" in result

    def test_file_path_in_header(self):
        result = format_tool_diff("agent", "src/utils.py", None, "content\n", colorize=False)
        assert "src/utils.py" in result

    def test_returns_string(self):
        assert isinstance(format_tool_diff("t", "f.py", None, "c\n"), str)


# ---------------------------------------------------------------------------
# DiffRenderer
# ---------------------------------------------------------------------------

class TestDiffRenderer:
    def test_empty_changes_returns_no_changes_message(self):
        renderer = DiffRenderer(colorize=False)
        result = renderer.render([])
        assert "No file changes" in result

    def test_renders_new_file(self):
        renderer = DiffRenderer(colorize=False)
        changes = [ToolDiff("agent", "new.py", None, "content\n")]
        result = renderer.render(changes)
        assert "[NEW FILE]" in result

    def test_renders_multiple_changes(self):
        renderer = DiffRenderer(colorize=False)
        changes = [
            ToolDiff("agent", "a.py", None, "content a\n"),
            ToolDiff("agent", "b.py", None, "content b\n"),
        ]
        result = renderer.render(changes)
        assert "a.py" in result
        assert "b.py" in result

    def test_identical_changes_returns_no_changes(self):
        renderer = DiffRenderer(colorize=False)
        changes = [ToolDiff("agent", "same.py", "same\n", "same\n")]
        result = renderer.render(changes)
        assert "No file changes" in result

    def test_colorize_true_adds_ansi(self):
        renderer = DiffRenderer(colorize=True)
        changes = [ToolDiff("agent", "f.py", None, "new\n")]
        result = renderer.render(changes)
        assert "\033[" in result or "\x1b[" in result
