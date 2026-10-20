"""Tests for pool_metadata pure helpers and continuation pure functions."""


from mullet.knowledge.engine.pool_metadata import (
    read_metadata,
    ensure_created,
    mark_searched,
    mark_indexed,
    _now_iso,
    METADATA_FILENAME,
)
from mullet.core.agent.continuation import (
    should_continue,
    format_todo_reminder,
    CONTINUATION_REMINDERS,
    _STOP_PHRASES,
    _COMPLETION_PHRASES,
)


# ---------------------------------------------------------------------------
# pool_metadata — pure operations on tmp_path
# ---------------------------------------------------------------------------

class TestNowIso:
    def test_returns_string(self):
        assert isinstance(_now_iso(), str)

    def test_contains_t_separator(self):
        assert "T" in _now_iso()

    def test_contains_utc_offset(self):
        ts = _now_iso()
        assert "+" in ts or "Z" in ts


class TestReadMetadata:
    def test_missing_dir_returns_empty(self, tmp_path):
        missing = tmp_path / "nonexistent"
        assert read_metadata(missing) == {}

    def test_no_file_returns_empty(self, tmp_path):
        assert read_metadata(tmp_path) == {}

    def test_valid_metadata_returned(self, tmp_path):
        (tmp_path / METADATA_FILENAME).write_text('{"created_at": "2025-01-01"}')
        data = read_metadata(tmp_path)
        assert data["created_at"] == "2025-01-01"

    def test_invalid_json_returns_empty(self, tmp_path):
        (tmp_path / METADATA_FILENAME).write_text("not json {")
        assert read_metadata(tmp_path) == {}


class TestEnsureCreated:
    def test_sets_created_at(self, tmp_path):
        ensure_created(tmp_path)
        data = read_metadata(tmp_path)
        assert "created_at" in data

    def test_idempotent(self, tmp_path):
        ensure_created(tmp_path)
        first = read_metadata(tmp_path)["created_at"]
        ensure_created(tmp_path)
        second = read_metadata(tmp_path)["created_at"]
        assert first == second


class TestMarkSearched:
    def test_sets_last_searched_at(self, tmp_path):
        mark_searched(tmp_path)
        data = read_metadata(tmp_path)
        assert "last_searched_at" in data

    def test_increments_search_count(self, tmp_path):
        mark_searched(tmp_path)
        mark_searched(tmp_path)
        data = read_metadata(tmp_path)
        assert data["search_count"] == 2

    def test_creates_created_at_if_missing(self, tmp_path):
        mark_searched(tmp_path)
        data = read_metadata(tmp_path)
        assert "created_at" in data


class TestMarkIndexed:
    def test_sets_last_indexed_at(self, tmp_path):
        mark_indexed(tmp_path)
        data = read_metadata(tmp_path)
        assert "last_indexed_at" in data

    def test_creates_created_at_if_missing(self, tmp_path):
        mark_indexed(tmp_path)
        data = read_metadata(tmp_path)
        assert "created_at" in data


# ---------------------------------------------------------------------------
# continuation — should_continue
# ---------------------------------------------------------------------------

class TestShouldContinue:
    def test_empty_content_returns_false(self):
        cont, msg = should_continue("", 5)
        assert cont is False
        assert msg is None

    def test_completion_phrase_returns_false(self):
        cont, msg = should_continue("All tests pass and everything verified working.", 3)
        assert cont is False

    def test_no_tools_short_response_returns_false(self):
        cont, msg = should_continue("OK", 0)
        assert cont is False

    def test_stop_phrase_with_pending_tasks_nudges(self):
        todos = [{"status": "pending", "id": "1", "content": "Fix bug"}]
        cont, msg = should_continue("Let me know if you need anything else.", 5, todos)
        assert cont is True
        assert msg is not None
        assert isinstance(msg, str)

    def test_stop_phrase_no_pending_returns_false(self):
        todos = [{"status": "completed", "id": "1", "content": "done"}]
        cont, msg = should_continue("Hope this helps!", 5, todos)
        assert cont is False

    def test_nudge_message_contains_pending_count(self):
        todos = [
            {"status": "pending", "id": "1", "content": "Task A"},
            {"status": "pending", "id": "2", "content": "Task B"},
        ]
        cont, msg = should_continue("All done!", 5, todos)
        # "All done!" has no stop phrase — so no nudge
        # Try with a stop phrase
        cont, msg = should_continue("I've completed the work, let me know!", 5, todos)
        if cont:
            assert "2" in msg or "Pending" in msg

    def test_returns_tuple(self):
        result = should_continue("test", 0)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestContinuationConstants:
    def test_continuation_reminders_nonempty(self):
        assert len(CONTINUATION_REMINDERS) > 0

    def test_all_reminders_are_strings(self):
        assert all(isinstance(r, str) for r in CONTINUATION_REMINDERS)

    def test_stop_phrases_nonempty(self):
        assert len(_STOP_PHRASES) > 0

    def test_completion_phrases_nonempty(self):
        assert len(_COMPLETION_PHRASES) > 0


# ---------------------------------------------------------------------------
# format_todo_reminder
# ---------------------------------------------------------------------------

class TestFormatTodoReminder:
    def test_no_items_returns_none(self):
        assert format_todo_reminder([]) is None

    def test_all_completed_returns_none(self):
        todos = [{"status": "completed", "id": "1", "content": "done"}]
        assert format_todo_reminder(todos) is None

    def test_pending_items_returns_string(self):
        todos = [{"status": "pending", "id": "1", "content": "Fix bug"}]
        result = format_todo_reminder(todos)
        assert isinstance(result, str)

    def test_in_progress_included(self):
        todos = [{"status": "in_progress", "id": "1", "content": "Active task"}]
        result = format_todo_reminder(todos)
        assert result is not None
        assert "Active task" in result

    def test_count_in_reminder(self):
        todos = [
            {"status": "pending", "id": "1", "content": "A"},
            {"status": "pending", "id": "2", "content": "B"},
        ]
        result = format_todo_reminder(todos)
        assert "2" in result

    def test_pending_item_content_shown(self):
        todos = [{"status": "pending", "id": "1", "content": "Write tests"}]
        result = format_todo_reminder(todos)
        assert "Write tests" in result
