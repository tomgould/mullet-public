"""Tests for mullet.core.agent.continuation — continuation enforcement helpers."""

from mullet.core.agent.continuation import (
    should_continue,
    format_todo_reminder,
    CONTINUATION_REMINDERS,
)


def _todo(id: str, status: str, content: str = "") -> dict:
    return {"id": id, "status": status, "content": content}


# ---------------------------------------------------------------------------
# CONTINUATION_REMINDERS constant
# ---------------------------------------------------------------------------

class TestContinuationReminders:
    def test_non_empty(self):
        assert len(CONTINUATION_REMINDERS) >= 2

    def test_all_strings(self):
        for msg in CONTINUATION_REMINDERS:
            assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# should_continue
# ---------------------------------------------------------------------------

class TestShouldContinue:
    def test_empty_response_returns_false(self):
        ok, nudge = should_continue("", tool_calls_made=5)
        assert ok is False
        assert nudge is None

    def test_genuine_completion_phrase_stops(self):
        ok, nudge = should_continue(
            "All tests pass and verified working.",
            tool_calls_made=3,
            todo_items=[_todo("t1", "pending")],
        )
        assert ok is False
        assert nudge is None

    def test_stop_phrase_with_pending_todos_nudges(self):
        ok, nudge = should_continue(
            "I've completed the first step, let me know if you need more help.",
            tool_calls_made=2,
            todo_items=[_todo("t1", "pending"), _todo("t2", "pending")],
        )
        assert ok is True
        assert nudge is not None
        assert "2" in nudge  # pending count in nudge

    def test_stop_phrase_no_pending_todos_no_nudge(self):
        ok, nudge = should_continue(
            "All done! Let me know if you need anything else.",
            tool_calls_made=5,
            todo_items=[_todo("t1", "completed")],
        )
        assert ok is False

    def test_no_stop_phrase_no_nudge(self):
        ok, nudge = should_continue(
            "I will now fix the authentication bug in auth.py.",
            tool_calls_made=5,
            todo_items=[_todo("t1", "pending")],
        )
        assert ok is False

    def test_zero_tool_calls_short_response_no_nudge(self):
        ok, nudge = should_continue(
            "I'm done.",
            tool_calls_made=0,
            todo_items=[_todo("t1", "pending")],
        )
        assert ok is False

    def test_nudge_rotates_by_iteration(self):
        content = "I'm done. Let me know if you need anything."
        todos = [_todo("t1", "pending")]
        _, nudge0 = should_continue(content, 3, todos, iteration=0)
        _, nudge1 = should_continue(content, 3, todos, iteration=1)
        assert nudge0 != nudge1

    def test_nudge_wraps_around(self):
        content = "Hope this helps! Let me know if you need more."
        todos = [_todo("t1", "pending")]
        n = len(CONTINUATION_REMINDERS)
        _, nudge0 = should_continue(content, 3, todos, iteration=0)
        _, nudgeN = should_continue(content, 3, todos, iteration=n)
        assert nudge0 == nudgeN  # modular rotation

    def test_in_progress_todos_counted_as_pending(self):
        ok, nudge = should_continue(
            "I'm done. Let me know if you need more.",
            tool_calls_made=2,
            todo_items=[_todo("t1", "in_progress")],
        )
        assert ok is True
        assert "1" in nudge

    def test_none_todo_items_handled(self):
        ok, nudge = should_continue(
            "Hope this helps! Let me know if you need more.",
            tool_calls_made=5,
            todo_items=None,
        )
        assert ok is False

    def test_all_completed_todos_no_nudge(self):
        ok, nudge = should_continue(
            "I'm done, let me know if you need anything.",
            tool_calls_made=5,
            todo_items=[
                _todo("t1", "completed"),
                _todo("t2", "completed"),
            ],
        )
        assert ok is False


# ---------------------------------------------------------------------------
# format_todo_reminder
# ---------------------------------------------------------------------------

class TestFormatTodoReminder:
    def test_empty_list_returns_none(self):
        assert format_todo_reminder([]) is None

    def test_all_completed_returns_none(self):
        todos = [_todo("t1", "completed"), _todo("t2", "completed")]
        assert format_todo_reminder(todos) is None

    def test_pending_todos_formatted(self):
        todos = [_todo("t1", "pending", "Fix the bug"), _todo("t2", "pending", "Write tests")]
        result = format_todo_reminder(todos)
        assert result is not None
        assert "Fix the bug" in result
        assert "Write tests" in result

    def test_in_progress_uses_arrow_marker(self):
        todos = [_todo("t1", "in_progress", "Fix auth")]
        result = format_todo_reminder(todos)
        assert "[>]" in result

    def test_pending_uses_empty_box(self):
        todos = [_todo("t1", "pending", "Write tests")]
        result = format_todo_reminder(todos)
        assert "[ ]" in result

    def test_reminder_header_included(self):
        todos = [_todo("t1", "pending")]
        result = format_todo_reminder(todos)
        assert "[Reminder:" in result

    def test_count_in_header(self):
        todos = [_todo("t1", "pending"), _todo("t2", "in_progress"), _todo("t3", "completed")]
        result = format_todo_reminder(todos)
        # 2 active items (pending + in_progress)
        assert "2" in result

    def test_completed_not_listed(self):
        todos = [
            _todo("t1", "pending", "active task"),
            _todo("t2", "completed", "done task"),
        ]
        result = format_todo_reminder(todos)
        assert "done task" not in result
        assert "active task" in result

    def test_returns_string(self):
        todos = [_todo("t1", "pending", "x")]
        assert isinstance(format_todo_reminder(todos), str)
