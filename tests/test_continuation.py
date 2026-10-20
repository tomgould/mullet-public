"""Tests for continuation enforcement."""

from mullet.core.agent.continuation import (
    should_continue,
    format_todo_reminder,
    CONTINUATION_REMINDERS,
)


# --- should_continue ---

def test_empty_response():
    cont, msg = should_continue("", 0)
    assert cont is False


def test_genuine_completion():
    cont, msg = should_continue("All tests pass and everything is verified working.", 5)
    assert cont is False


def test_no_stop_phrase():
    cont, msg = should_continue("I updated the auth module with OAuth2 support.", 3)
    assert cont is False


def test_stop_phrase_no_todos():
    cont, msg = should_continue("I've completed the task. Let me know if you need anything else.", 3)
    assert cont is False  # No pending todos, so OK to stop


def test_stop_phrase_with_pending_todos():
    todos = [
        {"id": "1", "content": "Fix auth", "status": "completed"},
        {"id": "2", "content": "Add tests", "status": "pending"},
        {"id": "3", "content": "Update docs", "status": "pending"},
    ]
    cont, msg = should_continue(
        "I've completed the auth fix. Let me know if you need anything else.",
        3,
        todo_items=todos,
    )
    assert cont is True
    assert msg is not None
    assert "2" in msg  # Pending count


def test_stop_phrase_all_completed():
    todos = [
        {"id": "1", "content": "Fix auth", "status": "completed"},
        {"id": "2", "content": "Add tests", "status": "completed"},
    ]
    cont, msg = should_continue(
        "That's all done. Feel free to ask if you need more.",
        5,
        todo_items=todos,
    )
    assert cont is False  # All completed, OK to stop


def test_in_progress_counts_as_pending():
    todos = [
        {"id": "1", "content": "Refactoring", "status": "in_progress"},
    ]
    cont, msg = should_continue(
        "I'm done with everything. Hope this helps!",
        2,
        todo_items=todos,
    )
    assert cont is True


def test_reminder_rotation():
    todos = [{"id": "1", "content": "task", "status": "pending"}]
    msgs = set()
    for i in range(len(CONTINUATION_REMINDERS)):
        _, msg = should_continue("I'm done. Let me know if you need more.", 1, todos, iteration=i)
        if msg:
            msgs.add(msg.split("\n")[0])  # First line is the reminder
    assert len(msgs) == len(CONTINUATION_REMINDERS)


def test_no_tools_short_response():
    """Short response with no tools should not trigger continuation."""
    todos = [{"id": "1", "content": "task", "status": "pending"}]
    cont, msg = should_continue("I'll fix that.", 0, todos)
    assert cont is False  # Let intermediate ack handler deal with this


# --- format_todo_reminder ---

def test_reminder_empty():
    assert format_todo_reminder([]) is None
    assert format_todo_reminder(None) is None


def test_reminder_all_completed():
    todos = [{"id": "1", "content": "Done", "status": "completed"}]
    assert format_todo_reminder(todos) is None


def test_reminder_active_items():
    todos = [
        {"id": "1", "content": "Fix auth", "status": "in_progress"},
        {"id": "2", "content": "Add tests", "status": "pending"},
        {"id": "3", "content": "Done thing", "status": "completed"},
    ]
    result = format_todo_reminder(todos)
    assert result is not None
    assert "2 tasks remaining" in result
    assert "[>]" in result  # in_progress marker
    assert "[ ]" in result  # pending marker
    assert "Done thing" not in result  # completed filtered


def test_reminder_format():
    todos = [{"id": "a", "content": "Write tests", "status": "pending"}]
    result = format_todo_reminder(todos)
    assert "1 tasks remaining" in result
    assert "Write tests" in result
