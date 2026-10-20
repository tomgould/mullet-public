"""Tests for the todo tool."""

import json
from unittest.mock import MagicMock

from mullet.core.tools.todo import TodoStore, handle_todo, register, VALID_STATUSES


# --- TodoStore ---

def test_store_empty():
    s = TodoStore()
    assert s.read() == []
    assert s.has_items() is False


def test_store_write_replace():
    s = TodoStore()
    items = s.write([
        {"id": "1", "content": "First task", "status": "pending"},
        {"id": "2", "content": "Second task", "status": "in_progress"},
    ])
    assert len(items) == 2
    assert items[0]["id"] == "1"
    assert items[1]["status"] == "in_progress"
    assert s.has_items() is True


def test_store_write_replaces_entirely():
    s = TodoStore()
    s.write([{"id": "1", "content": "Old", "status": "pending"}])
    items = s.write([{"id": "2", "content": "New", "status": "pending"}])
    assert len(items) == 1
    assert items[0]["id"] == "2"


def test_store_merge_update():
    s = TodoStore()
    s.write([
        {"id": "1", "content": "Task A", "status": "pending"},
        {"id": "2", "content": "Task B", "status": "pending"},
    ])
    items = s.write([
        {"id": "1", "content": "Task A updated", "status": "completed"},
    ], merge=True)
    assert len(items) == 2
    assert items[0]["content"] == "Task A updated"
    assert items[0]["status"] == "completed"
    assert items[1]["id"] == "2"  # Unchanged


def test_store_merge_add_new():
    s = TodoStore()
    s.write([{"id": "1", "content": "Existing", "status": "pending"}])
    items = s.write([
        {"id": "2", "content": "New task", "status": "pending"},
    ], merge=True)
    assert len(items) == 2
    assert items[1]["id"] == "2"


def test_store_merge_skip_no_id():
    s = TodoStore()
    s.write([{"id": "1", "content": "Task", "status": "pending"}])
    items = s.write([{"content": "No ID", "status": "pending"}], merge=True)
    assert len(items) == 1  # Didn't add the item without id


def test_store_merge_preserves_order():
    s = TodoStore()
    s.write([
        {"id": "a", "content": "First", "status": "pending"},
        {"id": "b", "content": "Second", "status": "pending"},
        {"id": "c", "content": "Third", "status": "pending"},
    ])
    items = s.write([
        {"id": "b", "status": "completed"},
    ], merge=True)
    assert items[0]["id"] == "a"
    assert items[1]["id"] == "b"
    assert items[1]["status"] == "completed"
    assert items[2]["id"] == "c"


def test_store_validate_defaults():
    s = TodoStore()
    items = s.write([{"id": "", "content": "", "status": "invalid"}])
    assert items[0]["id"] == "?"
    assert items[0]["content"] == "(no description)"
    assert items[0]["status"] == "pending"


def test_store_validate_status():
    s = TodoStore()
    for status in VALID_STATUSES:
        items = s.write([{"id": "1", "content": "test", "status": status}])
        assert items[0]["status"] == status


def test_store_read_returns_copy():
    s = TodoStore()
    s.write([{"id": "1", "content": "Task", "status": "pending"}])
    items = s.read()
    items[0]["content"] = "Modified"
    assert s.read()[0]["content"] == "Task"  # Original unchanged


# --- format_for_injection ---

def test_injection_empty():
    s = TodoStore()
    assert s.format_for_injection() is None


def test_injection_all_completed():
    s = TodoStore()
    s.write([{"id": "1", "content": "Done", "status": "completed"}])
    assert s.format_for_injection() is None  # No active items


def test_injection_active_items():
    s = TodoStore()
    s.write([
        {"id": "1", "content": "In progress", "status": "in_progress"},
        {"id": "2", "content": "Pending", "status": "pending"},
        {"id": "3", "content": "Done", "status": "completed"},
    ])
    result = s.format_for_injection()
    assert result is not None
    assert "[>]" in result  # in_progress marker
    assert "[ ]" in result  # pending marker
    assert "Done" not in result  # completed filtered out


# --- handle_todo ---

def test_handle_todo_read():
    store = TodoStore()
    store.write([{"id": "1", "content": "Task", "status": "pending"}])
    result = json.loads(handle_todo(_store=store))
    assert result["summary"]["total"] == 1
    assert result["summary"]["pending"] == 1


def test_handle_todo_write():
    store = TodoStore()
    result = json.loads(handle_todo(
        todos=[{"id": "1", "content": "New", "status": "in_progress"}],
        _store=store,
    ))
    assert result["summary"]["in_progress"] == 1
    assert result["todos"][0]["content"] == "New"


def test_handle_todo_merge():
    store = TodoStore()
    store.write([{"id": "1", "content": "Old", "status": "pending"}])
    result = json.loads(handle_todo(
        todos=[{"id": "1", "status": "completed"}],
        merge=True,
        _store=store,
    ))
    assert result["summary"]["completed"] == 1


# --- register ---

def test_register():
    registry = MagicMock()
    register(registry)
    registry.register.assert_called_once()
    call_kwargs = registry.register.call_args
    assert "todo" in str(call_kwargs)
