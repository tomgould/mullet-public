"""Tests for mullet.harness.sessions.todos, tools.todo_write, and commands.todos."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mullet.harness.commands.todos import handle_todos_command
from mullet.harness.sessions.store import SessionStore
from mullet.harness.sessions.todos import TodoStore
from mullet.harness.tools.todo_write import execute as todo_write_execute, SCHEMA, TOOL_NAME

pytestmark = pytest.mark.harness


class FakeSessionStore:
    """Minimal stand-in for SessionStore that keeps todos in one directory."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        d = self.base_dir / session_id
        d.mkdir(exist_ok=True)
        return d


@pytest.fixture
def todo_store(tmp_path: Path) -> TodoStore:
    store = FakeSessionStore(tmp_path / "sessions")
    return TodoStore(store, "sess-123")


@pytest.fixture
def real_session_store(tmp_path: Path) -> SessionStore:
    return SessionStore(base_dir=tmp_path / "sessions")


class TestTodoStore:
    def test_list_empty(self, todo_store: TodoStore) -> None:
        assert todo_store.list() == []

    def test_add_and_list(self, todo_store: TodoStore) -> None:
        item = todo_store.add("Build feature X", status="pending")
        assert item.content == "Build feature X"
        assert item.status == "pending"
        assert len(item.id) == 36  # UUID4
        items = todo_store.list()
        assert len(items) == 1
        assert items[0].content == "Build feature X"

    def test_update_status(self, todo_store: TodoStore) -> None:
        item = todo_store.add("Task A")
        updated = todo_store.update(item.id, status="in_progress")
        assert updated is not None
        assert updated.status == "in_progress"
        assert updated.id == item.id
        assert updated.updated_at >= item.updated_at

    def test_update_content(self, todo_store: TodoStore) -> None:
        item = todo_store.add("Old text")
        updated = todo_store.update(item.id, content="New text")
        assert updated is not None
        assert updated.content == "New text"

    def test_update_missing_returns_none(self, todo_store: TodoStore) -> None:
        assert todo_store.update("no-such-id", status="completed") is None

    def test_remove(self, todo_store: TodoStore) -> None:
        item = todo_store.add("Remove me")
        assert todo_store.remove(item.id) is True
        assert todo_store.list() == []

    def test_remove_missing_returns_false(self, todo_store: TodoStore) -> None:
        assert todo_store.remove("no-such-id") is False

    def test_clear(self, todo_store: TodoStore) -> None:
        todo_store.add("One")
        todo_store.add("Two")
        count = todo_store.clear()
        assert count == 2
        assert todo_store.list() == []

    def test_persists_to_file(self, todo_store: TodoStore) -> None:
        todo_store.add("Persisted")
        path = todo_store._file_path()
        assert path.exists()
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["content"] == "Persisted"

    def test_replace_all(self, todo_store: TodoStore) -> None:
        todo_store.add("Old 1")
        todo_store.add("Old 2")
        items = todo_store.replace_all([
            {"content": "New 1", "status": "pending"},
            {"content": "New 2", "status": "completed", "activeForm": "form-A"},
        ])
        assert len(items) == 2
        assert items[0].content == "New 1"
        assert items[1].status == "completed"
        assert items[1].activeForm == "form-A"
        loaded = todo_store.list()
        assert len(loaded) == 2
        assert loaded[1].content == "New 2"

    def test_invalid_status_defaults_to_pending(self, todo_store: TodoStore) -> None:
        item = todo_store.add("Bad status", status="not_real")
        assert item.status == "pending"
        items = todo_store.replace_all([{"content": "X", "status": "bogus"}])
        assert items[0].status == "pending"


class TestTodoWriteTool:
    def test_execute_replaces_todos(self, real_session_store: SessionStore, tmp_path: Path) -> None:
        meta = real_session_store.create(cwd=tmp_path / "proj")
        result = todo_write_execute(
            [
                {"content": "Write tests", "status": "in_progress"},
                {"content": "Ship it", "status": "pending"},
            ],
            session_id=meta.id,
            store=real_session_store,
        )
        assert result["total"] == 2
        assert len(result["todos"]) == 2
        assert result["todos"][0]["content"] == "Write tests"

    def test_schema_has_required_fields(self) -> None:
        assert SCHEMA["name"] == "TodoWrite"
        assert "todos" in SCHEMA["input_schema"]["properties"]
        assert "content" in SCHEMA["input_schema"]["properties"]["todos"]["items"]["properties"]
        assert "status" in SCHEMA["input_schema"]["properties"]["todos"]["items"]["properties"]

    def test_tool_name_constant(self) -> None:
        assert TOOL_NAME == "TodoWrite"


class TestTodosCommand:
    def test_list_empty(self, todo_store: TodoStore) -> None:
        out = handle_todos_command("list", store=todo_store._store, session_id=todo_store._session_id)
        assert out == "No todos."

    def test_add_and_list(self, todo_store: TodoStore) -> None:
        out = handle_todos_command("add Build widget", store=todo_store._store, session_id=todo_store._session_id)
        assert "Added todo" in out
        out2 = handle_todos_command("list", store=todo_store._store, session_id=todo_store._session_id)
        assert "Build widget" in out2

    def test_done_by_prefix(self, todo_store: TodoStore) -> None:
        item = todo_store.add("Finish this")
        prefix = item.id[:4]
        out = handle_todos_command(f"done {prefix}", store=todo_store._store, session_id=todo_store._session_id)
        assert "Marked" in out
        assert todo_store.list()[0].status == "completed"

    def test_done_ambiguous(self, todo_store: TodoStore) -> None:
        todo_store.add("Task 1")
        todo_store.add("Task 2")
        out = handle_todos_command("done 0000", store=todo_store._store, session_id=todo_store._session_id)
        assert "Ambiguous" in out or "No todo found" in out

    def test_clear(self, todo_store: TodoStore) -> None:
        todo_store.add("A")
        todo_store.add("B")
        out = handle_todos_command("clear", store=todo_store._store, session_id=todo_store._session_id)
        assert "Cleared 2" in out
        assert todo_store.list() == []

    def test_show_returns_text(self, todo_store: TodoStore) -> None:
        todo_store.add("Rich task")
        out = handle_todos_command("show", store=todo_store._store, session_id=todo_store._session_id)
        assert "Rich task" in out

    def test_unknown_subcommand(self, todo_store: TodoStore) -> None:
        out = handle_todos_command("nope", store=todo_store._store, session_id=todo_store._session_id)
        assert "Unknown subcommand" in out
