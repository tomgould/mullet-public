"""Tests for session branching, undo, and rewind."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from mullet.harness.commands.branch import handle_branch_command
from mullet.harness.commands.undo import handle_undo_command
from mullet.harness.sessions.branch import BranchResult, fork_session, rewind_to_index, undo_last_turn
from mullet.harness.sessions.store import MessageRecord, SessionStore


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(base_dir=tmp_path / "sessions")


@pytest.mark.harness
def test_undo_last_assistant_turn_drops_last_pair(store: SessionStore, tmp_path: Path) -> None:
    meta = store.create(cwd=tmp_path / "proj")
    store.append_message(meta.id, MessageRecord(role="user", content="u1", timestamp=time.time()))
    store.append_message(meta.id, MessageRecord(role="assistant", content="a1", timestamp=time.time()))
    store.append_message(
        meta.id,
        MessageRecord(role="tool", content="t1", timestamp=time.time(), tool_call_id="tc1"),
    )

    result = undo_last_turn(store, meta.id)
    assert result["ok"] is True
    assert result["removed_count"] == 2
    assert result["new_message_count"] == 1

    _meta, messages, _events = store.load(meta.id)
    assert len(messages) == 1
    assert messages[0].role == "user"


@pytest.mark.harness
def test_undo_with_no_assistant_message_is_noop(store: SessionStore, tmp_path: Path) -> None:
    meta = store.create(cwd=tmp_path / "proj")
    store.append_message(meta.id, MessageRecord(role="user", content="u1", timestamp=time.time()))

    result = undo_last_turn(store, meta.id)
    assert result["ok"] is True
    assert result["removed_count"] == 0
    assert result["new_message_count"] == 1

    _meta, messages, _events = store.load(meta.id)
    assert len(messages) == 1
    assert messages[0].role == "user"


@pytest.mark.harness
def test_undo_n_truncates_n_turns(store: SessionStore, tmp_path: Path) -> None:
    meta = store.create(cwd=tmp_path / "proj")
    store.append_message(meta.id, MessageRecord(role="user", content="u1", timestamp=time.time()))
    store.append_message(meta.id, MessageRecord(role="assistant", content="a1", timestamp=time.time()))
    store.append_message(
        meta.id,
        MessageRecord(role="tool", content="t1", timestamp=time.time(), tool_call_id="tc1"),
    )
    store.append_message(meta.id, MessageRecord(role="user", content="u2", timestamp=time.time()))
    store.append_message(meta.id, MessageRecord(role="assistant", content="a2", timestamp=time.time()))

    result = undo_last_turn(store, meta.id, turns=2)
    assert result["ok"] is True
    assert result["removed_count"] == 4
    assert result["new_message_count"] == 1

    _meta, messages, _events = store.load(meta.id)
    assert len(messages) == 1
    assert messages[0].content == "u1"


@pytest.mark.harness
def test_fork_session_copies_messages_up_to_index(store: SessionStore, tmp_path: Path) -> None:
    meta = store.create(cwd=tmp_path / "proj")
    for i in range(5):
        role = "user" if i % 2 == 0 else "assistant"
        store.append_message(meta.id, MessageRecord(role=role, content=f"m{i}", timestamp=time.time()))

    result = fork_session(store, meta.id, at_index=2)
    assert isinstance(result, BranchResult)
    assert result.new_session_id != meta.id
    assert result.branched_at_message_index == 2
    assert result.truncated_message_count == 2

    _meta, messages, _events = store.load(result.new_session_id)
    assert len(messages) == 3
    assert messages[2].content == "m2"


@pytest.mark.harness
def test_fork_creates_new_session_id_with_parent_link(store: SessionStore, tmp_path: Path) -> None:
    meta = store.create(cwd=tmp_path / "proj")
    store.append_message(meta.id, MessageRecord(role="user", content="u1", timestamp=time.time()))

    result = fork_session(store, meta.id, at_index=0)
    assert result.parent_session_id == meta.id

    new_meta, _messages, _events = store.load(result.new_session_id)
    assert new_meta.parent_session_id == meta.id


@pytest.mark.harness
def test_rewind_to_index_truncates_correctly(store: SessionStore, tmp_path: Path) -> None:
    meta = store.create(cwd=tmp_path / "proj")
    for i in range(5):
        role = "user" if i % 2 == 0 else "assistant"
        store.append_message(meta.id, MessageRecord(role=role, content=f"m{i}", timestamp=time.time()))

    result = rewind_to_index(store, meta.id, target_index=2)
    assert result["ok"] is True
    assert result["removed_count"] == 2
    assert result["new_message_count"] == 3

    _meta, messages, _events = store.load(meta.id)
    assert len(messages) == 3
    assert messages[-1].content == "m2"


@pytest.mark.harness
def test_branch_command_creates_fork(store: SessionStore, tmp_path: Path) -> None:
    meta = store.create(cwd=tmp_path / "proj")
    store.append_message(meta.id, MessageRecord(role="user", content="u1", timestamp=time.time()))

    engine = type("MockEngine", (), {"store": store, "session_id": meta.id})()
    out = handle_branch_command("", store=store, engine=engine, session_id=meta.id)
    assert "Branched session" in out
    assert "message" in out


@pytest.mark.harness
def test_branch_list_shows_genealogy(store: SessionStore, tmp_path: Path) -> None:
    meta = store.create(cwd=tmp_path / "proj")
    store.append_message(meta.id, MessageRecord(role="user", content="u1", timestamp=time.time()))
    fork_session(store, meta.id, at_index=0)

    out = handle_branch_command("list", store=store)
    assert "genealogy" in out.lower()
    assert "forked from" in out


@pytest.mark.harness
def test_undo_command_calls_undo_last_turn(store: SessionStore, tmp_path: Path) -> None:
    meta = store.create(cwd=tmp_path / "proj")
    store.append_message(meta.id, MessageRecord(role="user", content="u1", timestamp=time.time()))
    store.append_message(meta.id, MessageRecord(role="assistant", content="a1", timestamp=time.time()))

    out = handle_undo_command("", store=store, session_id=meta.id)
    assert "Undid" in out
    assert "message(s)" in out

    _meta, messages, _events = store.load(meta.id)
    assert len(messages) == 1
    assert messages[0].role == "user"


@pytest.mark.harness
def test_undo_logs_event(store: SessionStore, tmp_path: Path) -> None:
    meta = store.create(cwd=tmp_path / "proj")
    store.append_message(meta.id, MessageRecord(role="user", content="u1", timestamp=time.time()))
    store.append_message(meta.id, MessageRecord(role="assistant", content="a1", timestamp=time.time()))

    undo_last_turn(store, meta.id)
    _meta, _messages, events = store.load(meta.id)
    undo_events = [e for e in events if e.event_type == "session.undo"]
    assert len(undo_events) == 1
    assert undo_events[0].payload is not None
    assert undo_events[0].payload.get("removed_count") == 1
