"""Tests for mullet.harness.sessions.resume."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from mullet.harness.sessions.resume import (
    ResumeContext,
    resume,
    trim_to_last_user_turn,
)
from mullet.harness.sessions.store import (
    EventRecord,
    MessageRecord,
    SessionStore,
)

pytestmark = pytest.mark.harness


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(base_dir=tmp_path / "sessions")


def test_resume_loads_existing_session(store: SessionStore, tmp_path: Path) -> None:
    meta = store.create(cwd=tmp_path / "proj")
    store.append_message(
        meta.id,
        MessageRecord(role="user", content="hi", timestamp=time.time()),
    )
    store.append_message(
        meta.id,
        MessageRecord(role="assistant", content="hello", timestamp=time.time()),
    )
    store.append_event(
        meta.id,
        EventRecord(event_type="user.prompt", timestamp=time.time(), payload={}),
    )

    ctx = resume(meta.id, store=store)
    assert isinstance(ctx, ResumeContext)
    assert ctx.meta.id == meta.id
    assert len(ctx.messages) == 2
    assert ctx.messages[0].role == "user"
    assert ctx.messages[1].role == "assistant"
    assert len(ctx.events) == 1


def test_resume_missing_session_raises_filenotfounderror(
    store: SessionStore,
) -> None:
    with pytest.raises(FileNotFoundError):
        resume("sess-does-not-exist", store=store)


def test_trim_to_last_user_turn_drops_trailing_assistant() -> None:
    msgs = [
        MessageRecord(role="user", content="q1", timestamp=1.0),
        MessageRecord(role="assistant", content="a1", timestamp=2.0),
        MessageRecord(role="user", content="q2", timestamp=3.0),
        MessageRecord(role="assistant", content="a2", timestamp=4.0),
    ]
    trimmed = trim_to_last_user_turn(msgs)
    assert len(trimmed) == 3
    assert trimmed[-1].role == "user"
    assert trimmed[-1].content == "q2"


def test_trim_keeps_full_history_when_user_is_last() -> None:
    msgs = [
        MessageRecord(role="user", content="q1", timestamp=1.0),
        MessageRecord(role="assistant", content="a1", timestamp=2.0),
        MessageRecord(role="user", content="q2", timestamp=3.0),
    ]
    trimmed = trim_to_last_user_turn(msgs)
    assert len(trimmed) == 3
    assert trimmed == msgs
