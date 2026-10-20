"""Tests for mullet.harness.commands.resume."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mullet.harness.commands.resume import handle_resume_command
from mullet.harness.sessions.store import MessageRecord, SessionStore

pytestmark = pytest.mark.harness


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(base_dir=tmp_path / "sessions")


def _seed_session(store: SessionStore, cwd: Path, first_user: str) -> str:
    meta = store.create(cwd=cwd)
    store.append_message(
        meta.id,
        MessageRecord(role="user", content=first_user, timestamp=time.time()),
    )
    return meta.id


def test_resume_no_args_lists_sessions(store: SessionStore, tmp_path: Path) -> None:
    _seed_session(store, tmp_path / "proj_a", "first prompt A")
    # Ensure ordering by sleeping a tick.
    time.sleep(0.01)
    sid_b = _seed_session(store, tmp_path / "proj_b", "first prompt B")

    out = handle_resume_command("", store=store)
    # Most recent first.
    lines = out.splitlines()
    assert len(lines) == 2
    assert "[0]" in lines[0]
    assert "[1]" in lines[1]
    # The newer session (sid_b) should be at index 0; shortened id is the
    # final dash segment.
    short_b = sid_b.rsplit("-", 1)[-1]
    assert short_b in lines[0]
    assert "msg" in lines[0]
    assert "first prompt" in out


def test_resume_no_args_empty_returns_friendly_message(
    store: SessionStore,
) -> None:
    out = handle_resume_command("", store=store)
    assert out == "No sessions found for this project."


def test_resume_by_index_sets_session_id(
    store: SessionStore, tmp_path: Path
) -> None:
    sid = _seed_session(store, tmp_path / "proj", "hello")
    engine = SimpleNamespace()
    out = handle_resume_command("0", engine=engine, store=store)
    assert "Resumed" in out
    assert engine.session_id == sid


def test_resume_by_id_sets_session_id(
    store: SessionStore, tmp_path: Path
) -> None:
    sid = _seed_session(store, tmp_path / "proj", "hello")
    engine = SimpleNamespace()
    out = handle_resume_command(sid, engine=engine, store=store)
    assert "Resumed" in out
    assert engine.session_id == sid


def test_resume_unknown_id_returns_not_found(store: SessionStore) -> None:
    out = handle_resume_command("sess-does-not-exist", store=store)
    assert "Session not found." in out


def test_resume_search_filters(store: SessionStore, tmp_path: Path) -> None:
    _seed_session(store, tmp_path / "proj_a", "fix the auth bug")
    time.sleep(0.01)
    _seed_session(store, tmp_path / "proj_b", "refactor the database layer")

    out = handle_resume_command("search auth", store=store)
    assert "auth" in out.lower()
    assert "database" not in out.lower()
