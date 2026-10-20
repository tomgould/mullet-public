"""Tests for mullet.harness.sessions.search."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from mullet.harness.sessions.search import search_sessions
from mullet.harness.sessions.store import MessageRecord, SessionStore

pytestmark = pytest.mark.harness


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(base_dir=tmp_path / "sessions")


class TestSessionSearch:
    def test_finds_matching_user_messages(self, store: SessionStore, tmp_path: Path) -> None:
        meta = store.create(cwd=tmp_path / "proj")
        store.append_message(meta.id, MessageRecord(role="user", content="hello world", timestamp=time.time()))
        results = search_sessions("hello", store=store)
        assert len(results) == 1
        assert results[0].session_id == meta.id
        assert results[0].matched_in == "user"
        assert "hello" in results[0].matched_message.lower()

    def test_filters_by_role(self, store: SessionStore, tmp_path: Path) -> None:
        meta = store.create(cwd=tmp_path / "proj")
        store.append_message(meta.id, MessageRecord(role="user", content="search term", timestamp=time.time()))
        store.append_message(meta.id, MessageRecord(role="assistant", content="search term", timestamp=time.time() + 1))
        results = search_sessions("search", store=store, role_filter="user")
        assert len(results) == 1
        assert results[0].matched_in == "user"

    def test_orders_by_recency(self, store: SessionStore, tmp_path: Path) -> None:
        meta = store.create(cwd=tmp_path / "proj")
        t0 = time.time()
        store.append_message(meta.id, MessageRecord(role="user", content="old match", timestamp=t0))
        store.append_message(meta.id, MessageRecord(role="user", content="new match", timestamp=t0 + 10))
        results = search_sessions("match", store=store)
        assert len(results) == 2
        # Newer message should score higher and appear first
        assert results[0].timestamp > results[1].timestamp
        assert results[0].score >= results[1].score

    def test_limit_respected(self, store: SessionStore, tmp_path: Path) -> None:
        meta = store.create(cwd=tmp_path / "proj")
        for i in range(5):
            store.append_message(meta.id, MessageRecord(role="user", content=f"match {i}", timestamp=time.time() + i))
        results = search_sessions("match", store=store, limit=3)
        assert len(results) == 3

    def test_filter_by_project_hash(self, store: SessionStore, tmp_path: Path) -> None:
        from mullet.harness.sessions.project_hash import project_hash
        meta_a = store.create(cwd=tmp_path / "proj_a")
        meta_b = store.create(cwd=tmp_path / "proj_b")
        store.append_message(meta_a.id, MessageRecord(role="user", content="shared term", timestamp=time.time()))
        store.append_message(meta_b.id, MessageRecord(role="user", content="shared term", timestamp=time.time()))
        ph_a = project_hash(tmp_path / "proj_a")
        results = search_sessions("shared", store=store, project_hash=ph_a)
        assert len(results) == 1
        assert results[0].session_id == meta_a.id

    def test_no_match_returns_empty(self, store: SessionStore, tmp_path: Path) -> None:
        meta = store.create(cwd=tmp_path / "proj")
        store.append_message(meta.id, MessageRecord(role="user", content="nothing here", timestamp=time.time()))
        results = search_sessions("xyznonexistent", store=store)
        assert results == []
