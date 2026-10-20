"""Tests for mullet.harness.sessions.tags."""

from __future__ import annotations

from pathlib import Path

import pytest

from mullet.harness.sessions.store import SessionStore
from mullet.harness.sessions.tags import list_by_tag, list_pinned, remove_tag, tag_session

pytestmark = pytest.mark.harness


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(base_dir=tmp_path / "sessions")


class TestSessionTags:
    def test_tag_session_persists(self, store: SessionStore, tmp_path: Path) -> None:
        meta = store.create(cwd=tmp_path / "proj")
        result = tag_session(store, meta.id, tags=["bugfix", "auth"])
        assert "bugfix" in result["tags"]
        assert "auth" in result["tags"]
        # Verify by loading meta directly
        loaded_meta, _, _ = store.load(meta.id)
        assert "bugfix" in loaded_meta.tags
        assert "auth" in loaded_meta.tags

    def test_remove_tag_persists(self, store: SessionStore, tmp_path: Path) -> None:
        meta = store.create(cwd=tmp_path / "proj")
        tag_session(store, meta.id, tags=["keep", "remove"])
        result = remove_tag(store, meta.id, "remove")
        assert "remove" not in result["tags"]
        assert "keep" in result["tags"]
        loaded_meta, _, _ = store.load(meta.id)
        assert loaded_meta.tags == ["keep"]

    def test_pin_session_sets_meta(self, store: SessionStore, tmp_path: Path) -> None:
        meta = store.create(cwd=tmp_path / "proj")
        result = tag_session(store, meta.id, tags=[], pin=True)
        assert result["pinned"] is True
        loaded_meta, _, _ = store.load(meta.id)
        assert loaded_meta.pinned is True

    def test_list_pinned_returns_only_pinned(self, store: SessionStore, tmp_path: Path) -> None:
        meta_a = store.create(cwd=tmp_path / "proj_a")
        meta_b = store.create(cwd=tmp_path / "proj_b")
        tag_session(store, meta_a.id, tags=[], pin=True)
        pinned = list_pinned(store)
        assert len(pinned) == 1
        assert pinned[0].id == meta_a.id

    def test_list_by_tag_filters(self, store: SessionStore, tmp_path: Path) -> None:
        meta_a = store.create(cwd=tmp_path / "proj_a")
        meta_b = store.create(cwd=tmp_path / "proj_b")
        tag_session(store, meta_a.id, tags=["frontend"])
        tag_session(store, meta_b.id, tags=["backend"])
        frontend = list_by_tag(store, "frontend")
        assert len(frontend) == 1
        assert frontend[0].id == meta_a.id
