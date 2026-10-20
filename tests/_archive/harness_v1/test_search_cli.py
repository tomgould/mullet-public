"""Tests for mullet.cli.search_cli."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from mullet.cli.search_cli import cmd_search
from mullet.harness.sessions.store import MessageRecord, SessionStore

pytestmark = pytest.mark.harness


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(base_dir=tmp_path / "sessions")


class TestSearchCli:
    def test_search_outputs_results(self, store: SessionStore, tmp_path: Path, capsys) -> None:
        meta = store.create(cwd=tmp_path / "proj")
        store.append_message(meta.id, MessageRecord(role="user", content="find me", timestamp=time.time()))
        # Patch SessionStore instantiation inside search_cli to use our store
        orig_init = SessionStore.__init__

        def _patched_init(self, base_dir=None):
            orig_init(self, base_dir=store.base_dir)

        SessionStore.__init__ = _patched_init
        try:
            rc = cmd_search(["find"])
            captured = capsys.readouterr()
            assert rc == 0
            assert "find me" in captured.out or "Search results" in captured.out
        finally:
            SessionStore.__init__ = orig_init

    def test_search_no_results_friendly(self, store: SessionStore, tmp_path: Path, capsys) -> None:
        meta = store.create(cwd=tmp_path / "proj")
        store.append_message(meta.id, MessageRecord(role="user", content="nothing here", timestamp=time.time()))
        orig_init = SessionStore.__init__

        def _patched_init(self, base_dir=None):
            orig_init(self, base_dir=store.base_dir)

        SessionStore.__init__ = _patched_init
        try:
            rc = cmd_search(["xyznonexistent"])
            captured = capsys.readouterr()
            assert rc == 0
            assert "No results" in captured.out
        finally:
            SessionStore.__init__ = orig_init

    def test_search_json_format(self, store: SessionStore, tmp_path: Path, capsys) -> None:
        meta = store.create(cwd=tmp_path / "proj")
        store.append_message(meta.id, MessageRecord(role="user", content="json test", timestamp=time.time()))
        orig_init = SessionStore.__init__

        def _patched_init(self, base_dir=None):
            orig_init(self, base_dir=store.base_dir)

        SessionStore.__init__ = _patched_init
        try:
            rc = cmd_search(["--json", "json"])
            captured = capsys.readouterr()
            assert rc == 0
            data = json.loads(captured.out)
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["session_id"] == meta.id
            assert data[0]["matched_in"] == "user"
        finally:
            SessionStore.__init__ = orig_init

    def test_search_with_role_filter(self, store: SessionStore, tmp_path: Path, capsys) -> None:
        meta = store.create(cwd=tmp_path / "proj")
        store.append_message(meta.id, MessageRecord(role="user", content="role filter test", timestamp=time.time()))
        store.append_message(meta.id, MessageRecord(role="assistant", content="role filter test", timestamp=time.time() + 1))
        orig_init = SessionStore.__init__

        def _patched_init(self, base_dir=None):
            orig_init(self, base_dir=store.base_dir)

        SessionStore.__init__ = _patched_init
        try:
            rc = cmd_search(["--role", "user", "role filter"])
            captured = capsys.readouterr()
            assert rc == 0
            assert "[user]" in captured.out
            assert captured.out.count(meta.id) == 1
        finally:
            SessionStore.__init__ = orig_init
