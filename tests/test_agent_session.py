"""Tests for session.py missing coverage lines."""

import json
import os
import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from mullet.core.agent import session as session_module
from mullet.core.agent.session import (
    MAX_SESSIONS,
    Session,
    _cleanup_old_sessions,
    _index_session,
    _search_fts5,
    _search_naive,
    generate_title,
    list_recent_sessions,
    rebuild_fts_index,
    search_sessions,
)


@pytest.fixture(autouse=True)
def session_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("mullet.core.agent.session.SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr("mullet.core.agent.session.SESSION_DB", tmp_path / "sessions.db")
    return tmp_path / "sessions"


class TestAgentSessionMissingLines:
    def test_index_session_skips_short_content(self, session_dir):
        """Line 75: content < 5 chars is skipped."""
        conn = sqlite3.connect(str(session_module.SESSION_DB))
        session_module._get_db()  # ensure schema exists
        msgs = [
            {"role": "user", "content": "hi", "timestamp": 1},
            {"role": "user", "content": "hello world", "timestamp": 2},
        ]
        _index_session(conn, "s1", {}, msgs)
        rows = conn.execute(
            "SELECT content FROM session_fts WHERE session_id = ?", ("s1",)
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "hello world"
        conn.close()

    def test_save_fts_index_failure(self, session_dir):
        """Lines 124-125: exception in FTS5 index update is caught."""
        s = Session(session_id="fail_fts")
        s.add_message("user", "hello world")
        with patch("mullet.core.agent.session._get_db", side_effect=sqlite3.Error("boom")):
            s.save()  # should not raise

    def test_load_corrupted_json(self, session_dir):
        """Lines 141-142: corrupted JSON returns None."""
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "bad.json").write_text("not json")
        assert Session.load("bad") is None

    def test_generate_title_empty_user_content(self):
        """Line 151: empty user content is skipped."""
        msgs = [
            {"role": "user", "content": "   "},
            {"role": "user", "content": "real title"},
        ]
        title = generate_title(msgs)
        assert title == "real title"

    def test_search_sessions_fts_fallback(self, session_dir):
        """Lines 169-171: FTS5 search fails, falls back to naive."""
        session_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "id": "naive_session",
            "metadata": {"created_at": time.time()},
            "messages": [
                {
                    "role": "user",
                    "content": "fallback search term",
                    "timestamp": time.time(),
                }
            ],
        }
        (session_dir / "session_naive_session.json").write_text(json.dumps(data))

        with patch(
            "mullet.core.agent.session._search_fts5",
            side_effect=sqlite3.Error("fts5 down"),
        ):
            results = search_sessions("fallback search", max_results=5)
            assert len(results) > 0
            assert results[0]["session_id"] == "naive_session"

    def test_search_fts5_empty_query(self):
        """Line 181: empty query returns empty list."""
        assert _search_fts5("", 5) == []
        assert _search_fts5("   ", 5) == []

    def test_search_naive_no_session_dir(self):
        """Line 209: SESSION_DIR doesn't exist returns []."""
        with patch.object(
            session_module, "SESSION_DIR", Path("/nonexistent")
        ):
            assert _search_naive("query", 5) == []

    def test_search_naive_basic(self, session_dir):
        """Lines 209-241: naive search works."""
        session_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "id": "naive1",
            "metadata": {"created_at": time.time()},
            "messages": [
                {
                    "role": "user",
                    "content": "python testing guide",
                    "timestamp": time.time(),
                },
            ],
        }
        (session_dir / "session_naive1.json").write_text(json.dumps(data))
        results = _search_naive("python testing", 5)
        assert len(results) == 1
        assert results[0]["session_id"] == "naive1"

    def test_search_naive_empty_content(self, session_dir):
        """Lines 222-223: empty content is skipped in naive search."""
        session_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "id": "naive_empty",
            "metadata": {},
            "messages": [
                {"role": "user", "content": "", "timestamp": 1},
                {"role": "user", "content": "hello world", "timestamp": 2},
            ],
        }
        (session_dir / "session_naive_empty.json").write_text(json.dumps(data))
        results = _search_naive("hello world", 5)
        assert len(results) == 1

    def test_search_naive_break_limit(self, session_dir):
        """Lines 234-235: break when enough results collected."""
        session_dir.mkdir(parents=True, exist_ok=True)
        for i in range(20):
            data = {
                "id": f"naive_{i}",
                "metadata": {},
                "messages": [
                    {"role": "user", "content": f"common word {i}", "timestamp": 1}
                ],
            }
            (session_dir / f"session_naive_{i}.json").write_text(json.dumps(data))
        results = _search_naive("common word", 2)
        assert len(results) == 2

    def test_search_naive_exception_logging(self, session_dir, caplog):
        """Lines 236-238: exception during JSON load is logged and skipped."""
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "session_broken.json").write_text("bad json")
        with caplog.at_level("WARNING", logger="mullet.core.agent.session"):
            results = _search_naive("anything", 5)
        assert results == []
        assert "failed to load session" in caplog.text

    def test_list_recent_sessions_db_exception_fallback(self, session_dir):
        """Lines 262-283: DB exception falls back to JSON files."""
        session_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "id": "json_fallback",
            "metadata": {
                "title": "t",
                "created_at": 1.0,
                "provider": "p",
                "project": "pr",
            },
            "messages": [{"role": "user", "content": "hi", "timestamp": 1}],
        }
        (session_dir / "session_json_fallback.json").write_text(json.dumps(data))
        with patch(
            "mullet.core.agent.session._get_db", side_effect=Exception("db down")
        ):
            results = list_recent_sessions(limit=10)
        assert len(results) == 1
        assert results[0]["id"] == "json_fallback"

    def test_list_recent_sessions_json_exception(self, session_dir):
        """Lines 281-282: malformed JSON in fallback is skipped."""
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "session_bad.json").write_text("bad")
        with patch(
            "mullet.core.agent.session._get_db", side_effect=Exception("db down")
        ):
            results = list_recent_sessions(limit=10)
        assert results == []

    def test_list_recent_sessions_fallback_no_dir(self):
        """Line 267: fallback returns [] when SESSION_DIR does not exist."""
        with patch.object(
            session_module, "SESSION_DIR", Path("/nonexistent")
        ):
            with patch(
                "mullet.core.agent.session._get_db", side_effect=Exception("db down")
            ):
                assert list_recent_sessions(limit=10) == []

    def test_rebuild_fts_index_no_dir(self):
        """Line 289: no session dir returns 0."""
        with patch.object(
            session_module, "SESSION_DIR", Path("/nonexistent")
        ):
            assert rebuild_fts_index() == 0

    def test_rebuild_fts_index_malformed_json(self, session_dir, caplog):
        """Lines 300-301: malformed JSON is skipped during rebuild."""
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "session_bad.json").write_text("bad json")
        with caplog.at_level("DEBUG", logger="mullet.core.agent.session"):
            count = rebuild_fts_index()
        assert count == 0

    def test_cleanup_old_sessions_no_dir(self):
        """Line 405: no session dir returns early."""
        with patch.object(
            session_module, "SESSION_DIR", Path("/nonexistent")
        ):
            _cleanup_old_sessions()  # should not raise

    def test_cleanup_old_sessions_removes_oldest(self, session_dir):
        """Lines 408-409: oldest sessions beyond MAX_SESSIONS are removed."""
        session_dir.mkdir(parents=True, exist_ok=True)
        for i in range(MAX_SESSIONS + 5):
            (session_dir / f"session_{i}.json").write_text(f'{{"id":"s{i}"}}')
            os.utime(
                session_dir / f"session_{i}.json",
                (time.time() + i, time.time() + i),
            )
        _cleanup_old_sessions()
        remaining = list(session_dir.glob("session_*.json"))
        assert len(remaining) == MAX_SESSIONS

    # --- Happy-path & additional gap coverage ---

    def test_save_happy_path_updates_fts(self, session_dir):
        """Lines 122-123: save() opens DB and indexes session."""
        s = Session(session_id="save_fts")
        s.add_message("user", "hello world")
        s.save()
        # Verify FTS index was updated by searching
        results = _search_fts5("hello world", 5)
        assert len(results) >= 1
        assert results[0]["session_id"] == "save_fts"

    def test_load_missing_file(self):
        """Line 134: load() returns None for missing file."""
        assert Session.load("definitely_missing") is None

    def test_load_happy_path(self, session_dir):
        """Lines 137-140: load() parses JSON and returns session."""
        session_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "id": "load_test",
            "metadata": {"title": "t", "provider": "p"},
            "messages": [{"role": "user", "content": "hi", "timestamp": 1}],
        }
        (session_dir / "load_test.json").write_text(json.dumps(data))
        loaded = Session.load("load_test")
        assert loaded is not None
        assert loaded.id == "load_test"
        assert loaded.messages == data["messages"]
        assert loaded.metadata == data["metadata"]

    def test_generate_title_untitled(self):
        """Lines 156, 158: generate_title returns 'Untitled session'."""
        assert generate_title([]) == "Untitled session"
        assert generate_title([{"role": "assistant", "content": "hello"}]) == "Untitled session"

    def test_generate_title_truncated(self):
        """Line 156: generate_title appends '...' when truncated."""
        msgs = [{"role": "user", "content": "one two three four five six seven eight nine ten"}]
        title = generate_title(msgs, max_words=5)
        assert title.endswith("...")
        assert "one two three four five" in title

    def test_search_fts5_happy_path(self, session_dir):
        """Lines 182-193: _search_fts5 returns ranked results."""
        s = Session(session_id="fts_happy")
        s.add_message("user", "unique fts query test")
        s.save()
        results = _search_fts5("unique fts", 5)
        assert len(results) == 1
        assert results[0]["session_id"] == "fts_happy"
        assert "score" in results[0]

    def test_list_recent_sessions_db_happy_path(self, session_dir):
        """Lines 249-255: list_recent_sessions returns DB rows."""
        s = Session(session_id="recent_db")
        s.add_message("user", "msg")
        s.metadata["provider"] = "test"
        s.save()
        results = list_recent_sessions(limit=10)
        assert len(results) == 1
        assert results[0]["id"] == "recent_db"
        assert results[0]["provider"] == "test"

    def test_rebuild_fts_index_happy_path(self, session_dir):
        """Lines 298-299: rebuild indexes valid JSON files."""
        session_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "id": "rebuild_ok",
            "metadata": {"title": "t", "created_at": time.time()},
            "messages": [{"role": "user", "content": "rebuild content", "timestamp": time.time()}],
        }
        (session_dir / "session_rebuild_ok.json").write_text(json.dumps(data))
        count = rebuild_fts_index()
        assert count == 1
        results = _search_fts5("rebuild content", 5)
        assert len(results) == 1

    def test_get_session_aggregates_empty_dir(self, session_dir):
        """Lines 338-399: aggregates with empty/missing dir."""
        from mullet.core.agent.session import get_session_aggregates
        agg = get_session_aggregates()
        assert agg.total_sessions == 0
        assert agg.total_messages == 0

    def test_get_session_aggregates_basic(self, session_dir):
        """Lines 338-399: aggregates collect stats correctly."""
        from mullet.core.agent.session import get_session_aggregates
        session_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "id": "agg1",
            "metadata": {
                "created_at": time.time(),
                "provider": "openai",
                "tool_calls": 3,
                "tokens_in": 100,
                "tokens_out": 200,
                "cost_usd": 0.05,
                "tool_call_counts": {"search": 2, "read": 1},
            },
            "messages": [
                {"role": "user", "content": "m1", "timestamp": 1},
                {"role": "assistant", "content": "m2", "timestamp": 2},
            ],
        }
        (session_dir / "session_agg1.json").write_text(json.dumps(data))
        agg = get_session_aggregates()
        assert agg.total_sessions == 1
        assert agg.total_messages == 2
        assert agg.total_tool_calls == 3
        assert agg.total_tokens_in == 100
        assert agg.total_tokens_out == 200
        assert agg.total_cost_usd == 0.05
        assert agg.by_provider["openai"]["calls"] == 1
        assert agg.by_tool["search"] == 2
        assert agg.by_tool["read"] == 1
        assert agg.window_start is not None
        assert agg.window_end is not None

    def test_get_session_aggregates_days_filter(self, session_dir):
        """Lines 348-349: days filter excludes old sessions."""
        from mullet.core.agent.session import get_session_aggregates
        session_dir.mkdir(parents=True, exist_ok=True)
        old_data = {"id": "old", "metadata": {"created_at": 1}, "messages": []}
        (session_dir / "session_old.json").write_text(json.dumps(old_data))
        # Set old mtime so days filter catches it
        os.utime(session_dir / "session_old.json", (1, 1))
        agg = get_session_aggregates(days=1)
        assert agg.total_sessions == 0

    def test_get_session_aggregates_limit(self, session_dir):
        """Line 353: limit restricts number of sessions."""
        from mullet.core.agent.session import get_session_aggregates
        session_dir.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            data = {"id": f"lim{i}", "metadata": {"created_at": time.time()}, "messages": []}
            (session_dir / f"session_lim{i}.json").write_text(json.dumps(data))
            os.utime(session_dir / f"session_lim{i}.json", (time.time() + i, time.time() + i))
        agg = get_session_aggregates(limit=2)
        assert agg.total_sessions == 2

    def test_get_session_aggregates_malformed_file(self, session_dir, caplog):
        """Lines 362-364: malformed session file is skipped with debug log."""
        from mullet.core.agent.session import get_session_aggregates
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "session_bad.json").write_text("not json")
        with caplog.at_level("DEBUG", logger="mullet.core.agent.session"):
            agg = get_session_aggregates()
        assert agg.total_sessions == 0
        assert "skipping malformed" in caplog.text
