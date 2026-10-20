"""Tests for session management and search."""

import json
import time
import pytest
from mullet.core.agent.session import (
    Session, search_sessions, list_recent_sessions, generate_title,
    rebuild_fts_index, _search_fts5,
)


@pytest.fixture(autouse=True)
def session_dir(tmp_path, monkeypatch):
    """Use temp dir for sessions and FTS5 database."""
    monkeypatch.setattr("mullet.core.agent.session.SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr("mullet.core.agent.session.SESSION_DB", tmp_path / "sessions.db")
    return tmp_path / "sessions"


def test_session_create():
    s = Session()
    assert s.id.startswith("session_")
    assert s.messages == []


def test_session_add_message():
    s = Session()
    s.add_message("user", "hello")
    s.add_message("assistant", "hi there")
    assert len(s.messages) == 2
    assert s.messages[0]["role"] == "user"
    assert s.messages[1]["content"] == "hi there"


def test_session_save_and_load(session_dir):
    s = Session(session_id="test_session")
    s.add_message("user", "test message")
    s.metadata["provider"] = "local"
    s.save()

    loaded = Session.load("test_session")
    assert loaded is not None
    assert len(loaded.messages) == 1
    assert loaded.messages[0]["content"] == "test message"
    assert loaded.metadata["provider"] == "local"


def test_session_load_missing():
    loaded = Session.load("nonexistent")
    assert loaded is None


def test_search_sessions(session_dir):
    # Create a session with searchable content
    s = Session(session_id="session_search_test")
    s.add_message("user", "fix the authentication bug in login module")
    s.add_message("assistant", "I found the auth issue in login.py line 42")
    s.save()

    results = search_sessions("authentication login")
    assert len(results) > 0
    assert any("auth" in r["content"].lower() for r in results)


def test_search_sessions_no_match(session_dir):
    s = Session(session_id="session_nomatch")
    s.add_message("user", "hello world")
    s.save()

    results = search_sessions("quantum_mechanics_unlikely_match")
    assert len(results) == 0


def test_list_recent_sessions(session_dir):
    for i in range(3):
        s = Session(session_id=f"session_{1000+i}")
        s.add_message("user", f"message {i}")
        s.metadata["provider"] = "local"
        s.save()

    sessions = list_recent_sessions(limit=10)
    assert len(sessions) == 3
    assert all(s["provider"] == "local" for s in sessions)


# --- Title generation tests ---

def test_generate_title_simple():
    messages = [{"role": "user", "content": "fix the login bug"}]
    title = generate_title(messages)
    assert title == "fix the login bug"


def test_generate_title_truncated():
    messages = [{"role": "user", "content": "refactor the entire authentication system to support OAuth2 with JWT tokens and handle all security edge cases"}]
    title = generate_title(messages, max_words=7)
    assert title.endswith("...")
    assert len(title.split()) <= 8  # 7 words + "..."


def test_generate_title_strips_refs():
    messages = [{"role": "user", "content": "review @file:main.py and @diff please"}]
    title = generate_title(messages)
    assert "@" not in title
    assert "review" in title


def test_generate_title_no_user_messages():
    messages = [{"role": "assistant", "content": "hello"}]
    title = generate_title(messages)
    assert title == "Untitled session"


def test_generate_title_empty_messages():
    title = generate_title([])
    assert title == "Untitled session"


def test_session_auto_titles_on_save(session_dir):
    s = Session(session_id="titled_session")
    s.add_message("user", "implement user authentication")
    s.save()

    loaded = Session.load("titled_session")
    assert loaded.metadata.get("title") == "implement user authentication"


def test_list_sessions_shows_titles(session_dir):
    s = Session(session_id="session_with_title")
    s.add_message("user", "add dark mode toggle")
    s.save()

    sessions = list_recent_sessions(limit=10)
    assert any(s.get("title") == "add dark mode toggle" for s in sessions)


# --- FTS5 full-text search tests ---

def test_fts5_search_basic(session_dir):
    """FTS5 search should find sessions by content."""
    s = Session(session_id="fts_test_1")
    s.add_message("user", "fix the authentication bug in login module")
    s.add_message("assistant", "Found the issue in login.py at line 42")
    s.save()

    results = _search_fts5("authentication login", 5)
    assert len(results) > 0
    assert results[0]["session_id"] == "fts_test_1"


def test_fts5_search_no_match(session_dir):
    s = Session(session_id="fts_nomatch")
    s.add_message("user", "hello world")
    s.save()

    results = _search_fts5("quantum_mechanics_unlikely", 5)
    assert len(results) == 0


def test_fts5_search_returns_snippets(session_dir):
    s = Session(session_id="fts_snippet")
    s.add_message("user", "implement the payment processing module with Stripe integration")
    s.save()

    results = _search_fts5("payment Stripe", 5)
    assert len(results) > 0
    # Snippet should contain highlighted terms
    assert "payment" in results[0]["content"].lower() or "stripe" in results[0]["content"].lower()


def test_fts5_search_multiple_sessions(session_dir):
    for i in range(3):
        s = Session(session_id=f"fts_multi_{i}")
        s.add_message("user", f"database migration issue number {i}")
        s.save()

    results = _search_fts5("database migration", 10)
    assert len(results) == 3


def test_fts5_porter_stemming(session_dir):
    """FTS5 with porter tokenizer should match stemmed forms."""
    s = Session(session_id="fts_stem")
    s.add_message("user", "the authentication system is authenticating users")
    s.save()

    # "authenticate" should match "authenticating" and "authentication"
    results = _search_fts5("authenticate", 5)
    assert len(results) > 0


def test_rebuild_fts_index(session_dir):
    """rebuild_fts_index should re-index all JSON files."""
    # Create sessions directly as JSON (simulating pre-FTS5 sessions)
    session_dir.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        data = {
            "id": f"session_rebuild_{i}",
            "metadata": {"title": f"Session {i}", "created_at": time.time()},
            "messages": [{"role": "user", "content": f"rebuild test message {i}", "timestamp": time.time()}],
        }
        (session_dir / f"session_rebuild_{i}.json").write_text(json.dumps(data))

    count = rebuild_fts_index()
    assert count == 3

    # Should be searchable now
    results = _search_fts5("rebuild test", 10)
    assert len(results) == 3


def test_session_save_updates_fts_index(session_dir):
    """Saving a session should update the FTS5 index."""
    s = Session(session_id="fts_save_test")
    s.add_message("user", "unique searchable content xyz123")
    s.save()

    results = _search_fts5("xyz123", 5)
    assert len(results) == 1

    # Update the session and save again
    s.add_message("assistant", "I found the xyz123 issue")
    s.save()

    results = _search_fts5("xyz123", 5)
    assert len(results) == 2  # Both messages now indexed


def test_list_recent_sessions_from_db(session_dir):
    """list_recent_sessions should use DB when available."""
    for i in range(3):
        s = Session(session_id=f"db_list_{i}")
        s.add_message("user", f"message {i}")
        s.metadata["provider"] = "gemini"
        s.save()

    sessions = list_recent_sessions(limit=10)
    assert len(sessions) == 3
    assert all(s["provider"] == "gemini" for s in sessions)
