"""Tests for mullet.core.tools.session — session_search and session_list handlers."""

import json
from unittest.mock import MagicMock, patch

from mullet.core.tools.session import session_search, session_list, register


def test_session_search_no_results():
    with patch("mullet.core.tools.session.search_sessions", return_value=[]):
        result = session_search("missing query")
    assert result == "No relevant past sessions found."


def test_session_search_with_results():
    fake_results = [
        {"timestamp": 1700000000, "role": "user", "content": "hello world"},
        {"timestamp": 1700000010, "role": "assistant", "content": "hi there"},
    ]
    with patch("mullet.core.tools.session.search_sessions", return_value=fake_results):
        result = session_search("hello", max_results=2)
    assert "hello world" in result
    assert "hi there" in result
    assert "(user)" in result
    assert "(assistant)" in result


def test_session_search_null_timestamp():
    fake_results = [
        {"timestamp": None, "role": "user", "content": "no timestamp"},
    ]
    with patch("mullet.core.tools.session.search_sessions", return_value=fake_results):
        result = session_search("x")
    assert "?" in result
    assert "no timestamp" in result


def test_session_list_no_sessions():
    with patch("mullet.core.tools.session.list_recent_sessions", return_value=[]):
        result = session_list()
    assert result == "No past sessions found."


def test_session_list_with_sessions():
    fake_sessions = [
        {"id": "abc123", "title": "First session", "messages": 5},
        {"id": "def456", "title": "Second session", "messages": 3},
    ]
    with patch("mullet.core.tools.session.list_recent_sessions", return_value=fake_sessions):
        result = session_list(limit=2)
    parsed = json.loads(result)
    assert len(parsed) == 2
    assert parsed[0]["id"] == "abc123"


def test_register_adds_both_tools():
    registry = MagicMock()
    register(registry)
    calls = {c.kwargs["name"] for c in registry.register.call_args_list}
    assert "session_search" in calls
    assert "session_list" in calls
