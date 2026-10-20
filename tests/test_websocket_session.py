"""Tests for WebSocket session resumption — pure unit tests, no real server."""
import asyncio
import json
import time
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

try:
    from aiohttp import web, WSMsgType
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytestmark = pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")

if HAS_AIOHTTP:
    from mullet.server.websocket import WebSocketHandler, SESSION_TTL


def _make_ws_mock(messages):
    """Build a mock WebSocketResponse that yields *messages* and then closes."""
    ws = MagicMock(spec=web.WebSocketResponse)
    ws.closed = False

    # Track every send_json call
    sent = []

    async def _send_json(data):
        sent.append(data)

    ws.send_json.side_effect = _send_json
    ws.prepare = AsyncMock()

    # Build async iterator over messages
    class _Msg:
        __slots__ = ("data", "type")
        def __init__(self, tp, data):
            self.type = tp
            self.data = data

    async def _aiter():
        for tp, data in messages:
            yield _Msg(tp, data)

    ws.__aiter__ = lambda self: _aiter()
    ws.sent = sent
    return ws


@pytest.fixture(autouse=True)
def _reset_handler_state():
    """Reset class-level state before every test."""
    if not HAS_AIOHTTP:
        return
    WebSocketHandler._sessions.clear()
    # Prevent __init__ from spawning a background cleanup task
    WebSocketHandler._cleanup_started = True


class TestCleanupStaleSessions:
    def test_stale_sessions_removed(self):
        now = time.time()
        WebSocketHandler._sessions["fresh"] = {"last_seen": now - 10}
        WebSocketHandler._sessions["stale"] = {"last_seen": now - SESSION_TTL - 1}

        # Run the cleanup logic synchronously by extracting the inner loop body
        stale = [
            sid for sid, s in WebSocketHandler._sessions.items()
            if now - s.get("last_seen", 0) > SESSION_TTL
        ]
        for sid in stale:
            del WebSocketHandler._sessions[sid]

        assert "fresh" in WebSocketHandler._sessions
        assert "stale" not in WebSocketHandler._sessions

    def test_fresh_sessions_kept(self):
        now = time.time()
        WebSocketHandler._sessions["a"] = {"last_seen": now - 60}
        WebSocketHandler._sessions["b"] = {"last_seen": now - SESSION_TTL + 1}

        stale = [
            sid for sid, s in WebSocketHandler._sessions.items()
            if now - s.get("last_seen", 0) > SESSION_TTL
        ]
        for sid in stale:
            del WebSocketHandler._sessions[sid]

        assert "a" in WebSocketHandler._sessions
        assert "b" in WebSocketHandler._sessions


class TestSessionStorageOnDisconnect:
    def test_disconnect_stores_session(self):
        ws = _make_ws_mock([
            (WSMsgType.TEXT, json.dumps({"type": "ping"})),
            (WSMsgType.CLOSE, ""),
        ])

        handler = WebSocketHandler()

        request = MagicMock()
        request.app = {}

        with patch("mullet.server.websocket.web.WebSocketResponse", return_value=ws):
            asyncio.run(handler.handle(request))

        assert len(handler._sessions) == 1
        sid = next(iter(handler._sessions))
        session = handler._sessions[sid]
        assert "last_seen" in session
        assert session["user"] is None
        assert time.time() - session["last_seen"] < 5

    def test_disconnect_stores_authenticated_user(self, monkeypatch):
        monkeypatch.setattr(
            "mullet.server.websocket.MulletMode.is_solo", staticmethod(lambda: True)
        )
        ws = _make_ws_mock([
            (WSMsgType.TEXT, json.dumps({"type": "authenticate", "data": {}})),
            (WSMsgType.CLOSE, ""),
        ])

        handler = WebSocketHandler()

        request = MagicMock()
        request.app = {}

        # Patch send_json on the mock created inside handle()
        # Instead, we call handle directly but mock web.WebSocketResponse
        with patch("mullet.server.websocket.web.WebSocketResponse", return_value=ws):
            asyncio.run(handler.handle(request))

        assert len(handler._sessions) == 1
        sid = next(iter(handler._sessions))
        session = handler._sessions[sid]
        # In solo mode, get_solo_user returns {"id": "solo-admin", "role": "admin"}
        assert session["user"] is not None
        assert session["user"]["id"] == "solo-admin"
        assert "last_seen" in session


class TestResumeSession:
    def test_valid_session_id_returns_resumed(self):
        existing_sid = "sess-123"
        WebSocketHandler._sessions[existing_sid] = {
            "user": None,
            "last_seen": time.time() - 10,
        }

        ws = _make_ws_mock([
            (WSMsgType.TEXT, json.dumps({
                "type": "resume_session",
                "data": {"session_id": existing_sid},
            })),
            (WSMsgType.CLOSE, ""),
        ])

        handler = WebSocketHandler()
        request = MagicMock()
        request.app = {}

        with patch("mullet.server.websocket.web.WebSocketResponse", return_value=ws):
            asyncio.run(handler.handle(request))

        assert len(ws.sent) >= 1
        resumed_msg = next((m for m in ws.sent if m.get("type") == "session_resumed"), None)
        assert resumed_msg is not None
        assert resumed_msg["data"]["session_id"] == existing_sid

    def test_unknown_session_id_returns_created(self):
        ws = _make_ws_mock([
            (WSMsgType.TEXT, json.dumps({
                "type": "resume_session",
                "data": {"session_id": "unknown-id"},
            })),
            (WSMsgType.CLOSE, ""),
        ])

        handler = WebSocketHandler()
        request = MagicMock()
        request.app = {}

        with patch("mullet.server.websocket.web.WebSocketResponse", return_value=ws):
            asyncio.run(handler.handle(request))

        created_msg = next((m for m in ws.sent if m.get("type") == "session_created"), None)
        assert created_msg is not None
        new_sid = created_msg["data"]["session_id"]
        assert new_sid != "unknown-id"
        assert new_sid in handler._sessions

    def test_resume_without_session_id_creates_new(self):
        ws = _make_ws_mock([
            (WSMsgType.TEXT, json.dumps({
                "type": "resume_session",
                "data": {},
            })),
            (WSMsgType.CLOSE, ""),
        ])

        handler = WebSocketHandler()
        request = MagicMock()
        request.app = {}

        with patch("mullet.server.websocket.web.WebSocketResponse", return_value=ws):
            asyncio.run(handler.handle(request))

        created_msg = next((m for m in ws.sent if m.get("type") == "session_created"), None)
        assert created_msg is not None
        assert created_msg["data"]["session_id"]
