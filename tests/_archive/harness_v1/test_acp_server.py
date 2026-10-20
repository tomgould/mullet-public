"""Tests for the ACP server (Agent Client Protocol)."""

from __future__ import annotations

import asyncio
import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from mullet.harness.sessions.store import SessionStore
from mullet.server.acp_server import ACPServer, PROTOCOL_VERSION


class FakeEngine:
    """Fake HarnessEngine that records calls and returns canned responses."""

    def __init__(self, store: SessionStore) -> None:
        self.store = store
        self.calls: list[tuple] = []

    def run(self, prompt: str, session_id: str | None = None, **kwargs) -> dict:
        self.calls.append(("run", prompt, session_id))
        return {
            "session_id": session_id or "sess-fake",
            "response": f"Echo: {prompt}",
            "stop_reason": "stop",
            "input_tokens": 0,
            "output_tokens": 0,
            "iterations": 1,
            "message_count": 2,
        }


def _send(stdin: StringIO, stdout: StringIO, requests: list[dict]) -> list[dict]:
    """Write requests to stdin, run the server, and parse responses from stdout."""
    stdin.write("\n".join(json.dumps(r) for r in requests) + "\n")
    stdin.seek(0)
    with patch.object(sys, "stdin", stdin), patch.object(sys, "stdout", stdout):
        server = ACPServer(
            engine_factory=lambda: FakeEngine(SessionStore(base_dir=Path("/dev/null")))
        )
        asyncio.run(server.serve())
    lines = stdout.getvalue().strip().split("\n")
    return [json.loads(line) for line in lines if line.strip()]


@pytest.mark.harness
class TestInitialize:
    def test_initialize_returns_capabilities(self):
        stdin = StringIO()
        stdout = StringIO()
        responses = _send(
            stdin,
            stdout,
            [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}],
        )
        assert len(responses) == 1
        assert "error" not in responses[0]
        result = responses[0]["result"]
        assert result["protocolVersion"] == PROTOCOL_VERSION
        assert result["agentCapabilities"]["loadSession"] is True
        assert result["agentInfo"]["name"] == "mullet"


@pytest.mark.harness
class TestSessionNew:
    def test_session_new_returns_session_id(self, tmp_path: Path):
        store = SessionStore(base_dir=tmp_path / "sessions")
        engine = FakeEngine(store)
        stdin = StringIO()
        stdout = StringIO()
        server = ACPServer(engine_factory=lambda: engine)
        stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {}}) + "\n")
        stdin.seek(0)
        with patch.object(sys, "stdin", stdin), patch.object(sys, "stdout", stdout):
            asyncio.run(server.serve())
        responses = [json.loads(line) for line in stdout.getvalue().strip().split("\n") if line.strip()]
        assert len(responses) == 1
        assert "error" not in responses[0]
        assert responses[0]["result"]["sessionId"].startswith("acp-")

    def test_session_new_creates_internal_session(self, tmp_path: Path):
        store = SessionStore(base_dir=tmp_path / "sessions")
        engine = FakeEngine(store)
        stdin = StringIO()
        stdout = StringIO()
        server = ACPServer(engine_factory=lambda: engine)
        stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {}}) + "\n")
        stdin.seek(0)
        with patch.object(sys, "stdin", stdin), patch.object(sys, "stdout", stdout):
            asyncio.run(server.serve())
        responses = [json.loads(line) for line in stdout.getvalue().strip().split("\n") if line.strip()]
        acp_sid = responses[0]["result"]["sessionId"]
        assert acp_sid in server.sessions
        internal_id = server.sessions[acp_sid].internal_session_id
        assert internal_id is not None
        # Verify the internal session exists in the store
        recent = store.list_recent()
        assert len(recent) == 1
        assert recent[0].id == internal_id


@pytest.mark.harness
class TestSessionLoad:
    def test_session_load_existing_session(self, tmp_path: Path):
        store = SessionStore(base_dir=tmp_path / "sessions")
        meta = store.create(cwd=tmp_path)
        engine = FakeEngine(store)
        stdin = StringIO()
        stdout = StringIO()
        server = ACPServer(engine_factory=lambda: engine)
        stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "session/load",
                    "params": {"sessionId": meta.id},
                }
            )
            + "\n"
        )
        stdin.seek(0)
        with patch.object(sys, "stdin", stdin), patch.object(sys, "stdout", stdout):
            asyncio.run(server.serve())
        responses = [json.loads(line) for line in stdout.getvalue().strip().split("\n") if line.strip()]
        assert len(responses) == 1
        assert "error" not in responses[0]
        assert responses[0]["result"]["sessionId"].startswith("acp-")

    def test_session_load_unknown_returns_error(self, tmp_path: Path):
        store = SessionStore(base_dir=tmp_path / "sessions")
        engine = FakeEngine(store)
        stdin = StringIO()
        stdout = StringIO()
        server = ACPServer(engine_factory=lambda: engine)
        stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "session/load",
                    "params": {"sessionId": "nonexistent"},
                }
            )
            + "\n"
        )
        stdin.seek(0)
        with patch.object(sys, "stdin", stdin), patch.object(sys, "stdout", stdout):
            asyncio.run(server.serve())
        responses = [json.loads(line) for line in stdout.getvalue().strip().split("\n") if line.strip()]
        assert len(responses) == 1
        assert "error" in responses[0]
        assert "not found" in responses[0]["error"]["message"].lower()


@pytest.mark.harness
class TestSessionPrompt:
    def test_session_prompt_routes_to_engine(self, tmp_path: Path):
        store = SessionStore(base_dir=tmp_path / "sessions")
        engine = FakeEngine(store)
        server = ACPServer(engine_factory=lambda: engine)

        # Step 1: create session
        stdin = StringIO(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {}}) + "\n")
        stdout = StringIO()
        with patch.object(sys, "stdin", stdin), patch.object(sys, "stdout", stdout):
            asyncio.run(server.serve())
        responses = [json.loads(line) for line in stdout.getvalue().strip().split("\n") if line.strip()]
        acp_sid = responses[0]["result"]["sessionId"]

        # Step 2: prompt
        stdin = StringIO(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "session/prompt",
                    "params": {
                        "sessionId": acp_sid,
                        "prompt": [{"type": "text", "text": "hello world"}],
                    },
                }
            )
            + "\n"
        )
        stdout = StringIO()
        with patch.object(sys, "stdin", stdin), patch.object(sys, "stdout", stdout):
            asyncio.run(server.serve())

        # Verify run was called with the prompt text and the internal session id
        assert any(
            call[1] == "hello world" and call[2] == server.sessions[acp_sid].internal_session_id
            for call in engine.calls
        )

    def test_session_prompt_returns_text_content(self, tmp_path: Path):
        store = SessionStore(base_dir=tmp_path / "sessions")
        engine = FakeEngine(store)
        server = ACPServer(engine_factory=lambda: engine)

        # Step 1: create session
        stdin = StringIO(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {}}) + "\n")
        stdout = StringIO()
        with patch.object(sys, "stdin", stdin), patch.object(sys, "stdout", stdout):
            asyncio.run(server.serve())
        responses = [json.loads(line) for line in stdout.getvalue().strip().split("\n") if line.strip()]
        acp_sid = responses[0]["result"]["sessionId"]

        # Step 2: prompt
        stdin = StringIO(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "session/prompt",
                    "params": {
                        "sessionId": acp_sid,
                        "prompt": [{"type": "text", "text": "hi"}],
                    },
                }
            )
            + "\n"
        )
        stdout = StringIO()
        with patch.object(sys, "stdin", stdin), patch.object(sys, "stdout", stdout):
            asyncio.run(server.serve())
        responses = [json.loads(line) for line in stdout.getvalue().strip().split("\n") if line.strip()]
        prompt_response = responses[0]
        assert "error" not in prompt_response
        result = prompt_response["result"]
        assert result["stopReason"] == "stop"
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Echo: hi"


@pytest.mark.harness
class TestErrorHandling:
    def test_unknown_method_returns_error(self):
        stdin = StringIO()
        stdout = StringIO()
        responses = _send(
            stdin,
            stdout,
            [{"jsonrpc": "2.0", "id": 1, "method": "bogus", "params": {}}],
        )
        assert len(responses) == 1
        assert "error" in responses[0]
        assert responses[0]["error"]["code"] == -32601


@pytest.mark.harness
class TestSessionCancel:
    def test_session_cancel_returns_ok(self, tmp_path: Path):
        store = SessionStore(base_dir=tmp_path / "sessions")
        engine = FakeEngine(store)
        server = ACPServer(engine_factory=lambda: engine)

        # Step 1: create session
        stdin = StringIO(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {}}) + "\n")
        stdout = StringIO()
        with patch.object(sys, "stdin", stdin), patch.object(sys, "stdout", stdout):
            asyncio.run(server.serve())
        responses = [json.loads(line) for line in stdout.getvalue().strip().split("\n") if line.strip()]
        acp_sid = responses[0]["result"]["sessionId"]

        # Step 2: cancel
        stdin = StringIO(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "session/cancel",
                    "params": {"sessionId": acp_sid},
                }
            )
            + "\n"
        )
        stdout = StringIO()
        with patch.object(sys, "stdin", stdin), patch.object(sys, "stdout", stdout):
            asyncio.run(server.serve())
        responses = [json.loads(line) for line in stdout.getvalue().strip().split("\n") if line.strip()]
        cancel_response = responses[0]
        assert "error" not in cancel_response
        assert cancel_response["result"]["ok"] is True
        # Verify the session was marked cancelled
        assert server.sessions[acp_sid]._cancelled is True
