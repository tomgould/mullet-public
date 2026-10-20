"""Unit tests for the HarnessEngine agent loop (Phase 1)."""

from __future__ import annotations

import pytest

from mullet.core.providers.base import LLMResponse, Provider
from mullet.harness.engine.loop import HarnessEngine, _resolve_provider
from mullet.harness.hooks.registry import Block, HookRegistry
from mullet.harness.sessions.store import SessionStore


class _MockProvider(Provider):
    """Synchronous mock provider that records calls and returns a fixed response."""

    def __init__(self, content: str = "OK", finish_reason: str = "stop") -> None:
        self.calls: list[tuple] = []
        self._content = content
        self._finish_reason = finish_reason

    def create(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        reasoning_config: dict | None = None,
    ) -> LLMResponse:
        self.calls.append((messages, tools))
        return LLMResponse(
            content=self._content,
            finish_reason=self._finish_reason,
        )


@pytest.fixture
def mock_sync_provider():
    return _MockProvider()


@pytest.fixture
def engine(tmp_path, mock_sync_provider, monkeypatch):
    store = SessionStore(base_dir=tmp_path / "sessions")
    monkeypatch.setattr(
        "mullet.harness.engine.loop._resolve_provider",
        lambda name, model=None: mock_sync_provider,
    )
    # Fresh hooks registry so mutations don't leak across tests.
    return HarnessEngine(store=store, hooks=HookRegistry())


class TestHarnessEngineRun:
    def test_run_creates_session_and_returns_response(self, engine, mock_sync_provider):
        result = engine.run("hello")

        assert result["session_id"].startswith("sess-")
        assert result["response"] == "OK"
        assert result["stop_reason"] == "stop"
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0
        assert len(mock_sync_provider.calls) == 1

    def test_run_appends_user_and_assistant_messages(self, engine):
        result = engine.run("hello")
        sid = result["session_id"]

        meta, messages, events = engine.load_session(sid)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "hello"
        assert messages[1].role == "assistant"
        assert messages[1].content == "OK"

    def test_run_with_existing_session_id_reuses(self, engine, tmp_path):
        result1 = engine.run("hello")
        sid = result1["session_id"]

        result2 = engine.run("hello again", session_id=sid)
        assert result2["session_id"] == sid

        # Count session directories — should be exactly one
        session_dirs = []
        base = tmp_path / "sessions"
        for project_dir in base.iterdir():
            if project_dir.is_dir():
                session_dirs.extend([d for d in project_dir.iterdir() if d.is_dir()])
        assert len(session_dirs) == 1

    def test_params_hook_can_mutate_messages(self, engine, mock_sync_provider):
        def inject_system(payload):
            payload["messages"].insert(0, {"role": "system", "content": "SYS"})
            return payload

        engine.hooks.register("Params", inject_system)
        engine.run("hello")

        messages, _tools = mock_sync_provider.calls[-1]
        assert messages[0] == {"role": "system", "content": "SYS"}
        assert messages[-1] == {"role": "user", "content": "hello"}

    def test_params_hook_block_raises(self, engine):
        def block_hook(payload):
            return Block("nope")

        engine.hooks.register("Params", block_hook)
        with pytest.raises(RuntimeError, match="Params hook blocked the call: nope"):
            engine.run("hello")

    def test_event_hook_fires_on_message_complete(self, engine):
        captured = []

        def event_hook(payload):
            captured.append(payload)
            return None

        engine.hooks.register("Event", event_hook)
        engine.run("hello")

        assert any(c.get("event") == "session.message_complete" for c in captured)

    def test_unknown_provider_raises(self, engine, monkeypatch):
        # Restore the real resolver so we can test the error path.
        monkeypatch.setattr(
            "mullet.harness.engine.loop._resolve_provider",
            _resolve_provider,
        )
        with pytest.raises(NotImplementedError, match="Unknown provider 'bogus'"):
            engine.run("hello", provider="bogus")


class TestHarnessEngineQueries:
    def test_list_sessions_returns_recent(self, engine):
        result = engine.run("hello")
        sid = result["session_id"]

        sessions = engine.list_sessions(limit=10)
        assert any(s.id == sid for s in sessions)

    def test_load_session_returns_tuple(self, engine):
        result = engine.run("hello")
        sid = result["session_id"]

        meta, messages, events = engine.load_session(sid)
        assert meta.id == sid
        assert len(messages) == 2
        assert any(e.event_type == "user.prompt" for e in events)
