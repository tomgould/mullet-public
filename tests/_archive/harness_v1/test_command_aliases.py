"""Tests for harness slash-command aliases: /clear, /export, /compact, /cost."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mullet.harness.commands.clear import handle_clear_command
from mullet.harness.commands.compact import handle_compact_command
from mullet.harness.commands.cost import handle_cost_command
from mullet.harness.commands.export import handle_export_command
from mullet.harness.sessions.store import MessageRecord, SessionStore

pytestmark = pytest.mark.harness


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(base_dir=tmp_path / "sessions")


@pytest.fixture
def sample_session(store: SessionStore, tmp_path: Path) -> str:
    meta = store.create(cwd=tmp_path / "proj", model="kimi-k2.6", profile="kimi")
    for i in range(6):
        store.append_message(
            meta.id,
            MessageRecord(role="user" if i % 2 == 0 else "assistant", content=f"msg {i}", timestamp=time.time()),
        )
    return meta.id


class TestClear:
    def test_clear_returns_ok(self) -> None:
        out = handle_clear_command("")
        assert out == "Log cleared."

    def test_clear_all_clears_session(self) -> None:
        class FakeEngine:
            session_id = "sess-123"
            _last_session_id = "sess-123"

        engine = FakeEngine()
        out = handle_clear_command("all", engine=engine)
        assert engine.session_id is None
        assert engine._last_session_id is None
        assert "Cleared session and log." in out


class TestExport:
    def test_export_calls_share(self, monkeypatch) -> None:
        calls: list[str] = []

        def fake_share(args_line: str, **kwargs) -> str:
            calls.append(args_line)
            return "shared"

        monkeypatch.setattr(
            "mullet.harness.commands.export.handle_share_command", fake_share
        )
        out = handle_export_command("", session_id="sess-123")
        assert out == "shared"
        assert "--format markdown" in calls[0]
        assert "mullet-export-" in calls[0]

    def test_export_json_format(self, monkeypatch) -> None:
        calls: list[str] = []

        def fake_share(args_line: str, **kwargs) -> str:
            calls.append(args_line)
            return "shared"

        monkeypatch.setattr(
            "mullet.harness.commands.export.handle_share_command", fake_share
        )
        out = handle_export_command("json", session_id="sess-123")
        assert "--format json" in calls[0]


class TestCompact:
    def test_compact_returns_summary(self, monkeypatch, store: SessionStore, sample_session: str) -> None:
        mock_provider = MagicMock()
        mock_provider.create.return_value = SimpleNamespace(content="Summary")
        monkeypatch.setattr(
            "mullet.harness.engine.loop._resolve_provider",
            lambda name: mock_provider,
        )
        out = handle_compact_command("", store=store, session_id=sample_session)
        assert ("Compacted" in out) or ("Nothing to compact" in out)

    def test_compact_threshold_updates_session(self) -> None:
        engine = type("E", (), {})()
        out = handle_compact_command("threshold 0.5", engine=engine)
        assert "0.5" in out
        assert engine._compaction_threshold == 0.5


class TestCost:
    def test_cost_returns_session_totals(self, store: SessionStore, sample_session: str) -> None:
        out = handle_cost_command("", store=store, session_id=sample_session)
        assert "tokens" in out.lower()

    def test_cost_daily_aggregates(self, store: SessionStore, sample_session: str) -> None:
        out = handle_cost_command("daily", store=store)
        assert "Daily totals:" in out

    def test_cost_reset_zeroes(self) -> None:
        engine = type("E", (), {"_cost_counters": {}})()
        out = handle_cost_command("reset", engine=engine, session_id="sess-123")
        assert "reset" in out.lower()
        assert engine._cost_counters["sess-123"]["reset_at"] is not None
