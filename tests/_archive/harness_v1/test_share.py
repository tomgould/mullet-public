"""Tests for the /share command and session exporters."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from mullet.harness.commands.share import handle_share_command, _parse_share_args
from mullet.harness.sessions.exporter import export_markdown, export_html, export_json
from mullet.harness.sessions.store import EventRecord, MessageRecord, SessionStore


pytestmark = pytest.mark.harness


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(base_dir=tmp_path / "sessions")


@pytest.fixture
def sample_session(store: SessionStore, tmp_path: Path):
    """Create a session with a few messages and events."""
    meta = store.create(cwd=tmp_path / "my-project", model="kimi-k2.6", profile="kimi")
    store.append_message(meta.id, MessageRecord(role="user", content="Refactor auth.py", timestamp=time.time()))
    store.append_message(
        meta.id,
        MessageRecord(
            role="assistant",
            content="I'll read auth.py first.",
            timestamp=time.time(),
            tool_calls=[{"function": {"name": "Read", "arguments": '{"filePath": "auth.py"}'}}],
        ),
    )
    store.append_message(
        meta.id,
        MessageRecord(role="tool", content="def old_auth(): pass", timestamp=time.time(), tool_call_id="tc-1"),
    )
    store.append_event(meta.id, EventRecord(event_type="user.prompt", timestamp=time.time(), payload={"prompt": "Refactor auth.py"}))
    return meta.id


class TestParseShareArgs:
    def test_defaults(self) -> None:
        ns = _parse_share_args("")
        assert ns.session_id is None
        assert ns.format == "markdown"
        assert ns.out is None
        assert ns.anonymize is False

    def test_all_flags(self) -> None:
        ns = _parse_share_args("sess-abc --format html --out /tmp/out.html --anonymize")
        assert ns.session_id == "sess-abc"
        assert ns.format == "html"
        assert ns.out == "/tmp/out.html"
        assert ns.anonymize is True


class TestExportMarkdown:
    def test_contains_session_id(self, store: SessionStore, sample_session: str, tmp_path: Path) -> None:
        meta, messages, events = store.load(sample_session)
        md = export_markdown(meta, messages, events)
        assert f"# Mullet Session — {sample_session}" in md

    def test_contains_messages(self, store: SessionStore, sample_session: str, tmp_path: Path) -> None:
        meta, messages, events = store.load(sample_session)
        md = export_markdown(meta, messages, events)
        assert "Refactor auth.py" in md
        assert "I'll read auth.py first." in md

    def test_tool_calls_in_details(self, store: SessionStore, sample_session: str, tmp_path: Path) -> None:
        meta, messages, events = store.load(sample_session)
        md = export_markdown(meta, messages, events)
        assert "<details>" in md
        assert "Read" in md

    def test_anonymize_replaces_home_and_project(self, store: SessionStore, tmp_path: Path) -> None:
        home = Path.home()
        meta = store.create(cwd=home / "projects" / "secret", model="test", profile="local")
        store.append_message(meta.id, MessageRecord(role="user", content=str(home / "projects" / "secret" / "auth.py"), timestamp=time.time()))
        meta, messages, events = store.load(meta.id)
        md = export_markdown(meta, messages, events, anonymize=True)
        assert str(home) not in md
        assert "~/projects/secret" not in md
        assert "<project>" in md
        assert "~" in md or "<project>" in md

    def test_footer_present(self, store: SessionStore, sample_session: str) -> None:
        meta, messages, events = store.load(sample_session)
        md = export_markdown(meta, messages, events)
        assert "Exported by Mullet" in md


class TestExportHtml:
    def test_is_valid_html(self, store: SessionStore, sample_session: str) -> None:
        meta, messages, events = store.load(sample_session)
        html = export_html(meta, messages, events)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert sample_session in html

    def test_anonymize_applies(self, store: SessionStore, tmp_path: Path) -> None:
        home = Path.home()
        meta = store.create(cwd=home / "proj", model="test", profile="local")
        store.append_message(meta.id, MessageRecord(role="user", content=str(home / "proj" / "file.py"), timestamp=time.time()))
        meta, messages, events = store.load(meta.id)
        html_out = export_html(meta, messages, events, anonymize=True)
        assert str(home) not in html_out
        # HTML-escaped angle brackets
        assert "&lt;project&gt;" in html_out


class TestExportJson:
    def test_structure(self, store: SessionStore, sample_session: str) -> None:
        meta, messages, events = store.load(sample_session)
        data = export_json(meta, messages, events)
        assert "meta" in data
        assert "messages" in data
        assert "events" in data
        assert data["meta"]["id"] == sample_session
        assert len(data["messages"]) == 3
        assert len(data["events"]) == 1


class TestShareCommand:
    def test_share_with_explicit_session(self, store: SessionStore, sample_session: str, tmp_path: Path) -> None:
        out = handle_share_command(
            f"{sample_session} --out {tmp_path / 'share.md'}",
            store=store,
        )
        assert "Shared session" in out
        assert (tmp_path / "share.md").exists()

    def test_share_default_path(self, store: SessionStore, sample_session: str, tmp_path: Path) -> None:
        # Change cwd so the default path lands in tmp_path
        import os
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            out = handle_share_command(
                sample_session,
                store=store,
            )
            assert "Shared session" in out
            assert any(f.name.startswith(f"mullet-share-{sample_session}") for f in tmp_path.iterdir())
        finally:
            os.chdir(old_cwd)

    def test_share_json_format(self, store: SessionStore, sample_session: str, tmp_path: Path) -> None:
        out = handle_share_command(
            f"{sample_session} --format json --out {tmp_path / 'share.json'}",
            store=store,
        )
        assert "Shared session" in out
        data = json.loads((tmp_path / "share.json").read_text())
        assert data["meta"]["id"] == sample_session

    def test_share_anonymize(self, store: SessionStore, tmp_path: Path) -> None:
        home = Path.home()
        meta = store.create(cwd=home / "secret", model="test", profile="local")
        store.append_message(meta.id, MessageRecord(role="user", content="hello", timestamp=time.time()))
        import os
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            out = handle_share_command(
                f"{meta.id} --anonymize",
                store=store,
            )
            path = next(p for p in tmp_path.iterdir() if p.name.startswith("mullet-share-"))
            md = path.read_text()
            assert str(home) not in md
            assert "<project>" in md
        finally:
            os.chdir(old_cwd)

    def test_share_missing_session(self, store: SessionStore) -> None:
        out = handle_share_command("", store=store)
        assert "No active session" in out or "Usage:" in out

    def test_share_unknown_session(self, store: SessionStore) -> None:
        out = handle_share_command("sess-does-not-exist", store=store)
        assert "not found" in out
