import re
import time
from pathlib import Path

import pytest

from mullet.harness.sessions.project_hash import project_hash
from mullet.harness.sessions.store import EventRecord, MessageRecord, SessionMeta, SessionStore


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(base_dir=tmp_path / "sessions")


@pytest.mark.harness
class TestSessionStore:
    def test_create_session_writes_meta_messages_events_files(self, store: SessionStore, tmp_path: Path) -> None:
        meta = store.create(cwd=tmp_path / "proj")
        session_dir = store._session_path(meta.id)
        assert (session_dir / "meta.json").exists()
        assert (session_dir / "messages.jsonl").exists()
        assert (session_dir / "events.jsonl").exists()
        loaded = SessionMeta.model_validate_json((session_dir / "meta.json").read_text())
        assert loaded.id == meta.id
        assert loaded.project_hash == project_hash(tmp_path / "proj")
        assert loaded.cwd == str(tmp_path / "proj")
        assert loaded.message_count == 0
        assert loaded.status == "active"

    def test_append_message_persists_and_updates_meta(self, store: SessionStore, tmp_path: Path) -> None:
        meta = store.create(cwd=tmp_path / "proj")
        msg1 = MessageRecord(role="user", content="hello", timestamp=time.time())
        store.append_message(meta.id, msg1)
        loaded_meta = SessionMeta.model_validate_json((store._session_path(meta.id) / "meta.json").read_text())
        assert loaded_meta.message_count == 1

        msg2 = MessageRecord(role="assistant", content="hi", timestamp=time.time())
        store.append_message(meta.id, msg2)
        loaded_meta2 = SessionMeta.model_validate_json((store._session_path(meta.id) / "meta.json").read_text())
        assert loaded_meta2.message_count == 2
        assert loaded_meta2.updated_at >= loaded_meta.updated_at

        _, messages, _ = store.load(meta.id)
        assert len(messages) == 2

    def test_append_event_persists_in_events_jsonl(self, store: SessionStore, tmp_path: Path) -> None:
        meta = store.create(cwd=tmp_path / "proj")
        event = EventRecord(event_type="test", timestamp=time.time(), payload={"key": "val"})
        store.append_event(meta.id, event)
        events_path = store._session_path(meta.id) / "events.jsonl"
        lines = events_path.read_text().strip().splitlines()
        assert len(lines) == 1
        loaded_event = EventRecord.model_validate_json(lines[0])
        assert loaded_event.event_type == "test"

    def test_load_returns_meta_messages_events(self, store: SessionStore, tmp_path: Path) -> None:
        meta = store.create(cwd=tmp_path / "proj")
        for i in range(3):
            store.append_message(
                meta.id,
                MessageRecord(role="user", content=f"msg{i}", timestamp=time.time()),
            )
        for i in range(5):
            store.append_event(
                meta.id,
                EventRecord(event_type="evt", timestamp=time.time(), payload={"i": i}),
            )
        loaded_meta, messages, events = store.load(meta.id)
        assert loaded_meta.id == meta.id
        assert len(messages) == 3
        assert len(events) == 5
        assert messages[0].content == "msg0"

    def test_list_recent_orders_newest_first(self, store: SessionStore, tmp_path: Path) -> None:
        metas = []
        for i in range(3):
            metas.append(store.create(cwd=tmp_path / f"proj{i}"))
            time.sleep(0.01)
        recent = store.list_recent()
        assert len(recent) == 3
        assert recent[0].created_at >= recent[1].created_at >= recent[2].created_at
        assert [m.id for m in recent] == [m.id for m in reversed(metas)]

    def test_list_recent_filter_by_project_hash(self, store: SessionStore, tmp_path: Path) -> None:
        cwd_a = tmp_path / "project_a"
        cwd_b = tmp_path / "project_b"
        meta_a = store.create(cwd=cwd_a)
        meta_b = store.create(cwd=cwd_b)
        ph_a = project_hash(cwd_a)
        recent = store.list_recent(project_hash_filter=ph_a)
        assert len(recent) == 1
        assert recent[0].id == meta_a.id

    def test_session_id_format(self, store: SessionStore, tmp_path: Path) -> None:
        meta = store.create(cwd=tmp_path / "proj")
        assert re.fullmatch(r"sess-\d{8}-\d{6}-[a-z0-9]{6}", meta.id) is not None
