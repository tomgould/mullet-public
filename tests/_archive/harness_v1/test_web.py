"""Tests for the Mullet web session dashboard."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mullet.harness.sessions.project_hash import project_hash
from mullet.harness.sessions.store import EventRecord, MessageRecord, SessionStore
from mullet.server.web.app import create_app


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(base_dir=tmp_path / "sessions")


@pytest.fixture
def client(store: SessionStore) -> TestClient:
    app = create_app(base_dir=store.base_dir)
    return TestClient(app)


@pytest.fixture
def sample_session(store: SessionStore, tmp_path: Path) -> str:
    meta = store.create(cwd=tmp_path / "myproject", model="claude-sonnet", profile="sonnet")
    store.append_message(meta.id, MessageRecord(role="user", content="hello", timestamp=time.time()))
    store.append_message(meta.id, MessageRecord(role="assistant", content="hi there", timestamp=time.time()))
    store.append_event(meta.id, EventRecord(event_type="start", timestamp=time.time(), payload={}))
    return meta.id


class TestCreateApp:
    def test_create_app_returns_fastapi_app(self, store: SessionStore):
        app = create_app(base_dir=store.base_dir)
        assert app.title == "Mullet Sessions"


class TestIndex:
    def test_get_index_returns_200(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        assert "Mullet Sessions" in response.text


class TestApiProjects:
    def test_get_api_projects_returns_list(self, client: TestClient, sample_session: str):
        response = client.get("/api/projects")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1


class TestApiSessions:
    def test_get_api_sessions_filter_by_project(self, client: TestClient, sample_session: str, store: SessionStore, tmp_path: Path):
        # Create a second project with its own session
        meta2 = store.create(cwd=tmp_path / "otherproject", model="gemini-flash", profile="gemini")
        ph = project_hash(tmp_path / "myproject")
        response = client.get("/api/sessions", params={"project_hash": ph})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["id"] == sample_session


class TestApiSession:
    def test_get_api_session_unknown_returns_404(self, client: TestClient):
        response = client.get("/api/sessions/nonexistent-session-id")
        assert response.status_code == 404

    def test_get_api_session_known_returns_200(self, client: TestClient, sample_session: str):
        response = client.get(f"/api/sessions/{sample_session}")
        assert response.status_code == 200
        data = response.json()
        assert data["meta"]["id"] == sample_session
        assert len(data["messages"]) == 2
        assert len(data["events"]) == 1

    def test_get_session_view_returns_200(self, client: TestClient, sample_session: str):
        response = client.get(f"/sessions/{sample_session}")
        assert response.status_code == 200
        assert sample_session in response.text

    def test_get_project_view_returns_200(self, client: TestClient, sample_session: str, store: SessionStore, tmp_path: Path):
        ph = project_hash(tmp_path / "myproject")
        response = client.get(f"/projects/{ph}")
        assert response.status_code == 200
        assert sample_session in response.text
