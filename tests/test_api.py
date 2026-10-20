"""Tests for the REST API and WebSocket server."""

import pytest

# Check if aiohttp is available
try:
    from aiohttp import web  # noqa: F401  (importability probe)
    from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop  # noqa: F401
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytestmark = pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")


@pytest.fixture
def app():
    """Create a test app."""
    from mullet.server.app import create_app
    return create_app()


@pytest.fixture
def client(event_loop, aiohttp_client, app):
    """Create a test client."""
    return event_loop.run_until_complete(aiohttp_client(app))


async def test_health(aiohttp_client):
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    resp = await client.get("/health")
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "ok"
    assert "version" in data


async def test_models(aiohttp_client):
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    resp = await client.get("/v1/models")
    assert resp.status == 200
    data = await resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1


async def test_chat_completions_no_messages(aiohttp_client):
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    resp = await client.post("/v1/chat/completions", json={})
    assert resp.status == 400


async def test_chat_completions_no_user_message(aiohttp_client):
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    resp = await client.post("/v1/chat/completions", json={
        "messages": [{"role": "system", "content": "hello"}],
    })
    assert resp.status == 400


async def test_status(aiohttp_client):
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/status")
    assert resp.status == 200
    data = await resp.json()
    assert "active_tasks" in data


async def test_websocket_ping(aiohttp_client):
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    ws = await client.ws_connect("/ws")
    await ws.send_json({"type": "ping"})
    msg = await ws.receive_json()
    assert msg["type"] == "pong"
    await ws.close()


async def test_websocket_unknown_type(aiohttp_client):
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    ws = await client.ws_connect("/ws")
    await ws.send_json({"type": "bogus"})
    msg = await ws.receive_json()
    assert msg["type"] == "error"
    assert "Unknown" in msg["data"]["message"]
    await ws.close()


async def test_websocket_resume_session(aiohttp_client):
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    ws = await client.ws_connect("/ws")
    await ws.send_json({"type": "resume_session", "data": {"session_id": "nonexistent"}})
    msg = await ws.receive_json()
    assert msg["type"] == "session_created"
    assert "session_id" in msg["data"]
    await ws.close()


async def test_websocket_authenticate_solo(aiohttp_client):
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    ws = await client.ws_connect("/ws")
    await ws.send_json({"type": "authenticate", "data": {}})
    msg = await ws.receive_json()
    assert msg["type"] == "authenticated"
    assert "id" in msg["data"]
    await ws.close()


async def test_websocket_list_projects(aiohttp_client):
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    ws = await client.ws_connect("/ws")
    await ws.send_json({"type": "authenticate", "data": {}})
    auth_msg = await ws.receive_json()
    assert auth_msg["type"] == "authenticated"

    await ws.send_json({"type": "list_projects"})
    msg = await ws.receive_json()
    assert msg["type"] == "projects_list"
    assert "projects" in msg["data"]
    assert isinstance(msg["data"]["projects"], list)
    await ws.close()


async def test_websocket_kill_task_unknown(aiohttp_client):
    import asyncio
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    ws = await client.ws_connect("/ws")
    await ws.send_json({"type": "kill_task", "data": {"task_id": "does-not-exist"}})
    # No response expected for unknown task_id; verify connection stays open
    try:
        await asyncio.wait_for(ws.receive_json(), timeout=0.2)
        raise AssertionError("Expected no response for unknown task_id")
    except asyncio.TimeoutError:
        pass
    # Prove the handler did not crash by pinging
    await ws.send_json({"type": "ping"})
    pong = await ws.receive_json()
    assert pong["type"] == "pong"
    await ws.close()


async def test_cors_headers(aiohttp_client):
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    resp = await client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert resp.status == 200
    assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:5173"


async def test_cors_preflight(aiohttp_client):
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    resp = await client.options("/v1/chat/completions", headers={"Origin": "http://localhost:5173"})
    assert resp.status == 200
    assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:5173"
    assert "OPTIONS" in resp.headers.get("Access-Control-Allow-Methods", "")


async def test_projects_endpoint_solo(aiohttp_client):
    """GET /api/v1/projects returns projects list in solo mode."""
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)
    resp = await client.get("/api/v1/projects")
    assert resp.status == 200
    data = await resp.json()
    assert "projects" in data


async def test_search_endpoint_missing_fields(aiohttp_client):
    """POST /api/v1/search returns 400 when client/query missing."""
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)
    resp = await client.post("/api/v1/search", json={"query": "test"})
    assert resp.status == 400


async def test_memory_store_missing_fields(aiohttp_client):
    """POST /api/v1/memory returns 400 when client/key/content missing."""
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)
    resp = await client.post("/api/v1/memory", json={"key": "k"})
    assert resp.status == 400


async def test_memory_search_missing_client(aiohttp_client):
    """GET /api/v1/memory returns 400 when client param missing."""
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)
    resp = await client.get("/api/v1/memory")
    assert resp.status == 400


async def test_metrics_endpoint(aiohttp_client):
    """GET /metrics returns Prometheus text."""
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)
    resp = await client.get("/metrics")
    assert resp.status == 200
    text = await resp.text()
    assert "mullet_uptime_seconds" in text


async def test_info_endpoint(aiohttp_client):
    """GET /api/v1/info returns comprehensive system info."""
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)
    resp = await client.get("/api/v1/info")
    assert resp.status == 200
    data = await resp.json()
    assert "version" in data
    assert "tools" in data
    assert isinstance(data["tools"], list)


async def test_skills_endpoint(aiohttp_client):
    """GET /api/v1/skills returns skills list."""
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)
    resp = await client.get("/api/v1/skills")
    assert resp.status == 200
    data = await resp.json()
    assert "skills" in data
    assert isinstance(data["skills"], list)


async def test_tools_endpoint(aiohttp_client):
    """GET /api/v1/tools returns tool registry."""
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)
    resp = await client.get("/api/v1/tools")
    assert resp.status == 200
    data = await resp.json()
    assert "tools" in data
    assert "toolsets" in data
    assert isinstance(data["tools"], list)


async def test_explain_missing_task(aiohttp_client):
    """POST /api/v1/explain without task returns 400."""
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)
    resp = await client.post("/api/v1/explain", json={})
    assert resp.status == 400


async def test_explain_with_task(aiohttp_client):
    """POST /api/v1/explain returns complexity analysis."""
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)
    resp = await client.post("/api/v1/explain", json={"task": "write a hello world function"})
    assert resp.status == 200
    data = await resp.json()
    assert "tier" in data
    assert "complexity_score" in data
    assert "is_simple" in data
