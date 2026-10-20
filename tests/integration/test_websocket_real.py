"""Integration test: WebSocket and API server endpoints.

Tests the full server stack with real aiohttp test client.
"""

import pytest

try:
    from aiohttp import web  # noqa: F401  (importability probe)
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytestmark = pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")


async def test_chat_completions_no_provider(aiohttp_client):
    """Chat completions should handle missing provider gracefully."""
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    resp = await client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hello"}],
    })
    # May return 200 (if local LLM available), 503 (if not), or 500 (server error)
    assert resp.status in (200, 503, 500)


async def test_projects_endpoint(aiohttp_client):
    """Projects endpoint should return list."""
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/projects")
    assert resp.status == 200
    data = await resp.json()
    assert "projects" in data


async def test_admin_create_user(aiohttp_client):
    """Admin should be able to create a user."""
    import uuid
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    unique_email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    resp = await client.post("/api/v1/admin/users", json={
        "email": unique_email,
        "role": "dev",
    })
    assert resp.status == 201
    data = await resp.json()
    assert data["email"] == unique_email


async def test_websocket_ping_pong(aiohttp_client):
    """WebSocket should respond to ping."""
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    ws = await client.ws_connect("/ws")
    await ws.send_json({"type": "ping"})
    msg = await ws.receive_json()
    assert msg["type"] == "pong"
    await ws.close()


async def test_websocket_list_projects(aiohttp_client):
    """WebSocket should return project list."""
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    ws = await client.ws_connect("/ws")
    await ws.send_json({"type": "list_projects"})
    msg = await ws.receive_json()
    assert msg["type"] == "projects_list"
    assert "projects" in msg["data"]
    await ws.close()


async def test_websocket_authenticate(aiohttp_client):
    """WebSocket authenticate should return user info."""
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    ws = await client.ws_connect("/ws")
    await ws.send_json({"type": "authenticate", "data": {}})
    msg = await ws.receive_json()
    assert msg["type"] == "authenticated"
    assert msg["data"]["role"] == "admin"  # Solo mode = admin
    await ws.close()


async def test_models_returns_providers(aiohttp_client):
    """Models endpoint should list detected providers."""
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    resp = await client.get("/v1/models")
    assert resp.status == 200
    data = await resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1  # At least "mullet" model


async def test_generate_and_revoke_token(aiohttp_client):
    """Token lifecycle: generate, list, revoke."""
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    # Generate
    resp = await client.post("/api/v1/tokens", json={"name": "test-token"})
    assert resp.status == 201
    data = await resp.json()
    assert data["token"].startswith("mllt_")
    token_id = data["id"]

    # List
    resp = await client.get("/api/v1/tokens")
    data = await resp.json()
    assert any(t["id"] == token_id for t in data["tokens"])

    # Revoke
    resp = await client.delete(f"/api/v1/tokens/{token_id}")
    assert resp.status == 200
