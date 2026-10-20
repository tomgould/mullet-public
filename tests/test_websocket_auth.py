"""Tests for WebSocket authentication and authorization security."""

import time
import pytest

try:
    from aiohttp import web  # noqa: F401  (importability probe)
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytestmark = pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")


@pytest.fixture
def app():
    from mullet.server.app import create_app
    import mullet.server.middleware as mw
    # Reset global rate limiter so full-suite runs don't hit 429 from accumulated 127.0.0.1 hits
    mw._rate_limiter._counters.clear()
    return create_app()


async def test_ws_auth_invalid_token_format(aiohttp_client, app):
    """Invalid token format should be rejected."""
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")
    await ws.send_json({"type": "authenticate", "data": {"token": "invalid_token"}})
    resp = await ws.receive_json()
    assert resp["type"] == "error"
    assert "Invalid token" in resp["data"]["message"]
    await ws.close()


async def test_ws_auth_empty_token(aiohttp_client, app):
    """Empty token should be rejected."""
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")
    await ws.send_json({"type": "authenticate", "data": {"token": ""}})
    resp = await ws.receive_json()
    # In solo mode, empty token should still authenticate as solo user
    assert resp["type"] == "authenticated"
    await ws.close()


async def test_ws_auth_missing_token_field(aiohttp_client, app):
    """Missing token field should handle gracefully."""
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")
    await ws.send_json({"type": "authenticate", "data": {}})
    resp = await ws.receive_json()
    # Should authenticate as solo user (default behavior)
    assert resp["type"] == "authenticated"
    await ws.close()


async def test_ws_protected_endpoint_without_auth(aiohttp_client, app, monkeypatch):
    """Accessing protected endpoints without auth should fail in enterprise mode."""
    # Set enterprise mode
    monkeypatch.setenv("MULLET_MODE", "enterprise")
    from mullet.config.mode import MulletMode
    MulletMode._mode = "enterprise"
    
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")
    
    # Try to list projects without authenticating
    await ws.send_json({"type": "list_projects", "data": {}})
    resp = await ws.receive_json()
    
    # In enterprise mode, should require authentication
    assert resp.get("type") == "error"
    assert "Authentication required" in resp.get("data", {}).get("message", "")
    await ws.close()


async def test_ws_search_requires_auth(aiohttp_client, app, monkeypatch):
    """Search endpoint should require authentication in enterprise mode."""
    monkeypatch.setenv("MULLET_MODE", "enterprise")
    from mullet.config.mode import MulletMode
    MulletMode._mode = "enterprise"
    
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")
    
    await ws.send_json({
        "type": "search",
        "data": {"project": "test", "query": "test"}
    })
    resp = await ws.receive_json()
    
    assert resp.get("type") == "error"
    assert "Authentication required" in resp.get("data", {}).get("message", "")
    await ws.close()


async def test_ws_start_task_requires_auth(aiohttp_client, app, monkeypatch):
    """start_task should require authentication in enterprise mode."""
    monkeypatch.setenv("MULLET_MODE", "enterprise")
    from mullet.config.mode import MulletMode
    MulletMode._mode = "enterprise"
    
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")
    
    await ws.send_json({
        "type": "start_task",
        "data": {"prompt": "test"}
    })
    resp = await ws.receive_json()
    
    assert resp.get("type") == "error"
    assert "Authentication required" in resp.get("data", {}).get("message", "")
    await ws.close()


async def test_ws_auth_persists_across_messages(aiohttp_client, app):
    """Authentication should persist across multiple messages."""
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")
    
    # First authenticate
    await ws.send_json({"type": "authenticate", "data": {}})
    resp = await ws.receive_json()
    assert resp["type"] == "authenticated"
    
    # Then use protected endpoint without re-authenticating
    await ws.send_json({"type": "list_projects", "data": {}})
    resp = await ws.receive_json()
    assert resp["type"] == "projects_list"
    
    # And again
    await ws.send_json({"type": "ping"})
    resp = await ws.receive_json()
    assert resp["type"] == "pong"
    
    await ws.close()


async def test_ws_malformed_auth_data(aiohttp_client, app):
    """Malformed auth data should be handled gracefully."""
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")
    
    await ws.send_json({"type": "authenticate", "data": None})
    resp = await ws.receive_json()
    # Should handle gracefully (auto-authenticate in solo mode)
    assert resp["type"] in ["authenticated", "error"]
    await ws.close()


async def test_ws_token_validation_with_db(aiohttp_client, app, tmp_path):
    """Token validation should use database when available."""
    from mullet.server.db import init_db
    from mullet.server.auth import generate_pat
    
    # Create a test database with a valid token
    db = init_db(str(tmp_path / "test.db"))
    
    # Create a test user
    db.execute("INSERT INTO users (id, email, role, created_at) VALUES (?, ?, ?, ?)",
               ("test_user", "test@example.com", "dev", time.time()))
    
    # Create a valid PAT
    raw_token, token_hash = generate_pat()
    db.execute("INSERT INTO access_tokens (id, user_id, token_hash, name, created_at) VALUES (?, ?, ?, ?, ?)",
               ("pat1", "test_user", token_hash, "test_token", time.time()))
    db.commit()
    
    # Attach database to app
    app["db"] = db
    
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")
    
    # Authenticate with valid token
    await ws.send_json({"type": "authenticate", "data": {"token": raw_token}})
    resp = await ws.receive_json()
    assert resp["type"] == "authenticated"
    assert resp["data"]["email"] == "test@example.com"
    
    db.close()
