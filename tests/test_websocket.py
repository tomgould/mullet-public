"""Tests for the WebSocket handler."""

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
    return create_app()


async def test_ws_ping(aiohttp_client, app):
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")
    await ws.send_json({"type": "ping"})
    resp = await ws.receive_json()
    assert resp["type"] == "pong"
    await ws.close()


async def test_ws_unknown_type(aiohttp_client, app):
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")
    await ws.send_json({"type": "bogus_type"})
    resp = await ws.receive_json()
    assert resp["type"] == "error"
    assert "Unknown type" in resp["data"]["message"]
    await ws.close()


async def test_ws_invalid_json(aiohttp_client, app):
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")
    await ws.send_str("not json at all")
    resp = await ws.receive_json()
    assert resp["type"] == "error"
    assert "Invalid JSON" in resp["data"]["message"]
    await ws.close()


async def test_ws_authenticate_solo(aiohttp_client, app):
    """Without a DB, authenticate should return solo user."""
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")
    await ws.send_json({"type": "authenticate", "data": {}})
    resp = await ws.receive_json()
    assert resp["type"] == "authenticated"
    assert resp["data"]["id"] == "solo-admin"
    assert resp["data"]["role"] == "admin"
    await ws.close()


async def test_ws_list_projects_no_db(aiohttp_client, app):
    """list_projects should return a projects list."""
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")
    await ws.send_json({"type": "list_projects", "data": {}})
    resp = await ws.receive_json()
    assert resp["type"] == "projects_list"
    assert isinstance(resp["data"]["projects"], list)
    await ws.close()


async def test_ws_search_missing_params(aiohttp_client, app):
    """search without project/query should return error."""
    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")
    await ws.send_json({"type": "search", "data": {}})
    resp = await ws.receive_json()
    assert resp["type"] == "error"
    assert "required" in resp["data"]["message"]
    await ws.close()


async def test_ws_max_concurrent_tasks(aiohttp_client, app):
    """Starting more than MAX_TASKS_PER_CONNECTION should error."""
    from mullet.server.websocket import MAX_TASKS_PER_CONNECTION

    client = await aiohttp_client(app)
    ws = await client.ws_connect("/ws")

    # Start max tasks (they'll fail quickly but occupy slots briefly)
    for i in range(MAX_TASKS_PER_CONNECTION):
        await ws.send_json({"type": "start_task", "data": {"prompt": f"task {i}"}})

    # Next one should be rejected
    await ws.send_json({"type": "start_task", "data": {"prompt": "one too many"}})

    # Drain messages until we find the max-tasks error
    found_error = False
    for _ in range(20):  # Safety limit
        try:
            resp = await ws.receive_json(timeout=2)
            if resp.get("type") == "error" and "concurrent" in resp.get("data", {}).get("message", "").lower():
                found_error = True
                break
        except Exception:
            break

    assert found_error, "Should have received max concurrent tasks error"
    await ws.close()
