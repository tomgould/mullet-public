"""End-to-end server tests with aiohttp test client.

Tests the full API stack including auth, routing, and responses.
"""

import pytest

try:
    from aiohttp import web  # noqa: F401  (importability probe)
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytestmark = pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")


async def test_full_health_check(aiohttp_client):
    """Health check returns all expected fields."""
    from mullet.server.app import create_app
    client = await aiohttp_client(create_app())

    resp = await client.get("/health")
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "providers" in data
    assert data["version"] != "0.1.0"  # Should be dynamic


async def test_full_admin_workflow(aiohttp_client):
    """Full admin workflow: create user → create project → grant access."""
    import uuid
    from mullet.server.app import create_app
    client = await aiohttp_client(create_app())

    # Create user
    email = f"admin_test_{uuid.uuid4().hex[:6]}@test.com"
    resp = await client.post("/api/v1/admin/users", json={"email": email, "role": "dev"})
    assert resp.status == 201
    user = await resp.json()
    user_id = user["id"]

    # Create project
    proj_name = f"proj_{uuid.uuid4().hex[:6]}"
    resp = await client.post("/api/v1/admin/projects", json={
        "name": proj_name, "pool_name": proj_name,
    })
    assert resp.status == 201
    proj = await resp.json()
    project_id = proj["id"]

    # Grant access
    resp = await client.post("/api/v1/admin/access", json={
        "user_id": user_id, "project_id": project_id, "access_level": "write",
    })
    assert resp.status == 200
    data = await resp.json()
    assert data["granted"] is True

    # Verify access
    resp = await client.get(f"/api/v1/admin/access/{project_id}")
    assert resp.status == 200
    access = await resp.json()
    assert len(access["access"]) >= 1

    # Clean up
    await client.delete(f"/api/v1/admin/users/{user_id}")
    await client.delete(f"/api/v1/admin/projects/{project_id}")


async def test_token_lifecycle(aiohttp_client):
    """Generate → verify → list → revoke token."""
    from mullet.server.app import create_app
    client = await aiohttp_client(create_app())

    # Generate
    resp = await client.post("/api/v1/tokens", json={"name": "e2e-test"})
    assert resp.status == 201
    data = await resp.json()
    assert data["token"].startswith("mllt_")
    token = data["token"]
    token_id = data["id"]

    # List
    resp = await client.get("/api/v1/tokens")
    data = await resp.json()
    assert any(t["id"] == token_id for t in data["tokens"])

    # Revoke
    resp = await client.delete(f"/api/v1/tokens/{token_id}")
    assert resp.status == 200

    # Verify gone
    resp = await client.get("/api/v1/tokens")
    data = await resp.json()
    assert not any(t["id"] == token_id for t in data["tokens"])


async def test_models_endpoint(aiohttp_client):
    """Models endpoint returns provider-aware list."""
    from mullet.server.app import create_app
    client = await aiohttp_client(create_app())

    resp = await client.get("/v1/models")
    assert resp.status == 200
    data = await resp.json()
    assert data["object"] == "list"
    # At minimum should have the "mullet" model
    names = [m["id"] for m in data["data"]]
    assert any("mullet" in n for n in names)


async def test_rate_limiting(aiohttp_client):
    """Rate limiter should return headers."""
    from mullet.server.app import create_app
    client = await aiohttp_client(create_app())

    resp = await client.get("/health")
    assert "X-RateLimit-Remaining" in resp.headers
    assert "X-RateLimit-Limit" in resp.headers
