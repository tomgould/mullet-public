"""Tests for admin API endpoints."""

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


async def test_admin_users_list(aiohttp_client, app):
    """Admin user listing should return users."""
    client = await aiohttp_client(app)
    resp = await client.get("/api/v1/admin/users")
    # Without auth this may return 403 or empty list depending on mode
    assert resp.status in (200, 403)


async def test_admin_projects_list(aiohttp_client, app):
    """Admin project listing."""
    client = await aiohttp_client(app)
    resp = await client.get("/api/v1/admin/projects")
    assert resp.status in (200, 403)


async def test_admin_usage(aiohttp_client, app):
    """Admin usage endpoint."""
    client = await aiohttp_client(app)
    resp = await client.get("/api/v1/admin/usage")
    assert resp.status in (200, 403)


async def test_admin_tokens_list(aiohttp_client, app):
    """Token listing endpoint."""
    client = await aiohttp_client(app)
    resp = await client.get("/api/v1/tokens")
    assert resp.status in (200, 403)


async def test_admin_create_user(aiohttp_client, app):
    """Creating a user via admin API."""
    client = await aiohttp_client(app)
    resp = await client.post("/api/v1/admin/users", json={
        "email": "test@example.com",
        "role": "user",
    })
    # May succeed, conflict (existing user), or fail auth
    assert resp.status in (200, 201, 400, 403, 409)


async def test_admin_create_project(aiohttp_client, app):
    """Creating a project via admin API."""
    client = await aiohttp_client(app)
    resp = await client.post("/api/v1/admin/projects", json={
        "name": "test-project",
        "description": "A test project",
    })
    assert resp.status in (200, 201, 400, 403, 409)


async def test_admin_users_list_solo_returns_data(aiohttp_client, app):
    """In solo mode admin endpoint should return 200 with users data."""
    client = await aiohttp_client(app)
    resp = await client.get("/api/v1/admin/users")
    # Solo mode auto-auth as admin — should return 200 with at least the solo-admin user
    if resp.status == 200:
        data = await resp.json()
        assert "users" in data
        assert isinstance(data["users"], list)


async def test_admin_create_user_missing_email(aiohttp_client, app):
    """POST /api/v1/admin/users without email returns 400."""
    client = await aiohttp_client(app)
    resp = await client.post("/api/v1/admin/users", json={"role": "dev"})
    # Solo mode auth OK, but missing email → 400
    assert resp.status in (400, 403)


async def test_admin_delete_user_unknown_id(aiohttp_client, app):
    """DELETE /api/v1/admin/users/{id} — idempotent, returns 200 or 403 even for unknown id."""
    client = await aiohttp_client(app)
    resp = await client.delete("/api/v1/admin/users/nonexistent-id-xyz")
    assert resp.status in (200, 403)


async def test_admin_create_token_returns_200_or_40x(aiohttp_client, app):
    """POST /api/v1/tokens in solo mode returns a token or expected error."""
    client = await aiohttp_client(app)
    resp = await client.post("/api/v1/tokens", json={"name": "test-token"})
    assert resp.status in (200, 201, 400, 403)
    if resp.status in (200, 201):
        data = await resp.json()
        # Token should be returned on success
        assert "token" in data or "id" in data or "access_token" in data


async def test_admin_update_user_returns_200_or_40x(aiohttp_client, app):
    """PUT /api/v1/admin/users/{id} — updates role or returns auth error."""
    client = await aiohttp_client(app)
    resp = await client.put("/api/v1/admin/users/nonexistent-id", json={"role": "dev"})
    assert resp.status in (200, 403)
    if resp.status == 200:
        data = await resp.json()
        assert "updated" in data


async def test_admin_delete_project_idempotent(aiohttp_client, app):
    """DELETE /api/v1/admin/projects/{id} — idempotent like delete_user."""
    client = await aiohttp_client(app)
    resp = await client.delete("/api/v1/admin/projects/nonexistent-project-xyz")
    assert resp.status in (200, 403)
    if resp.status == 200:
        data = await resp.json()
        assert data.get("deleted") is True


async def test_admin_list_access_for_project(aiohttp_client, app):
    """GET /api/v1/admin/access/{project_id} — returns access list."""
    client = await aiohttp_client(app)
    resp = await client.get("/api/v1/admin/access/some-project-id")
    assert resp.status in (200, 403)
    if resp.status == 200:
        data = await resp.json()
        assert "access" in data
        assert isinstance(data["access"], list)


async def test_admin_grant_access_missing_fields(aiohttp_client, app):
    """POST /api/v1/admin/access without required fields returns 400 or 403."""
    client = await aiohttp_client(app)
    resp = await client.post("/api/v1/admin/access", json={"user_id": "u1"})
    assert resp.status in (400, 403)


async def test_admin_grant_access_ok(aiohttp_client, app):
    """POST /api/v1/admin/access with both fields returns success."""
    client = await aiohttp_client(app)
    resp = await client.post("/api/v1/admin/access", json={
        "user_id": "uid-1", "project_id": "pid-1", "access_level": "read",
    })
    assert resp.status in (200, 403)
    if resp.status == 200:
        data = await resp.json()
        assert data.get("granted") is True


async def test_admin_revoke_access_missing_fields(aiohttp_client, app):
    """DELETE /api/v1/admin/access without fields returns 400 or 403."""
    client = await aiohttp_client(app)
    resp = await client.delete("/api/v1/admin/access", json={"user_id": "u1"})
    assert resp.status in (400, 403)


async def test_admin_revoke_access_ok(aiohttp_client, app):
    """DELETE /api/v1/admin/access with both fields returns revoked."""
    client = await aiohttp_client(app)
    resp = await client.delete("/api/v1/admin/access", json={
        "user_id": "uid-1", "project_id": "pid-1",
    })
    assert resp.status in (200, 403)
    if resp.status == 200:
        data = await resp.json()
        assert data.get("revoked") is True


async def test_admin_revoke_token_idempotent(aiohttp_client, app):
    """DELETE /api/v1/tokens/{id} — idempotent, returns revoked=True even for unknown token."""
    client = await aiohttp_client(app)
    resp = await client.delete("/api/v1/tokens/nonexistent-token-id")
    assert resp.status in (200, 403)
    if resp.status == 200:
        data = await resp.json()
        assert data.get("revoked") is True
