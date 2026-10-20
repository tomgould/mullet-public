"""Tests for admin API endpoints."""

from unittest.mock import MagicMock

import pytest

try:
    from aiohttp import web  # noqa: F401  (importability probe)
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytestmark = pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app(route_handler, method, path, *, db=None, role="admin"):
    """Build a minimal aiohttp app with a single route and inline middleware."""
    from aiohttp import web as _web

    @_web.middleware
    async def _auth(request, handler):
        request["user"] = {"id": "test-user", "role": role}
        return await handler(request)

    app = _web.Application(middlewares=[_auth])
    getattr(app.router, f"add_{method}")(path, route_handler)
    app["db"] = db
    return app


async def test_admin_users_list(aiohttp_client):
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/admin/users")
    assert resp.status == 200
    data = await resp.json()
    assert "users" in data


async def test_admin_projects_list(aiohttp_client):
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/admin/projects")
    assert resp.status == 200
    data = await resp.json()
    assert "projects" in data


async def test_projects_api(aiohttp_client):
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/projects")
    assert resp.status == 200
    data = await resp.json()
    assert "projects" in data


async def test_status_api(aiohttp_client):
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/status")
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "ok"
    assert "version" in data


async def test_tokens_api(aiohttp_client):
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    # List tokens (should be empty)
    resp = await client.get("/api/v1/tokens")
    assert resp.status == 200
    data = await resp.json()
    assert "tokens" in data


async def test_admin_usage(aiohttp_client):
    from mullet.server.app import create_app
    app = create_app()
    client = await aiohttp_client(app)

    resp = await client.get("/api/v1/admin/usage")
    assert resp.status == 200
    data = await resp.json()
    assert "usage" in data


# ---------------------------------------------------------------------------
# Error-path tests: 403 (non-admin) and 500 (no DB)
# These cover the missing branches on lines 57,60,70,73,96,99,112,115,
# 129,132,142,145,151,170,173,186,189,203,206,226,229,245,248,265,280,299
# ---------------------------------------------------------------------------

# --- list_users ---

async def test_list_users_non_admin(aiohttp_client):
    from mullet.server.admin import list_users
    app = _make_app(list_users, "get", "/admin/users", db=MagicMock(), role="dev")
    client = await aiohttp_client(app)
    resp = await client.get("/admin/users")
    assert resp.status == 403
    data = await resp.json()
    assert "Admin required" in data["error"]


async def test_list_users_no_db(aiohttp_client):
    from mullet.server.admin import list_users
    app = _make_app(list_users, "get", "/admin/users", db=None, role="admin")
    client = await aiohttp_client(app)
    resp = await client.get("/admin/users")
    assert resp.status == 500


# --- create_user ---

async def test_create_user_non_admin(aiohttp_client):
    from mullet.server.admin import create_user
    app = _make_app(create_user, "post", "/admin/users", db=MagicMock(), role="dev")
    client = await aiohttp_client(app)
    resp = await client.post("/admin/users", json={"email": "x@example.com"})
    assert resp.status == 403


async def test_create_user_no_db(aiohttp_client):
    from mullet.server.admin import create_user
    app = _make_app(create_user, "post", "/admin/users", db=None, role="admin")
    client = await aiohttp_client(app)
    resp = await client.post("/admin/users", json={"email": "x@example.com"})
    assert resp.status == 500


# --- update_user ---

async def test_update_user_non_admin(aiohttp_client):
    from mullet.server.admin import update_user
    app = _make_app(update_user, "put", "/admin/users/{user_id}", db=MagicMock(), role="dev")
    client = await aiohttp_client(app)
    resp = await client.put("/admin/users/uid-1", json={"role": "admin"})
    assert resp.status == 403


async def test_update_user_no_db(aiohttp_client):
    from mullet.server.admin import update_user
    app = _make_app(update_user, "put", "/admin/users/{user_id}", db=None, role="admin")
    client = await aiohttp_client(app)
    resp = await client.put("/admin/users/uid-1", json={"role": "admin"})
    assert resp.status == 500


# --- delete_user ---

async def test_delete_user_non_admin(aiohttp_client):
    from mullet.server.admin import delete_user
    app = _make_app(delete_user, "delete", "/admin/users/{user_id}", db=MagicMock(), role="dev")
    client = await aiohttp_client(app)
    resp = await client.delete("/admin/users/uid-1")
    assert resp.status == 403


async def test_delete_user_no_db(aiohttp_client):
    from mullet.server.admin import delete_user
    app = _make_app(delete_user, "delete", "/admin/users/{user_id}", db=None, role="admin")
    client = await aiohttp_client(app)
    resp = await client.delete("/admin/users/uid-1")
    assert resp.status == 500


# --- list_projects ---

async def test_list_projects_non_admin(aiohttp_client):
    from mullet.server.admin import list_projects
    app = _make_app(list_projects, "get", "/admin/projects", db=MagicMock(), role="dev")
    client = await aiohttp_client(app)
    resp = await client.get("/admin/projects")
    assert resp.status == 403


async def test_list_projects_no_db(aiohttp_client):
    from mullet.server.admin import list_projects
    app = _make_app(list_projects, "get", "/admin/projects", db=None, role="admin")
    client = await aiohttp_client(app)
    resp = await client.get("/admin/projects")
    assert resp.status == 500


# --- create_project ---

async def test_create_project_non_admin(aiohttp_client):
    from mullet.server.admin import create_project
    app = _make_app(create_project, "post", "/admin/projects", db=MagicMock(), role="dev")
    client = await aiohttp_client(app)
    resp = await client.post("/admin/projects", json={"name": "proj"})
    assert resp.status == 403


async def test_create_project_no_db(aiohttp_client):
    from mullet.server.admin import create_project
    app = _make_app(create_project, "post", "/admin/projects", db=None, role="admin")
    client = await aiohttp_client(app)
    resp = await client.post("/admin/projects", json={"name": "proj"})
    assert resp.status == 500


# --- delete_project ---

async def test_delete_project_non_admin(aiohttp_client):
    from mullet.server.admin import delete_project
    app = _make_app(delete_project, "delete", "/admin/projects/{project_id}", db=MagicMock(), role="dev")
    client = await aiohttp_client(app)
    resp = await client.delete("/admin/projects/proj-1")
    assert resp.status == 403


async def test_delete_project_no_db(aiohttp_client):
    from mullet.server.admin import delete_project
    app = _make_app(delete_project, "delete", "/admin/projects/{project_id}", db=None, role="admin")
    client = await aiohttp_client(app)
    resp = await client.delete("/admin/projects/proj-1")
    assert resp.status == 500


# --- list_access ---

async def test_list_access_non_admin(aiohttp_client):
    from mullet.server.admin import list_access
    app = _make_app(list_access, "get", "/admin/access/{project_id}", db=MagicMock(), role="dev")
    client = await aiohttp_client(app)
    resp = await client.get("/admin/access/proj-1")
    assert resp.status == 403


async def test_list_access_no_db(aiohttp_client):
    from mullet.server.admin import list_access
    app = _make_app(list_access, "get", "/admin/access/{project_id}", db=None, role="admin")
    client = await aiohttp_client(app)
    resp = await client.get("/admin/access/proj-1")
    assert resp.status == 500


# --- grant_access ---

async def test_grant_access_non_admin(aiohttp_client):
    from mullet.server.admin import grant_access
    app = _make_app(grant_access, "post", "/admin/access", db=MagicMock(), role="dev")
    client = await aiohttp_client(app)
    resp = await client.post("/admin/access", json={"user_id": "u1", "project_id": "p1"})
    assert resp.status == 403


async def test_grant_access_no_db(aiohttp_client):
    from mullet.server.admin import grant_access
    app = _make_app(grant_access, "post", "/admin/access", db=None, role="admin")
    client = await aiohttp_client(app)
    resp = await client.post("/admin/access", json={"user_id": "u1", "project_id": "p1"})
    assert resp.status == 500


# --- revoke_access ---

async def test_revoke_access_non_admin(aiohttp_client):
    from mullet.server.admin import revoke_access
    app = _make_app(revoke_access, "delete", "/admin/access", db=MagicMock(), role="dev")
    client = await aiohttp_client(app)
    resp = await client.delete("/admin/access", json={"user_id": "u1", "project_id": "p1"})
    assert resp.status == 403


async def test_revoke_access_no_db(aiohttp_client):
    from mullet.server.admin import revoke_access
    app = _make_app(revoke_access, "delete", "/admin/access", db=None, role="admin")
    client = await aiohttp_client(app)
    resp = await client.delete("/admin/access", json={"user_id": "u1", "project_id": "p1"})
    assert resp.status == 500


# --- get_usage ---

async def test_get_usage_non_admin(aiohttp_client):
    from mullet.server.admin import get_usage
    app = _make_app(get_usage, "get", "/admin/usage", db=MagicMock(), role="dev")
    client = await aiohttp_client(app)
    resp = await client.get("/admin/usage")
    assert resp.status == 403


async def test_get_usage_no_db(aiohttp_client):
    from mullet.server.admin import get_usage
    app = _make_app(get_usage, "get", "/admin/usage", db=None, role="admin")
    client = await aiohttp_client(app)
    resp = await client.get("/admin/usage")
    assert resp.status == 500


# --- list_tokens (no admin check, only DB check) ---

async def test_list_tokens_no_db(aiohttp_client):
    from mullet.server.admin import list_tokens
    app = _make_app(list_tokens, "get", "/tokens", db=None, role="admin")
    client = await aiohttp_client(app)
    resp = await client.get("/tokens")
    assert resp.status == 500


# --- create_token (no admin check, only DB check) ---

async def test_create_token_no_db(aiohttp_client):
    from mullet.server.admin import create_token
    app = _make_app(create_token, "post", "/tokens", db=None, role="admin")
    client = await aiohttp_client(app)
    resp = await client.post("/tokens", json={"name": "my-token"})
    assert resp.status == 500


# --- revoke_token (no admin check, only DB check) ---

async def test_revoke_token_no_db(aiohttp_client):
    from mullet.server.admin import revoke_token
    app = _make_app(revoke_token, "delete", "/tokens/{token_id}", db=None, role="admin")
    client = await aiohttp_client(app)
    resp = await client.delete("/tokens/tok-1")
    assert resp.status == 500


# ---------------------------------------------------------------------------
# Happy-path tests (cover DB query bodies and return values)
# ---------------------------------------------------------------------------

def _mock_db(rows=None):
    """Return a MagicMock that mimics sqlite3 cursor/connection."""
    db = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = rows or []
    db.execute.return_value = cursor
    return db


async def test_list_users_happy(aiohttp_client):
    from mullet.server.admin import list_users
    db = _mock_db([("u1", "a@b.com", "Alice", "admin", 1.0, 2.0)])
    app = _make_app(list_users, "get", "/admin/users", db=db, role="admin")
    client = await aiohttp_client(app)
    resp = await client.get("/admin/users")
    assert resp.status == 200
    data = await resp.json()
    assert data["users"][0]["email"] == "a@b.com"


async def test_create_user_happy(aiohttp_client):
    from mullet.server.admin import create_user
    app = _make_app(create_user, "post", "/admin/users", db=_mock_db(), role="admin")
    client = await aiohttp_client(app)
    resp = await client.post("/admin/users", json={"email": "new@example.com", "role": "dev"})
    assert resp.status == 201
    data = await resp.json()
    assert data["email"] == "new@example.com"


async def test_create_user_missing_email(aiohttp_client):
    from mullet.server.admin import create_user
    app = _make_app(create_user, "post", "/admin/users", db=_mock_db(), role="admin")
    client = await aiohttp_client(app)
    resp = await client.post("/admin/users", json={})
    assert resp.status == 400


async def test_create_user_duplicate(aiohttp_client):
    from mullet.server.admin import create_user
    db = _mock_db()
    db.execute.side_effect = Exception("UNIQUE constraint failed")
    app = _make_app(create_user, "post", "/admin/users", db=db, role="admin")
    client = await aiohttp_client(app)
    resp = await client.post("/admin/users", json={"email": "dup@example.com"})
    assert resp.status == 409


async def test_update_user_happy(aiohttp_client):
    from mullet.server.admin import update_user
    app = _make_app(update_user, "put", "/admin/users/{user_id}", db=_mock_db(), role="admin")
    client = await aiohttp_client(app)
    resp = await client.put("/admin/users/uid-1", json={"role": "admin"})
    assert resp.status == 200
    data = await resp.json()
    assert data["updated"] is True


async def test_delete_user_happy(aiohttp_client):
    from mullet.server.admin import delete_user
    app = _make_app(delete_user, "delete", "/admin/users/{user_id}", db=_mock_db(), role="admin")
    client = await aiohttp_client(app)
    resp = await client.delete("/admin/users/uid-1")
    assert resp.status == 200
    data = await resp.json()
    assert data["deleted"] is True


async def test_list_projects_happy(aiohttp_client):
    from mullet.server.admin import list_projects
    db = _mock_db([("p1", "MyProj", "pool-a", "desc", 1.0)])
    app = _make_app(list_projects, "get", "/admin/projects", db=db, role="admin")
    client = await aiohttp_client(app)
    resp = await client.get("/admin/projects")
    assert resp.status == 200
    data = await resp.json()
    assert data["projects"][0]["name"] == "MyProj"


async def test_create_project_happy(aiohttp_client):
    from mullet.server.admin import create_project
    app = _make_app(create_project, "post", "/admin/projects", db=_mock_db(), role="admin")
    client = await aiohttp_client(app)
    resp = await client.post("/admin/projects", json={"name": "NewProject"})
    assert resp.status == 201
    data = await resp.json()
    assert data["name"] == "NewProject"


async def test_create_project_missing_name(aiohttp_client):
    from mullet.server.admin import create_project
    app = _make_app(create_project, "post", "/admin/projects", db=_mock_db(), role="admin")
    client = await aiohttp_client(app)
    resp = await client.post("/admin/projects", json={})
    assert resp.status == 400


async def test_create_project_duplicate(aiohttp_client):
    from mullet.server.admin import create_project
    db = _mock_db()
    db.execute.side_effect = Exception("UNIQUE constraint failed")
    app = _make_app(create_project, "post", "/admin/projects", db=db, role="admin")
    client = await aiohttp_client(app)
    resp = await client.post("/admin/projects", json={"name": "dup"})
    assert resp.status == 409


async def test_delete_project_happy(aiohttp_client):
    from mullet.server.admin import delete_project
    app = _make_app(delete_project, "delete", "/admin/projects/{project_id}", db=_mock_db(), role="admin")
    client = await aiohttp_client(app)
    resp = await client.delete("/admin/projects/proj-1")
    assert resp.status == 200
    data = await resp.json()
    assert data["deleted"] is True


async def test_list_access_happy(aiohttp_client):
    from mullet.server.admin import list_access
    db = _mock_db([("u1", "a@b.com", "Alice", "read")])
    app = _make_app(list_access, "get", "/admin/access/{project_id}", db=db, role="admin")
    client = await aiohttp_client(app)
    resp = await client.get("/admin/access/proj-1")
    assert resp.status == 200
    data = await resp.json()
    assert data["access"][0]["email"] == "a@b.com"


async def test_grant_access_happy(aiohttp_client):
    from mullet.server.admin import grant_access
    app = _make_app(grant_access, "post", "/admin/access", db=_mock_db(), role="admin")
    client = await aiohttp_client(app)
    resp = await client.post("/admin/access", json={"user_id": "u1", "project_id": "p1"})
    assert resp.status == 200
    data = await resp.json()
    assert data["granted"] is True


async def test_grant_access_missing_fields(aiohttp_client):
    from mullet.server.admin import grant_access
    app = _make_app(grant_access, "post", "/admin/access", db=_mock_db(), role="admin")
    client = await aiohttp_client(app)
    resp = await client.post("/admin/access", json={})
    assert resp.status == 400


async def test_revoke_access_happy(aiohttp_client):
    from mullet.server.admin import revoke_access
    app = _make_app(revoke_access, "delete", "/admin/access", db=_mock_db(), role="admin")
    client = await aiohttp_client(app)
    resp = await client.delete("/admin/access", json={"user_id": "u1", "project_id": "p1"})
    assert resp.status == 200
    data = await resp.json()
    assert data["revoked"] is True


async def test_revoke_access_missing_fields(aiohttp_client):
    from mullet.server.admin import revoke_access
    app = _make_app(revoke_access, "delete", "/admin/access", db=_mock_db(), role="admin")
    client = await aiohttp_client(app)
    resp = await client.delete("/admin/access", json={})
    assert resp.status == 400


async def test_get_usage_happy(aiohttp_client):
    from mullet.server.admin import get_usage
    db = _mock_db([("u1", "anthropic", 1000, 500, 0.05, 10)])
    app = _make_app(get_usage, "get", "/admin/usage", db=db, role="admin")
    client = await aiohttp_client(app)
    resp = await client.get("/admin/usage")
    assert resp.status == 200
    data = await resp.json()
    assert data["usage"][0]["provider"] == "anthropic"


async def test_list_tokens_happy(aiohttp_client):
    from mullet.server.admin import list_tokens
    db = _mock_db([("tok1", "CLI Token", 1.0, 2.0, None)])
    app = _make_app(list_tokens, "get", "/tokens", db=db, role="admin")
    client = await aiohttp_client(app)
    resp = await client.get("/tokens")
    assert resp.status == 200
    data = await resp.json()
    assert data["tokens"][0]["name"] == "CLI Token"


async def test_create_token_happy(aiohttp_client):
    from mullet.server.admin import create_token
    app = _make_app(create_token, "post", "/tokens", db=_mock_db(), role="admin")
    client = await aiohttp_client(app)
    resp = await client.post("/tokens", json={"name": "my-pat"})
    assert resp.status == 201
    data = await resp.json()
    assert data["name"] == "my-pat"
    assert "token" in data


async def test_revoke_token_happy(aiohttp_client):
    from mullet.server.admin import revoke_token
    app = _make_app(revoke_token, "delete", "/tokens/{token_id}", db=_mock_db(), role="admin")
    client = await aiohttp_client(app)
    resp = await client.delete("/tokens/tok-1")
    assert resp.status == 200
    data = await resp.json()
    assert data["revoked"] is True
