"""Admin REST endpoints for enterprise mode.

User CRUD, project management, access control, usage tracking.
All endpoints require admin role (enforced by middleware).
"""

import logging
import time
import uuid

from aiohttp import web

logger = logging.getLogger(__name__)


def setup_admin_routes(app: web.Application):
    """Register admin API routes."""
    # Users
    app.router.add_get("/api/v1/admin/users", list_users)
    app.router.add_post("/api/v1/admin/users", create_user)
    app.router.add_put("/api/v1/admin/users/{user_id}", update_user)
    app.router.add_delete("/api/v1/admin/users/{user_id}", delete_user)

    # Projects
    app.router.add_get("/api/v1/admin/projects", list_projects)
    app.router.add_post("/api/v1/admin/projects", create_project)
    app.router.add_delete("/api/v1/admin/projects/{project_id}", delete_project)

    # Access control
    app.router.add_get("/api/v1/admin/access/{project_id}", list_access)
    app.router.add_post("/api/v1/admin/access", grant_access)
    app.router.add_delete("/api/v1/admin/access", revoke_access)

    # Usage
    app.router.add_get("/api/v1/admin/usage", get_usage)

    # Tokens
    app.router.add_get("/api/v1/tokens", list_tokens)
    app.router.add_post("/api/v1/tokens", create_token)
    app.router.add_delete("/api/v1/tokens/{token_id}", revoke_token)


def _require_admin(request) -> bool:
    user = request.get("user", {})
    return user.get("role") == "admin"


def _get_db(request):
    return request.app.get("db")


# --- Users ---

async def list_users(request: web.Request) -> web.Response:
    """List all users in the system."""
    if not _require_admin(request):
        return web.json_response({"error": "Admin required"}, status=403)
    db = _get_db(request)
    if not db:
        return web.json_response({"error": "Database not initialized"}, status=500)
    rows = db.execute("SELECT id, email, display_name, role, created_at, last_login_at FROM users").fetchall()
    users = [{"id": r[0], "email": r[1], "display_name": r[2], "role": r[3],
              "created_at": r[4], "last_login_at": r[5]} for r in rows]
    return web.json_response({"users": users})


async def create_user(request: web.Request) -> web.Response:
    """Create a new user with the provided email and role."""
    if not _require_admin(request):
        return web.json_response({"error": "Admin required"}, status=403)
    db = _get_db(request)
    if not db:
        return web.json_response({"error": "Database not initialized"}, status=500)
    body = await request.json()
    email = body.get("email")
    role = body.get("role", "dev")
    display_name = body.get("display_name", "")
    if not email:
        return web.json_response({"error": "email required"}, status=400)
    user_id = str(uuid.uuid4())
    try:
        db.execute(
            "INSERT INTO users (id, email, display_name, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, email, display_name, role, time.time()),
        )
        db.commit()
        return web.json_response({"id": user_id, "email": email, "role": role}, status=201)
    except Exception as e:
        logger.warning("create_user failed: %s", e)
        return web.json_response({"error": "A record with that identifier already exists"}, status=409)


async def update_user(request: web.Request) -> web.Response:
    """Update a user's role by ID."""
    if not _require_admin(request):
        return web.json_response({"error": "Admin required"}, status=403)
    db = _get_db(request)
    if not db:
        return web.json_response({"error": "Database not initialized"}, status=500)
    user_id = request.match_info["user_id"]
    body = await request.json()
    role = body.get("role")
    if role:
        db.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
        db.commit()
    return web.json_response({"id": user_id, "updated": True})


async def delete_user(request: web.Request) -> web.Response:
    """Delete a user and all associated data by ID."""
    if not _require_admin(request):
        return web.json_response({"error": "Admin required"}, status=403)
    db = _get_db(request)
    if not db:
        return web.json_response({"error": "Database not initialized"}, status=500)
    user_id = request.match_info["user_id"]
    db.execute("DELETE FROM user_projects WHERE user_id = ?", (user_id,))
    db.execute("DELETE FROM access_tokens WHERE user_id = ?", (user_id,))
    db.execute("DELETE FROM users WHERE id = ?", (user_id,))
    db.commit()
    return web.json_response({"deleted": True})


# --- Projects ---

async def list_projects(request: web.Request) -> web.Response:
    """List all projects in the system."""
    if not _require_admin(request):
        return web.json_response({"error": "Admin required"}, status=403)
    db = _get_db(request)
    if not db:
        return web.json_response({"error": "Database not initialized"}, status=500)
    rows = db.execute("SELECT id, name, pool_name, description, created_at FROM projects").fetchall()
    projects = [{"id": r[0], "name": r[1], "pool": r[2],
                 "description": r[3], "created_at": r[4]} for r in rows]
    return web.json_response({"projects": projects})


async def create_project(request: web.Request) -> web.Response:
    """Create a new project with the provided name and description."""
    if not _require_admin(request):
        return web.json_response({"error": "Admin required"}, status=403)
    db = _get_db(request)
    if not db:
        return web.json_response({"error": "Database not initialized"}, status=500)
    body = await request.json()
    name = body.get("name")
    pool = body.get("pool_name", name)
    description = body.get("description", "")
    if not name:
        return web.json_response({"error": "name required"}, status=400)
    project_id = str(uuid.uuid4())
    user = request.get("user", {})
    try:
        db.execute(
            "INSERT INTO projects (id, name, pool_name, description, created_by, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (project_id, name, pool, description, user.get("id"), time.time()),
        )
        db.commit()
        return web.json_response({"id": project_id, "name": name}, status=201)
    except Exception as e:
        logger.warning("create_project failed: %s", e)
        return web.json_response({"error": "A record with that identifier already exists"}, status=409)


async def delete_project(request: web.Request) -> web.Response:
    """Delete a project and all associated access by ID."""
    if not _require_admin(request):
        return web.json_response({"error": "Admin required"}, status=403)
    db = _get_db(request)
    if not db:
        return web.json_response({"error": "Database not initialized"}, status=500)
    project_id = request.match_info["project_id"]
    db.execute("DELETE FROM user_projects WHERE project_id = ?", (project_id,))
    db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    db.commit()
    return web.json_response({"deleted": True})


# --- Access Control ---

async def list_access(request: web.Request) -> web.Response:
    """List all users with access to a specific project."""
    if not _require_admin(request):
        return web.json_response({"error": "Admin required"}, status=403)
    db = _get_db(request)
    if not db:
        return web.json_response({"error": "Database not initialized"}, status=500)
    project_id = request.match_info["project_id"]
    rows = db.execute(
        "SELECT u.id, u.email, u.display_name, up.access_level FROM user_projects up "
        "JOIN users u ON up.user_id = u.id WHERE up.project_id = ?",
        (project_id,),
    ).fetchall()
    access = [{"user_id": r[0], "email": r[1], "display_name": r[2], "access_level": r[3]} for r in rows]
    return web.json_response({"access": access})


async def grant_access(request: web.Request) -> web.Response:
    """Grant a user access to a project with specified access level."""
    if not _require_admin(request):
        return web.json_response({"error": "Admin required"}, status=403)
    db = _get_db(request)
    if not db:
        return web.json_response({"error": "Database not initialized"}, status=500)
    body = await request.json()
    user_id = body.get("user_id")
    project_id = body.get("project_id")
    access_level = body.get("access_level", "read")
    if not user_id or not project_id:
        return web.json_response({"error": "user_id and project_id required"}, status=400)
    granter = request.get("user", {})
    db.execute(
        "INSERT OR REPLACE INTO user_projects (user_id, project_id, access_level, granted_at, granted_by) "
        "VALUES (?, ?, ?, ?, ?)",
        (user_id, project_id, access_level, time.time(), granter.get("id")),
    )
    db.commit()
    return web.json_response({"granted": True, "access_level": access_level})


async def revoke_access(request: web.Request) -> web.Response:
    """Revoke a user's access to a project."""
    if not _require_admin(request):
        return web.json_response({"error": "Admin required"}, status=403)
    db = _get_db(request)
    if not db:
        return web.json_response({"error": "Database not initialized"}, status=500)
    body = await request.json()
    user_id = body.get("user_id")
    project_id = body.get("project_id")
    if not user_id or not project_id:
        return web.json_response({"error": "user_id and project_id required"}, status=400)
    db.execute("DELETE FROM user_projects WHERE user_id = ? AND project_id = ?", (user_id, project_id))
    db.commit()
    return web.json_response({"revoked": True})


# --- Usage ---

async def get_usage(request: web.Request) -> web.Response:
    """Get aggregated usage statistics across all users and providers."""
    if not _require_admin(request):
        return web.json_response({"error": "Admin required"}, status=403)
    db = _get_db(request)
    if not db:
        return web.json_response({"error": "Database not initialized"}, status=500)
    rows = db.execute(
        "SELECT user_id, provider, SUM(input_tokens), SUM(output_tokens), "
        "SUM(estimated_cost), COUNT(*) FROM usage_log GROUP BY user_id, provider"
    ).fetchall()
    usage = [{"user_id": r[0], "provider": r[1], "input_tokens": r[2],
              "output_tokens": r[3], "total_cost": round(r[4], 4), "requests": r[5]} for r in rows]
    return web.json_response({"usage": usage})


# --- Tokens ---

async def list_tokens(request: web.Request) -> web.Response:
    """List all access tokens for the current user."""
    user = request.get("user", {})
    db = _get_db(request)
    if not db:
        return web.json_response({"error": "Database not initialized"}, status=500)
    rows = db.execute(
        "SELECT id, name, created_at, last_used_at, expires_at FROM access_tokens WHERE user_id = ?",
        (user.get("id"),),
    ).fetchall()
    tokens = [{"id": r[0], "name": r[1], "created_at": r[2],
               "last_used_at": r[3], "expires_at": r[4]} for r in rows]
    return web.json_response({"tokens": tokens})


async def create_token(request: web.Request) -> web.Response:
    """Create a new personal access token for the current user."""
    user = request.get("user", {})
    db = _get_db(request)
    if not db:
        return web.json_response({"error": "Database not initialized"}, status=500)
    body = await request.json()
    name = body.get("name", "CLI Token")
    from mullet.server.auth import generate_pat
    raw_token, token_hash = generate_pat()
    token_id = str(uuid.uuid4())
    db.execute(
        "INSERT INTO access_tokens (id, user_id, token_hash, name, created_at) VALUES (?, ?, ?, ?, ?)",
        (token_id, user.get("id"), token_hash, name, time.time()),
    )
    db.commit()
    return web.json_response({"id": token_id, "token": raw_token, "name": name}, status=201)


async def revoke_token(request: web.Request) -> web.Response:
    """Revoke a specific access token by ID for the current user."""
    user = request.get("user", {})
    db = _get_db(request)
    if not db:
        return web.json_response({"error": "Database not initialized"}, status=500)
    token_id = request.match_info["token_id"]
    db.execute("DELETE FROM access_tokens WHERE id = ? AND user_id = ?", (token_id, user.get("id")))
    db.commit()
    return web.json_response({"revoked": True})
