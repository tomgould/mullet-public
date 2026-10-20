"""Mullet API server — aiohttp application setup.

Single process serving REST API + WebSocket on one port.
Started via `mullet serve`.
"""

import logging
from aiohttp import web

from mullet.server.api import setup_api_routes
from mullet.server.admin import setup_admin_routes
from mullet.server.websocket import WebSocketHandler

logger = logging.getLogger(__name__)


def create_app(
    host: str = "127.0.0.1",
    port: int = 3000,
    cors_origins: list[str] | None = None,
) -> web.Application:
    """Create the aiohttp application with all routes."""
    app = web.Application(client_max_size=1024 * 1024)  # 1MB request limit

    # Store config in app
    app["host"] = host
    app["port"] = port
    app["cors_origins"] = cors_origins or ["http://localhost:*", "http://127.0.0.1:*"]
    app["active_tasks"] = {}

    # Initialize database
    _init_database(app)

    # Middleware (order matters: error handling → rate limiting → CORS → auth)
    from mullet.server.middleware import error_handling_middleware, rate_limit_middleware
    app.middlewares.append(error_handling_middleware)
    app.middlewares.append(rate_limit_middleware)
    app.middlewares.append(cors_middleware)
    app.middlewares.append(auth_middleware)

    # REST routes
    setup_api_routes(app)
    setup_admin_routes(app)

    # WebSocket
    ws_handler = WebSocketHandler()
    app.router.add_get("/ws", ws_handler.handle)

    # Startup/cleanup hooks
    app.on_startup.append(_start_ws_cleanup)
    app.on_cleanup.append(_cleanup)

    return app


def _init_database(app: web.Application):
    """Initialize database and bootstrap solo mode."""
    try:
        from mullet.server.db import init_db, bootstrap_solo_mode
        db = init_db()
        bootstrap_solo_mode(db)
        app["db"] = db
        logger.info("Database initialized")
    except Exception as e:
        logger.warning("Database initialization failed: %s", e)
        app["db"] = None


async def _start_ws_cleanup(app: web.Application):
    """Start WebSocket stale-session cleanup when the event loop is running."""
    WebSocketHandler.start_cleanup()


async def _cleanup(app: web.Application):
    """Graceful shutdown — close DB, cancel active tasks."""
    # Cancel active tasks
    active = app.get("active_tasks", {})
    for task_id, task in list(active.items()):
        task.cancel()
        logger.info("Cancelled active task: %s", task_id)
    active.clear()

    # Close database
    db = app.get("db")
    if db:
        try:
            db.close()
            logger.info("Database connection closed")
        except Exception as _exc:
            logger.warning("app: Failed to close database connection: %s", _exc)


@web.middleware
async def cors_middleware(request, handler):
    """CORS middleware for local dev and configured origins."""
    # Handle preflight
    if request.method == "OPTIONS":
        response = web.Response()
    else:
        response = await handler(request)

    origin = request.headers.get("Origin", "")
    if origin:
        # Validate origin against allowlist
        import fnmatch
        allowed = request.app.get("cors_origins", ["http://localhost:*", "http://127.0.0.1:*"])
        origin_allowed = any(fnmatch.fnmatch(origin, pattern) for pattern in allowed)
        if origin_allowed:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            response.headers["Access-Control-Max-Age"] = "3600"
    return response


@web.middleware
async def auth_middleware(request, handler):
    """Authentication middleware — solo bypasses, enterprise requires PAT."""
    from mullet.config.mode import MulletMode
    from mullet.server.auth import get_solo_user, verify_pat

    # Public endpoints — no auth required
    public_paths = {"/health", "/v1/models", "/ws"}
    if request.path in public_paths:
        request["user"] = get_solo_user()
        return await handler(request)

    if MulletMode.is_solo():
        request["user"] = get_solo_user()
        return await handler(request)

    # Enterprise mode — require Bearer token
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return web.json_response(
            {"error": {"message": "Authentication required", "type": "auth_error"}},
            status=401,
        )

    token = auth_header[7:]
    db = request.app.get("db")
    if db:
        user = verify_pat(token, db)
        if user:
            request["user"] = user
            return await handler(request)

    return web.json_response(
        {"error": {"message": "Invalid or expired token", "type": "auth_error"}},
        status=401,
    )


def run_server(host: str = "127.0.0.1", port: int = 3000):
    """Start the Mullet API server."""
    app = create_app(host=host, port=port)
    logger.info("Starting Mullet server on %s:%d", host, port)
    print(f"Mullet API server running on http://{host}:{port}")
    print(f"  REST: http://{host}:{port}/v1/chat/completions")
    print(f"  WebSocket: ws://{host}:{port}/ws")
    print(f"  Health: http://{host}:{port}/health")
    print("  MCP: mullet mcp (separate process)")
    web.run_app(app, host=host, port=port, print=None)
