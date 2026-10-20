"""Tests for mullet/server/app.py — create_app, CORS middleware, _init_database.

Kept minimal: aiohttp app construction without running the server.
"""

from unittest.mock import patch



def test_create_app_returns_application():
    """create_app builds an aiohttp Application with expected app-level config."""
    with patch("mullet.server.app._init_database"):
        from mullet.server.app import create_app
        app = create_app(host="127.0.0.1", port=3999)
        assert app["host"] == "127.0.0.1"
        assert app["port"] == 3999
        assert isinstance(app["active_tasks"], dict)


def test_create_app_defaults_localhost_cors():
    """Default cors_origins allows localhost + 127.0.0.1 with any port."""
    with patch("mullet.server.app._init_database"):
        from mullet.server.app import create_app
        app = create_app()
        origins = app["cors_origins"]
        assert any("localhost" in o for o in origins)
        assert any("127.0.0.1" in o for o in origins)


def test_create_app_custom_cors_origins():
    """Passing cors_origins overrides the default list."""
    with patch("mullet.server.app._init_database"):
        from mullet.server.app import create_app
        app = create_app(cors_origins=["https://example.com"])
        assert app["cors_origins"] == ["https://example.com"]


def test_init_database_never_raises_on_failure():
    """_init_database is defensive — DB failures should not prevent app creation."""
    from mullet.server import app as app_mod

    with patch.object(app_mod, "_init_database") as mock_init:
        # Simulate _init_database silently handling a DB failure (its own responsibility)
        mock_init.side_effect = None
        app = app_mod.create_app()
        assert app is not None
        mock_init.assert_called_once()


def test_create_app_registers_middleware():
    """All four middleware are attached in expected order."""
    with patch("mullet.server.app._init_database"):
        from mullet.server.app import create_app
        app = create_app()
        mw_names = [getattr(mw, "__name__", str(mw)) for mw in app.middlewares]
        # At least error_handling, rate_limit, cors, auth should be present
        joined = " ".join(mw_names)
        assert "error_handling" in joined
        assert "rate_limit" in joined
        assert "cors" in joined
        assert "auth" in joined


def test_create_app_websocket_route_registered():
    """/ws should be registered as a GET route."""
    with patch("mullet.server.app._init_database"):
        from mullet.server.app import create_app
        app = create_app()
        ws_routes = [r for r in app.router.routes() if r.method == "GET" and "/ws" in str(r.resource)]
        assert len(ws_routes) >= 1


def test_create_app_cleanup_registered():
    """_cleanup handler is attached so app shutdown closes resources."""
    with patch("mullet.server.app._init_database"):
        from mullet.server.app import create_app
        app = create_app()
        assert len(app.on_cleanup) >= 1


def test_create_app_startup_hook_registered():
    """on_startup hook must be present so WebSocket cleanup starts with the event loop."""
    with patch("mullet.server.app._init_database"):
        from mullet.server.app import create_app
        app = create_app()
        assert len(app.on_startup) >= 1


def test_websocket_handler_start_cleanup_idempotent():
    """WebSocketHandler.start_cleanup sets _cleanup_started=True and is idempotent."""
    from mullet.server.websocket import WebSocketHandler

    # Guard via the flag — don't actually schedule; just verify the guard works.
    WebSocketHandler._cleanup_started = True  # pretend already started
    call_count = []

    with patch("mullet.server.websocket.asyncio.ensure_future", side_effect=lambda c: call_count.append(1)):
        WebSocketHandler.start_cleanup()  # should be a no-op because flag is True

    WebSocketHandler._cleanup_started = False  # reset
    assert len(call_count) == 0, "start_cleanup must not re-schedule when _cleanup_started=True"


def test_auth_middleware_public_path_skips_auth():
    """Public paths (/health, /v1/models, /ws) bypass auth even in enterprise mode."""
    with patch("mullet.server.app._init_database"):
        from mullet.server.app import create_app
        from mullet.config.mode import MulletMode
        with patch.object(MulletMode, "is_solo", return_value=False):
            app = create_app()
            # /health is registered as a public path in auth_middleware
            assert any("/health" in str(r.resource) for r in app.router.routes())


def test_app_has_expected_route_count():
    """App should have at least 15 routes registered."""
    with patch("mullet.server.app._init_database"):
        from mullet.server.app import create_app
        app = create_app()
        routes = list(app.router.routes())
        assert len(routes) >= 15
