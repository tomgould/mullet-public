"""Tests covering uncovered paths in mullet/server/app.py.

Lines targeted: 58-66, 71, 77-90, 97-113, 119-148, 156-163.
Uses pytest-aiohttp (aiohttp_client fixture) and asyncio_mode=auto from pyproject.toml.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web


# ---------------------------------------------------------------------------
# _init_database (lines 58-66)
# ---------------------------------------------------------------------------

class TestInitDatabase:
    """Tests for the _init_database helper."""

    def test_success_sets_db_on_app(self):
        """When init_db and bootstrap_solo_mode succeed, app['db'] is set."""
        from mullet.server.app import _init_database

        app = web.Application()
        mock_db = MagicMock()

        with patch("mullet.server.db.init_db", return_value=mock_db) as mock_init, \
             patch("mullet.server.db.bootstrap_solo_mode") as mock_boot:
            _init_database(app)

        assert app["db"] is mock_db
        mock_init.assert_called_once()
        mock_boot.assert_called_once_with(mock_db)

    def test_exception_sets_db_none(self):
        """When init_db raises, app['db'] is set to None (no re-raise)."""
        from mullet.server.app import _init_database

        app = web.Application()

        with patch("mullet.server.db.init_db", side_effect=Exception("DB offline")):
            _init_database(app)  # must not raise

        assert app["db"] is None

    def test_bootstrap_exception_sets_db_none(self):
        """When bootstrap_solo_mode raises, app['db'] is also set to None."""
        from mullet.server.app import _init_database

        app = web.Application()
        mock_db = MagicMock()

        with patch("mullet.server.db.init_db", return_value=mock_db), \
             patch("mullet.server.db.bootstrap_solo_mode", side_effect=RuntimeError("seed fail")):
            _init_database(app)

        assert app["db"] is None

    def test_logs_warning_on_failure(self, caplog):
        """A warning is logged when DB init fails."""
        from mullet.server.app import _init_database

        app = web.Application()

        with caplog.at_level(logging.WARNING, logger="mullet.server.app"), \
             patch("mullet.server.db.init_db", side_effect=Exception("boom")):
            _init_database(app)

        assert any("boom" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _start_ws_cleanup (line 71)
# ---------------------------------------------------------------------------

class TestStartWsCleanup:
    """Tests for the _start_ws_cleanup startup hook."""

    async def test_calls_websocket_handler_start_cleanup(self):
        """_start_ws_cleanup delegates to WebSocketHandler.start_cleanup."""
        from mullet.server.app import _start_ws_cleanup
        from mullet.server.websocket import WebSocketHandler

        app = web.Application()

        with patch.object(WebSocketHandler, "start_cleanup") as mock_cleanup:
            await _start_ws_cleanup(app)

        mock_cleanup.assert_called_once()

    async def test_start_ws_cleanup_returns_none(self):
        """_start_ws_cleanup is a coroutine that returns None."""
        from mullet.server.app import _start_ws_cleanup
        from mullet.server.websocket import WebSocketHandler

        with patch.object(WebSocketHandler, "start_cleanup"):
            result = await _start_ws_cleanup({})

        assert result is None


# ---------------------------------------------------------------------------
# _cleanup (lines 77-90)
# ---------------------------------------------------------------------------

class TestCleanup:
    """Tests for the _cleanup shutdown hook."""

    async def test_no_active_tasks_no_db(self):
        """Empty app state completes without error."""
        from mullet.server.app import _cleanup

        app = {"active_tasks": {}, "db": None}
        await _cleanup(app)  # must not raise

    async def test_active_tasks_are_cancelled(self):
        """Each active task has cancel() called and the dict is cleared."""
        from mullet.server.app import _cleanup

        mock_task_a = MagicMock()
        mock_task_b = MagicMock()
        active = {"t1": mock_task_a, "t2": mock_task_b}
        app = {"active_tasks": active, "db": None}

        await _cleanup(app)

        mock_task_a.cancel.assert_called_once()
        mock_task_b.cancel.assert_called_once()
        assert len(active) == 0

    async def test_db_close_called(self):
        """When db is present its close() method is called."""
        from mullet.server.app import _cleanup

        mock_db = MagicMock()
        app = {"active_tasks": {}, "db": mock_db}

        await _cleanup(app)

        mock_db.close.assert_called_once()

    async def test_db_close_exception_does_not_propagate(self):
        """If db.close() raises, _cleanup swallows the error."""
        from mullet.server.app import _cleanup

        mock_db = MagicMock()
        mock_db.close.side_effect = Exception("close failed")
        app = {"active_tasks": {}, "db": mock_db}

        await _cleanup(app)  # must not raise

    async def test_db_close_exception_logs_warning(self, caplog):
        """If db.close() raises, a warning is logged."""
        from mullet.server.app import _cleanup

        mock_db = MagicMock()
        mock_db.close.side_effect = Exception("close failed")
        app = {"active_tasks": {}, "db": mock_db}

        with caplog.at_level(logging.WARNING, logger="mullet.server.app"):
            await _cleanup(app)

        assert any("close" in r.message.lower() for r in caplog.records)

    async def test_missing_active_tasks_key(self):
        """If active_tasks is absent, _cleanup uses .get() safely."""
        from mullet.server.app import _cleanup

        app = {}  # no active_tasks key
        await _cleanup(app)  # must not raise


# ---------------------------------------------------------------------------
# cors_middleware (lines 97-113)
# ---------------------------------------------------------------------------

class TestCorsMiddleware:
    """Tests for the cors_middleware using aiohttp test client."""

    @pytest.fixture
    def cors_app(self):
        """Minimal app with only cors_middleware wired up."""
        from mullet.server.app import cors_middleware

        app = web.Application(middlewares=[cors_middleware])
        app["cors_origins"] = ["http://localhost:*", "http://127.0.0.1:*"]

        async def handler(request):
            return web.Response(text="ok")

        app.router.add_get("/test", handler)
        app.router.add_route("OPTIONS", "/test", handler)
        return app

    async def test_options_returns_response_without_calling_handler(self, aiohttp_client, cors_app):
        """OPTIONS preflight returns a response immediately."""
        client = await aiohttp_client(cors_app)
        resp = await client.options(
            "/test",
            headers={"Origin": "http://localhost:3000"},
        )
        assert resp.status == 200

    async def test_allowed_origin_adds_cors_headers(self, aiohttp_client, cors_app):
        """GET with an allowed origin receives Access-Control-Allow-Origin header."""
        client = await aiohttp_client(cors_app)
        resp = await client.get(
            "/test",
            headers={"Origin": "http://localhost:3000"},
        )
        assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
        assert "GET" in resp.headers.get("Access-Control-Allow-Methods", "")
        assert "Content-Type" in resp.headers.get("Access-Control-Allow-Headers", "")

    async def test_disallowed_origin_no_cors_headers(self, aiohttp_client, cors_app):
        """GET from a disallowed origin does not receive CORS headers."""
        client = await aiohttp_client(cors_app)
        resp = await client.get(
            "/test",
            headers={"Origin": "https://evil.com"},
        )
        assert "Access-Control-Allow-Origin" not in resp.headers

    async def test_no_origin_header_no_cors_headers(self, aiohttp_client, cors_app):
        """GET without Origin header receives no CORS headers."""
        client = await aiohttp_client(cors_app)
        resp = await client.get("/test")
        assert "Access-Control-Allow-Origin" not in resp.headers

    async def test_cors_max_age_set(self, aiohttp_client, cors_app):
        """Access-Control-Max-Age is set for allowed origins."""
        client = await aiohttp_client(cors_app)
        resp = await client.get(
            "/test",
            headers={"Origin": "http://127.0.0.1:8080"},
        )
        assert resp.headers.get("Access-Control-Max-Age") == "3600"

    async def test_options_preflight_allowed_origin(self, aiohttp_client, cors_app):
        """OPTIONS from allowed origin also gets CORS headers."""
        client = await aiohttp_client(cors_app)
        resp = await client.options(
            "/test",
            headers={"Origin": "http://localhost:5000"},
        )
        assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:5000"


# ---------------------------------------------------------------------------
# auth_middleware (lines 119-148)
# ---------------------------------------------------------------------------

class TestAuthMiddleware:
    """Tests for auth_middleware."""

    @pytest.fixture
    def auth_app(self):
        """Minimal app with only auth_middleware."""
        from mullet.server.app import auth_middleware

        app = web.Application(middlewares=[auth_middleware])
        app["db"] = None  # default: no db

        async def handler(request):
            return web.Response(text="ok")

        for path in ["/health", "/v1/models", "/ws", "/protected"]:
            app.router.add_get(path, handler)

        return app

    async def test_public_path_health_passes_without_auth(self, aiohttp_client, auth_app):
        """/health is public — no auth header required, returns 200."""
        with patch("mullet.server.auth.get_solo_user", return_value={"id": "solo", "email": "a@b", "role": "admin"}):
            client = await aiohttp_client(auth_app)
            resp = await client.get("/health")

        assert resp.status == 200

    async def test_public_path_models_passes_without_auth(self, aiohttp_client, auth_app):
        """/v1/models is public — no auth header required."""
        with patch("mullet.server.auth.get_solo_user", return_value={"id": "solo"}):
            client = await aiohttp_client(auth_app)
            resp = await client.get("/v1/models")

        assert resp.status == 200

    async def test_solo_mode_passes_without_token(self, aiohttp_client, auth_app):
        """In solo mode any path passes without a Bearer token."""
        with patch("mullet.config.mode.MulletMode.is_solo", return_value=True), \
             patch("mullet.server.auth.get_solo_user", return_value={"id": "solo"}):
            client = await aiohttp_client(auth_app)
            resp = await client.get("/protected")

        assert resp.status == 200

    async def test_enterprise_mode_no_bearer_returns_401(self, aiohttp_client, auth_app):
        """Enterprise mode: missing Bearer token → 401."""
        with patch("mullet.config.mode.MulletMode.is_solo", return_value=False):
            client = await aiohttp_client(auth_app)
            resp = await client.get("/protected")

        assert resp.status == 401
        body = await resp.json()
        assert body["error"]["type"] == "auth_error"

    async def test_enterprise_mode_malformed_auth_returns_401(self, aiohttp_client, auth_app):
        """Authorization header without 'Bearer ' prefix → 401."""
        with patch("mullet.config.mode.MulletMode.is_solo", return_value=False):
            client = await aiohttp_client(auth_app)
            resp = await client.get("/protected", headers={"Authorization": "Basic abc"})

        assert resp.status == 401

    async def test_enterprise_mode_valid_pat_passes(self, aiohttp_client, auth_app):
        """Valid Bearer PAT in enterprise mode → 200."""
        mock_db = MagicMock()
        auth_app["db"] = mock_db

        with patch("mullet.config.mode.MulletMode.is_solo", return_value=False), \
             patch("mullet.server.auth.verify_pat", return_value={"id": "u1", "role": "admin"}):
            client = await aiohttp_client(auth_app)
            resp = await client.get(
                "/protected",
                headers={"Authorization": "Bearer valid-token"},
            )

        assert resp.status == 200

    async def test_enterprise_mode_invalid_pat_returns_401(self, aiohttp_client, auth_app):
        """Invalid Bearer PAT in enterprise mode → 401."""
        mock_db = MagicMock()
        auth_app["db"] = mock_db

        with patch("mullet.config.mode.MulletMode.is_solo", return_value=False), \
             patch("mullet.server.auth.verify_pat", return_value=None):
            client = await aiohttp_client(auth_app)
            resp = await client.get(
                "/protected",
                headers={"Authorization": "Bearer bad-token"},
            )

        assert resp.status == 401
        body = await resp.json()
        assert "expired" in body["error"]["message"].lower() or "invalid" in body["error"]["message"].lower()

    async def test_enterprise_mode_bearer_no_db_returns_401(self, aiohttp_client, auth_app):
        """When db is None in enterprise mode, Bearer tokens are rejected."""
        auth_app["db"] = None  # no db

        with patch("mullet.config.mode.MulletMode.is_solo", return_value=False):
            client = await aiohttp_client(auth_app)
            resp = await client.get(
                "/protected",
                headers={"Authorization": "Bearer some-token"},
            )

        assert resp.status == 401

    async def test_public_ws_path_passes(self, aiohttp_client, auth_app):
        """/ws is a public path — no auth needed."""
        with patch("mullet.server.auth.get_solo_user", return_value={"id": "solo"}):
            client = await aiohttp_client(auth_app)
            resp = await client.get("/ws")

        assert resp.status == 200


# ---------------------------------------------------------------------------
# run_server (lines 156-163)
# ---------------------------------------------------------------------------

class TestRunServer:
    """Tests for the run_server entry-point."""

    def test_run_server_calls_create_app_and_web_run_app(self, capsys):
        """run_server builds the app then delegates to web.run_app."""
        from mullet.server.app import run_server

        mock_app = MagicMock()

        with patch("mullet.server.app.create_app", return_value=mock_app) as mock_create, \
             patch("aiohttp.web.run_app") as mock_run_app:
            run_server(host="0.0.0.0", port=9999)

        mock_create.assert_called_once_with(host="0.0.0.0", port=9999)
        mock_run_app.assert_called_once()
        call_kwargs = mock_run_app.call_args
        assert call_kwargs[0][0] is mock_app or mock_app in call_kwargs[0]

    def test_run_server_prints_port(self, capsys):
        """run_server prints the URL including the port number."""
        from mullet.server.app import run_server

        with patch("mullet.server.app.create_app", return_value=MagicMock()), \
             patch("aiohttp.web.run_app"):
            run_server(host="127.0.0.1", port=7777)

        out = capsys.readouterr().out
        assert "7777" in out

    def test_run_server_prints_all_endpoint_hints(self, capsys):
        """Startup output includes REST, WebSocket, and health endpoint hints."""
        from mullet.server.app import run_server

        with patch("mullet.server.app.create_app", return_value=MagicMock()), \
             patch("aiohttp.web.run_app"):
            run_server(host="127.0.0.1", port=3000)

        out = capsys.readouterr().out
        assert "/v1/chat/completions" in out
        assert "/ws" in out
        assert "/health" in out

    def test_run_server_default_args(self, capsys):
        """run_server uses 127.0.0.1:3000 as defaults."""
        from mullet.server.app import run_server

        with patch("mullet.server.app.create_app", return_value=MagicMock()) as mock_create, \
             patch("aiohttp.web.run_app"):
            run_server()

        mock_create.assert_called_once_with(host="127.0.0.1", port=3000)

    def test_run_server_passes_print_none_to_web_run_app(self):
        """run_server suppresses aiohttp's own startup banner (print=None)."""
        from mullet.server.app import run_server

        with patch("mullet.server.app.create_app", return_value=MagicMock()), \
             patch("aiohttp.web.run_app") as mock_run_app:
            run_server()

        _, kwargs = mock_run_app.call_args
        assert kwargs.get("print") is None
