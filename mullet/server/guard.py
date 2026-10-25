"""Keeps web pages and other machines from driving ``mullet serve``.

A browser sends any page's requests to ``localhost`` as readily as to its own site, and
CORS headers only stop the page reading the answer, not the server acting on the request.
So the server refuses, before anything else runs:

- a request whose ``Host`` is not one of its own names (DNS rebinding),
- a request carrying an ``Origin`` that is not on the allowlist (any other web page),
- in solo mode, a request without this install's token (everything except ``/health``).

The token is made on first use, kept in ``serve-token`` in the Mullet config directory
(readable only by the user), and sent as ``Authorization: Bearer <token>`` or
``X-Mullet-Token``. A WebSocket may also put it in ``?token=``, since browsers cannot set
headers there. ``server.require_token``, ``server.allowed_origins`` and
``server.allowed_hosts`` in the config file change all of this.
"""

from __future__ import annotations

import fnmatch
import hmac
import os
import secrets
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import web

TOKEN_FILE = "serve-token"
PUBLIC_PATHS = frozenset({"/health"})
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")
DEFAULT_ORIGINS = ["http://localhost:*", "http://127.0.0.1:*"]


def token_path() -> Path:
    """Where this install's server token is kept."""
    from mullet.core.paths import config_file

    return config_file(TOKEN_FILE)


def get_or_create_token(path: Path | None = None) -> str:
    """This install's server token, made and stored (mode 0600) on first use."""
    path = path or token_path()
    try:
        existing = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        existing = ""
    if existing:
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(token + "\n")
    return token


def _hostname(host_header: str) -> str:
    """The name in a ``Host`` header, without port or brackets."""
    parts = urlsplit("//" + host_header)
    return (parts.hostname or "").lower()


def _presented_token(request: web.Request) -> str:
    header = request.headers.get("Authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    found = request.headers.get("X-Mullet-Token", "").strip()
    if found:
        return found
    return request.query.get("token", "") if request.path == "/ws" else ""


def _refuse(status: int, message: str) -> web.Response:
    return web.json_response({"error": {"message": message, "type": "guard_error"}}, status=status)


@web.middleware
async def guard_middleware(request: web.Request, handler):
    """Refuse a foreign Host or Origin, and a request without the token when one is required."""
    app = request.app
    allowed_hosts = set(LOCAL_HOSTS) | {h.lower() for h in app.get("allowed_hosts", ())}
    if app.get("host") not in (None, "", "0.0.0.0", "::"):
        allowed_hosts.add(str(app["host"]).lower())
    if _hostname(request.headers.get("Host", "")) not in allowed_hosts:
        return _refuse(403, "Mullet: this server doesn't answer to that host name")
    origin = request.headers.get("Origin", "")
    if origin:
        patterns = app.get("cors_origins") or DEFAULT_ORIGINS
        if not any(fnmatch.fnmatch(origin, p) for p in patterns):
            return _refuse(403, "Mullet: requests from that web page aren't allowed (server.allowed_origins)")
    if app.get("require_token") and request.path not in PUBLIC_PATHS and request.method != "OPTIONS":
        from mullet.config.mode import MulletMode

        if MulletMode.is_solo():
            expected = app.get("token") or ""
            given = _presented_token(request)
            if not expected or not hmac.compare_digest(given.encode(), expected.encode()):
                return web.json_response(
                    {"error": {"message": "Mullet: this server needs its token. It is in the file "
                                          f"{token_path()}; send it as 'Authorization: Bearer <token>'.",
                               "type": "auth_error"}},
                    status=401,
                )
    return await handler(request)
