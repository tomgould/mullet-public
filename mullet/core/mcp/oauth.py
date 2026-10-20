
"""OAuth token management for MCP HTTP servers."""

import base64
import hashlib
import json
import logging
import os
import secrets
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse, urlencode

import asyncio

import httpx

logger = logging.getLogger(__name__)

_TOKEN_DIR_NAME = "mcp-tokens"

def _sanitize_server_name(name: str) -> str:
    """Sanitize server name for safe use as a filename."""
    import re
    clean = re.sub(r"[^\w\-]", "-", name.strip().lower())
    clean = re.sub(r"-+", "-", clean).strip("-")
    return clean[:60] or "unnamed"

class MulletTokenStorage:
    """File-backed token storage for Mullet."""

    def __init__(self, server_name: str):
        """Initialize storage for the given server name."""
        self._server_name = _sanitize_server_name(server_name)

    def _base_dir(self) -> Path:
        home = Path.home() / ".mullet"
        d = home / _TOKEN_DIR_NAME
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _tokens_path(self) -> Path:
        return self._base_dir() / f"{self._server_name}.json"

    def _client_path(self) -> Path:
        return self._base_dir() / f"{self._server_name}.client.json"

    def get_tokens(self) -> dict | None:
        """Return stored tokens, or None if none exist."""
        return self._read_json(self._tokens_path())

    def set_tokens(self, tokens: dict) -> None:
        """Persist the token dictionary to disk."""
        self._write_json(self._tokens_path(), tokens)

    def get_client_info(self) -> dict | None:
        """Return stored client info, or None if none exists."""
        return self._read_json(self._client_path())

    def set_client_info(self, info: dict) -> None:
        """Persist the client info dictionary to disk."""
        self._write_json(self._client_path(), info)

    def clear(self) -> None:
        """Remove all stored data for this server."""
        for p in (self._tokens_path(), self._client_path()):
            try:
                p.unlink(missing_ok=True)
            except OSError as _exc:
                logger.warning("mcp.oauth: failed to unlink path %s: %s", p, _exc)

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as _exc:
            logger.warning("mcp.oauth: failed to read JSON file %s: %s", path, _exc)
            return None

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:
            pass


_oauth_port: int | None = None

class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Handles the OAuth redirect callback on localhost."""
    _result: dict = {"auth_code": None, "state": None}

    def do_GET(self):
        """Handle the OAuth redirect GET request."""
        qs = parse_qs(urlparse(self.path).query)
        OAuthCallbackHandler._result["auth_code"] = (qs.get("code") or [None])[0]
        OAuthCallbackHandler._result["state"] = (qs.get("state") or [None])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<html><body><h3>Authorization complete. You can close this tab.</h3></body></html>")

    def log_message(self, *_args: Any) -> None:
        """Suppress default HTTP request logging."""
        pass

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def _can_open_browser() -> bool:
    if os.environ.get("SSH_CLIENT") or os.environ.get("SSH_TTY"):
        return False
    if not os.environ.get("DISPLAY") and os.name != "nt" and "darwin" not in os.uname().sysname.lower():
        return False
    return True

async def _redirect_to_browser(auth_url: str) -> None:
    """Open the authorization URL in the user's browser."""
    try:
        if _can_open_browser():
            webbrowser.open(auth_url)
            print("  Opened browser for authorization...")
        else:
            print(f"\n  Open this URL to authorize:\n  {auth_url}\n")
    except Exception:
        print(f"\n  Open this URL to authorize:\n  {auth_url}\n")

async def _wait_for_callback() -> tuple[str, str | None]:
    """Start a local HTTP server on the pre-registered port and wait for the OAuth redirect."""
    global _oauth_port
    port = _oauth_port or _find_free_port()
    _oauth_port = port # Store the chosen port
    server = HTTPServer(("127.0.0.1", port), OAuthCallbackHandler)

    def _serve():
        server.timeout = 120
        server.handle_request()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    for _ in range(1200):  # 120 seconds
        await asyncio.sleep(0.1)
        if OAuthCallbackHandler._result["auth_code"] is not None:
            break

    server.server_close()
    code = OAuthCallbackHandler._result["auth_code"] or ""
    state = OAuthCallbackHandler._result["state"]
    if not code:
        print("  Browser callback timed out. Paste the authorization code manually:")
        code = input("  Code: ").strip()
    return code, state

async def run_oauth_flow(server_name: str, server_url: str, client_id: str, auth_url: str, token_url: str, scopes: list[str] = None) -> dict:
    """
    Generates PKCE code_verifier + code_challenge, opens browser to auth_url,
    starts local HTTP server to catch redirect, exchanges auth code for tokens,
    stores tokens, and returns the token dict.
    """
    if scopes is None:
        scopes = ["openid", "profile", "email", "offline_access"]

    storage = MulletTokenStorage(server_name)

    # PKCE
    code_verifier = secrets.token_urlsafe(96)
    code_challenge = base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest()).decode().rstrip("=")

    # Ensure a port is selected before building the redirect URI
    global _oauth_port
    if _oauth_port is None:
        _oauth_port = _find_free_port()

    redirect_uri = f"http://127.0.0.1:{_oauth_port}"

    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": " ".join(scopes),
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": secrets.token_urlsafe(32),
    }
    full_auth_url = f"{auth_url}?{urlencode(params)}"

    await _redirect_to_browser(full_auth_url)
    auth_code, state = await _wait_for_callback()

    if not auth_code:
        raise RuntimeError("OAuth: failed to get authorization code.")
    if state != params["state"]:
        raise RuntimeError("OAuth: state parameter mismatch (possible CSRF).")

    # Exchange code for tokens
    token_params = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "code": auth_code,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(token_url, data=token_params)
        response.raise_for_status()
        tokens = response.json()

    storage.set_tokens(tokens)
    return tokens

def build_oauth_auth(server_name: str, server_url: str) -> dict | None:
    """Convenience wrapper that loads existing tokens or returns None."""
    storage = MulletTokenStorage(server_name)
    return storage.get_tokens()

