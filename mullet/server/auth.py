"""Authentication — PAT generation, verification, middleware.

Solo mode: auto-authenticate as admin, no auth required.
Enterprise mode: require Bearer token (PAT) on every request.
"""

import hashlib
import logging
import secrets
import sqlite3
import time
from typing import Dict, List

from mullet.config.mode import MulletMode

_audit_log = logging.getLogger("mullet.security.audit")

# Rate limiting state
_failed_attempts: Dict[str, List[float]] = {}


def generate_pat() -> tuple[str, str]:
    """Generate a Personal Access Token.

    Returns (raw_token, token_hash). Raw shown to user once, hash stored in DB.
    """
    raw = f"mllt_{secrets.token_urlsafe(24)}"
    hashed = hashlib.sha256(raw.encode()).hexdigest()
    return raw, hashed


def verify_pat(raw_token: str, db: sqlite3.Connection) -> dict | None:
    """Verify a PAT and return the user. Returns None if invalid."""
    hashed = hashlib.sha256(raw_token.encode()).hexdigest()
    row = db.execute(
        """SELECT u.id, u.email, u.role FROM access_tokens t
           JOIN users u ON t.user_id = u.id
           WHERE t.token_hash = ? AND (t.expires_at IS NULL OR t.expires_at > ?)""",
        (hashed, time.time()),
    ).fetchone()
    if row:
        db.execute("UPDATE access_tokens SET last_used_at = ? WHERE token_hash = ?",
                    (time.time(), hashed))
        db.commit()
        return {"id": row[0], "email": row[1], "role": row[2]}
    return None


def get_solo_user() -> dict:
    """Return the auto-created solo admin user."""
    return {"id": "solo-admin", "email": "admin@localhost", "role": "admin"}


def _record_failure(identifier: str) -> None:
    """Record a failed authentication attempt."""
    now = time.time()
    if identifier not in _failed_attempts:
        _failed_attempts[identifier] = []
    _failed_attempts[identifier].append(now)
    # Clean old entries (> 60s)
    _failed_attempts[identifier] = [
        t for t in _failed_attempts[identifier]
        if now - t < 60
    ]


def _rate_limit_check(identifier: str) -> float:
    """Check if identifier should be rate limited. Returns seconds to wait."""
    now = time.time()
    if identifier not in _failed_attempts:
        return 0.0
    
    # Clean old entries
    attempts = [
        t for t in _failed_attempts[identifier]
        if now - t < 60
    ]
    _failed_attempts[identifier] = attempts
    
    if len(attempts) >= 20:
        return 60.0
    elif len(attempts) >= 10:
        return 30.0
    elif len(attempts) >= 5:
        return 5.0
    return 0.0


class AuthMiddleware:
    """Request authentication middleware for the API server."""

    def __init__(self, db: sqlite3.Connection | None = None):
        """Bind the middleware to *db*; left None for solo mode where auth is bypassed."""
        self.db = db

    async def __call__(self, request, handler):
        if MulletMode.is_solo():
            request["user"] = get_solo_user()
            return await handler(request)

        # Get client identifier (IP address)
        identifier = request.remote or "unknown"
        
        # Check rate limiting
        wait_time = _rate_limit_check(identifier)
        if wait_time > 0:
            _audit_log.warning("auth.rate_limited ip=%s wait_seconds=%d", identifier, int(wait_time))
            from aiohttp.web import json_response
            return json_response(
                {"error": "Rate limit exceeded"},
                status=429,
                headers={"Retry-After": str(int(wait_time))}
            )

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            _record_failure(identifier)
            _audit_log.warning("auth.missing_token ip=%s path=%s", identifier, request.path)
            from aiohttp.web import json_response
            return json_response({"error": "Authentication required"}, status=401)

        token = auth_header[7:]
        if self.db:
            user = verify_pat(token, self.db)
            if not user:
                _record_failure(identifier)
                _audit_log.warning("auth.invalid_token ip=%s path=%s", identifier, request.path)
                from aiohttp.web import json_response
                return json_response({"error": "Invalid token"}, status=401)
            request["user"] = user
        else:
            # No database available in enterprise mode — deny access
            _record_failure(identifier)
            _audit_log.error("auth.db_unavailable ip=%s", identifier)
            from aiohttp.web import json_response
            return json_response({"error": "Authentication service unavailable"}, status=503)

        return await handler(request)
