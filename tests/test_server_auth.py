"""Tests for mullet/server/auth.py — PAT generation, verification, middleware."""

import hashlib
import sqlite3
import time
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

import mullet.server.auth as auth
from mullet.server.db import SQLITE_SCHEMA


@pytest.fixture(autouse=True)
def clear_failed_attempts(monkeypatch):
    """Reset module-level rate-limiting state between tests."""
    monkeypatch.setattr(auth, "_failed_attempts", {})


@pytest.fixture
def db():
    """In-memory SQLite DB with full schema."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(SQLITE_SCHEMA)
    conn.commit()
    return conn


@pytest.fixture
def test_user(db):
    """Insert a test user and return its id."""
    user_id = "test-user-1"
    db.execute(
        "INSERT INTO users (id, email, display_name, role, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, "test@example.com", "Test User", "dev", time.time()),
    )
    db.commit()
    return user_id


@pytest.fixture
def valid_token(db, test_user):
    """Insert a valid access token and return (raw_token, token_hash)."""
    raw, hashed = auth.generate_pat()
    db.execute(
        "INSERT INTO access_tokens (id, user_id, token_hash, name, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("tok-1", test_user, hashed, "test-token", time.time(), None),
    )
    db.commit()
    return raw, hashed


@pytest.fixture
def expired_token(db, test_user):
    """Insert an expired access token and return (raw_token, token_hash)."""
    raw, hashed = auth.generate_pat()
    db.execute(
        "INSERT INTO access_tokens (id, user_id, token_hash, name, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("tok-expired", test_user, hashed, "expired-token", time.time() - 120, time.time() - 60),
    )
    db.commit()
    return raw, hashed


# ---------------------------------------------------------------------------
# generate_pat
# ---------------------------------------------------------------------------

class TestGeneratePat:
    def test_returns_tuple(self):
        result = auth.generate_pat()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_raw_starts_with_prefix(self):
        raw, _ = auth.generate_pat()
        assert raw.startswith("mllt_")

    def test_hash_is_sha256_hex_of_raw(self):
        raw, hashed = auth.generate_pat()
        expected = hashlib.sha256(raw.encode()).hexdigest()
        assert hashed == expected

    def test_two_calls_produce_different_tokens(self):
        raw1, _ = auth.generate_pat()
        raw2, _ = auth.generate_pat()
        assert raw1 != raw2


# ---------------------------------------------------------------------------
# verify_pat
# ---------------------------------------------------------------------------

class TestVerifyPat:
    def test_valid_token_returns_user_dict(self, db, valid_token):
        raw, _ = valid_token
        user = auth.verify_pat(raw, db)
        assert user is not None
        assert user["id"] == "test-user-1"
        assert user["email"] == "test@example.com"
        assert user["role"] == "dev"

    def test_expired_token_returns_none(self, db, expired_token):
        raw, _ = expired_token
        assert auth.verify_pat(raw, db) is None

    def test_wrong_token_returns_none(self, db):
        assert auth.verify_pat("mllt_totallywrong", db) is None

    def test_updates_last_used_at_on_success(self, db, valid_token):
        raw, hashed = valid_token
        before = time.time()
        auth.verify_pat(raw, db)
        after = time.time()

        row = db.execute("SELECT last_used_at FROM access_tokens WHERE token_hash = ?", (hashed,)).fetchone()
        assert row is not None
        assert before <= row[0] <= after


# ---------------------------------------------------------------------------
# get_solo_user
# ---------------------------------------------------------------------------

class TestGetSoloUser:
    def test_returns_expected_dict(self):
        user = auth.get_solo_user()
        assert user == {"id": "solo-admin", "email": "admin@localhost", "role": "admin"}


# ---------------------------------------------------------------------------
# _record_failure
# ---------------------------------------------------------------------------

class TestRecordFailure:
    def test_adds_entry(self):
        assert auth._failed_attempts == {}
        auth._record_failure("ip-1")
        assert "ip-1" in auth._failed_attempts
        assert len(auth._failed_attempts["ip-1"]) == 1

    def test_cleans_entries_older_than_60s(self, monkeypatch):
        now = 1000.0
        monkeypatch.setattr(auth, "_failed_attempts", {"ip-1": [now - 61, now - 30, now - 10]})

        with patch("mullet.server.auth.time.time", return_value=now):
            auth._record_failure("ip-1")

        # Old entry (now - 61) should be dropped; the two recent ones plus the new one remain
        assert len(auth._failed_attempts["ip-1"]) == 3
        assert (now - 61) not in auth._failed_attempts["ip-1"]


# ---------------------------------------------------------------------------
# _rate_limit_check
# ---------------------------------------------------------------------------

class TestRateLimitCheck:
    def test_unknown_identifier_returns_zero(self):
        assert auth._rate_limit_check("unknown-ip") == 0.0

    def test_five_attempts_returns_five_seconds(self, monkeypatch):
        now = 500.0
        monkeypatch.setattr(auth, "_failed_attempts", {"ip-1": [now - i for i in range(5)]})
        with patch("mullet.server.auth.time.time", return_value=now):
            assert auth._rate_limit_check("ip-1") == 5.0

    def test_ten_attempts_returns_thirty_seconds(self, monkeypatch):
        now = 500.0
        monkeypatch.setattr(auth, "_failed_attempts", {"ip-1": [now - i for i in range(10)]})
        with patch("mullet.server.auth.time.time", return_value=now):
            assert auth._rate_limit_check("ip-1") == 30.0

    def test_twenty_attempts_returns_sixty_seconds(self, monkeypatch):
        now = 500.0
        monkeypatch.setattr(auth, "_failed_attempts", {"ip-1": [now - i for i in range(20)]})
        with patch("mullet.server.auth.time.time", return_value=now):
            assert auth._rate_limit_check("ip-1") == 60.0

    def test_cleans_old_entries_before_checking(self, monkeypatch):
        now = 1000.0
        # 5 attempts, but 4 are older than 60s
        monkeypatch.setattr(
            auth,
            "_failed_attempts",
            {"ip-1": [now - 65, now - 70, now - 80, now - 90, now - 10]},
        )
        with patch("mullet.server.auth.time.time", return_value=now):
            assert auth._rate_limit_check("ip-1") == 0.0


# ---------------------------------------------------------------------------
# AuthMiddleware — solo mode
# ---------------------------------------------------------------------------

class TestAuthMiddlewareSolo:
    @pytest.fixture(autouse=True)
    def solo_mode(self, monkeypatch):
        """Force solo mode for every test in this class."""
        monkeypatch.setenv("MULLET_MODE", "solo")

    @pytest.mark.asyncio
    async def test_sets_request_user_without_token(self):
        request = MagicMock()
        request.headers = {}
        handler = AsyncMock(return_value="ok")

        middleware = auth.AuthMiddleware(db=None)
        result = await middleware(request, handler)

        assert request.__setitem__.call_args_list == [(("user", auth.get_solo_user()),)]
        handler.assert_awaited_once_with(request)
        assert result == "ok"


# ---------------------------------------------------------------------------
# AuthMiddleware — enterprise mode
# ---------------------------------------------------------------------------

class TestAuthMiddlewareEnterprise:
    @pytest.fixture(autouse=True)
    def enterprise_mode(self, monkeypatch):
        """Force enterprise mode for every test in this class."""
        monkeypatch.setenv("MULLET_MODE", "enterprise")

    def _make_request(self, remote="1.2.3.4", auth_header=""):
        request = MagicMock()
        request.remote = remote
        request.headers = {"Authorization": auth_header} if auth_header else {}
        return request

    @pytest.mark.asyncio
    async def test_rate_limit_returns_429(self):
        # Pre-seed 20 failed attempts for the IP
        now = time.time()
        auth._failed_attempts["1.2.3.4"] = [now - i for i in range(20)]

        request = self._make_request()
        handler = AsyncMock()

        with patch("aiohttp.web.json_response") as mock_json_resp:
            mock_json_resp.return_value = {"status": 429}
            middleware = auth.AuthMiddleware(db=None)
            result = await middleware(request, handler)

        handler.assert_not_awaited()
        mock_json_resp.assert_called_once()
        call_args = mock_json_resp.call_args
        assert call_args.kwargs["status"] == 429
        assert call_args.kwargs["headers"]["Retry-After"] == "60"
        assert result == {"status": 429}

    @pytest.mark.asyncio
    async def test_missing_bearer_returns_401(self):
        request = self._make_request(auth_header="Basic foo")
        handler = AsyncMock()

        with patch("aiohttp.web.json_response") as mock_json_resp:
            mock_json_resp.return_value = {"status": 401}
            middleware = auth.AuthMiddleware(db=None)
            result = await middleware(request, handler)

        handler.assert_not_awaited()
        mock_json_resp.assert_called_once()
        assert mock_json_resp.call_args.kwargs["status"] == 401
        assert result == {"status": 401}

    @pytest.mark.asyncio
    async def test_invalid_token_returns_401(self, db):
        request = self._make_request(auth_header="Bearer mllt_invalidtoken")
        handler = AsyncMock()

        with patch("aiohttp.web.json_response") as mock_json_resp:
            mock_json_resp.return_value = {"status": 401}
            middleware = auth.AuthMiddleware(db=db)
            result = await middleware(request, handler)

        handler.assert_not_awaited()
        mock_json_resp.assert_called_once()
        assert mock_json_resp.call_args.kwargs["status"] == 401
        assert result == {"status": 401}

    @pytest.mark.asyncio
    async def test_no_db_returns_503(self):
        request = self._make_request(auth_header="Bearer mllt_something")
        handler = AsyncMock()

        with patch("aiohttp.web.json_response") as mock_json_resp:
            mock_json_resp.return_value = {"status": 503}
            middleware = auth.AuthMiddleware(db=None)
            result = await middleware(request, handler)

        handler.assert_not_awaited()
        mock_json_resp.assert_called_once()
        assert mock_json_resp.call_args.kwargs["status"] == 503
        assert result == {"status": 503}

    @pytest.mark.asyncio
    async def test_valid_token_sets_user_and_calls_handler(self, db, valid_token):
        raw, _ = valid_token
        request = self._make_request(auth_header=f"Bearer {raw}")
        handler = AsyncMock(return_value="ok")

        middleware = auth.AuthMiddleware(db=db)
        result = await middleware(request, handler)

        handler.assert_awaited_once_with(request)
        assert request.__setitem__.call_args_list == [(("user", {"id": "test-user-1", "email": "test@example.com", "role": "dev"}),)]
        assert result == "ok"
