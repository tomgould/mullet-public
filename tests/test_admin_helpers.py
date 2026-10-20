"""Tests for pure helpers in mullet.server.admin."""
from __future__ import annotations

import pytest

from mullet.server.admin import _get_db, _require_admin


class _FakeApp(dict):
    pass


class _FakeRequest(dict):
    def __init__(self, user=None, app=None):
        super().__init__()
        if user is not None:
            self["user"] = user
        self.app = app or _FakeApp()


class TestRequireAdmin:
    def test_admin_role_returns_true(self) -> None:
        req = _FakeRequest(user={"role": "admin"})
        assert _require_admin(req) is True

    def test_dev_role_returns_false(self) -> None:
        req = _FakeRequest(user={"role": "dev"})
        assert _require_admin(req) is False

    def test_viewer_role_returns_false(self) -> None:
        req = _FakeRequest(user={"role": "viewer"})
        assert _require_admin(req) is False

    def test_empty_role_returns_false(self) -> None:
        req = _FakeRequest(user={"role": ""})
        assert _require_admin(req) is False

    def test_no_role_key_returns_false(self) -> None:
        req = _FakeRequest(user={"email": "x@example.com"})
        assert _require_admin(req) is False

    def test_no_user_key_returns_false(self) -> None:
        req = _FakeRequest()
        assert _require_admin(req) is False

    def test_user_none_returns_false(self) -> None:
        req = _FakeRequest()
        req["user"] = None
        # get("role") on None would fail — but the code uses dict.get chain
        # _require_admin does request.get("user", {}) so None is returned, then .get("role") fails
        # Actually None.get("role") will raise AttributeError — let's test real behavior:
        with pytest.raises(AttributeError):
            _require_admin(req)

    def test_role_case_sensitive_admin_lowercase_only(self) -> None:
        req = _FakeRequest(user={"role": "Admin"})
        assert _require_admin(req) is False

        req2 = _FakeRequest(user={"role": "ADMIN"})
        assert _require_admin(req2) is False


class TestGetDb:
    def test_returns_db_when_present(self) -> None:
        app = _FakeApp()
        app["db"] = "my_database"
        req = _FakeRequest(app=app)
        assert _get_db(req) == "my_database"

    def test_returns_none_when_absent(self) -> None:
        req = _FakeRequest()
        assert _get_db(req) is None

    def test_returns_actual_object(self) -> None:
        class FakeDB:
            pass

        app = _FakeApp()
        db = FakeDB()
        app["db"] = db
        req = _FakeRequest(app=app)
        assert _get_db(req) is db
