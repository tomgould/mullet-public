"""Tests for auth and database modules."""

import time
from mullet.server.auth import generate_pat, verify_pat, get_solo_user
from mullet.server.db import init_db, bootstrap_solo_mode
from mullet.server.rbac import check_project_access


def test_generate_pat_format():
    raw, hashed = generate_pat()
    assert raw.startswith("mllt_")
    assert len(raw) > 20
    assert len(hashed) == 64


def test_solo_user():
    user = get_solo_user()
    assert user["role"] == "admin"
    assert user["id"] == "solo-admin"


def test_init_db(tmp_path):
    db = init_db(str(tmp_path / "test.db"))
    tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    names = {t[0] for t in tables}
    assert "users" in names
    assert "access_tokens" in names
    assert "projects" in names
    assert "user_projects" in names
    assert "usage_log" in names


def test_bootstrap_solo(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_MODE", "solo")
    db = init_db(str(tmp_path / "test.db"))
    bootstrap_solo_mode(db)
    user = db.execute("SELECT * FROM users WHERE id='solo-admin'").fetchone()
    assert user is not None


def test_verify_valid_pat(tmp_path):
    db = init_db(str(tmp_path / "test.db"))
    db.execute("INSERT INTO users (id, email, role, created_at) VALUES (?, ?, ?, ?)",
               ("u1", "test@test.com", "dev", time.time()))
    raw, hashed = generate_pat()
    db.execute("INSERT INTO access_tokens (id, user_id, token_hash, name, created_at) VALUES (?, ?, ?, ?, ?)",
               ("t1", "u1", hashed, "test", time.time()))
    db.commit()

    user = verify_pat(raw, db)
    assert user is not None
    assert user["email"] == "test@test.com"


def test_verify_invalid_pat(tmp_path):
    db = init_db(str(tmp_path / "test.db"))
    assert verify_pat("mllt_invalid", db) is None


def test_admin_bypasses_rbac():
    admin = {"id": "a", "role": "admin"}
    assert check_project_access(admin, "any", "owner") is True


def test_rbac_no_db():
    user = {"id": "u1", "role": "dev"}
    assert check_project_access(user, "proj", "read", db=None) is False
