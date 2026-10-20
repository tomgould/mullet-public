"""Tests for enterprise features — auth, RBAC, database, mode toggle."""

import os
import time

from mullet.config.mode import MulletMode
from mullet.server.auth import generate_pat, verify_pat, get_solo_user
from mullet.server.db import init_db, bootstrap_solo_mode
from mullet.server.rbac import check_project_access, get_user_projects


# --- Mode toggle ---

def test_default_mode_is_solo():
    # Default (no env var) should be solo
    old = os.environ.pop("MULLET_MODE", None)
    try:
        assert MulletMode.is_solo() is True
        assert MulletMode.is_enterprise() is False
    finally:
        if old:
            os.environ["MULLET_MODE"] = old


def test_enterprise_mode():
    os.environ["MULLET_MODE"] = "enterprise"
    try:
        assert MulletMode.is_enterprise() is True
        assert MulletMode.is_solo() is False
    finally:
        os.environ["MULLET_MODE"] = "solo"


# --- PAT ---

def test_generate_pat():
    raw, hashed = generate_pat()
    assert raw.startswith("mllt_")
    assert len(raw) > 20
    assert len(hashed) == 64  # SHA-256 hex


def test_verify_pat(tmp_path):
    db = init_db(str(tmp_path / "test.db"))

    # Create user
    db.execute("INSERT INTO users (id, email, role, created_at) VALUES (?, ?, ?, ?)",
               ("user1", "test@test.com", "dev", time.time()))

    # Create token
    raw, hashed = generate_pat()
    db.execute("INSERT INTO access_tokens (id, user_id, token_hash, name, created_at) "
               "VALUES (?, ?, ?, ?, ?)",
               ("tok1", "user1", hashed, "test token", time.time()))
    db.commit()

    # Verify
    user = verify_pat(raw, db)
    assert user is not None
    assert user["email"] == "test@test.com"


def test_verify_invalid_pat(tmp_path):
    db = init_db(str(tmp_path / "test.db"))
    user = verify_pat("mllt_invalid_token", db)
    assert user is None


def test_verify_expired_pat(tmp_path):
    db = init_db(str(tmp_path / "test.db"))

    db.execute("INSERT INTO users (id, email, role, created_at) VALUES (?, ?, ?, ?)",
               ("user1", "test@test.com", "dev", time.time()))

    raw, hashed = generate_pat()
    db.execute("INSERT INTO access_tokens (id, user_id, token_hash, name, created_at, expires_at) "
               "VALUES (?, ?, ?, ?, ?, ?)",
               ("tok1", "user1", hashed, "expired", time.time(), time.time() - 3600))
    db.commit()

    user = verify_pat(raw, db)
    assert user is None  # Expired


def test_solo_user():
    user = get_solo_user()
    assert user["role"] == "admin"
    assert user["id"] == "solo-admin"


# --- Database ---

def test_init_db(tmp_path):
    db = init_db(str(tmp_path / "test.db"))
    # Should create all tables
    tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = {t[0] for t in tables}
    assert "users" in table_names
    assert "access_tokens" in table_names
    assert "projects" in table_names
    assert "user_projects" in table_names
    assert "usage_log" in table_names
    assert "spend_limits" in table_names


def test_bootstrap_solo(tmp_path):
    os.environ["MULLET_MODE"] = "solo"
    try:
        db = init_db(str(tmp_path / "test.db"))
        bootstrap_solo_mode(db)

        user = db.execute("SELECT * FROM users WHERE id = ?", ("solo-admin",)).fetchone()
        assert user is not None
    finally:
        os.environ["MULLET_MODE"] = "solo"


# --- RBAC ---

def test_admin_bypasses_rbac():
    admin = {"id": "admin1", "role": "admin"}
    assert check_project_access(admin, "any-project", "owner") is True


def test_rbac_with_access(tmp_path):
    db = init_db(str(tmp_path / "test.db"))

    db.execute("INSERT INTO users (id, email, role, created_at) VALUES (?, ?, ?, ?)",
               ("user1", "dev@co.com", "dev", time.time()))
    db.execute("INSERT INTO projects (id, name, pool_name, created_at) VALUES (?, ?, ?, ?)",
               ("proj1", "backend", "backend", time.time()))
    db.execute("INSERT INTO user_projects (user_id, project_id, access_level, granted_at) "
               "VALUES (?, ?, ?, ?)",
               ("user1", "proj1", "write", time.time()))
    db.commit()

    user = {"id": "user1", "role": "dev"}
    assert check_project_access(user, "proj1", "read", db) is True
    assert check_project_access(user, "proj1", "write", db) is True
    assert check_project_access(user, "proj1", "owner", db) is False


def test_rbac_no_access(tmp_path):
    db = init_db(str(tmp_path / "test.db"))
    user = {"id": "user1", "role": "dev"}
    assert check_project_access(user, "proj1", "read", db) is False


def test_get_user_projects(tmp_path):
    db = init_db(str(tmp_path / "test.db"))

    db.execute("INSERT INTO users (id, email, role, created_at) VALUES (?, ?, ?, ?)",
               ("user1", "dev@co.com", "dev", time.time()))
    for i in range(3):
        db.execute("INSERT INTO projects (id, name, pool_name, description, created_at) "
                   "VALUES (?, ?, ?, ?, ?)",
                   (f"proj{i}", f"project-{i}", f"client-{i}", f"Desc {i}", time.time()))
        db.execute("INSERT INTO user_projects (user_id, project_id, access_level, granted_at) "
                   "VALUES (?, ?, ?, ?)",
                   ("user1", f"proj{i}", "write" if i < 2 else "read", time.time()))
    db.commit()

    projects = get_user_projects("user1", db)
    assert len(projects) == 3
    assert projects[0]["access"] == "write"
