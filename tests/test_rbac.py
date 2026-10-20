"""Tests for RBAC — role-based access control."""
import sqlite3
import pytest
from mullet.server.rbac import check_project_access, get_user_projects, ACCESS_LEVELS


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE projects (id TEXT, name TEXT, pool_name TEXT, description TEXT)")
    conn.execute("CREATE TABLE user_projects (user_id TEXT, project_id TEXT, access_level TEXT)")
    conn.execute("INSERT INTO projects VALUES ('p1', 'mullet-dev', 'mullet-dev', 'Test project')")
    conn.execute("INSERT INTO user_projects VALUES ('u1', 'p1', 'write')")
    conn.execute("INSERT INTO user_projects VALUES ('u2', 'p1', 'read')")
    conn.commit()
    return conn


def test_admin_bypasses():
    assert check_project_access({"role": "admin"}, "anything") is True


def test_no_db_denies():
    assert check_project_access({"id": "u1"}, "mullet-dev") is False


def test_write_access(db):
    assert check_project_access({"id": "u1"}, "mullet-dev", "write", db) is True


def test_read_access(db):
    assert check_project_access({"id": "u2"}, "mullet-dev", "read", db) is True


def test_insufficient_access(db):
    assert check_project_access({"id": "u2"}, "mullet-dev", "write", db) is False


def test_unknown_project(db):
    assert check_project_access({"id": "u1"}, "nonexistent", "read", db) is False


def test_get_user_projects(db):
    projects = get_user_projects("u1", db)
    assert len(projects) == 1
    assert projects[0]["name"] == "mullet-dev"
    assert projects[0]["access"] == "write"


def test_access_levels():
    assert ACCESS_LEVELS["read"] < ACCESS_LEVELS["write"]
    assert ACCESS_LEVELS["write"] < ACCESS_LEVELS["owner"]
