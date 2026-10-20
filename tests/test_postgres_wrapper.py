"""Tests for PostgreSQL wrapper and database initialization.

Tests the PostgresWrapper class with mocked psycopg2, and verifies
the init_db function dispatches correctly based on DATABASE_URL.
"""

import sqlite3
import pytest
from unittest.mock import MagicMock, patch

from mullet.server.db import (
    PostgresWrapper,
    init_db,
    get_db_path,
    SQLITE_SCHEMA,
    POSTGRES_SCHEMA,
)


# --- PostgresWrapper ---

def test_wrapper_execute():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    w = PostgresWrapper(mock_conn)
    w.execute("SELECT * FROM users WHERE id = ?", ("user1",))

    # Should convert ? to %s
    mock_cursor.execute.assert_called_once_with(
        "SELECT * FROM users WHERE id = %s", ("user1",)
    )


def test_wrapper_execute_no_params():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    w = PostgresWrapper(mock_conn)
    w.execute("SELECT count(*) FROM users")

    mock_cursor.execute.assert_called_once_with("SELECT count(*) FROM users")


def test_wrapper_execute_multiple_params():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    w = PostgresWrapper(mock_conn)
    w.execute(
        "INSERT INTO users (id, email, role) VALUES (?, ?, ?)",
        ("u1", "a@b.com", "admin"),
    )

    mock_cursor.execute.assert_called_once_with(
        "INSERT INTO users (id, email, role) VALUES (%s, %s, %s)",
        ("u1", "a@b.com", "admin"),
    )


def test_wrapper_executescript():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    w = PostgresWrapper(mock_conn)
    w.executescript("CREATE TABLE t1 (id TEXT); CREATE TABLE t2 (id TEXT)")

    assert mock_cursor.execute.call_count == 2
    mock_conn.commit.assert_called_once()


def test_wrapper_commit():
    mock_conn = MagicMock()
    w = PostgresWrapper(mock_conn)
    w.commit()
    mock_conn.commit.assert_called_once()


def test_wrapper_close():
    mock_conn = MagicMock()
    w = PostgresWrapper(mock_conn)
    w.close()
    mock_conn.close.assert_called_once()


def test_wrapper_get_returns_default():
    mock_conn = MagicMock()
    w = PostgresWrapper(mock_conn)
    assert w.get("key") is None
    assert w.get("key", "default") == "default"


# --- init_db dispatch ---

def test_init_db_sqlite(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = init_db(db_path)
    assert isinstance(conn, sqlite3.Connection)
    # Verify schema was applied
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert "users" in tables
    assert "access_tokens" in tables
    assert "projects" in tables
    conn.close()


@patch.dict("os.environ", {"DATABASE_URL": ""})
def test_get_db_path_default():
    path = get_db_path()
    assert path.endswith("mullet.db")


@patch.dict("os.environ", {"DATABASE_URL": "postgresql://localhost/mullet"})
def test_get_db_path_postgres():
    path = get_db_path()
    assert path == "postgresql://localhost/mullet"


@patch("mullet.server.db.psycopg2", create=True)
def test_init_db_postgres_dispatch(mock_psycopg2):
    """init_db with postgres URL should create PostgresWrapper."""
    mock_conn = MagicMock()
    mock_psycopg2.connect.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
        result = init_db("postgresql://localhost/mullet")

    assert isinstance(result, PostgresWrapper)
    mock_psycopg2.connect.assert_called_once_with("postgresql://localhost/mullet")


def test_init_db_postgres_no_psycopg2():
    """init_db with postgres URL should raise ImportError if psycopg2 missing."""
    with patch.dict("sys.modules", {"psycopg2": None}):
        with pytest.raises(ImportError, match="psycopg2"):
            init_db("postgresql://localhost/mullet")


# --- Schema compatibility ---

def test_sqlite_schema_has_all_tables():
    assert "users" in SQLITE_SCHEMA
    assert "access_tokens" in SQLITE_SCHEMA
    assert "projects" in SQLITE_SCHEMA
    assert "user_projects" in SQLITE_SCHEMA
    assert "usage_log" in SQLITE_SCHEMA
    assert "spend_limits" in SQLITE_SCHEMA


def test_postgres_schema_has_all_tables():
    assert "users" in POSTGRES_SCHEMA
    assert "access_tokens" in POSTGRES_SCHEMA
    assert "projects" in POSTGRES_SCHEMA
    assert "user_projects" in POSTGRES_SCHEMA
    assert "usage_log" in POSTGRES_SCHEMA
    assert "spend_limits" in POSTGRES_SCHEMA


def test_postgres_schema_uses_serial():
    assert "SERIAL" in POSTGRES_SCHEMA
    assert "AUTOINCREMENT" not in POSTGRES_SCHEMA


def test_postgres_schema_uses_double_precision():
    assert "DOUBLE PRECISION" in POSTGRES_SCHEMA


def test_sqlite_schema_uses_real():
    assert "REAL" in SQLITE_SCHEMA
    assert "DOUBLE PRECISION" not in SQLITE_SCHEMA
