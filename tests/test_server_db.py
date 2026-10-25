"""Tests for mullet/server/db.py."""

import sqlite3
import sys
from unittest.mock import MagicMock

import pytest

from mullet.server.db import SCHEMA, SQLITE_SCHEMA, _init_sqlite, bootstrap_solo_mode, init_db


def test_sqlite_schema_is_nonempty():
    assert len(SQLITE_SCHEMA.strip()) > 0


def test_schema_alias():
    assert SCHEMA is SQLITE_SCHEMA


class TestInitSQLite:
    def test_creates_tables(self):
        conn = _init_sqlite(":memory:")
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence' ORDER BY name"
        ).fetchall()
        assert tables == [
            ("access_tokens",),
            ("projects",),
            ("spend_limits",),
            ("usage_log",),
            ("user_projects",),
            ("users",),
        ]
        conn.close()

    def test_init_db_with_memory_path(self):
        conn = init_db(":memory:")
        assert isinstance(conn, sqlite3.Connection)
        row = conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='users'"
        ).fetchone()
        assert row[0] == 1
        conn.close()


class TestBootstrapSoloMode:
    def test_creates_admin_user(self, monkeypatch):
        monkeypatch.setenv("MULLET_MODE", "solo")
        conn = _init_sqlite(":memory:")
        bootstrap_solo_mode(conn)
        row = conn.execute(
            "SELECT id, email, display_name, role FROM users WHERE id = 'solo-admin'"
        ).fetchone()
        assert row == ("solo-admin", "admin@localhost", "Admin", "admin")
        conn.close()

    def test_skips_when_not_solo(self, monkeypatch):
        monkeypatch.setenv("MULLET_MODE", "enterprise")
        conn = _init_sqlite(":memory:")
        bootstrap_solo_mode(conn)
        row = conn.execute("SELECT count(*) FROM users").fetchone()
        assert row[0] == 0
        conn.close()


# ---------------------------------------------------------------------------
# get_db_path() — PostgreSQL URL pass-through
# ---------------------------------------------------------------------------

class TestGetDbPathPostgres:
    def test_returns_postgresql_url(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host/db")
        # Force reimport so env change is picked up
        import importlib
        import mullet.server.db as db_mod
        importlib.reload(db_mod)
        from mullet.server.db import get_db_path
        assert get_db_path() == "postgresql://user:pass@host/db"

    def test_returns_postgres_variant(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgres://user:pass@host/db")
        import importlib
        import mullet.server.db as db_mod
        importlib.reload(db_mod)
        from mullet.server.db import get_db_path
        assert get_db_path() == "postgres://user:pass@host/db"

    def test_returns_sqlite_path_when_no_url(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        import importlib
        import mullet.server.db as db_mod
        importlib.reload(db_mod)
        from mullet.server.db import get_db_path
        path = get_db_path()
        assert path.endswith("mullet.db")


# ---------------------------------------------------------------------------
# PostgresWrapper
# ---------------------------------------------------------------------------

class TestPostgresWrapper:
    def _make_wrapper(self):
        from mullet.server.db import PostgresWrapper
        mock_conn = MagicMock()
        wrapper = PostgresWrapper(mock_conn)
        return wrapper, mock_conn

    def test_sets_autocommit_false(self):
        from mullet.server.db import PostgresWrapper
        mock_conn = MagicMock()
        PostgresWrapper(mock_conn)
        assert mock_conn.autocommit is False

    def test_execute_converts_placeholders(self):
        wrapper, mock_conn = self._make_wrapper()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        wrapper.execute("SELECT ? FROM t WHERE id=?", ("a", "b"))
        mock_cursor.execute.assert_called_once_with("SELECT %s FROM t WHERE id=%s", ("a", "b"))

    def test_execute_no_params(self):
        wrapper, mock_conn = self._make_wrapper()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        wrapper.execute("SELECT 1")
        mock_cursor.execute.assert_called_once_with("SELECT 1")

    def test_execute_returns_cursor(self):
        wrapper, mock_conn = self._make_wrapper()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        result = wrapper.execute("SELECT 1")
        assert result is mock_cursor

    def test_executescript_runs_each_statement(self):
        wrapper, mock_conn = self._make_wrapper()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        wrapper.executescript("CREATE TABLE a (id INT); CREATE TABLE b (id INT);")
        assert mock_cursor.execute.call_count == 2
        mock_conn.commit.assert_called_once()

    def test_executescript_skips_blank_statements(self):
        wrapper, mock_conn = self._make_wrapper()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        wrapper.executescript("  \n  ;  \n  ")
        mock_cursor.execute.assert_not_called()
        # commit still called even for empty script
        mock_conn.commit.assert_called_once()

    def test_commit_delegates_to_conn(self):
        wrapper, mock_conn = self._make_wrapper()
        wrapper.commit()
        mock_conn.commit.assert_called_once()

    def test_close_delegates_to_conn(self):
        wrapper, mock_conn = self._make_wrapper()
        wrapper.close()
        mock_conn.close.assert_called_once()

    def test_get_returns_none_by_default(self):
        wrapper, _ = self._make_wrapper()
        assert wrapper.get("anything") is None

    def test_get_returns_supplied_default(self):
        wrapper, _ = self._make_wrapper()
        assert wrapper.get("anything", "fallback") == "fallback"


# ---------------------------------------------------------------------------
# _init_postgres()
# ---------------------------------------------------------------------------

class TestInitPostgres:
    def test_success_returns_postgres_wrapper(self, monkeypatch):
        mock_psycopg2 = MagicMock()
        mock_conn = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn
        monkeypatch.setitem(sys.modules, "psycopg2", mock_psycopg2)

        import importlib
        import mullet.server.db as db_mod
        importlib.reload(db_mod)

        result = db_mod._init_postgres("postgresql://user:pass@host/db")
        mock_psycopg2.connect.assert_called_once_with("postgresql://user:pass@host/db")
        assert isinstance(result, db_mod.PostgresWrapper)

    def test_import_error_raises_with_message(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "psycopg2", None)

        import importlib
        import mullet.server.db as db_mod
        importlib.reload(db_mod)

        with pytest.raises(ImportError, match="psycopg2"):
            db_mod._init_postgres("postgresql://user:pass@host/db")

    def test_init_db_routes_to_postgres(self, monkeypatch):
        mock_psycopg2 = MagicMock()
        mock_conn = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn
        monkeypatch.setitem(sys.modules, "psycopg2", mock_psycopg2)

        import importlib
        import mullet.server.db as db_mod
        importlib.reload(db_mod)

        result = db_mod.init_db("postgresql://user:pass@host/db")
        assert isinstance(result, db_mod.PostgresWrapper)

    def test_init_db_postgres_variant_url(self, monkeypatch):
        mock_psycopg2 = MagicMock()
        mock_conn = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn
        monkeypatch.setitem(sys.modules, "psycopg2", mock_psycopg2)

        import importlib
        import mullet.server.db as db_mod
        importlib.reload(db_mod)

        result = db_mod.init_db("postgres://user:pass@host/db")
        assert isinstance(result, db_mod.PostgresWrapper)


# ---------------------------------------------------------------------------
# bootstrap_solo_mode() — PostgresWrapper branch
# ---------------------------------------------------------------------------

class TestBootstrapSoloModePostgres:
    def test_postgres_branch_uses_percent_s_and_on_conflict(self, monkeypatch):
        monkeypatch.setenv("MULLET_MODE", "solo")
        from mullet.server.db import PostgresWrapper
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        wrapper = PostgresWrapper(mock_conn)

        bootstrap_solo_mode(wrapper)

        assert mock_cursor.execute.called
        sql = mock_cursor.execute.call_args[0][0]
        assert "%s" in sql
        assert "ON CONFLICT" in sql
        mock_conn.commit.assert_called()

    def test_postgres_branch_skips_when_not_solo(self, monkeypatch):
        monkeypatch.setenv("MULLET_MODE", "enterprise")
        from mullet.server.db import PostgresWrapper
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        wrapper = PostgresWrapper(mock_conn)

        bootstrap_solo_mode(wrapper)

        mock_cursor.execute.assert_not_called()
        mock_conn.commit.assert_not_called()
