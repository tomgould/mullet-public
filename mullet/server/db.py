"""Database schema, migrations, and solo bootstrapping.

SQLite for solo/dev, PostgreSQL for production via DATABASE_URL.
Both use the same schema and a thin wrapper for interface compatibility.
"""

import logging
import os
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Schema uses SQL subset compatible with both SQLite and PostgreSQL.
# Key differences handled: AUTOINCREMENT → SERIAL, REAL → DOUBLE PRECISION.
SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    display_name TEXT,
    role TEXT NOT NULL DEFAULT 'dev',
    created_at REAL NOT NULL,
    last_login_at REAL
);

CREATE TABLE IF NOT EXISTS access_tokens (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    token_hash TEXT NOT NULL,
    name TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_used_at REAL,
    expires_at REAL
);

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    pool_name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_by TEXT REFERENCES users(id),
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS user_projects (
    user_id TEXT NOT NULL REFERENCES users(id),
    project_id TEXT NOT NULL REFERENCES projects(id),
    access_level TEXT NOT NULL DEFAULT 'read',
    granted_at REAL NOT NULL,
    granted_by TEXT REFERENCES users(id),
    PRIMARY KEY (user_id, project_id)
);

CREATE TABLE IF NOT EXISTS usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL REFERENCES users(id),
    project_id TEXT REFERENCES projects(id),
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    estimated_cost REAL NOT NULL,
    task_summary TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS spend_limits (
    user_id TEXT PRIMARY KEY REFERENCES users(id),
    daily_max_dollars REAL,
    monthly_max_dollars REAL,
    per_task_max_tokens INTEGER
);
"""

# Keep SCHEMA as alias for backwards compatibility
SCHEMA = SQLITE_SCHEMA

POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    display_name TEXT,
    role TEXT NOT NULL DEFAULT 'dev',
    created_at DOUBLE PRECISION NOT NULL,
    last_login_at DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS access_tokens (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    token_hash TEXT NOT NULL,
    name TEXT NOT NULL,
    created_at DOUBLE PRECISION NOT NULL,
    last_used_at DOUBLE PRECISION,
    expires_at DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    pool_name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_by TEXT REFERENCES users(id),
    created_at DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS user_projects (
    user_id TEXT NOT NULL REFERENCES users(id),
    project_id TEXT NOT NULL REFERENCES projects(id),
    access_level TEXT NOT NULL DEFAULT 'read',
    granted_at DOUBLE PRECISION NOT NULL,
    granted_by TEXT REFERENCES users(id),
    PRIMARY KEY (user_id, project_id)
);

CREATE TABLE IF NOT EXISTS usage_log (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    project_id TEXT REFERENCES projects(id),
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    estimated_cost DOUBLE PRECISION NOT NULL,
    task_summary TEXT,
    created_at DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS spend_limits (
    user_id TEXT PRIMARY KEY REFERENCES users(id),
    daily_max_dollars DOUBLE PRECISION,
    monthly_max_dollars DOUBLE PRECISION,
    per_task_max_tokens INTEGER
);
"""


class PostgresWrapper:
    """Thin wrapper around psycopg2 connection to match sqlite3 interface.

    SQLite returns conn.execute().fetchall() as lists of tuples.
    psycopg2 returns the same, so the wrapper is minimal — mainly
    handling executescript and commit semantics.
    """

    def __init__(self, conn):
        self._conn = conn
        self._conn.autocommit = False

    def execute(self, sql: str, params=None):
        """Execute SQL with optional parameters. Returns cursor."""
        cursor = self._conn.cursor()
        # Convert SQLite-style ? placeholders to PostgreSQL %s
        sql = sql.replace("?", "%s")
        if params:
            cursor.execute(sql, params)
        else:
            cursor.execute(sql)
        return cursor

    def executescript(self, sql: str):
        """Execute multiple SQL statements."""
        cursor = self._conn.cursor()
        # Split on semicolons and execute each statement
        for statement in sql.split(";"):
            statement = statement.strip()
            if statement:
                cursor.execute(statement)
        self._conn.commit()

    def commit(self):
        """Commit the current transaction."""
        self._conn.commit()

    def close(self):
        """Close the database connection."""
        self._conn.close()

    def get(self, key: str, default=None):
        """Dict-like access for aiohttp app compatibility."""
        return default


def get_db_path() -> str:
    """Get the database path from environment or default."""
    db_url = os.getenv("DATABASE_URL", "")
    if db_url.startswith("postgresql://") or db_url.startswith("postgres://"):
        return db_url  # Return URL for Postgres
    # Default: SQLite in ~/.mullet/
    db_dir = Path.home() / ".mullet"
    db_dir.mkdir(parents=True, exist_ok=True)
    return str(db_dir / "mullet.db")


def init_db(db_path: str | None = None):
    """Initialize the database with schema.

    Returns sqlite3.Connection for SQLite or PostgresWrapper for PostgreSQL.
    Detects backend from DATABASE_URL env var or path format.
    """
    path = db_path or get_db_path()

    if path.startswith("postgresql://") or path.startswith("postgres://"):
        return _init_postgres(path)

    return _init_sqlite(path)


def _init_sqlite(path: str) -> sqlite3.Connection:
    """Initialize SQLite database."""
    conn = sqlite3.connect(path)
    conn.executescript(SQLITE_SCHEMA)
    conn.commit()
    return conn


def _init_postgres(url: str) -> PostgresWrapper:
    """Initialize PostgreSQL database."""
    try:
        import psycopg2
    except ImportError as exc:
        raise ImportError(
            "PostgreSQL support requires psycopg2: pip install psycopg2-binary"
        ) from exc

    conn = psycopg2.connect(url)
    wrapper = PostgresWrapper(conn)
    wrapper.executescript(POSTGRES_SCHEMA)
    logger.info("PostgreSQL database initialized")
    return wrapper


def bootstrap_solo_mode(conn):
    """Auto-create admin user for solo mode."""
    from mullet.config.mode import MulletMode
    if not MulletMode.is_solo():
        return

    is_postgres = isinstance(conn, PostgresWrapper)
    if is_postgres:
        conn.execute(
            "INSERT INTO users (id, email, display_name, role, created_at) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
            ("solo-admin", "admin@localhost", "Admin", "admin", time.time()),
        )
    else:
        conn.execute(
            "INSERT OR IGNORE INTO users (id, email, display_name, role, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("solo-admin", "admin@localhost", "Admin", "admin", time.time()),
        )
    conn.commit()
    logger.info("Solo mode: admin user bootstrapped")
