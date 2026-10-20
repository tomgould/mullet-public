"""Tests for mullet.harness.workflows.store.WorkflowStore."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mullet.harness.workflows.store import WorkflowStore

pytestmark = pytest.mark.harness


def test_init_creates_db_file_and_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    store = WorkflowStore(db_path=db_path)

    assert db_path.exists(), "WorkflowStore did not create the db file"

    # Sanity: connect with a fresh connection and confirm the tables exist.
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    table_names = {r[0] for r in rows}
    assert {"workflow_runs", "workflow_events", "workflow_messages"}.issubset(table_names)
    store.close()


def test_idempotent_init(tmp_path: Path) -> None:
    db_path = tmp_path / "double.db"
    store_a = WorkflowStore(db_path=db_path)
    store_b = WorkflowStore(db_path=db_path)
    # No exception, file exists, both stores share schema.
    assert db_path.exists()
    store_a.close()
    store_b.close()


def test_db_schema_has_correct_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "schema.db"
    store = WorkflowStore(db_path=db_path)
    conn = store._conn()
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    names = {row["name"] for row in rows}
    assert "workflow_runs" in names
    assert "workflow_events" in names
    assert "workflow_messages" in names

    # Index sanity-check (mirrors DDL).
    idx_rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    ).fetchall()
    idx_names = {row["name"] for row in idx_rows}
    expected_indexes = {
        "idx_runs_type",
        "idx_runs_status",
        "idx_runs_created",
        "idx_events_run",
        "idx_messages_run",
    }
    assert expected_indexes.issubset(idx_names)
    store.close()


def test_default_db_path_is_under_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Override HOME so the default path lands in tmp_path.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))  # type: ignore[arg-type]
    store = WorkflowStore()
    expected = tmp_path / ".config/mullet/workflows.db"
    assert store.db_path == expected
    assert expected.exists()
    store.close()


def test_foreign_keys_enabled(tmp_path: Path) -> None:
    store = WorkflowStore(db_path=tmp_path / "fk.db")
    conn = store._conn()
    fk = conn.execute("PRAGMA foreign_keys").fetchone()
    assert int(fk[0]) == 1
    store.close()


def test_tx_rollback_on_exception(tmp_path: Path) -> None:
    """Failed transaction inside ``_tx`` should not leave partial writes."""
    store = WorkflowStore(db_path=tmp_path / "rollback.db")
    conn = store._conn()
    conn.execute(
        "INSERT INTO workflow_runs (run_id, workflow_type, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("r1", "t", "pending", 1.0, 1.0),
    )

    with pytest.raises(RuntimeError):
        with store._tx() as tx:
            tx.execute("DELETE FROM workflow_runs WHERE run_id = 'r1'")
            raise RuntimeError("boom")

    rows = conn.execute("SELECT run_id FROM workflow_runs").fetchall()
    assert [r["run_id"] for r in rows] == ["r1"]
    store.close()
