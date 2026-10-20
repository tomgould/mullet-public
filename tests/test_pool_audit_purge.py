"""Tests for _purge_pool_sidecars and _force_rmtree in pool_audit."""

import sqlite3

import pytest

from mullet.cli.pool_audit import _purge_pool_sidecars, _force_rmtree


@pytest.fixture()
def mullet_home(tmp_path):
    h = tmp_path / ".mullet"
    h.mkdir()
    return h


def _make_db(mullet_home, db_name, pool_name):
    """Create a tiny SQLite DB with a pool_name column to test deletion."""
    db = mullet_home / db_name
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE usage_log (id INTEGER PRIMARY KEY, pool_name TEXT, data TEXT)")
    conn.execute("INSERT INTO usage_log (pool_name, data) VALUES (?, ?)", (pool_name, "record1"))
    conn.execute("INSERT INTO usage_log (pool_name, data) VALUES (?, ?)", ("other_pool", "record2"))
    conn.commit()
    conn.close()
    return db


def test_purge_deletes_rows_by_pool_name(mullet_home):
    _make_db(mullet_home, "mullet.db", "MyPool")
    rows, warnings = _purge_pool_sidecars("MyPool", "mypool", mullet_home)
    assert rows >= 1
    assert warnings == []
    # Verify the remaining row is for other_pool
    conn = sqlite3.connect(str(mullet_home / "mullet.db"))
    remaining = conn.execute("SELECT pool_name FROM usage_log").fetchall()
    conn.close()
    assert all(r[0] == "other_pool" for r in remaining)


def test_purge_returns_zero_when_no_db(mullet_home):
    rows, warnings = _purge_pool_sidecars("NonExistent", "nonexistent", mullet_home)
    assert rows == 0
    assert warnings == []


def test_purge_deletes_sidecar_files(mullet_home):
    analytics = mullet_home / "analytics"
    analytics.mkdir()
    sidecar = analytics / "mypool.json"
    sidecar.write_text('{"data": 1}')
    rows, warnings = _purge_pool_sidecars("MyPool", "mypool", mullet_home)
    assert not sidecar.exists()
    assert warnings == []


def test_force_rmtree_removes_directory(tmp_path):
    target = tmp_path / "pool_to_remove"
    target.mkdir()
    (target / "file.txt").write_text("data")
    success, err = _force_rmtree(target)
    assert success is True
    assert err is None
    assert not target.exists()


def test_force_rmtree_nonexistent_returns_success(tmp_path):
    target = tmp_path / "does_not_exist"
    success, err = _force_rmtree(target)
    assert success is True
    assert err is None


def test_force_rmtree_nested_directory(tmp_path):
    target = tmp_path / "nested"
    (target / "sub").mkdir(parents=True)
    (target / "sub" / "file.bin").write_bytes(b"\x00" * 100)
    success, err = _force_rmtree(target)
    assert success is True
    assert not target.exists()
