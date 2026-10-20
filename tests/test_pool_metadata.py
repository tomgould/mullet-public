"""Tests for mullet.knowledge.engine.pool_metadata — per-pool usage tracking."""

import json
from pathlib import Path

import pytest

from mullet.knowledge.engine.pool_metadata import (
    METADATA_FILENAME,
    delete_pool,
    ensure_created,
    mark_indexed,
    mark_searched,
    read_metadata,
)


def _meta_path(pool_dir: Path) -> Path:
    return pool_dir / METADATA_FILENAME


# --- read_metadata ---------------------------------------------------------


def test_read_metadata_missing_file_returns_empty(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    assert read_metadata(pool) == {}


def test_read_metadata_valid_file(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    _meta_path(pool).write_text(json.dumps({"created_at": "2026-01-01T00:00:00+00:00"}))
    data = read_metadata(pool)
    assert data["created_at"] == "2026-01-01T00:00:00+00:00"


def test_read_metadata_malformed_json_returns_empty(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    _meta_path(pool).write_text("not valid json {{{")
    assert read_metadata(pool) == {}


# --- ensure_created --------------------------------------------------------


def test_ensure_created_stamps_first_time(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    ensure_created(pool)
    data = read_metadata(pool)
    assert "created_at" in data


def test_ensure_created_is_idempotent(tmp_path):
    """Second call must not overwrite the original created_at."""
    pool = tmp_path / "pool"
    pool.mkdir()
    ensure_created(pool)
    first_ts = read_metadata(pool)["created_at"]

    # Sleep not needed — ensure_created checks membership, not value
    ensure_created(pool)
    second_ts = read_metadata(pool)["created_at"]
    assert second_ts == first_ts


# --- mark_searched ---------------------------------------------------------


def test_mark_searched_increments_count(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    mark_searched(pool)
    mark_searched(pool)
    mark_searched(pool)
    data = read_metadata(pool)
    assert data["search_count"] == 3
    assert "last_searched_at" in data


def test_mark_searched_sets_created_at_if_missing(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    mark_searched(pool)
    data = read_metadata(pool)
    assert "created_at" in data
    assert data["created_at"] == data["last_searched_at"]


# --- mark_indexed ----------------------------------------------------------


def test_mark_indexed_stamps_timestamp(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    mark_indexed(pool)
    data = read_metadata(pool)
    assert "last_indexed_at" in data


def test_mark_indexed_sets_created_at_if_missing(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    mark_indexed(pool)
    data = read_metadata(pool)
    assert data["created_at"] == data["last_indexed_at"]


def test_mark_indexed_preserves_existing_created_at(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    _meta_path(pool).write_text(json.dumps({"created_at": "2024-01-01T00:00:00+00:00"}))
    mark_indexed(pool)
    data = read_metadata(pool)
    assert data["created_at"] == "2024-01-01T00:00:00+00:00"
    assert "last_indexed_at" in data


# --- delete_pool -----------------------------------------------------------


def test_delete_pool_removes_directory(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "file.txt").write_text("content")
    delete_pool(pool)
    assert not pool.exists()


def test_delete_pool_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        delete_pool(tmp_path / "nope")


# --- atomic write ----------------------------------------------------------


def test_mark_indexed_noop_on_missing_dir(tmp_path):
    """mark_indexed must NOT conjure a pool dir for a non-existent pool.

    Usage tracking is a sidecar to an existing pool; creating the directory
    as a side effect spawns empty stub pools when callers pass an unresolved
    display name. It must be a clean no-op instead.
    """
    pool = tmp_path / "new_pool"
    # Don't mkdir first
    mark_indexed(pool)
    assert not pool.exists()
    assert not _meta_path(pool).exists()


def test_mark_searched_noop_on_missing_dir(tmp_path):
    """mark_searched must NOT create a pool dir for a non-existent pool."""
    pool = tmp_path / "new_pool"
    mark_searched(pool)
    assert not pool.exists()
    assert not _meta_path(pool).exists()


def test_write_never_raises_on_permission_error(tmp_path, monkeypatch):
    """Write failures are swallowed (best-effort) — must not raise."""
    pool = tmp_path / "pool"
    pool.mkdir()

    # Monkeypatch os.replace to raise PermissionError
    import os as _os
    original_replace = _os.replace

    def failing_replace(src, dst):
        raise PermissionError("simulated read-only filesystem")

    monkeypatch.setattr(_os, "replace", failing_replace)

    # Must not raise — metadata write is best-effort
    mark_indexed(pool)
    monkeypatch.setattr(_os, "replace", original_replace)
