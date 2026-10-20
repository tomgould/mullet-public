"""Tests for mullet.knowledge.engine.pool_metadata — filesystem metadata operations."""

import json
import pytest
from datetime import datetime

from mullet.knowledge.engine.pool_metadata import (
    read_metadata,
    ensure_created,
    mark_searched,
    mark_indexed,
    delete_pool,
    _now_iso,
    _metadata_path,
    METADATA_FILENAME,
)


# ---------------------------------------------------------------------------
# _now_iso
# ---------------------------------------------------------------------------

class TestNowIso:
    def test_returns_string(self):
        assert isinstance(_now_iso(), str)

    def test_format_matches_iso8601(self):
        ts = _now_iso()
        # Should parse as ISO 8601 with timezone
        dt = datetime.fromisoformat(ts)
        assert dt.tzinfo is not None

    def test_seconds_precision(self):
        # No microseconds in output
        ts = _now_iso()
        assert "." not in ts or ts.count(".") == 0  # isoformat with timespec="seconds"


# ---------------------------------------------------------------------------
# _metadata_path
# ---------------------------------------------------------------------------

class TestMetadataPath:
    def test_returns_path_in_pool_dir(self, tmp_path):
        result = _metadata_path(tmp_path)
        assert result.parent == tmp_path

    def test_filename_is_correct(self, tmp_path):
        result = _metadata_path(tmp_path)
        assert result.name == METADATA_FILENAME


# ---------------------------------------------------------------------------
# read_metadata
# ---------------------------------------------------------------------------

class TestReadMetadata:
    def test_missing_file_returns_empty_dict(self, tmp_path):
        result = read_metadata(tmp_path)
        assert result == {}

    def test_valid_file_returns_data(self, tmp_path):
        data = {"created_at": "2024-01-01T00:00:00+00:00", "search_count": 5}
        (tmp_path / METADATA_FILENAME).write_text(json.dumps(data))
        result = read_metadata(tmp_path)
        assert result["search_count"] == 5

    def test_corrupted_file_returns_empty(self, tmp_path):
        (tmp_path / METADATA_FILENAME).write_text("not json")
        result = read_metadata(tmp_path)
        assert result == {}


# ---------------------------------------------------------------------------
# ensure_created
# ---------------------------------------------------------------------------

class TestEnsureCreated:
    def test_creates_metadata_with_created_at(self, tmp_path):
        ensure_created(tmp_path)
        data = read_metadata(tmp_path)
        assert "created_at" in data

    def test_idempotent_does_not_overwrite(self, tmp_path):
        ensure_created(tmp_path)
        ts1 = read_metadata(tmp_path)["created_at"]
        ensure_created(tmp_path)
        ts2 = read_metadata(tmp_path)["created_at"]
        assert ts1 == ts2

    def test_creates_pool_dir_if_missing(self, tmp_path):
        pool_dir = tmp_path / "new_pool"
        ensure_created(pool_dir)
        assert pool_dir.exists()


# ---------------------------------------------------------------------------
# mark_searched
# ---------------------------------------------------------------------------

class TestMarkSearched:
    def test_creates_last_searched_at(self, tmp_path):
        mark_searched(tmp_path)
        data = read_metadata(tmp_path)
        assert "last_searched_at" in data

    def test_increments_search_count(self, tmp_path):
        mark_searched(tmp_path)
        mark_searched(tmp_path)
        mark_searched(tmp_path)
        data = read_metadata(tmp_path)
        assert data["search_count"] == 3

    def test_sets_created_at_if_missing(self, tmp_path):
        mark_searched(tmp_path)
        data = read_metadata(tmp_path)
        assert "created_at" in data

    def test_does_not_overwrite_existing_created_at(self, tmp_path):
        ensure_created(tmp_path)
        ts1 = read_metadata(tmp_path)["created_at"]
        mark_searched(tmp_path)
        ts2 = read_metadata(tmp_path)["created_at"]
        assert ts1 == ts2

    def test_search_count_starts_at_one(self, tmp_path):
        mark_searched(tmp_path)
        assert read_metadata(tmp_path)["search_count"] == 1


# ---------------------------------------------------------------------------
# mark_indexed
# ---------------------------------------------------------------------------

class TestMarkIndexed:
    def test_creates_last_indexed_at(self, tmp_path):
        mark_indexed(tmp_path)
        data = read_metadata(tmp_path)
        assert "last_indexed_at" in data

    def test_sets_created_at_if_missing(self, tmp_path):
        mark_indexed(tmp_path)
        data = read_metadata(tmp_path)
        assert "created_at" in data

    def test_does_not_overwrite_existing_created_at(self, tmp_path):
        ensure_created(tmp_path)
        ts1 = read_metadata(tmp_path)["created_at"]
        mark_indexed(tmp_path)
        ts2 = read_metadata(tmp_path)["created_at"]
        assert ts1 == ts2

    def test_last_indexed_at_is_iso(self, tmp_path):
        mark_indexed(tmp_path)
        ts = read_metadata(tmp_path)["last_indexed_at"]
        dt = datetime.fromisoformat(ts)
        assert dt.tzinfo is not None


# ---------------------------------------------------------------------------
# delete_pool
# ---------------------------------------------------------------------------

class TestDeletePool:
    def test_removes_directory(self, tmp_path):
        pool_dir = tmp_path / "pool"
        pool_dir.mkdir()
        (pool_dir / "file.txt").write_text("data")
        delete_pool(pool_dir)
        assert not pool_dir.exists()

    def test_raises_if_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            delete_pool(tmp_path / "nonexistent")

    def test_removes_nested_contents(self, tmp_path):
        pool_dir = tmp_path / "pool"
        nested = pool_dir / "sub" / "deep"
        nested.mkdir(parents=True)
        (nested / "file.json").write_text("{}")
        delete_pool(pool_dir)
        assert not pool_dir.exists()
