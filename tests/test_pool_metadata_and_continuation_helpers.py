"""Tests for pool_metadata pure helpers."""


from mullet.knowledge.engine.pool_metadata import (
    read_metadata,
    ensure_created,
    mark_searched,
    mark_indexed,
    _now_iso,
    METADATA_FILENAME,
)


# ---------------------------------------------------------------------------
# pool_metadata — pure operations on tmp_path
# ---------------------------------------------------------------------------

class TestNowIso:
    def test_returns_string(self):
        assert isinstance(_now_iso(), str)

    def test_contains_t_separator(self):
        assert "T" in _now_iso()

    def test_contains_utc_offset(self):
        ts = _now_iso()
        assert "+" in ts or "Z" in ts


class TestReadMetadata:
    def test_missing_dir_returns_empty(self, tmp_path):
        missing = tmp_path / "nonexistent"
        assert read_metadata(missing) == {}

    def test_no_file_returns_empty(self, tmp_path):
        assert read_metadata(tmp_path) == {}

    def test_valid_metadata_returned(self, tmp_path):
        (tmp_path / METADATA_FILENAME).write_text('{"created_at": "2025-01-01"}')
        data = read_metadata(tmp_path)
        assert data["created_at"] == "2025-01-01"

    def test_invalid_json_returns_empty(self, tmp_path):
        (tmp_path / METADATA_FILENAME).write_text("not json {")
        assert read_metadata(tmp_path) == {}


class TestEnsureCreated:
    def test_sets_created_at(self, tmp_path):
        ensure_created(tmp_path)
        data = read_metadata(tmp_path)
        assert "created_at" in data

    def test_idempotent(self, tmp_path):
        ensure_created(tmp_path)
        first = read_metadata(tmp_path)["created_at"]
        ensure_created(tmp_path)
        second = read_metadata(tmp_path)["created_at"]
        assert first == second


class TestMarkSearched:
    def test_sets_last_searched_at(self, tmp_path):
        mark_searched(tmp_path)
        data = read_metadata(tmp_path)
        assert "last_searched_at" in data

    def test_increments_search_count(self, tmp_path):
        mark_searched(tmp_path)
        mark_searched(tmp_path)
        data = read_metadata(tmp_path)
        assert data["search_count"] == 2

    def test_creates_created_at_if_missing(self, tmp_path):
        mark_searched(tmp_path)
        data = read_metadata(tmp_path)
        assert "created_at" in data


class TestMarkIndexed:
    def test_sets_last_indexed_at(self, tmp_path):
        mark_indexed(tmp_path)
        data = read_metadata(tmp_path)
        assert "last_indexed_at" in data

    def test_creates_created_at_if_missing(self, tmp_path):
        mark_indexed(tmp_path)
        data = read_metadata(tmp_path)
        assert "created_at" in data
