"""Tests for review_session_cmd pure helpers and LMStudioManager init."""

from datetime import datetime, timezone

from mullet.cli.review_session_cmd import _parse_ts, _section, _utcnow
from mullet.core.providers.lmstudio_manager import LMStudioManager, _LMS_SEARCH_PATHS


# ---------------------------------------------------------------------------
# _utcnow
# ---------------------------------------------------------------------------

class TestUtcNow:
    def test_returns_datetime(self):
        result = _utcnow()
        assert isinstance(result, datetime)

    def test_has_utc_timezone(self):
        result = _utcnow()
        assert result.tzinfo is not None
        assert result.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# _parse_ts
# ---------------------------------------------------------------------------

class TestParseTs:
    def test_z_suffix_parsed(self):
        result = _parse_ts("2025-01-15T10:30:45Z")
        assert result is not None
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 15

    def test_utc_offset_parsed(self):
        result = _parse_ts("2025-06-01T12:00:00+00:00")
        assert result is not None
        assert result.hour == 12

    def test_returns_utc_aware_datetime(self):
        result = _parse_ts("2025-01-15T10:30:45Z")
        assert result.tzinfo == timezone.utc

    def test_empty_string_returns_none(self):
        assert _parse_ts("") is None

    def test_invalid_format_returns_none(self):
        assert _parse_ts("not-a-date") is None

    def test_none_like_format_handled(self):
        # Various malformed inputs
        assert _parse_ts("2025-13-99T99:99:99Z") is None

    def test_iso_without_z_also_works(self):
        # fromisoformat path
        result = _parse_ts("2025-01-15T10:30:45")
        assert result is not None


# ---------------------------------------------------------------------------
# _section
# ---------------------------------------------------------------------------

class TestSection:
    def test_returns_string(self):
        assert isinstance(_section("Test Section"), str)

    def test_title_in_output(self):
        result = _section("My Title")
        assert "My Title" in result

    def test_contains_separator_chars(self):
        result = _section("Title")
        # Should contain the ─ bar characters
        assert "─" in result

    def test_starts_with_newline(self):
        result = _section("Title")
        assert result.startswith("\n")


# ---------------------------------------------------------------------------
# LMStudioManager init
# ---------------------------------------------------------------------------

class TestLMStudioManagerInit:
    def test_can_instantiate_default_port(self):
        mgr = LMStudioManager()
        assert mgr.port == 1234

    def test_custom_port(self):
        mgr = LMStudioManager(port=5678)
        assert mgr.port == 5678

    def test_base_url_contains_port(self):
        mgr = LMStudioManager(port=1234)
        assert "1234" in mgr.base_url

    def test_base_url_is_localhost(self):
        mgr = LMStudioManager()
        assert "localhost" in mgr.base_url


class TestLmsSearchPaths:
    def test_is_list(self):
        assert isinstance(_LMS_SEARCH_PATHS, list)

    def test_non_empty(self):
        assert len(_LMS_SEARCH_PATHS) > 0

    def test_all_strings(self):
        assert all(isinstance(p, str) for p in _LMS_SEARCH_PATHS)

    def test_contains_cache_path(self):
        assert any(".cache" in p for p in _LMS_SEARCH_PATHS)
