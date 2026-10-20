"""Tests for mullet.knowledge.sources.web_cache."""
from __future__ import annotations

import hashlib
import json
import os
from unittest.mock import patch


from mullet.knowledge.sources.web_cache import (
    CACHE_FILENAME,
    CACHE_VERSION,
    CrawlCache,
    _get_header,
)


# ---------------------------------------------------------------------------
# _get_header
# ---------------------------------------------------------------------------

class TestGetHeader:
    def test_exact_match(self) -> None:
        assert _get_header({"etag": "abc"}, "etag") == "abc"

    def test_case_insensitive_lookup(self) -> None:
        assert _get_header({"ETag": "xyz"}, "etag") == "xyz"
        assert _get_header({"etag": "xyz"}, "ETag") == "xyz"

    def test_mixed_case_header(self) -> None:
        assert _get_header({"Last-Modified": "Mon, 01 Jan 2024"}, "last-modified") == "Mon, 01 Jan 2024"

    def test_missing_header_returns_empty(self) -> None:
        assert _get_header({"content-type": "text/html"}, "etag") == ""

    def test_empty_dict_returns_empty(self) -> None:
        assert _get_header({}, "etag") == ""


# ---------------------------------------------------------------------------
# CrawlCache
# ---------------------------------------------------------------------------

class TestCrawlCacheInit:
    def test_cache_path_set_correctly(self, tmp_path) -> None:
        cache = CrawlCache(str(tmp_path))
        assert cache.cache_path == os.path.join(str(tmp_path), CACHE_FILENAME)

    def test_empty_cache_has_no_pages(self, tmp_path) -> None:
        cache = CrawlCache(str(tmp_path))
        assert cache.page_count == 0


class TestCrawlCacheLoad:
    def test_load_nonexistent_file_is_noop(self, tmp_path) -> None:
        cache = CrawlCache(str(tmp_path))
        cache.load()  # Should not raise
        assert cache.page_count == 0

    def test_load_valid_cache(self, tmp_path) -> None:
        data = {
            "version": CACHE_VERSION,
            "pages": {
                "/page1": {
                    "etag": "abc",
                    "last_modified": "",
                    "content_hash": "sha256:xxx",
                    "links": [],
                    "timestamp": "2024-01-01T00:00:00+00:00",
                }
            },
        }
        cache_file = tmp_path / CACHE_FILENAME
        cache_file.write_text(json.dumps(data))
        cache = CrawlCache(str(tmp_path))
        cache.load()
        assert cache.page_count == 1

    def test_load_version_mismatch_starts_fresh(self, tmp_path) -> None:
        data = {"version": 999, "pages": {"/old": {}}}
        (tmp_path / CACHE_FILENAME).write_text(json.dumps(data))
        cache = CrawlCache(str(tmp_path))
        cache.load()
        assert cache.page_count == 0

    def test_load_invalid_json_is_silent(self, tmp_path) -> None:
        (tmp_path / CACHE_FILENAME).write_text("{ not valid json")
        cache = CrawlCache(str(tmp_path))
        cache.load()  # Should not raise
        assert cache.page_count == 0


class TestCrawlCacheSave:
    def test_save_creates_file(self, tmp_path) -> None:
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/test", {}, "hello")
        cache.save()
        assert (tmp_path / CACHE_FILENAME).exists()

    def test_save_round_trip(self, tmp_path) -> None:
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/index", {"ETag": "abc"}, "page content")
        cache.save()

        cache2 = CrawlCache(str(tmp_path))
        cache2.load()
        assert cache2.page_count == 1
        headers = cache2.get_page_headers("/index")
        assert headers.get("If-None-Match") == "abc"


class TestCrawlCacheUpdatePage:
    def test_update_page_stores_entry(self, tmp_path) -> None:
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/api/v1", {"etag": "def"}, "content")
        assert cache.page_count == 1

    def test_update_page_stores_content_hash(self, tmp_path) -> None:
        cache = CrawlCache(str(tmp_path))
        content = "hello world"
        cache.update_page("/p", {}, content)
        expected_hash = f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"
        assert cache.is_page_unchanged("/p", content) is True

    def test_update_page_stores_links(self, tmp_path) -> None:
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/home", {}, "body", links=["/about", "/contact"])
        links = cache.get_page_links("/home")
        assert "/about" in links
        assert "/contact" in links

    def test_update_page_no_links_returns_empty(self, tmp_path) -> None:
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/p", {}, "content")
        assert cache.get_page_links("/p") == []


class TestCrawlCacheGetPageHeaders:
    def test_returns_if_none_match_for_etag(self, tmp_path) -> None:
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/doc", {"etag": "abc123"}, "content")
        headers = cache.get_page_headers("/doc")
        assert headers.get("If-None-Match") == "abc123"

    def test_returns_if_modified_since(self, tmp_path) -> None:
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/doc", {"last-modified": "Mon, 01 Jan 2024 00:00:00 GMT"}, "content")
        headers = cache.get_page_headers("/doc")
        assert "If-Modified-Since" in headers

    def test_empty_for_unknown_page(self, tmp_path) -> None:
        cache = CrawlCache(str(tmp_path))
        assert cache.get_page_headers("/unknown") == {}

    def test_no_etag_no_if_none_match(self, tmp_path) -> None:
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/doc", {}, "content")
        headers = cache.get_page_headers("/doc")
        assert "If-None-Match" not in headers


class TestCrawlCacheIsPageUnchanged:
    def test_unchanged_page_returns_true(self, tmp_path) -> None:
        cache = CrawlCache(str(tmp_path))
        content = "test page content"
        cache.update_page("/p", {}, content)
        assert cache.is_page_unchanged("/p", content) is True

    def test_changed_page_returns_false(self, tmp_path) -> None:
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/p", {}, "original content")
        assert cache.is_page_unchanged("/p", "new content") is False

    def test_unknown_page_returns_false(self, tmp_path) -> None:
        cache = CrawlCache(str(tmp_path))
        assert cache.is_page_unchanged("/new", "content") is False


class TestCrawlCachePageCount:
    def test_page_count_increments(self, tmp_path) -> None:
        cache = CrawlCache(str(tmp_path))
        assert cache.page_count == 0
        cache.update_page("/a", {}, "x")
        assert cache.page_count == 1
        cache.update_page("/b", {}, "y")
        assert cache.page_count == 2

    def test_overwrite_does_not_increment(self, tmp_path) -> None:
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/a", {}, "first")
        cache.update_page("/a", {}, "second")
        assert cache.page_count == 1


class TestCrawlCacheSaveExceptionPath:
    def test_save_os_error_is_silenced(self, tmp_path) -> None:
        """Lines 68-74: OSError during save is caught and logged (no raise)."""
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/a", {}, "content")
        with patch("builtins.open", side_effect=OSError("disk full")):
            # Should not raise
            cache.save()

    def test_save_cleanup_unlink_failure_is_silenced(self, tmp_path) -> None:
        """Lines 70-73: when cleanup os.unlink also raises, it is silenced."""
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/a", {}, "content")
        with patch("builtins.open", side_effect=OSError("disk full")), \
             patch("os.unlink", side_effect=OSError("unlink failed")):
            cache.save()  # Must not raise


class TestIsPageUnchangedMissingEntry:
    def test_no_entry_returns_false(self, tmp_path) -> None:
        """Line 108: entry doesn't exist → returns False."""
        cache = CrawlCache(str(tmp_path))
        assert cache.is_page_unchanged("/nonexistent", "anything") is False

    def test_entry_without_content_hash_returns_false(self, tmp_path) -> None:
        """Line 108: entry exists but has no content_hash → returns False."""
        cache = CrawlCache(str(tmp_path))
        # Inject a page entry without content_hash
        cache._data["pages"]["/page"] = {"etag": "abc"}
        assert cache.is_page_unchanged("/page", "anything") is False


class TestGetPageLinksWithEntry:
    def test_returns_links_when_entry_exists(self, tmp_path) -> None:
        """get_page_links: entry with links → returns them."""
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/a", {}, "content", links=["http://x.com", "http://y.com"])
        links = cache.get_page_links("/a")
        assert links == ["http://x.com", "http://y.com"]

    def test_returns_empty_list_when_no_entry(self, tmp_path) -> None:
        """Line 108: get_page_links with unknown URL → early return []."""
        cache = CrawlCache(str(tmp_path))
        assert cache.get_page_links("/never-updated") == []
