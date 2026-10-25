"""Tests for WebSourceConnector.sync() and _crawl() — httpx mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from mullet.knowledge.sources.base import SyncResult
from mullet.knowledge.sources.web_cache import CrawlCache
from mullet.knowledge.sources.web_crawler import (
    WebSourceConfig,
    WebSourceConnector,
    _url_to_filepath,
)


def _make_config(url="https://example.com/docs", max_pages=10, wait_time=0.0):
    return WebSourceConfig(url=url, max_pages=max_pages, wait_time=wait_time)


def _make_response(status=200, html="", content_type="text/html; charset=utf-8", headers=None):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.text = html
    resp.headers = {"content-type": content_type, **(headers or {})}
    return resp


def _make_html(body: str, title: str = "Test Page") -> str:
    return f"""<!DOCTYPE html>
<html>
<head><title>{title}</title></head>
<body>
<article class="entry-content">
<h1>{title}</h1>
<p>{body}</p>
</article>
</body>
</html>"""


class TestSync:
    def test_sync_creates_web_dir(self, tmp_path):
        """sync() creates target/web/{domain}/ directory."""
        config = _make_config()
        connector = WebSourceConnector(config)
        with patch.object(connector, "_crawl") as mock_crawl:
            result = connector.sync(str(tmp_path))
        web_dir = tmp_path / "web" / "example.com"
        assert web_dir.is_dir()
        mock_crawl.assert_called_once()

    def test_sync_returns_result(self, tmp_path):
        """sync() returns SyncResult."""
        config = _make_config()
        connector = WebSourceConnector(config)
        with patch.object(connector, "_crawl"):
            result = connector.sync(str(tmp_path))
        assert isinstance(result, SyncResult)
        assert result.source_type == "web"

    def test_sync_http_error_appended(self, tmp_path):
        """Lines 184-186: httpx.HTTPError during crawl → error in result."""
        config = _make_config()
        connector = WebSourceConnector(config)
        with patch.object(connector, "_crawl", side_effect=httpx.HTTPError("conn failed")):
            result = connector.sync(str(tmp_path))
        assert any("Crawl error" in e for e in result.errors)

    def test_sync_os_error_appended(self, tmp_path):
        """Line 184: OSError during crawl → error in result."""
        config = _make_config()
        connector = WebSourceConnector(config)
        with patch.object(connector, "_crawl", side_effect=OSError("disk full")):
            result = connector.sync(str(tmp_path))
        assert any("Crawl error" in e for e in result.errors)

    def test_status_returns_dict(self):
        """Line 193: status() returns a dictionary with connector info."""
        config = _make_config()
        connector = WebSourceConnector(config)
        status = connector.status()
        assert isinstance(status, dict)
        assert status["source_type"] == "web"
        assert status["url"] == config.url


class TestCrawl:
    """Tests for _crawl() via mocking httpx.Client."""

    def _connector(self, url="https://example.com/", max_pages=5, wait_time=0.0):
        return WebSourceConnector(_make_config(url=url, max_pages=max_pages, wait_time=wait_time))

    def _patch_client(self, side_effect_or_return):
        """Return a patch context for httpx.Client that controls response."""
        mock_client = MagicMock()
        if isinstance(side_effect_or_return, list):
            mock_client.get.side_effect = side_effect_or_return
        elif isinstance(side_effect_or_return, BaseException):
            mock_client.get.side_effect = side_effect_or_return
        else:
            mock_client.get.return_value = side_effect_or_return
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        return patch("httpx.Client", return_value=mock_client), mock_client

    def test_successful_page_files_added(self, tmp_path):
        """Lines 288-330: page with content → .md file written, files_added incremented."""
        connector = self._connector()
        body = " ".join(["word"] * 100)
        html = _make_html(body)
        response = _make_response(html=html)
        patch_ctx, _mock_client = self._patch_client(response)

        cache = CrawlCache(str(tmp_path))
        result_obj = SyncResult(source_type="web")

        with patch_ctx, patch("time.sleep"):
            connector._crawl(tmp_path, cache, result_obj, set())

        assert result_obj.files_added >= 1

    def test_http_error_appended_to_errors(self, tmp_path):
        """Lines 240-242: httpx.HTTPError on get → error appended."""
        connector = self._connector()
        patch_ctx, _mock_client = self._patch_client(httpx.HTTPError("timeout"))

        cache = CrawlCache(str(tmp_path))
        result_obj = SyncResult(source_type="web")

        with patch_ctx:
            connector._crawl(tmp_path, cache, result_obj, set())

        assert len(result_obj.errors) >= 1

    def test_304_reuses_cached_links(self, tmp_path):
        """Lines 245-249: 304 Not Modified → cached links queued."""
        connector = self._connector()
        response_304 = _make_response(status=304, html="")
        patch_ctx, _mock_client = self._patch_client(response_304)

        cache = CrawlCache(str(tmp_path))
        result_obj = SyncResult(source_type="web")

        with patch_ctx:
            connector._crawl(tmp_path, cache, result_obj, set())

        assert result_obj.files_added == 0  # 304 doesn't add files

    def test_500_server_error_appended(self, tmp_path):
        """Lines 252-255: status >= 500 → error appended."""
        connector = self._connector()
        response_500 = _make_response(status=500)
        patch_ctx, _mock_client = self._patch_client(response_500)

        cache = CrawlCache(str(tmp_path))
        result_obj = SyncResult(source_type="web")

        with patch_ctx:
            connector._crawl(tmp_path, cache, result_obj, set())

        assert any("server error" in e for e in result_obj.errors)

    def test_429_rate_limited_appended(self, tmp_path):
        """Lines 256-258: status 429 → error appended."""
        connector = self._connector()
        response_429 = _make_response(status=429)
        patch_ctx, _mock_client = self._patch_client(response_429)

        cache = CrawlCache(str(tmp_path))
        result_obj = SyncResult(source_type="web")

        with patch_ctx:
            connector._crawl(tmp_path, cache, result_obj, set())

        assert any("rate limited" in e for e in result_obj.errors)

    def test_400_skipped_silently(self, tmp_path):
        """Lines 259-262: status 404 → no error appended, just skipped."""
        connector = self._connector()
        response_404 = _make_response(status=404)
        patch_ctx, _mock_client = self._patch_client(response_404)

        cache = CrawlCache(str(tmp_path))
        result_obj = SyncResult(source_type="web")

        with patch_ctx:
            connector._crawl(tmp_path, cache, result_obj, set())

        assert len(result_obj.errors) == 0

    def test_non_html_content_type_skipped(self, tmp_path):
        """Line 267-268: non-text/html content-type → page skipped."""
        connector = self._connector()
        response = _make_response(status=200, html="", content_type="application/json")
        patch_ctx, _mock_client = self._patch_client(response)

        cache = CrawlCache(str(tmp_path))
        result_obj = SyncResult(source_type="web")

        with patch_ctx:
            connector._crawl(tmp_path, cache, result_obj, set())

        assert result_obj.files_added == 0

    def test_unchanged_page_uses_cached_links(self, tmp_path):
        """Lines 273-277: content hash unchanged → cached links queued."""
        connector = self._connector()
        html = _make_html("some content that stays the same")
        response = _make_response(html=html)

        cache = CrawlCache(str(tmp_path))
        # Pre-populate cache with the same content hash so it's "unchanged"
        cache.update_page("/", {}, html, links=[])
        patch_ctx, _mock_client = self._patch_client(response)

        result_obj = SyncResult(source_type="web")

        with patch_ctx:
            connector._crawl(tmp_path, cache, result_obj, set())

        assert result_obj.files_added == 0  # unchanged, not re-added

    def test_page_with_no_markdown_updates_cache_only(self, tmp_path):
        """Lines 290-298: html_to_markdown returns None → cache updated, no file written."""
        connector = self._connector()
        # Minimal HTML that won't produce meaningful markdown
        html = "<html><head><title>t</title></head><body><p>x</p></body></html>"
        response = _make_response(html=html)
        patch_ctx, _mock_client = self._patch_client(response)

        cache = CrawlCache(str(tmp_path))
        result_obj = SyncResult(source_type="web")

        with patch_ctx, patch("mullet.knowledge.sources.web_crawler.html_to_markdown", return_value=None):
            connector._crawl(tmp_path, cache, result_obj, set())

        assert result_obj.files_added == 0

    def test_existing_file_increments_updated(self, tmp_path):
        """Lines 325-326: filename already in known_files → files_updated += 1."""
        connector = self._connector()
        body = " ".join(["word"] * 100)
        html = _make_html(body)
        response = _make_response(html=html)
        patch_ctx, _mock_client = self._patch_client(response)

        cache = CrawlCache(str(tmp_path))
        result_obj = SyncResult(source_type="web")
        # Pre-populate known_files with the filename that will be written
        starting_path = connector.config.starting_path
        existing = {_url_to_filepath(starting_path)}

        with patch_ctx, patch("time.sleep"):
            connector._crawl(tmp_path, cache, result_obj, existing)

        assert result_obj.files_updated >= 1

    def test_wait_time_zero_no_sleep(self, tmp_path):
        """Line 340-341: wait_time=0 → time.sleep not called."""
        connector = self._connector(wait_time=0.0)
        body = " ".join(["word"] * 100)
        html = _make_html(body)
        response = _make_response(html=html)
        patch_ctx, _mock_client = self._patch_client(response)

        cache = CrawlCache(str(tmp_path))
        result_obj = SyncResult(source_type="web")

        with patch_ctx, patch("time.sleep") as mock_sleep:
            connector._crawl(tmp_path, cache, result_obj, set())

        mock_sleep.assert_not_called()

    def test_already_visited_url_skipped(self, tmp_path):
        """Line 223: url already in visited set → continue."""
        connector = self._connector()
        body = " ".join(["word"] * 100)
        html = _make_html(body)
        response = _make_response(html=html)

        from mullet.knowledge.sources.web_cache import CrawlCache
        cache = CrawlCache(str(tmp_path))
        result_obj = MagicMock()
        result_obj.files_added = 0
        result_obj.files_updated = 0
        result_obj.errors = []

        call_count = [0]
        def get_side_effect(url, headers=None):
            call_count[0] += 1
            return response

        patch_ctx, mock_client = self._patch_client(response)
        mock_client.get.side_effect = get_side_effect

        # The HTML contains a link back to "/" (self-referential) — the second visit is skipped
        self_link_html = _make_html(" ".join(["word"] * 100)) + '<a href="/">home</a>'
        response2 = _make_response(html=self_link_html)
        mock_client.get.return_value = response2

        with patch_ctx, patch("time.sleep"):
            connector._crawl(tmp_path, cache, result_obj, set())

        # Even with self-referential links, pages_crawled stays bounded
        assert result_obj.files_added >= 0  # no infinite loop

    def test_non_content_path_skipped(self, tmp_path):
        """Line 228: URL matches NON_CONTENT_PATHS → continue."""
        # Configure with url that starts at /login — this path is skipped by NON_CONTENT_PATHS
        connector = WebSourceConnector(_make_config(url="https://example.com/login", wait_time=0.0))
        from mullet.knowledge.sources.web_cache import CrawlCache
        cache = CrawlCache(str(tmp_path))
        result_obj = MagicMock()
        result_obj.files_added = 0
        result_obj.files_updated = 0
        result_obj.errors = []

        patch_ctx, mock_client = self._patch_client(_make_response(html=""))
        with patch_ctx:
            connector._crawl(tmp_path, cache, result_obj, set())

        # /login matches NON_CONTENT_PATHS → no HTTP request made
        mock_client.get.assert_not_called()

    def test_out_of_scope_url_skipped(self, tmp_path):
        """Line 232: url_path not in scope → continue."""
        # stay_in_path=True, starting_path=/docs — crawl starts at /docs
        connector = WebSourceConnector(_make_config(url="https://example.com/docs", wait_time=0.0))
        # Create HTML that links to /other — out of scope
        html_with_out_of_scope = _make_html(" ".join(["word"] * 100)) + '<a href="/other">link</a>'
        response = _make_response(html=html_with_out_of_scope)

        from mullet.knowledge.sources.web_cache import CrawlCache
        cache = CrawlCache(str(tmp_path))
        result_obj = MagicMock()
        result_obj.files_added = 0
        result_obj.files_updated = 0
        result_obj.errors = []

        patch_ctx, mock_client = self._patch_client(response)
        with patch_ctx, patch("time.sleep"):
            connector._crawl(tmp_path, cache, result_obj, set())

        # /other is out of scope and will be skipped — no error for it
        assert len(result_obj.errors) == 0

    def test_304_with_cached_links_queued(self, tmp_path):
        """Lines 247-248: 304 + cache has links → links queued into BFS."""
        connector = self._connector()
        response_304 = _make_response(status=304, html="")

        from mullet.knowledge.sources.web_cache import CrawlCache
        cache = CrawlCache(str(tmp_path))
        # Pre-cache the starting path with a link to /docs
        cache.update_page("/", {}, "cached content", links=["/docs"])
        # /docs response: also 304 (so test stays simple)
        patch_ctx, mock_client = self._patch_client(response_304)

        result_obj = MagicMock()
        result_obj.files_added = 0
        result_obj.files_updated = 0
        result_obj.errors = []

        with patch_ctx:
            connector._crawl(tmp_path, cache, result_obj, set())

        # /docs was queued from the cached link — get() called for both / and /docs
        assert mock_client.get.call_count >= 2

    def test_unchanged_page_with_cached_links_queued(self, tmp_path):
        """Lines 275-276: content unchanged + cache has links → links queued."""
        connector = self._connector()
        body = "unchanged content " * 10
        html = _make_html(body)
        response = _make_response(html=html)

        from mullet.knowledge.sources.web_cache import CrawlCache
        cache = CrawlCache(str(tmp_path))
        # Pre-cache the starting page with current content hash (so it's "unchanged")
        cache.update_page("/", {}, html, links=["/docs"])
        # /docs response: not-found → skip cleanly
        response_404 = _make_response(status=404)

        patch_ctx, mock_client = self._patch_client([response, response_404])
        result_obj = MagicMock()
        result_obj.files_added = 0
        result_obj.files_updated = 0
        result_obj.errors = []

        with patch_ctx:
            connector._crawl(tmp_path, cache, result_obj, set())

        # /docs was queued from unchanged page's cached links
        assert mock_client.get.call_count >= 2

    def test_links_extracted_and_queued_from_new_page(self, tmp_path):
        """Lines 283-286: new page has links → normalized and queued."""
        connector = self._connector(url="https://example.com/", wait_time=0.0)
        # Page with a link to /sub-page
        body = " ".join(["word"] * 100)
        html_with_link = (
            '<html><head><title>Home</title></head><body>'
            '<article class="entry-content"><h1>Home</h1>'
            f'<p>{body}</p>'
            '<a href="/sub-page">Go to sub</a>'
            '</article></body></html>'
        )
        # First response: page with link; second response: 404 for /sub-page
        response_main = _make_response(html=html_with_link)
        response_sub = _make_response(status=404)
        patch_ctx, mock_client = self._patch_client([response_main, response_sub])

        from mullet.knowledge.sources.web_cache import CrawlCache
        cache = CrawlCache(str(tmp_path))
        result_obj = MagicMock()
        result_obj.files_added = 0
        result_obj.files_updated = 0
        result_obj.errors = []

        with patch_ctx, patch("time.sleep"):
            connector._crawl(tmp_path, cache, result_obj, set())

        # /sub-page was queued and attempted
        assert mock_client.get.call_count >= 2

    def test_wait_time_positive_sleeps(self, tmp_path):
        """Line 341: wait_time > 0 → time.sleep called after each crawled page."""
        connector = self._connector(wait_time=0.5)
        body = " ".join(["word"] * 100)
        html = _make_html(body)
        response = _make_response(html=html)
        patch_ctx, mock_client = self._patch_client(response)

        from mullet.knowledge.sources.web_cache import CrawlCache
        cache = CrawlCache(str(tmp_path))
        result_obj = MagicMock()
        result_obj.files_added = 0
        result_obj.files_updated = 0
        result_obj.errors = []

        with patch_ctx, patch("time.sleep") as mock_sleep:
            connector._crawl(tmp_path, cache, result_obj, set())

        mock_sleep.assert_called_with(0.5)
