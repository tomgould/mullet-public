"""Comprehensive tests for mullet.knowledge.sources.web_crawler."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
from mullet.knowledge.sources.web_crawler import (
    WebSourceConfig,
    WebSourceConnector,
    normalize_link,
    _is_in_scope,
    _extract_links,
    _url_to_filepath,
)
from mullet.knowledge.sources.base import SyncResult


class TestWebSourceConfig:
    def test_from_dict_defaults(self):
        c = WebSourceConfig.from_dict({})
        assert c.url == ""
        assert c.mode == "crawl"
        assert c.stay_in_path is True
        assert c.max_pages == 500
        assert c.wait_time == 1.0
        assert c.timeout == 30

    def test_from_dict_custom(self):
        c = WebSourceConfig.from_dict(
            {"url": "https://example.com", "mode": "docs", "max_pages": 100, "wait_time": 0.5}
        )
        assert c.url == "https://example.com"
        assert c.mode == "docs"
        assert c.max_pages == 100
        assert c.wait_time == 0.5

    def test_domain_strips_www(self):
        c = WebSourceConfig(url="https://www.example.com/docs")
        assert c.domain == "example.com"

    def test_domain_no_www(self):
        c = WebSourceConfig(url="https://example.com/docs")
        assert c.domain == "example.com"

    def test_base_url(self):
        c = WebSourceConfig(url="https://example.com/docs/path")
        assert c.base_url == "https://example.com"

    def test_starting_path(self):
        c = WebSourceConfig(url="https://example.com/docs/path")
        assert c.starting_path == "/docs/path"

    def test_starting_path_empty(self):
        c = WebSourceConfig(url="https://example.com")
        assert c.starting_path == "/"


class TestNormalizeLink:
    def test_empty_returns_none(self):
        assert normalize_link("", "/page", "https://example.com") is None

    def test_mailto_returns_none(self):
        assert normalize_link("mailto:test@example.com", "/page", "https://example.com") is None

    def test_tel_returns_none(self):
        assert normalize_link("tel:+1234567890", "/page", "https://example.com") is None

    def test_javascript_returns_none(self):
        assert normalize_link("javascript:void(0)", "/page", "https://example.com") is None

    def test_skip_extension_returns_none(self):
        assert normalize_link("image.jpg", "/page", "https://example.com") is None
        assert normalize_link("file.pdf", "/page", "https://example.com") is None
        assert normalize_link("style.css", "/page", "https://example.com") is None

    def test_external_http_returns_none(self):
        assert normalize_link("https://other.com/page", "/page", "https://example.com") is None
        assert normalize_link("http://other.com/page", "/page", "https://example.com") is None

    def test_absolute_same_domain_strips_base(self):
        result = normalize_link("https://example.com/about", "/page", "https://example.com")
        assert result == "/about"

    def test_relative_link_resolved(self):
        result = normalize_link("child", "/parent/page", "https://example.com")
        assert result == "/parent/child"

    def test_path_normalized(self):
        result = normalize_link("/a/../b", "/", "https://example.com")
        assert result == "/b"

    def test_query_and_fragment_stripped(self):
        result = normalize_link("/page?id=1", "/", "https://example.com")
        assert result == "/page"
        result = normalize_link("/page#section", "/", "https://example.com")
        assert result == "/page"
        result = normalize_link("/page?id=1#section", "/", "https://example.com")
        assert result == "/page"

    def test_root_slash_link(self):
        result = normalize_link("/", "/page", "https://example.com")
        assert result == "/"

    def test_anchors_only_returns_none(self):
        assert normalize_link("#section", "/page", "https://example.com") is None


class TestIsInScope:
    def test_no_stay_in_path_always_true(self):
        config = WebSourceConfig(url="https://example.com/docs", stay_in_path=False)
        assert _is_in_scope("/anything", config) is True

    def test_root_starting_path_true(self):
        config = WebSourceConfig(url="https://example.com/", stay_in_path=True)
        assert _is_in_scope("/anything", config) is True

    def test_prefix_match_true(self):
        config = WebSourceConfig(url="https://example.com/docs", stay_in_path=True)
        assert _is_in_scope("/docs", config) is True
        assert _is_in_scope("/docs/page1", config) is True
        assert _is_in_scope("/docs/sub/page", config) is True

    def test_out_of_scope_false(self):
        config = WebSourceConfig(url="https://example.com/docs", stay_in_path=True)
        assert _is_in_scope("/other", config) is False
        assert _is_in_scope("/doc", config) is False
        assert _is_in_scope("/docs", config) is True


class TestExtractLinks:
    def test_double_quotes(self):
        html = '<a href="/page1">link1</a><a href="/page2">link2</a>'
        links = _extract_links(html)
        assert "/page1" in links
        assert "/page2" in links

    def test_single_quotes(self):
        html = "<a href='/page1'>link1</a><a href='/page2'>link2</a>"
        links = _extract_links(html)
        assert "/page1" in links
        assert "/page2" in links

    def test_deduplicates(self):
        html = '<a href="/page1">link1</a><a href="/page1">link1 again</a>'
        links = _extract_links(html)
        assert links.count("/page1") == 1
        assert len(links) == 1

    def test_no_links(self):
        html = "<p>No links here</p>"
        links = _extract_links(html)
        assert links == []


class TestUrlToFilepath:
    def test_empty_path_is_index(self):
        assert _url_to_filepath("") == "index.md"
        assert _url_to_filepath("/") == "index.md"

    def test_path_separators_to_underscores(self):
        assert _url_to_filepath("/docs/getting-started") == "docs_getting-started.md"

    def test_adds_md_extension(self):
        assert _url_to_filepath("about").endswith(".md")

    def test_strips_slashes(self):
        assert _url_to_filepath("/path/") == "path.md"

    def test_unsafe_chars_replaced(self):
        assert _url_to_filepath("/path<bad>") == "path_bad_.md"
        assert _url_to_filepath('/path"quotes"') == "path_quotes_.md"
        assert _url_to_filepath("/path|pipe") == "path_pipe.md"


class TestWebSourceConnector:
    def _make_config(self, **kwargs):
        defaults = {"url": "https://example.com/docs", "wait_time": 0, "max_pages": 500}
        defaults.update(kwargs)
        return WebSourceConfig(**defaults)

    def test_status_returns_dict(self):
        c = WebSourceConnector(self._make_config())
        s = c.status()
        assert s["source_type"] == "web"
        assert "url" in s
        assert s["url"] == "https://example.com/docs"
        assert s["domain"] == "example.com"

    def test_sync_crawl_error(self, tmp_path):
        config = self._make_config()
        connector = WebSourceConnector(config)

        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.HTTPError("Connection refused")

        with patch("mullet.knowledge.sources.web_crawler.httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            result = connector.sync(str(tmp_path))

        assert result.source_type == "web"
        assert len(result.errors) == 1
        assert "Connection refused" in result.errors[0]

    def test_sync_success(self, tmp_path):
        config = self._make_config()
        connector = WebSourceConnector(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = "<html><body>Hello</body></html>"

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        with patch("mullet.knowledge.sources.web_crawler.httpx.Client") as MockClient, \
             patch("mullet.knowledge.sources.web_crawler.html_to_markdown", return_value="# Hello\n\nContent"), \
             patch("mullet.knowledge.sources.web_crawler.extract_title", return_value="Hello"):
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            result = connector.sync(str(tmp_path))

        assert result.files_added == 1
        assert result.errors == []

        web_dir = tmp_path / "web" / "example.com"
        assert (web_dir / "docs.md").exists()
        content = (web_dir / "docs.md").read_text()
        assert "Hello" in content
        assert "source: web_crawler" in content

    def test_crawl_skips_visited(self, tmp_path):
        config = self._make_config()
        connector = WebSourceConnector(config)

        html = '<a href="/docs">self</a>'
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = html

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        result = SyncResult(source_type="web")
        cache = MagicMock()
        cache.get_page_headers.return_value = {}
        cache.is_page_unchanged.return_value = False

        with patch("mullet.knowledge.sources.web_crawler.httpx.Client") as MockClient, \
             patch("mullet.knowledge.sources.web_crawler.html_to_markdown", return_value="content"), \
             patch("mullet.knowledge.sources.web_crawler.extract_title", return_value="Title"):
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            pages = connector._crawl(tmp_path, cache, result)

        # Starting path /docs is visited once; self-link /docs is skipped as visited
        assert pages == 1
        assert mock_client.get.call_count == 1

    def test_crawl_304_reuses_cached_links(self, tmp_path):
        config = self._make_config()
        connector = WebSourceConnector(config)

        mock_response = MagicMock()
        mock_response.status_code = 304
        mock_response.headers = {}
        mock_response.text = ""

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        result = SyncResult(source_type="web")
        cache = MagicMock()
        cache.get_page_headers.return_value = {"If-None-Match": "abc"}
        cache.get_page_links.return_value = ["/docs/page1", "/docs/page2"]

        with patch("mullet.knowledge.sources.web_crawler.httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            pages = connector._crawl(tmp_path, cache, result)

        assert pages == 0
        cache.get_page_links.assert_any_call("/docs")

    def test_crawl_500_records_error(self, tmp_path):
        config = self._make_config()
        connector = WebSourceConnector(config)

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = "Error"

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        result = SyncResult(source_type="web")
        cache = MagicMock()
        cache.get_page_headers.return_value = {}

        with patch("mullet.knowledge.sources.web_crawler.httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            pages = connector._crawl(tmp_path, cache, result)

        assert pages == 0
        assert len(result.errors) == 1
        assert "server error (500)" in result.errors[0]

    def test_crawl_429_records_error(self, tmp_path):
        config = self._make_config()
        connector = WebSourceConnector(config)

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = "Rate limited"

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        result = SyncResult(source_type="web")
        cache = MagicMock()
        cache.get_page_headers.return_value = {}

        with patch("mullet.knowledge.sources.web_crawler.httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            pages = connector._crawl(tmp_path, cache, result)

        assert pages == 0
        assert len(result.errors) == 1
        assert "rate limited (429)" in result.errors[0]

    def test_crawl_non_html_content_type_skipped(self, tmp_path):
        config = self._make_config()
        connector = WebSourceConnector(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = "{}"

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        result = SyncResult(source_type="web")
        cache = MagicMock()
        cache.get_page_headers.return_value = {}

        with patch("mullet.knowledge.sources.web_crawler.httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            pages = connector._crawl(tmp_path, cache, result)

        assert pages == 0
        assert result.errors == []

    def test_crawl_unchanged_content_reuses_links(self, tmp_path):
        config = self._make_config()
        connector = WebSourceConnector(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = "<html>same</html>"

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        result = SyncResult(source_type="web")
        cache = MagicMock()
        cache.get_page_headers.return_value = {}
        cache.is_page_unchanged.return_value = True
        cache.get_page_links.return_value = ["/docs/other"]

        with patch("mullet.knowledge.sources.web_crawler.httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            pages = connector._crawl(tmp_path, cache, result)

        assert pages == 0
        cache.get_page_links.assert_any_call("/docs")
        assert not (tmp_path / "docs.md").exists()

    def test_crawl_max_pages_limit(self, tmp_path):
        config = self._make_config(max_pages=2)
        connector = WebSourceConnector(config)

        html = '<a href="/docs/1">1</a><a href="/docs/2">2</a><a href="/docs/3">3</a>'
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = html

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        result = SyncResult(source_type="web")
        cache = MagicMock()
        cache.get_page_headers.return_value = {}
        cache.is_page_unchanged.return_value = False

        with patch("mullet.knowledge.sources.web_crawler.httpx.Client") as MockClient, \
             patch("mullet.knowledge.sources.web_crawler.html_to_markdown", return_value="content"), \
             patch("mullet.knowledge.sources.web_crawler.extract_title", return_value="Title"):
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            pages = connector._crawl(tmp_path, cache, result)

        assert pages == 2

    def test_crawl_files_updated_for_existing(self, tmp_path):
        config = self._make_config()
        connector = WebSourceConnector(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = "<html>update</html>"

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        result = SyncResult(source_type="web")
        cache = MagicMock()
        cache.get_page_headers.return_value = {}
        cache.is_page_unchanged.return_value = False

        with patch("mullet.knowledge.sources.web_crawler.httpx.Client") as MockClient, \
             patch("mullet.knowledge.sources.web_crawler.html_to_markdown", return_value="content"), \
             patch("mullet.knowledge.sources.web_crawler.extract_title", return_value="Title"):
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            pages = connector._crawl(tmp_path, cache, result, existing_files={"docs.md"})

        assert pages == 1
        assert result.files_updated == 1
        assert result.files_added == 0

    def test_crawl_skips_non_content_paths(self, tmp_path):
        config = self._make_config(url="https://example.com/login")
        connector = WebSourceConnector(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = "<html>login</html>"

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        result = SyncResult(source_type="web")
        cache = MagicMock()
        cache.get_page_headers.return_value = {}

        with patch("mullet.knowledge.sources.web_crawler.httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            pages = connector._crawl(tmp_path, cache, result)

        assert pages == 0
        assert mock_client.get.called is False

    def test_crawl_skips_out_of_scope(self, tmp_path):
        config = self._make_config(url="https://example.com/docs", stay_in_path=True)
        connector = WebSourceConnector(config)

        html = '<a href="/other/page">out</a>'
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = html

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        result = SyncResult(source_type="web")
        cache = MagicMock()
        cache.get_page_headers.return_value = {}
        cache.is_page_unchanged.return_value = False

        with patch("mullet.knowledge.sources.web_crawler.httpx.Client") as MockClient, \
             patch("mullet.knowledge.sources.web_crawler.html_to_markdown", return_value="content"), \
             patch("mullet.knowledge.sources.web_crawler.extract_title", return_value="Title"):
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            pages = connector._crawl(tmp_path, cache, result)

        assert pages == 1
        # /other/page should be skipped as out of scope
        assert mock_client.get.call_count == 1  # only the starting /docs page

    def test_crawl_empty_markdown_updates_cache(self, tmp_path):
        config = self._make_config()
        connector = WebSourceConnector(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = "<html>nothing</html>"

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        result = SyncResult(source_type="web")
        cache = MagicMock()
        cache.get_page_headers.return_value = {}
        cache.is_page_unchanged.return_value = False

        with patch("mullet.knowledge.sources.web_crawler.httpx.Client") as MockClient, \
             patch("mullet.knowledge.sources.web_crawler.html_to_markdown", return_value=""), \
             patch("mullet.knowledge.sources.web_crawler.extract_title", return_value="Title"):
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            pages = connector._crawl(tmp_path, cache, result)

        assert pages == 0
        cache.update_page.assert_called_once()
        assert not (tmp_path / "docs.md").exists()
