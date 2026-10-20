"""Tests for web crawler utility functions — normalize, scope, link extraction."""

from mullet.knowledge.sources.web_crawler import (
    normalize_link, _is_in_scope, _extract_links, _url_to_filepath, WebSourceConfig,
)


class TestNormalizeLink:
    def test_absolute_same_domain(self):
        result = normalize_link("https://docs.example.com/api", "/", "https://docs.example.com")
        assert result == "/api"

    def test_relative_link(self):
        result = normalize_link("/guide/intro", "/", "https://docs.example.com")
        assert result == "/guide/intro"

    def test_relative_from_path(self):
        result = normalize_link("overview", "/guide/", "https://docs.example.com")
        assert result == "/guide/overview"

    def test_external_link_returns_none(self):
        result = normalize_link("https://other.com/page", "/", "https://docs.example.com")
        assert result is None

    def test_mailto_returns_none(self):
        result = normalize_link("mailto:hello@example.com", "/", "https://docs.example.com")
        assert result is None

    def test_javascript_returns_none(self):
        result = normalize_link("javascript:void(0)", "/", "https://docs.example.com")
        assert result is None

    def test_empty_link_returns_none(self):
        assert normalize_link("", "/", "https://docs.example.com") is None
        assert normalize_link(None, "/", "https://docs.example.com") is None

    def test_skip_extension_pdf_returns_none(self):
        result = normalize_link("/doc/file.pdf", "/", "https://docs.example.com")
        assert result is None

    def test_fragment_stripped(self):
        result = normalize_link("/page#section", "/", "https://docs.example.com")
        assert result == "/page"

    def test_query_string_stripped(self):
        result = normalize_link("/search?q=test", "/", "https://docs.example.com")
        assert result == "/search"


class TestIsInScope:
    def _cfg(self, starting_path: str, stay_in_path: bool = True) -> WebSourceConfig:
        return WebSourceConfig(
            url=f"https://example.com{starting_path}",
            stay_in_path=stay_in_path,
        )

    def test_exact_match_is_in_scope(self):
        cfg = self._cfg("/docs")
        assert _is_in_scope("/docs", cfg) is True

    def test_subpath_is_in_scope(self):
        cfg = self._cfg("/docs")
        assert _is_in_scope("/docs/api", cfg) is True

    def test_sibling_path_out_of_scope(self):
        cfg = self._cfg("/docs")
        assert _is_in_scope("/blog", cfg) is False

    def test_stay_in_path_false_allows_all(self):
        cfg = self._cfg("/docs", stay_in_path=False)
        assert _is_in_scope("/anywhere", cfg) is True


class TestExtractLinks:
    def test_extracts_href_links(self):
        html = '<a href="/page1">link1</a> <a href="/page2">link2</a>'
        links = _extract_links(html)
        assert "/page1" in links
        assert "/page2" in links

    def test_deduplicates_links(self):
        html = '<a href="/page">1</a> <a href="/page">2</a>'
        links = _extract_links(html)
        assert links.count("/page") == 1

    def test_empty_html_returns_empty(self):
        assert _extract_links("") == []

    def test_no_links_returns_empty(self):
        assert _extract_links("<p>Hello world</p>") == []


class TestUrlToFilepath:
    def test_root_becomes_index(self):
        assert _url_to_filepath("/") == "index.md"

    def test_path_becomes_underscored(self):
        result = _url_to_filepath("/docs/api/intro")
        assert result == "docs_api_intro.md"

    def test_adds_md_extension(self):
        result = _url_to_filepath("/page")
        assert result.endswith(".md")

    def test_already_has_md_no_double(self):
        result = _url_to_filepath("/page.md")
        assert result.endswith(".md")
        assert not result.endswith(".md.md")
