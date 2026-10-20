"""Tests for web_crawler pure helpers: WebSourceConfig, normalize_link, _is_in_scope, _extract_links, _url_to_filepath."""


from mullet.knowledge.sources.web_crawler import (
    WebSourceConfig,
    normalize_link,
    _is_in_scope,
    _extract_links,
    _url_to_filepath,
    SKIP_EXTENSIONS,
    NON_CONTENT_PATHS,
)


# ---------------------------------------------------------------------------
# WebSourceConfig defaults
# ---------------------------------------------------------------------------

class TestWebSourceConfigDefaults:
    def test_mode_default_crawl(self):
        assert WebSourceConfig().mode == "crawl"

    def test_stay_in_path_default_true(self):
        assert WebSourceConfig().stay_in_path is True

    def test_max_pages_default_500(self):
        assert WebSourceConfig().max_pages == 500

    def test_wait_time_default(self):
        assert WebSourceConfig().wait_time > 0

    def test_timeout_default_30(self):
        assert WebSourceConfig().timeout == 30


class TestWebSourceConfigFromDict:
    def test_url_set(self):
        c = WebSourceConfig.from_dict({"url": "https://example.com"})
        assert c.url == "https://example.com"

    def test_mode_set(self):
        c = WebSourceConfig.from_dict({"mode": "docs"})
        assert c.mode == "docs"

    def test_max_pages_set(self):
        c = WebSourceConfig.from_dict({"max_pages": 100})
        assert c.max_pages == 100

    def test_defaults_when_empty_dict(self):
        c = WebSourceConfig.from_dict({})
        assert c.mode == "crawl"
        assert c.max_pages == 500


class TestWebSourceConfigProperties:
    def test_domain_strips_www(self):
        c = WebSourceConfig(url="https://www.example.com/docs")
        assert c.domain == "example.com"

    def test_domain_without_www(self):
        c = WebSourceConfig(url="https://docs.example.com")
        assert c.domain == "docs.example.com"

    def test_base_url(self):
        c = WebSourceConfig(url="https://example.com/docs/guide")
        assert c.base_url == "https://example.com"

    def test_starting_path_with_path(self):
        c = WebSourceConfig(url="https://example.com/docs")
        assert c.starting_path == "/docs"

    def test_starting_path_root(self):
        c = WebSourceConfig(url="https://example.com/")
        assert c.starting_path == "/"


# ---------------------------------------------------------------------------
# normalize_link
# ---------------------------------------------------------------------------

class TestNormalizeLink:
    BASE = "https://example.com"

    def test_returns_string_or_none(self):
        result = normalize_link("/page", "/", self.BASE)
        assert result is None or isinstance(result, str)

    def test_absolute_path_returned(self):
        assert normalize_link("/docs/guide", "/", self.BASE) == "/docs/guide"

    def test_relative_link_resolved(self):
        result = normalize_link("about", "/docs/", self.BASE)
        assert result is not None
        assert "about" in result

    def test_external_link_returns_none(self):
        assert normalize_link("https://other.com/page", "/", self.BASE) is None

    def test_mailto_returns_none(self):
        assert normalize_link("mailto:user@example.com", "/", "") is None

    def test_tel_returns_none(self):
        assert normalize_link("tel:+1234567890", "/", "") is None

    def test_javascript_returns_none(self):
        assert normalize_link("javascript:void(0)", "/", "") is None

    def test_empty_link_returns_none(self):
        assert normalize_link("", "/", self.BASE) is None

    def test_anchor_only_returns_none(self):
        assert normalize_link("#section", "/page/", self.BASE) is None

    def test_pdf_returns_none(self):
        assert normalize_link("/docs/file.pdf", "/", self.BASE) is None

    def test_image_returns_none(self):
        assert normalize_link("/images/logo.png", "/", self.BASE) is None

    def test_same_domain_full_url_stripped(self):
        result = normalize_link("https://example.com/docs", "/", self.BASE)
        assert result == "/docs"

    def test_query_stripped(self):
        result = normalize_link("/page?q=1&p=2", "/", self.BASE)
        assert "?" not in result

    def test_hash_stripped(self):
        result = normalize_link("/page#anchor", "/", self.BASE)
        assert "#" not in result

    def test_starts_with_slash(self):
        result = normalize_link("/docs/guide", "/", self.BASE)
        assert result.startswith("/")


# ---------------------------------------------------------------------------
# _is_in_scope
# ---------------------------------------------------------------------------

class TestIsInScope:
    def test_stay_in_path_false_always_in_scope(self):
        config = WebSourceConfig(url="https://example.com/docs", stay_in_path=False)
        assert _is_in_scope("/anything/here", config) is True

    def test_root_start_path_always_in_scope(self):
        config = WebSourceConfig(url="https://example.com/", stay_in_path=True)
        assert _is_in_scope("/any/path", config) is True

    def test_exact_match_in_scope(self):
        config = WebSourceConfig(url="https://example.com/docs", stay_in_path=True)
        assert _is_in_scope("/docs", config) is True

    def test_sub_path_in_scope(self):
        config = WebSourceConfig(url="https://example.com/docs", stay_in_path=True)
        assert _is_in_scope("/docs/guide", config) is True

    def test_different_path_out_of_scope(self):
        config = WebSourceConfig(url="https://example.com/docs", stay_in_path=True)
        assert _is_in_scope("/blog/post", config) is False

    def test_partial_prefix_out_of_scope(self):
        config = WebSourceConfig(url="https://example.com/docs", stay_in_path=True)
        assert _is_in_scope("/documents", config) is False


# ---------------------------------------------------------------------------
# _extract_links
# ---------------------------------------------------------------------------

class TestExtractLinks:
    def test_returns_list(self):
        assert isinstance(_extract_links(""), list)

    def test_empty_html_returns_empty(self):
        assert _extract_links("") == []

    def test_extracts_href_links(self):
        html = '<a href="/page1">link</a><a href="/page2">link2</a>'
        links = _extract_links(html)
        assert "/page1" in links
        assert "/page2" in links

    def test_double_quoted_hrefs(self):
        links = _extract_links('<a href="/page">text</a>')
        assert "/page" in links

    def test_single_quoted_hrefs(self):
        links = _extract_links("<a href='/page'>text</a>")
        assert "/page" in links

    def test_no_duplicate_links(self):
        html = '<a href="/page">1</a><a href="/page">2</a>'
        links = _extract_links(html)
        assert links.count("/page") == 1

    def test_no_links_returns_empty(self):
        assert _extract_links("<p>No links here</p>") == []


# ---------------------------------------------------------------------------
# _url_to_filepath
# ---------------------------------------------------------------------------

class TestUrlToFilepath:
    def test_returns_string(self):
        assert isinstance(_url_to_filepath("/page"), str)

    def test_root_returns_index_md(self):
        assert _url_to_filepath("/") == "index.md"

    def test_ends_with_md(self):
        result = _url_to_filepath("/docs/guide")
        assert result.endswith(".md")

    def test_slashes_replaced_with_underscores(self):
        result = _url_to_filepath("/docs/guide")
        assert "/" not in result

    def test_simple_path(self):
        assert _url_to_filepath("/about") == "about.md"

    def test_nested_path(self):
        result = _url_to_filepath("/docs/api/reference")
        assert "docs" in result
        assert "api" in result
        assert "reference" in result

    def test_already_md_no_double_extension(self):
        result = _url_to_filepath("/page.md")
        assert result.count(".md") == 1


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestWebCrawlerConstants:
    def test_skip_extensions_compiled(self):
        assert hasattr(SKIP_EXTENSIONS, "search")

    def test_skip_extensions_matches_pdf(self):
        assert SKIP_EXTENSIONS.search("/file.pdf")

    def test_skip_extensions_matches_image(self):
        assert SKIP_EXTENSIONS.search("/logo.png")

    def test_skip_extensions_no_match_html(self):
        assert not SKIP_EXTENSIONS.search("/page.html")

    def test_non_content_paths_compiled(self):
        assert hasattr(NON_CONTENT_PATHS, "search")

    def test_non_content_paths_matches_login(self):
        assert NON_CONTENT_PATHS.search("/login")

    def test_non_content_paths_matches_admin(self):
        assert NON_CONTENT_PATHS.search("/admin/")


class TestNormalizeLinkLeadingSlash:
    def test_relative_link_from_shallow_path_gets_slash_prefix(self):
        """Line 125: normpath('page.html') has no leading slash → '/' prepended."""
        result = normalize_link("page.html", "/docs", "https://example.com")
        assert result is not None
        assert result.startswith("/")


class TestUrlToFilepathRootPath:
    def test_root_path_becomes_index(self):
        """Line 167: url_path='/' → strip → '' → path='index'."""
        assert _url_to_filepath("/") == "index.md"
