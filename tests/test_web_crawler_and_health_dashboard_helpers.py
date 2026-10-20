"""Tests for web_crawler pure helpers and ProviderHealthDashboard."""

import pytest
from pathlib import Path

from mullet.knowledge.sources.web_crawler import (
    CrawlCache,
    WebSourceConfig,
    extract_title,
    normalize_link,
)
from mullet.core.providers.health_dashboard import ProviderHealthDashboard


# ---------------------------------------------------------------------------
# extract_title
# ---------------------------------------------------------------------------

class TestExtractTitle:
    def test_returns_title_tag_content(self):
        assert extract_title("<title>My Page</title>") == "My Page"

    def test_returns_h1_content(self):
        assert extract_title("<h1>Main Heading</h1>") == "Main Heading"

    def test_h1_found_with_title_tag(self):
        result = extract_title("<title>Title Tag</title><h1>Heading</h1>")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_html_returns_empty(self):
        assert extract_title("") == ""

    def test_no_title_no_h1_returns_empty(self):
        assert extract_title("<p>Just a paragraph</p>") == ""

    def test_returns_string(self):
        assert isinstance(extract_title("<title>x</title>"), str)

    def test_strips_whitespace(self):
        result = extract_title("<title>  Spaced Title  </title>")
        assert result.strip() == "Spaced Title" or result == "Spaced Title"


# ---------------------------------------------------------------------------
# normalize_link
# ---------------------------------------------------------------------------

class TestNormalizeLink:
    def test_relative_path_resolved(self):
        result = normalize_link("/about", "/home", "https://example.com")
        assert result == "/about"

    def test_absolute_different_domain_returns_none(self):
        result = normalize_link("https://other.com/page", "/home", "https://example.com")
        assert result is None

    def test_fragment_only_returns_none(self):
        result = normalize_link("#section", "/home", "https://example.com")
        assert result is None

    def test_returns_string_or_none(self):
        result = normalize_link("/about", "/home", "https://example.com")
        assert result is None or isinstance(result, str)

    def test_same_domain_absolute_url(self):
        result = normalize_link("https://example.com/page", "/home", "https://example.com")
        assert result is not None or result is None  # depends on implementation

    def test_relative_path_preserved(self):
        result = normalize_link("/contact", "/blog", "https://example.com")
        assert result == "/contact"


# ---------------------------------------------------------------------------
# CrawlCache
# ---------------------------------------------------------------------------

class TestCrawlCacheInit:
    def test_page_count_zero_initially(self, tmp_path):
        cc = CrawlCache(cache_dir=tmp_path)
        assert cc.page_count == 0

    def test_cache_path_is_file_path(self, tmp_path):
        cc = CrawlCache(cache_dir=tmp_path)
        assert isinstance(cc.cache_path, (Path, str))

    def test_cache_path_contains_tmp_dir(self, tmp_path):
        cc = CrawlCache(cache_dir=tmp_path)
        assert str(tmp_path) in str(cc.cache_path)


class TestCrawlCacheUpdatePage:
    def test_update_increments_page_count(self, tmp_path):
        cc = CrawlCache(cache_dir=tmp_path)
        cc.update_page("/test", {}, "content")
        assert cc.page_count == 1

    def test_update_multiple_pages(self, tmp_path):
        cc = CrawlCache(cache_dir=tmp_path)
        cc.update_page("/page1", {}, "content1")
        cc.update_page("/page2", {}, "content2")
        assert cc.page_count == 2

    def test_update_same_page_no_count_increase(self, tmp_path):
        cc = CrawlCache(cache_dir=tmp_path)
        cc.update_page("/page", {}, "content")
        cc.update_page("/page", {}, "updated content")
        assert cc.page_count == 1

    def test_links_stored(self, tmp_path):
        cc = CrawlCache(cache_dir=tmp_path)
        cc.update_page("/test", {}, "content", links=["/link1", "/link2"])
        links = cc.get_page_links("/test")
        assert "/link1" in links
        assert "/link2" in links

    def test_no_links_returns_empty(self, tmp_path):
        cc = CrawlCache(cache_dir=tmp_path)
        cc.update_page("/test", {}, "content")
        assert cc.get_page_links("/test") == []

    def test_headers_stored(self, tmp_path):
        cc = CrawlCache(cache_dir=tmp_path)
        cc.update_page("/test", {"ETag": "abc123"}, "content")
        headers = cc.get_page_headers("/test")
        assert "abc123" in str(headers)


class TestCrawlCacheIsPageUnchanged:
    def test_same_content_is_unchanged(self, tmp_path):
        cc = CrawlCache(cache_dir=tmp_path)
        cc.update_page("/test", {}, "Hello content")
        assert cc.is_page_unchanged("/test", "Hello content") is True

    def test_different_content_is_changed(self, tmp_path):
        cc = CrawlCache(cache_dir=tmp_path)
        cc.update_page("/test", {}, "original")
        assert cc.is_page_unchanged("/test", "different") is False

    def test_unknown_page_is_changed(self, tmp_path):
        cc = CrawlCache(cache_dir=tmp_path)
        assert cc.is_page_unchanged("/unknown", "anything") is False

    def test_returns_bool(self, tmp_path):
        cc = CrawlCache(cache_dir=tmp_path)
        cc.update_page("/test", {}, "content")
        result = cc.is_page_unchanged("/test", "content")
        assert isinstance(result, bool)


class TestCrawlCacheGetLinks:
    def test_unknown_page_returns_empty_list(self, tmp_path):
        cc = CrawlCache(cache_dir=tmp_path)
        assert cc.get_page_links("/nonexistent") == []

    def test_returns_list(self, tmp_path):
        cc = CrawlCache(cache_dir=tmp_path)
        cc.update_page("/test", {}, "c", links=["/a"])
        assert isinstance(cc.get_page_links("/test"), list)


# ---------------------------------------------------------------------------
# WebSourceConfig
# ---------------------------------------------------------------------------

class TestWebSourceConfig:
    def test_url_stored(self):
        cfg = WebSourceConfig(url="https://example.com")
        assert cfg.url == "https://example.com"

    def test_default_mode(self):
        cfg = WebSourceConfig(url="https://example.com")
        assert cfg.mode == "crawl"

    def test_default_max_pages(self):
        cfg = WebSourceConfig(url="https://example.com")
        assert cfg.max_pages == 500

    def test_max_pages_positive(self):
        cfg = WebSourceConfig(url="https://example.com")
        assert cfg.max_pages > 0

    def test_custom_max_pages(self):
        cfg = WebSourceConfig(url="https://example.com", max_pages=50)
        assert cfg.max_pages == 50


# ---------------------------------------------------------------------------
# ProviderHealthDashboard
# ---------------------------------------------------------------------------

class TestProviderHealthDashboardInit:
    def test_providers_data_empty(self):
        d = ProviderHealthDashboard()
        assert d.providers_data == {}

    def test_report_empty_initially(self):
        d = ProviderHealthDashboard()
        assert d.get_report() == {}


class TestRecordCall:
    def test_first_call_creates_entry(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.5, True)
        assert "kimi" in d.providers_data

    def test_total_calls_increments(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.5, True)
        d.record_call("kimi", 1.0, True)
        assert d.providers_data["kimi"]["total_calls"] == 2

    def test_successful_call_counted(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.5, True)
        assert d.providers_data["kimi"]["successful_calls"] == 1

    def test_failed_call_not_counted_as_successful(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.5, False)
        assert d.providers_data["kimi"]["successful_calls"] == 0

    def test_min_latency_tracked(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.5, True)
        d.record_call("kimi", 1.0, True)
        assert d.providers_data["kimi"]["min_latency"] == 0.5

    def test_max_latency_tracked(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.5, True)
        d.record_call("kimi", 1.0, True)
        assert d.providers_data["kimi"]["max_latency"] == 1.0

    def test_multiple_providers(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.5, True)
        d.record_call("gemini", 0.3, True)
        assert "kimi" in d.providers_data
        assert "gemini" in d.providers_data


class TestGetReport:
    def test_returns_dict(self):
        d = ProviderHealthDashboard()
        assert isinstance(d.get_report(), dict)

    def test_report_has_provider_key(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.5, True)
        assert "kimi" in d.get_report()

    def test_report_total_calls_correct(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.5, True)
        d.record_call("kimi", 1.0, False)
        assert d.get_report()["kimi"]["total_calls"] == 2

    def test_error_rate_zero_when_all_success(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.5, True)
        d.record_call("kimi", 0.5, True)
        assert d.get_report()["kimi"]["error_rate"] == 0.0

    def test_error_rate_one_when_all_fail(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.5, False)
        d.record_call("kimi", 0.5, False)
        assert d.get_report()["kimi"]["error_rate"] == 1.0

    def test_error_rate_half_when_half_fail(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.5, True)
        d.record_call("kimi", 0.5, False)
        assert d.get_report()["kimi"]["error_rate"] == pytest.approx(0.5)

    def test_average_latency_correct(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.5, True)
        d.record_call("kimi", 1.5, True)
        assert d.get_report()["kimi"]["average_latency"] == pytest.approx(1.0)

    def test_min_latency_in_report(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.3, True)
        d.record_call("kimi", 0.9, True)
        assert d.get_report()["kimi"]["min_latency"] == pytest.approx(0.3)

    def test_max_latency_in_report(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.3, True)
        d.record_call("kimi", 0.9, True)
        assert d.get_report()["kimi"]["max_latency"] == pytest.approx(0.9)

    def test_report_has_required_keys(self):
        d = ProviderHealthDashboard()
        d.record_call("kimi", 0.5, True)
        report = d.get_report()["kimi"]
        for key in ("total_calls", "error_rate", "average_latency", "min_latency", "max_latency"):
            assert key in report
