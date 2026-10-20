"""Tests for credential_guard pure helpers and web_cache pure functions."""


from mullet.core.tools.credential_guard import (
    CredentialGuard,
    CREDENTIAL_FILENAMES,
    CREDENTIAL_EXTENSIONS,
)
from mullet.knowledge.sources.web_cache import (
    _get_header,
    CrawlCache,
    CACHE_FILENAME,
)


# ---------------------------------------------------------------------------
# _get_header
# ---------------------------------------------------------------------------

class TestGetHeader:
    def test_lowercase_key_found(self):
        headers = {"content-type": "text/html"}
        assert _get_header(headers, "content-type") == "text/html"

    def test_case_insensitive_lookup(self):
        headers = {"Content-Type": "text/html"}
        assert _get_header(headers, "content-type") == "text/html"

    def test_missing_key_returns_empty(self):
        headers = {"x-foo": "bar"}
        assert _get_header(headers, "etag") == ""

    def test_empty_headers_returns_empty(self):
        assert _get_header({}, "etag") == ""

    def test_returns_string(self):
        assert isinstance(_get_header({"a": "b"}, "a"), str)


# ---------------------------------------------------------------------------
# CrawlCache
# ---------------------------------------------------------------------------

class TestCrawlCache:
    def test_page_count_starts_zero(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        assert cache.page_count == 0

    def test_update_page_increases_count(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/page", {}, "content", ["/a", "/b"])
        assert cache.page_count == 1

    def test_get_page_links_returns_links(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/page", {}, "content", ["/a", "/b"])
        links = cache.get_page_links("/page")
        assert "/a" in links
        assert "/b" in links

    def test_get_page_links_unknown_page_returns_empty(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        assert cache.get_page_links("/unknown") == []

    def test_get_page_headers_with_etag(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/page", {"etag": "abc123"}, "content")
        headers = cache.get_page_headers("/page")
        assert headers.get("If-None-Match") == "abc123"

    def test_get_page_headers_unknown_returns_empty(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        assert cache.get_page_headers("/unknown") == {}

    def test_is_page_unchanged_same_content(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/page", {}, "hello world")
        assert cache.is_page_unchanged("/page", "hello world") is True

    def test_is_page_unchanged_different_content(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/page", {}, "hello world")
        assert cache.is_page_unchanged("/page", "different content") is False

    def test_is_page_unchanged_unknown_page(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        assert cache.is_page_unchanged("/unknown", "content") is False

    def test_save_and_load(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        cache.update_page("/page", {"etag": "v1"}, "hello", ["/link"])
        cache.save()

        cache2 = CrawlCache(str(tmp_path))
        cache2.load()
        assert cache2.page_count == 1
        assert cache2.get_page_links("/page") == ["/link"]

    def test_load_missing_file_ok(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        cache.load()  # Should not raise
        assert cache.page_count == 0

    def test_cache_path_uses_constant(self, tmp_path):
        cache = CrawlCache(str(tmp_path))
        assert cache.cache_path.endswith(CACHE_FILENAME)


# ---------------------------------------------------------------------------
# CredentialGuard
# ---------------------------------------------------------------------------

class TestCredentialGuard:
    def setup_method(self):
        self.guard = CredentialGuard()

    def test_env_file_is_credential(self):
        assert self.guard.is_credential_file(".env") is True

    def test_pem_extension_is_credential(self):
        assert self.guard.is_credential_file("server.pem") is True

    def test_id_rsa_is_credential(self):
        assert self.guard.is_credential_file("id_rsa") is True

    def test_regular_py_not_credential(self):
        assert self.guard.is_credential_file("main.py") is False

    def test_readme_not_credential(self):
        assert self.guard.is_credential_file("README.md") is False

    def test_warn_if_credential_env_returns_warning(self):
        result = self.guard.warn_if_credential("/home/user/.env")
        assert result is not None
        assert "WARNING" in result

    def test_warn_if_credential_py_returns_none(self):
        assert self.guard.warn_if_credential("main.py") is None

    def test_scan_content_clean_text_returns_empty(self):
        result = self.guard.scan_content("Hello world, no secrets here")
        assert result == []

    def test_scan_content_github_pat_detected(self):
        token = "ghp_" + "A" * 36
        result = self.guard.scan_content(f"TOKEN={token}")
        assert len(result) > 0

    def test_scan_content_openai_key_detected(self):
        key = "sk-" + "a" * 40
        result = self.guard.scan_content(f"key={key}")
        assert len(result) > 0

    def test_scan_content_aws_access_key_detected(self):
        result = self.guard.scan_content("AKIA1234567890ABCDEF")
        assert len(result) > 0

    def test_scan_content_private_key_header_detected(self):
        result = self.guard.scan_content("-----BEGIN RSA PRIVATE KEY-----")
        assert len(result) > 0

    def test_scan_content_returns_list(self):
        assert isinstance(self.guard.scan_content("text"), list)

    def test_credential_filenames_nonempty(self):
        assert len(CREDENTIAL_FILENAMES) > 0

    def test_credential_extensions_nonempty(self):
        assert len(CREDENTIAL_EXTENSIONS) > 0
