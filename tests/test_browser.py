"""Tests for browser tool."""

from mullet.core.tools.browser import browse, _is_safe_url, _extract_links, _extract_element


def test_safe_url_public():
    assert _is_safe_url("https://example.com") is True
    assert _is_safe_url("https://google.com") is True


def test_unsafe_url_localhost():
    assert _is_safe_url("http://localhost:8080") is False
    assert _is_safe_url("http://127.0.0.1:3000") is False


def test_unsafe_url_private():
    assert _is_safe_url("http://10.0.0.1") is False
    assert _is_safe_url("http://192.168.1.1") is False
    assert _is_safe_url("http://172.16.0.1") is False


def test_unsafe_url_metadata():
    assert _is_safe_url("http://169.254.169.254") is False


def test_extract_links():
    html = '<a class="result__a" href="https://example.com">Example</a>'
    result = _extract_links(html, "https://base.com")
    assert "example.com" in result


def test_extract_element_title():
    html = "<html><title>Test Page</title></html>"
    result = _extract_element(html, "title")
    assert result == "Test Page"


def test_extract_element_meta():
    html = '<meta name="description" content="A test page">'
    result = _extract_element(html, "meta:description")
    assert "test page" in result


def test_extract_element_h1():
    html = "<h1>Header One</h1><h1>Header Two</h1>"
    result = _extract_element(html, "h1")
    assert "Header One" in result
    assert "Header Two" in result


def test_browse_blocked_url():
    result = browse("http://localhost:8080", action="read")
    assert "blocked" in result.lower()
