"""Tests for web search and fetch_page tools."""

from mullet.core.tools.web import _html_to_text, _strip_html


def test_strip_html():
    assert _strip_html("<b>bold</b>") == "bold"
    assert _strip_html("<a href='x'>link</a>") == "link"


def test_html_to_text():
    html = "<html><body><p>Hello</p><p>World</p></body></html>"
    text = _html_to_text(html)
    assert "Hello" in text
    assert "World" in text


def test_html_to_text_strips_scripts():
    html = "<script>alert('xss')</script><p>Content</p>"
    text = _html_to_text(html)
    assert "alert" not in text
    assert "Content" in text


def test_html_to_text_strips_styles():
    html = "<style>.x{color:red}</style><p>Visible</p>"
    text = _html_to_text(html)
    assert "color" not in text
    assert "Visible" in text


def test_html_entities():
    text = _html_to_text("&amp; &lt; &gt; &quot; &#39; &nbsp;")
    assert "&" in text
    assert "<" in text
    assert ">" in text


import json
from unittest.mock import patch, MagicMock
from mullet.core.tools.web import (
    web_search, fetch_page, _ddg_html_search, _ddg_api_search,
    register, WEB_SEARCH_SCHEMA, FETCH_PAGE_SCHEMA,
)


def test_web_search_html_results():
    with patch("mullet.core.tools.web._ddg_html_search", return_value=[
        {"title": "T1", "snippet": "S1", "url": "http://example.com/1"},
    ]):
        result = web_search("query", max_results=1)
    assert "T1" in result
    assert "S1" in result
    assert "http://example.com/1" in result


def test_web_search_fallback_api():
    with patch("mullet.core.tools.web._ddg_html_search", return_value=[]), \
         patch("mullet.core.tools.web._ddg_api_search", return_value=[
             {"title": "T2", "snippet": "S2", "url": "http://example.com/2"},
         ]):
        result = web_search("query", max_results=1)
    assert "T2" in result


def test_web_search_no_results():
    with patch("mullet.core.tools.web._ddg_html_search", return_value=[]), \
         patch("mullet.core.tools.web._ddg_api_search", return_value=[]):
        result = web_search("query", max_results=1)
    assert "No results found" in result


def test_web_search_exception():
    with patch("mullet.core.tools.web._ddg_html_search", side_effect=Exception("boom")):
        result = web_search("query", max_results=1)
    assert "[error]" in result


def test_fetch_page_html():
    html = "<html><body><p>Hello World</p></body></html>"
    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Type": "text/html"}
    mock_resp.read.return_value = html.encode("utf-8")
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_resp)
    mock_cm.__exit__ = MagicMock(return_value=False)
    with patch("urllib.request.urlopen", return_value=mock_cm), \
         patch("mullet.core.tools.url_safety.is_safe_url", return_value=True):
        result = fetch_page("http://example.com")
    assert "Hello World" in result


def test_fetch_page_text():
    text = "plain text content"
    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Type": "text/plain"}
    mock_resp.read.return_value = text.encode("utf-8")
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_resp)
    mock_cm.__exit__ = MagicMock(return_value=False)
    with patch("urllib.request.urlopen", return_value=mock_cm), \
         patch("mullet.core.tools.url_safety.is_safe_url", return_value=True):
        result = fetch_page("http://example.com")
    assert "plain text content" in result


def test_fetch_page_truncation():
    text = "x" * 15_000
    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Type": "text/plain"}
    mock_resp.read.return_value = text.encode("utf-8")
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_resp)
    mock_cm.__exit__ = MagicMock(return_value=False)
    with patch("urllib.request.urlopen", return_value=mock_cm), \
         patch("mullet.core.tools.url_safety.is_safe_url", return_value=True):
        result = fetch_page("http://example.com", max_chars=10000)
    assert "[...truncated at 10000 chars]" in result


def test_fetch_page_blocked():
    with patch("mullet.core.tools.url_safety.is_safe_url", return_value=False):
        result = fetch_page("http://internal.local")
    assert "blocked by SSRF protection" in result


def test_fetch_page_exception():
    with patch("urllib.request.urlopen", side_effect=Exception("connection error")), \
         patch("mullet.core.tools.url_safety.is_safe_url", return_value=True):
        result = fetch_page("http://example.com")
    assert "[error]" in result


def test_ddg_html_search():
    html = (
        '<a class="result__a" href="http://example.com/1">Title 1</a>'
        '<a class="result__snippet">Snippet 1</a>'
    )
    mock_resp = MagicMock()
    mock_resp.read.return_value = html.encode("utf-8")
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_resp)
    mock_cm.__exit__ = MagicMock(return_value=False)
    with patch("urllib.request.urlopen", return_value=mock_cm):
        result = _ddg_html_search("query", max_results=1)
    assert len(result) == 1
    assert result[0]["title"] == "Title 1"
    assert result[0]["snippet"] == "Snippet 1"
    assert result[0]["url"] == "http://example.com/1"


def test_ddg_html_search_uddg():
    html = (
        '<a class="result__a" href="/l/?uddg=http%3A%2F%2Fexample.com%2F1">Title 1</a>'
        '<a class="result__snippet">Snippet 1</a>'
    )
    mock_resp = MagicMock()
    mock_resp.read.return_value = html.encode("utf-8")
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_resp)
    mock_cm.__exit__ = MagicMock(return_value=False)
    with patch("urllib.request.urlopen", return_value=mock_cm):
        result = _ddg_html_search("query", max_results=1)
    assert result[0]["url"] == "http://example.com/1"


def test_ddg_api_search():
    data = {
        "AbstractText": "Abstract",
        "Heading": "Heading",
        "AbstractURL": "http://example.com/abstract",
        "RelatedTopics": [
            {"Text": "Related text", "FirstURL": "http://example.com/related"},
        ],
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(data).encode("utf-8")
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_resp)
    mock_cm.__exit__ = MagicMock(return_value=False)
    with patch("urllib.request.urlopen", return_value=mock_cm):
        result = _ddg_api_search("query", max_results=2)
    assert len(result) == 2
    assert result[0]["title"] == "Heading"
    assert result[1]["title"] == "Related text"


def test_ddg_api_search_no_abstract():
    data = {"RelatedTopics": []}
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(data).encode("utf-8")
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_resp)
    mock_cm.__exit__ = MagicMock(return_value=False)
    with patch("urllib.request.urlopen", return_value=mock_cm):
        result = _ddg_api_search("query", max_results=2)
    assert result == []


def test_register_web_tools():
    registry = MagicMock()
    register(registry)
    assert registry.register.call_count == 2
