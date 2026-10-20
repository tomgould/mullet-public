"""Tests for TrawlDocsLookup — updated for real Context7Client implementation."""

from unittest.mock import MagicMock, patch

from mullet.core.tools.trawl_docs_lookup import TrawlDocsLookup

# Patch Context7Client at module level in tests that don't need it,
# to prevent lazy instantiation from spawning a real npx process.
_NO_C7 = patch('mullet.core.tools.trawl_docs_lookup.Context7Client', side_effect=RuntimeError("no c7"))


def test_lookup_with_web_fallback():
    """Web source calls web_search and returns source='web'."""
    with _NO_C7, patch('mullet.core.tools.trawl_docs_lookup.web_search') as mock_web:
        mock_web.return_value = [{"title": "T", "url": "https://x.com", "snippet": "s"}]
        lookup = TrawlDocsLookup(context7_client=None, source_preference=["web"])
        result = lookup.trawl_docs_lookup("requests", "2.31.0", "authentication")

    assert result["source"] == "web"
    assert result["library"] == "requests"
    assert result["version"] == "2.31.0"
    mock_web.assert_called_once_with(query="requests 2.31.0 authentication", max_results=3)


def test_lookup_with_trawl_fallback():
    """Trawl source calls _search_trawl_pools and returns source='trawl'."""
    with _NO_C7:
        lookup = TrawlDocsLookup(context7_client=None, source_preference=["trawl"])
        with patch.object(lookup, '_search_trawl_pools', return_value=[{"content": "docs", "_pool": "p"}]):
            result = lookup.trawl_docs_lookup("requests", "2.31.0", "authentication")

    assert result["source"] == "trawl"
    assert result["library"] == "requests"
    assert len(result["results"]) == 1


def test_lookup_returns_empty_when_all_fail():
    """Returns source='none' when all sources fail."""
    with _NO_C7, patch('mullet.core.tools.trawl_docs_lookup.web_search') as mock_web:
        mock_web.side_effect = Exception("Network error")
        lookup = TrawlDocsLookup(context7_client=None, source_preference=["web"])
        result = lookup.trawl_docs_lookup("nonexistent", "1.0.0", "docs")

    assert result["source"] == "none"
    assert result["results"] == []


def test_auto_suggest_versions_without_context7():
    """auto_suggest_versions returns empty list when Context7 unavailable."""
    with _NO_C7:
        lookup = TrawlDocsLookup(context7_client=None, source_preference=["web"])
        versions = lookup.auto_suggest_versions("requests")
    assert versions == []


def test_lookup_with_context7_success():
    """Successful lookup using a mock context7 client."""
    mock_c7 = MagicMock()
    mock_c7.search_docs.return_value = [{"title": "R", "url": "https://c7.example.com"}]
    lookup = TrawlDocsLookup(context7_client=mock_c7, source_preference=["context7"])
    result = lookup.trawl_docs_lookup("fastapi", "0.104.0", "middleware")

    assert result["source"] == "context7"
    assert result["library"] == "fastapi"
    mock_c7.search_docs.assert_called_once_with(library="fastapi", version="0.104.0", query="middleware")


def test_context7_failure_falls_back_to_web():
    """Context7 exception falls through to web source."""
    mock_c7 = MagicMock()
    mock_c7.search_docs.side_effect = Exception("Context7 unavailable")
    lookup = TrawlDocsLookup(context7_client=mock_c7, source_preference=["context7", "web"])
    with patch('mullet.core.tools.trawl_docs_lookup.web_search') as mock_web:
        mock_web.return_value = [{"snippet": "auth guide"}]
        result = lookup.trawl_docs_lookup("requests", "2.0", "auth")

    assert result["source"] == "web"
    assert len(result["results"]) == 1


def test_auto_suggest_libraries_no_client():
    with _NO_C7:
        lookup = TrawlDocsLookup(context7_client=None, source_preference=["web"])
        assert lookup.auto_suggest_libraries("http client") == []


def test_auto_suggest_libraries_with_client():
    mock_c7 = MagicMock()
    mock_c7.search_libraries.return_value = ["requests", "httpx"]
    lookup = TrawlDocsLookup(context7_client=mock_c7)
    assert "httpx" in lookup.auto_suggest_libraries("http client")
