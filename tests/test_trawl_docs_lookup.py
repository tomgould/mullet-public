"""Tests for TrawlDocsLookup and the trawl_docs_lookup MCP handler."""
import json
from unittest.mock import MagicMock, patch


from mullet.core.tools.trawl_docs_lookup import TrawlDocsLookup, DEFAULT_SOURCES


# ---------------------------------------------------------------------------
# TrawlDocsLookup tests
# ---------------------------------------------------------------------------


def test_trawl_docs_lookup_context7_success():
    """Context7 returns results → used directly."""
    mock_c7 = MagicMock()
    mock_c7.search_docs.return_value = [
        {"content": "react hooks docs", "library": "react", "library_id": "/react"}
    ]

    lookup = TrawlDocsLookup(context7_client=mock_c7)
    result = lookup.trawl_docs_lookup("react", "18.0", "hooks API")

    assert result["source"] == "context7"
    assert result["library"] == "react"
    assert result["version"] == "18.0"
    assert result["query"] == "hooks API"
    assert len(result["results"]) == 1
    mock_c7.search_docs.assert_called_once_with(library="react", version="18.0", query="hooks API")


def test_trawl_docs_lookup_context7_raises_falls_through():
    """Context7 raises → falls through to trawl (and then web if trawl empty)."""
    mock_c7 = MagicMock()
    mock_c7.search_docs.side_effect = RuntimeError("context7 down")

    lookup = TrawlDocsLookup(context7_client=mock_c7)

    with patch.object(lookup, "_search_trawl_pools", return_value=None):
        with patch("mullet.core.tools.trawl_docs_lookup.web_search", return_value="web results"):
            result = lookup.trawl_docs_lookup("react", "18.0", "hooks")

    assert result["source"] == "web"
    mock_c7.search_docs.assert_called_once()


def test_trawl_docs_lookup_all_fail_returns_none():
    """All sources fail → returns source=none."""
    mock_c7 = MagicMock()
    mock_c7.search_docs.return_value = []

    lookup = TrawlDocsLookup(context7_client=mock_c7)

    with patch.object(lookup, "_search_trawl_pools", return_value=None):
        with patch("mullet.core.tools.trawl_docs_lookup.web_search", return_value=""):
            result = lookup.trawl_docs_lookup("react", "18.0", "hooks")

    assert result["source"] == "none"
    assert result["results"] == []


def test_trawl_docs_lookup_skips_context7_when_sources_trawl_only():
    """sources=['trawl'] bypasses Context7 entirely."""
    mock_c7 = MagicMock()
    mock_c7.search_docs.return_value = [{"content": "c7"}]

    lookup = TrawlDocsLookup(
        context7_client=mock_c7, source_preference=["trawl", "web"]
    )

    with patch.object(lookup, "_search_trawl_pools", return_value=[{"content": "trawl"}]):
        result = lookup.trawl_docs_lookup("react", "18.0", "hooks")

    assert result["source"] == "trawl"
    mock_c7.search_docs.assert_not_called()


def test_trawl_docs_lookup_default_sources():
    """Default sources match expected order."""
    lookup = TrawlDocsLookup()
    assert lookup.source_preference == DEFAULT_SOURCES
    assert DEFAULT_SOURCES == ["context7", "trawl", "web"]


def test_context7_client_lazy_instantiation():
    """Context7Client is created lazily on first access."""
    lookup = TrawlDocsLookup(context7_client=None)
    assert lookup._context7_client is None

    mock_c7 = MagicMock()
    with patch("mullet.core.tools.trawl_docs_lookup.Context7Client", return_value=mock_c7):
        client = lookup.context7_client

    assert client is mock_c7
    assert lookup._context7_client is mock_c7


# ---------------------------------------------------------------------------
# MCP handler tests
# ---------------------------------------------------------------------------


def test_handle_trawl_docs_lookup_success():
    """Handler returns JSON string with results."""
    from mullet.server.mcp_knowledge import handle_trawl_docs_lookup

    mock_lookup = MagicMock()
    mock_lookup.trawl_docs_lookup.return_value = {
        "source": "context7",
        "library": "fastapi",
        "version": "latest",
        "query": "middleware",
        "results": [{"content": "docs"}],
    }

    with patch("mullet.server.mcp_knowledge._trawl_docs_lookup_instance", mock_lookup):
        result = handle_trawl_docs_lookup(
            {"library": "fastapi", "version": "latest", "query": "middleware"}
        )

    parsed = json.loads(result)
    assert parsed["source"] == "context7"
    assert parsed["library"] == "fastapi"
    mock_lookup.trawl_docs_lookup.assert_called_once_with(
        library="fastapi", version="latest", query="middleware"
    )


def test_handle_trawl_docs_lookup_missing_args():
    """Handler returns error when library or query missing."""
    from mullet.server.mcp_knowledge import handle_trawl_docs_lookup

    result = handle_trawl_docs_lookup({"library": "fastapi"})
    parsed = json.loads(result)
    assert "error" in parsed
    assert "library and query are required" in parsed["error"]


def test_handle_trawl_docs_lookup_custom_sources():
    """Handler passes custom sources to a new TrawlDocsLookup instance."""
    from mullet.server.mcp_knowledge import handle_trawl_docs_lookup

    mock_lookup = MagicMock()
    mock_lookup.context7_client = MagicMock()
    mock_lookup.trawl_docs_lookup.return_value = {"source": "web", "results": []}

    with patch("mullet.server.mcp_knowledge._trawl_docs_lookup_instance", mock_lookup):
        with patch("mullet.core.tools.trawl_docs_lookup.TrawlDocsLookup") as MockLookup:
            MockLookup.return_value = mock_lookup
            result = handle_trawl_docs_lookup(
                {
                    "library": "numpy",
                    "query": "array",
                    "sources": ["web"],
                }
            )

    MockLookup.assert_called_once_with(
        context7_client=mock_lookup.context7_client,
        source_preference=["web"],
    )
    parsed = json.loads(result)
    assert parsed["source"] == "web"


def test_handle_trawl_docs_lookup_exception():
    """Handler catches exceptions and returns JSON error."""
    from mullet.server.mcp_knowledge import handle_trawl_docs_lookup

    mock_lookup = MagicMock()
    mock_lookup.trawl_docs_lookup.side_effect = RuntimeError("boom")

    with patch("mullet.server.mcp_knowledge._trawl_docs_lookup_instance", mock_lookup):
        result = handle_trawl_docs_lookup(
            {"library": "x", "query": "y"}
        )

    parsed = json.loads(result)
    assert "error" in parsed
    assert "boom" in parsed["error"]


# ---------------------------------------------------------------------------
# auto_suggest_* tests
# ---------------------------------------------------------------------------


def test_auto_suggest_versions_returns_context7_versions():
    """auto_suggest_versions delegates to context7_client.get_versions."""
    mock_c7 = MagicMock()
    mock_c7.get_versions.return_value = ["18.0", "17.0", "16.0"]
    lookup = TrawlDocsLookup(context7_client=mock_c7)
    result = lookup.auto_suggest_versions("react")
    assert result == ["18.0", "17.0", "16.0"]
    mock_c7.get_versions.assert_called_once_with(library="react")


def test_auto_suggest_versions_returns_empty_when_context7_fails():
    """auto_suggest_versions returns [] when context7 raises."""
    mock_c7 = MagicMock()
    mock_c7.get_versions.side_effect = RuntimeError("c7 down")
    lookup = TrawlDocsLookup(context7_client=mock_c7)
    result = lookup.auto_suggest_versions("react")
    assert result == []


def test_auto_suggest_versions_returns_empty_when_no_client():
    """auto_suggest_versions returns [] when context7_client is None."""
    lookup = TrawlDocsLookup(context7_client=None)
    with patch("mullet.core.tools.trawl_docs_lookup.Context7Client", side_effect=RuntimeError("no c7")):
        result = lookup.auto_suggest_versions("react")
    assert result == []


def test_auto_suggest_libraries_returns_matches():
    """auto_suggest_libraries delegates to context7_client.search_libraries."""
    mock_c7 = MagicMock()
    mock_c7.search_libraries.return_value = ["react", "react-dom", "react-router"]
    lookup = TrawlDocsLookup(context7_client=mock_c7)
    result = lookup.auto_suggest_libraries("react")
    assert "react" in result
    mock_c7.search_libraries.assert_called_once_with(query="react")


def test_auto_suggest_libraries_returns_empty_on_error():
    """auto_suggest_libraries returns [] when context7 raises."""
    mock_c7 = MagicMock()
    mock_c7.search_libraries.side_effect = ValueError("bad query")
    lookup = TrawlDocsLookup(context7_client=mock_c7)
    result = lookup.auto_suggest_libraries("???")
    assert result == []


def test_trawl_docs_lookup_web_search_raises_returns_none():
    """web_search raising triggers the except branch (lines 87-88) → source=none."""
    mock_c7 = MagicMock()
    mock_c7.search_docs.return_value = []

    lookup = TrawlDocsLookup(context7_client=mock_c7)

    with patch.object(lookup, "_search_trawl_pools", return_value=None):
        with patch("mullet.core.tools.trawl_docs_lookup.web_search", side_effect=RuntimeError("net fail")):
            result = lookup.trawl_docs_lookup("react", "18", "hooks")

    assert result["source"] == "none"


def test_context7_client_setter():
    """context7_client setter (line 37) stores the injected client."""
    lookup = TrawlDocsLookup()
    mock_c7 = MagicMock()
    lookup.context7_client = mock_c7
    assert lookup._context7_client is mock_c7


def test_trawl_docs_lookup_trawl_pool_raises_continues_to_web():
    """When _search_trawl_pools raises, log + fall through to web (lines 74-75)."""
    mock_c7 = MagicMock()
    mock_c7.search_docs.return_value = []  # no context7 results

    lookup = TrawlDocsLookup(context7_client=mock_c7)

    with patch.object(lookup, "_search_trawl_pools", side_effect=RuntimeError("pool error")):
        with patch("mullet.core.tools.trawl_docs_lookup.web_search", return_value=["web hit"]):
            result = lookup.trawl_docs_lookup("react", "18", "hooks")

    assert result["source"] == "web"


def test_search_trawl_pools_import_error_returns_none():
    """ImportError in _search_trawl_pools returns None (lines 134-136)."""
    lookup = TrawlDocsLookup()
    with patch.dict("sys.modules", {
        "mullet.knowledge.engine.client": None,
        "mullet.knowledge.engine.search.searcher": None,
    }):
        result = lookup._search_trawl_pools("some query")
    assert result is None


def test_search_trawl_pools_no_pools_dir(tmp_path):
    """Missing pools dir returns None (lines 106-107)."""
    lookup = TrawlDocsLookup()
    nonexistent = tmp_path / "does_not_exist"

    with patch("mullet.core.env.get_pools_dir", return_value=nonexistent):
        result = lookup._search_trawl_pools("query")

    assert result is None


def test_search_trawl_pools_no_pools_with_index(tmp_path):
    """Pools dir exists but no indexed pools returns None (lines 113-114)."""
    lookup = TrawlDocsLookup()
    pools_dir = tmp_path / "pools"
    pools_dir.mkdir()
    (pools_dir / "mypool").mkdir()  # pool dir, but no 'index' subdir

    with patch("mullet.core.env.get_pools_dir", return_value=pools_dir):
        with patch("mullet.knowledge.engine.client.discover_clients", return_value=["mypool"]):
            result = lookup._search_trawl_pools("query")

    assert result is None


def test_search_trawl_pools_returns_results(tmp_path):
    """Pools with index return search results (lines 116-133)."""
    lookup = TrawlDocsLookup()
    pools_dir = tmp_path / "pools"
    pools_dir.mkdir()
    pool_dir = pools_dir / "mypool"
    pool_dir.mkdir()
    (pool_dir / "index").mkdir()

    mock_client = MagicMock()
    mock_client.index_path = pool_dir / "index"
    mock_client.machine_name = "mypool"
    mock_client.embedding_config = {}
    mock_client.tokenization_config = {}

    mock_searcher = MagicMock()
    mock_searcher.search.return_value = [{"content": "found it", "score": 0.9}]

    with patch("mullet.core.env.get_pools_dir", return_value=pools_dir):
        with patch("mullet.knowledge.engine.client.discover_clients", return_value=["mypool"]):
            with patch("mullet.knowledge.engine.client.load_client", return_value=mock_client):
                with patch(
                    "mullet.knowledge.engine.search.searcher.HybridSearcher",
                    return_value=mock_searcher,
                ):
                    result = lookup._search_trawl_pools("react hooks")

    assert result is not None
    assert result[0]["content"] == "found it"
    assert result[0]["_pool"] == "mypool"


def test_search_trawl_pools_individual_pool_error_is_logged(tmp_path):
    """Single pool error is caught per pool, continues (lines 130-131)."""
    lookup = TrawlDocsLookup()
    pools_dir = tmp_path / "pools"
    pools_dir.mkdir()
    pool_dir = pools_dir / "mypool"
    pool_dir.mkdir()
    (pool_dir / "index").mkdir()

    with patch("mullet.core.env.get_pools_dir", return_value=pools_dir):
        with patch("mullet.knowledge.engine.client.discover_clients", return_value=["mypool"]):
            with patch("mullet.knowledge.engine.client.load_client", side_effect=RuntimeError("load fail")):
                result = lookup._search_trawl_pools("query")

    assert result is None  # no results since pool failed
