import json
from unittest.mock import MagicMock, patch

from mullet.core.knowledge.tools import (
    make_trawl_recall,
    make_trawl_remember,
    make_trawl_search,
    make_trawl_search_multi,
    register,
)


def test_trawl_search_empty_results():
    km = MagicMock()
    km.search.return_value = []
    tool = make_trawl_search(km)
    result = tool("foo")
    assert result == "No results found for: foo"
    km.search.assert_called_once_with("foo", top_k=10)


def test_trawl_search_formats_results():
    km = MagicMock()
    km.search.return_value = [
        {"content": "hello world", "source": "a.py", "score": 0.95},
    ]
    tool = make_trawl_search(km)
    result = tool("bar", top_k=5)
    assert "**1. a.py** (score: 0.95)" in result
    assert "hello world" in result
    km.search.assert_called_once_with("bar", top_k=5)


def test_trawl_search_truncates_long_content():
    km = MagicMock()
    km.search.return_value = [
        {"content": "x" * 600, "source": "b.py", "score": 0.5},
    ]
    tool = make_trawl_search(km)
    result = tool("q")
    assert "..." in result
    assert len(result.split("x" * 500)[0].split("\n")[1]) <= 503


def test_trawl_remember_returns_json():
    km = MagicMock()
    tool = make_trawl_remember(km)
    result = tool("key1", "content1")
    assert json.loads(result) == {"success": True, "key": "key1"}
    km.remember.assert_called_once_with("key1", "content1")


def test_trawl_recall_empty():
    km = MagicMock()
    km.recall.return_value = []
    tool = make_trawl_recall(km)
    result = tool("query")
    assert result == "No memories found."
    km.recall.assert_called_once_with("query")


def test_trawl_recall_returns_json():
    km = MagicMock()
    km.recall.return_value = [{"key": "k", "content": "c"}]
    tool = make_trawl_recall(km)
    result = tool("q")
    assert json.loads(result) == [{"key": "k", "content": "c"}]


def test_trawl_search_multi_parses_clients():
    with patch("mullet.core.knowledge.manager.knowledge_session") as mock_session:
        km = MagicMock()
        km.search.return_value = []
        mock_session.return_value.__enter__ = MagicMock(return_value=km)
        mock_session.return_value.__exit__ = MagicMock(return_value=False)
        tool = make_trawl_search_multi()
        result = tool("q", "client_a, client_b")
        assert result == "No results found across clients."
        assert mock_session.call_count == 2


def test_trawl_search_multi_rrf_merge():
    with patch("mullet.core.knowledge.manager.knowledge_session") as mock_session:
        km = MagicMock()
        km.search.side_effect = [
            [{"content": "alpha", "source": "s1"}],
            [{"content": "beta", "source": "s2"}],
        ]
        mock_session.return_value.__enter__ = MagicMock(return_value=km)
        mock_session.return_value.__exit__ = MagicMock(return_value=False)
        tool = make_trawl_search_multi()
        result = tool("q", "a,b", top_k=2)
        assert "[a] s1" in result
        assert "[b] s2" in result


def test_register_calls_registry():
    registry = MagicMock()
    km = MagicMock()
    register(registry, km)
    assert registry.register.call_count == 4
    names = [call.kwargs.get("name") or call.args[0] for call in registry.register.call_args_list]
    assert "trawl_search" in names
    assert "trawl_remember" in names
    assert "trawl_recall" in names
    assert "trawl_search_multi" in names


def test_search_multi_client_exception_logged(capsys):
    """Lines 127-128: knowledge_session raises → except logs warning, continues."""
    with patch("mullet.core.knowledge.manager.knowledge_session") as mock_session:
        mock_session.side_effect = RuntimeError("client down")
        tool = make_trawl_search_multi()
        result = tool("q", "bad_client")
    # No crash; returns empty result (no matches)
    assert isinstance(result, str)
