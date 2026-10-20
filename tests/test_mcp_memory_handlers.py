"""Tests for memory + graph MCP handlers in mcp_knowledge."""

from unittest.mock import MagicMock, patch


def _mock_store():
    s = MagicMock()
    s.search.return_value = []
    s.list.return_value = []
    return s


# --- memory_store ---

def test_memory_store_success():
    store = _mock_store()
    with patch("mullet.server.mcp_knowledge._get_memory_store", return_value=store):
        from mullet.server.mcp_knowledge import handle_memory_store
        result = handle_memory_store({"key": "mykey", "content": "myval"})
    assert "mykey" in result
    store.store.assert_called_once()


def test_memory_store_with_ttl():
    store = _mock_store()
    with patch("mullet.server.mcp_knowledge._get_memory_store", return_value=store):
        from mullet.server.mcp_knowledge import handle_memory_store
        result = handle_memory_store({"key": "k", "content": "v", "ttl": "60"})
    assert "TTL: 60s" in result


# --- memory_search ---

def test_memory_search_no_results():
    store = _mock_store()
    with patch("mullet.server.mcp_knowledge._get_memory_store", return_value=store):
        from mullet.server.mcp_knowledge import handle_memory_search
        result = handle_memory_search({"query": "anything"})
    assert "No memories" in result


def test_memory_search_with_results():
    store = _mock_store()
    store.search.return_value = [{"key": "k1", "namespace": "mcp", "content": "hello world"}]
    with patch("mullet.server.mcp_knowledge._get_memory_store", return_value=store):
        from mullet.server.mcp_knowledge import handle_memory_search
        result = handle_memory_search({"query": "hello"})
    assert "k1" in result


# --- memory_list ---

def test_memory_list_empty():
    store = _mock_store()
    with patch("mullet.server.mcp_knowledge._get_memory_store", return_value=store):
        from mullet.server.mcp_knowledge import handle_memory_list
        result = handle_memory_list({})
    assert "No memories" in result


def test_memory_list_with_records():
    store = _mock_store()
    store.list.return_value = [{"namespace": "mcp", "key": "key1", "content": "val1", "updated_at": 1700000000}]
    with patch("mullet.server.mcp_knowledge._get_memory_store", return_value=store):
        from mullet.server.mcp_knowledge import handle_memory_list
        result = handle_memory_list({})
    assert "key1" in result


# --- memory_delete ---

def test_memory_delete_success():
    store = _mock_store()
    with patch("mullet.server.mcp_knowledge._get_memory_store", return_value=store):
        from mullet.server.mcp_knowledge import handle_memory_delete
        result = handle_memory_delete({"key": "mykey"})
    assert "mykey" in result
    store.delete.assert_called_once()
