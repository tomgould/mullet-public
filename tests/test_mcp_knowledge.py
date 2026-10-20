"""Tests for MCP knowledge tools (TRAG-compatible)."""
from unittest.mock import patch, MagicMock
from mullet.server import mcp_knowledge


def test_handle_list_pools_with_mock_dir():
    mock_dir = MagicMock()
    mock_dir.exists.return_value = True
    client1 = MagicMock()
    client1.is_dir.return_value = True
    client1.name = "test-client"
    # __truediv__ mock must return an object whose read_text() returns a real
    # string — otherwise yaml.safe_load(MagicMock()) enters an infinite read
    # loop that consumes all available RAM.
    yaml_mock = MagicMock()
    yaml_mock.exists.return_value = True
    yaml_mock.read_text.return_value = 'name: test-client\ndescription: A test client\n'
    client1.__truediv__ = lambda self, x: yaml_mock
    mock_dir.iterdir.return_value = [client1]
    with patch.object(mcp_knowledge, '_get_pools_dir', return_value=mock_dir):
        result = mcp_knowledge.handle_list_pools({})
    assert "test-client" in result


def test_handle_search_requires_query():
    result = mcp_knowledge.handle_search({"pool_name": "x"})
    assert "Error" in result


def test_handle_search_requires_client():
    mcp_knowledge._current_client = None
    result = mcp_knowledge.handle_search({"query": "test"})
    assert "Error" in result or "No client" in result


def test_handle_load_pool_requires_name():
    result = mcp_knowledge.handle_load_pool({})
    assert "Error" in result


def test_handle_memory_store_requires_key_content():
    result = mcp_knowledge.handle_memory_store({})
    assert "Error" in result


def test_handle_memory_search_requires_query():
    result = mcp_knowledge.handle_memory_search({})
    assert "Error" in result


def test_handle_load_pool_unknown_pool_returns_friendly_error(tmp_path):
    """Unknown pool names must produce a clean string error, not a traceback.

    Regression: earlier versions created a phantom dir for unknown names
    and reported 0 chunks. Now load_client raises ValueError with a
    helpful hint; the MCP handler wraps it into a user-facing string.
    """
    import yaml
    real = tmp_path / "real_pool"
    real.mkdir()
    (real / "config.yaml").write_text(yaml.safe_dump({"machine_name": "real_pool"}))
    # Clear any cached pools from prior tests
    mcp_knowledge._pool_cache._cache.clear()
    mcp_knowledge._pool_cache.current_name = None

    with patch.object(mcp_knowledge, '_get_pools_dir', return_value=tmp_path):
        result = mcp_knowledge.handle_load_pool({"pool_name": "Claude Code Ecosystem"})

    assert isinstance(result, str)
    assert "Traceback" not in result
    assert "Error" in result
    assert "Claude Code Ecosystem" in result
    assert "real_pool" in result  # available-pools list surfaced
    # Phantom dir must NOT have been created
    assert not (tmp_path / "Claude Code Ecosystem").exists()


def test_handle_memory_delete_requires_key():
    result = mcp_knowledge.handle_memory_delete({})
    assert "Error" in result


def test_handle_reindex_requires_client():
    result = mcp_knowledge.handle_reindex_pool({})
    assert "Error" in result


def test_handle_ingest_status_no_job():
    result = mcp_knowledge.handle_ingest_status({})
    assert "No ingestion job" in result


def test_handle_ingest_status_with_job():
    import time
    mcp_knowledge._jobs["test123"] = {
        "client": "test", "status": "running",
        "progress": 50, "total": 100, "chunks": 200,
        "started": time.time(),
    }
    result = mcp_knowledge.handle_ingest_status({"job_id": "test123"})
    assert "running" in result
    assert "50" in result
    del mcp_knowledge._jobs["test123"]


def test_handle_find_symbol_requires_name():
    result = mcp_knowledge.handle_find_symbol({})
    assert "Error" in result


def test_handle_search_code_requires_query():
    result = mcp_knowledge.handle_search_code({})
    assert "Error" in result


def test_handle_build_graph_requires_client():
    result = mcp_knowledge.handle_build_graph({})
    assert "Error" in result


def test_handle_connect_github_requires_owner():
    result = mcp_knowledge.handle_connect_github({"pool_name": "x"})
    assert "Error" in result


def test_handle_summarize_requires_content():
    result = mcp_knowledge.handle_summarize({})
    assert "Error" in result


def test_handler_dispatch_has_all_tools():
    for tool in mcp_knowledge.MCP_KNOWLEDGE_TOOLS:
        name = tool["name"]
        assert name in mcp_knowledge.MCP_KNOWLEDGE_HANDLERS, f"Missing handler for {name}"


def test_tool_count():
    assert len(mcp_knowledge.MCP_KNOWLEDGE_TOOLS) == 43
    assert len(mcp_knowledge.MCP_KNOWLEDGE_HANDLERS) == 43


def test_build_local_provider_returns_generate_interface():
    """Regression: _build_local_provider must return a provider with generate(), not create()."""
    from mullet.server.mcp_knowledge import _build_local_provider
    provider = _build_local_provider({"lmstudio_host": "http://localhost:1234", "lmstudio_model": "test"})
    assert hasattr(provider, "generate"), "provider missing generate() — knowledge engine HyDE/multi-query will fail"
    assert callable(provider.generate)
