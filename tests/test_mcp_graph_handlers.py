"""Tests for graph MCP handlers requiring-field validation."""


def test_handle_graph_stats_missing_pool():
    from mullet.server.mcp_knowledge import handle_graph_stats
    result = handle_graph_stats({})
    assert "Error" in result


def test_handle_graph_prune_missing_pool():
    from mullet.server.mcp_knowledge import handle_graph_prune
    result = handle_graph_prune({})
    assert "Error" in result


def test_handle_search_entities_missing_query():
    from mullet.server.mcp_knowledge import handle_search_entities
    result = handle_search_entities({"pool_name": "mypool"})
    assert "Error" in result and "query" in result.lower()


def test_handle_search_entities_missing_pool():
    from mullet.server.mcp_knowledge import handle_search_entities
    result = handle_search_entities({"query": "test"})
    assert "Error" in result


def test_handle_search_relationships_missing_entity():
    from mullet.server.mcp_knowledge import handle_search_relationships
    result = handle_search_relationships({"pool_name": "mypool"})
    assert "Error" in result and "entity" in result.lower()


def test_handle_graph_export_missing_pool():
    from mullet.server.mcp_knowledge import handle_graph_export
    result = handle_graph_export({})
    assert "Error" in result


def test_handle_ingest_file_missing_path():
    from mullet.server.mcp_knowledge import handle_ingest_file
    result = handle_ingest_file({"pool_name": "mypool"})
    assert "Error" in result


def test_handle_ingest_url_missing_url():
    from mullet.server.mcp_knowledge import handle_ingest_url
    result = handle_ingest_url({"pool_name": "mypool"})
    assert "Error" in result


def test_handle_list_sources_missing_pool():
    from mullet.server.mcp_knowledge import handle_list_sources
    result = handle_list_sources({})
    assert "Error" in result and "pool_name" in result.lower()


def test_handle_sync_sources_missing_pool():
    from mullet.server.mcp_knowledge import handle_sync_sources
    result = handle_sync_sources({})
    assert "Error" in result and "pool_name" in result.lower()


def test_handle_sync_sources_nonexistent_pool():
    from mullet.server.mcp_knowledge import handle_sync_sources
    result = handle_sync_sources({"pool_name": "pool_that_does_not_exist_xyz"})
    assert "Error" in result


def test_handle_search_code_missing_query():
    from mullet.server.mcp_knowledge import handle_search_code
    result = handle_search_code({"pool_name": "mypool"})
    assert "Error" in result and "query" in result.lower()


def test_handle_search_code_missing_pool():
    from mullet.server.mcp_knowledge import handle_search_code
    result = handle_search_code({"query": "test"})
    assert "Error" in result


def test_handle_search_multi_missing_query():
    from mullet.server.mcp_knowledge import handle_search_multi
    result = handle_search_multi({"clients": ["mypool"]})
    assert "Error" in result and "query" in result.lower()
