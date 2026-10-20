# Quick tests for handle_prompts_search_live, handle_prompts_find, handle_prompts_get
# in mullet/server/mcp_knowledge

from unittest.mock import patch


def test_handle_prompts_search_live_missing_query():
    from mullet.server.mcp_knowledge import handle_prompts_search_live
    result = handle_prompts_search_live({})
    assert "Error" in result and "query" in result.lower()


def test_handle_prompts_find_missing_query():
    from mullet.server.mcp_knowledge import handle_prompts_find
    result = handle_prompts_find({})
    assert "Error" in result and "query" in result.lower()


def test_handle_prompts_get_missing_id():
    from mullet.server.mcp_knowledge import handle_prompts_get
    result = handle_prompts_get({})
    assert "Error" in result and "prompt_id" in result.lower()


def test_handle_prompts_search_live_network_error():
    from mullet.server.mcp_knowledge import handle_prompts_search_live
    with patch("urllib.request.urlopen", side_effect=Exception("network error")):
        result = handle_prompts_search_live({"query": "test"})
    assert "error" in result.lower()


def test_handle_prompts_get_network_error():
    from mullet.server.mcp_knowledge import handle_prompts_get
    with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
        result = handle_prompts_get({"prompt_id": "123"})
    assert "error" in result.lower()
