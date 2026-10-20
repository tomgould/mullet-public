"""Tests for key mullet_* MCP handlers — missing field validation."""
import json


def test_handle_mullet_remember_missing_key():
    from mullet.server.mcp_handlers import handle_mullet_remember
    result = json.loads(handle_mullet_remember({"content": "c", "client": "p"}))
    assert "error" in result


def test_handle_mullet_remember_missing_client():
    from mullet.server.mcp_handlers import handle_mullet_remember
    result = json.loads(handle_mullet_remember({"key": "k", "content": "c"}))
    assert "error" in result


def test_handle_mullet_analyze_missing_task():
    from mullet.server.mcp_handlers import handle_mullet_analyze
    result = json.loads(handle_mullet_analyze({}))
    assert "error" in result


def test_handle_mullet_analyze_with_task():
    from mullet.server.mcp_handlers import handle_mullet_analyze
    result = json.loads(handle_mullet_analyze({"task": "summarise this document"}))
    assert "tier" in result or "complexity" in result or "task" in result


def test_handle_mullet_budget_returns_dict():
    from mullet.server.mcp_handlers import handle_mullet_budget
    result = json.loads(handle_mullet_budget({}))
    assert isinstance(result, dict)


def test_handle_mullet_status_returns_ok():
    from mullet.server.mcp_handlers import handle_mullet_status
    result = json.loads(handle_mullet_status({}))
    assert "status" in result
    assert result["status"] == "ok"
