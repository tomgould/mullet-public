"""Tests for MCP server and handlers."""

import json
from mullet.server.mcp import handle_request
from mullet.server.mcp_handlers import (
    MCP_TOOLS, MCP_HANDLERS,
)


def test_mcp_initialize():
    resp = handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert resp["result"]["protocolVersion"] == "2024-11-05"
    assert resp["result"]["serverInfo"]["name"] == "mullet"


def test_mcp_tools_list():
    resp = handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    tools = resp["result"]["tools"]
    assert len(tools) >= 8  # Mullet tools + knowledge/trag tools
    names = [t["name"] for t in tools]
    assert "mullet_run" in names
    assert "mullet_search" in names
    assert "mullet_analyze" in names
    assert "mullet_remember" in names


def test_mcp_ping():
    resp = handle_request({"jsonrpc": "2.0", "id": 3, "method": "ping", "params": {}})
    assert resp["result"] == {}


def test_mcp_unknown_method():
    resp = handle_request({"jsonrpc": "2.0", "id": 4, "method": "bogus", "params": {}})
    assert "error" in resp


def test_mcp_tool_call_status():
    resp = handle_request({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                           "params": {"name": "mullet_status", "arguments": {}}})
    text = resp["result"]["content"][0]["text"]
    data = json.loads(text)
    assert data["status"] == "ok"


def test_mcp_tool_call_providers():
    resp = handle_request({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                           "params": {"name": "mullet_providers", "arguments": {}}})
    text = resp["result"]["content"][0]["text"]
    data = json.loads(text)
    assert "providers" in data


def test_mcp_tool_call_analyze():
    resp = handle_request({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                           "params": {"name": "mullet_analyze", "arguments": {
                               "task": "refactor the auth module for OAuth2"
                           }}})
    text = resp["result"]["content"][0]["text"]
    data = json.loads(text)
    assert data["tier"] in ("STANDARD", "ADVANCED", "CRITICAL")
    assert data["complexity_score"] > 30
    assert data["signals"]["architecture_keywords"] is True


def test_mcp_tool_call_analyze_simple():
    resp = handle_request({"jsonrpc": "2.0", "id": 8, "method": "tools/call",
                           "params": {"name": "mullet_analyze", "arguments": {
                               "task": "list files"
                           }}})
    text = resp["result"]["content"][0]["text"]
    data = json.loads(text)
    assert data["is_simple"] is True
    assert data["tier"] == "LOW"


def test_mcp_tool_call_unknown():
    resp = handle_request({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                           "params": {"name": "nonexistent", "arguments": {}}})
    assert resp["result"]["isError"] is True


def test_mcp_handlers_registered():
    assert len(MCP_HANDLERS) == 10
    assert all(callable(h) for h in MCP_HANDLERS.values())


def test_mcp_tools_have_schemas():
    for tool in MCP_TOOLS:
        assert "name" in tool
        assert "description" in tool
        assert "inputSchema" in tool
        assert tool["inputSchema"]["type"] == "object"
