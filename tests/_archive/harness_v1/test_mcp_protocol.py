"""Tests for mullet.harness.mcp.protocol — JSON-RPC 2.0 envelopes and constants."""

from __future__ import annotations

import pytest

from mullet.harness.mcp.protocol import (
    INITIALIZE,
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    PING,
    PROMPTS_LIST,
    PROTOCOL_VERSION,
    RESOURCES_LIST,
    RESOURCES_READ,
    TOOLS_CALL,
    TOOLS_LIST,
)

pytestmark = pytest.mark.harness


def test_jsonrpc_request_serialises_correctly() -> None:
    req = JsonRpcRequest(id=1, method="initialize", params={"protocolVersion": PROTOCOL_VERSION})
    dumped = req.model_dump()
    assert dumped["jsonrpc"] == "2.0"
    assert dumped["id"] == 1
    assert dumped["method"] == "initialize"
    assert dumped["params"]["protocolVersion"] == PROTOCOL_VERSION


def test_jsonrpc_request_default_params_is_empty_dict() -> None:
    req = JsonRpcRequest(id="abc", method="ping")
    assert req.params == {}


def test_jsonrpc_response_round_trip() -> None:
    payload = {
        "jsonrpc": "2.0",
        "id": 7,
        "result": {"tools": [{"name": "read_file"}]},
    }
    resp = JsonRpcResponse(**payload)
    assert resp.id == 7
    assert resp.result == {"tools": [{"name": "read_file"}]}
    assert resp.error is None
    assert resp.model_dump()["result"]["tools"][0]["name"] == "read_file"


def test_jsonrpc_response_error_branch() -> None:
    resp = JsonRpcResponse(id=2, error={"code": -32601, "message": "Method not found"})
    assert resp.result is None
    assert resp.error is not None
    assert resp.error["code"] == -32601


def test_jsonrpc_error_carries_code_message_data() -> None:
    err = JsonRpcError(-32000, "Boom", data={"detail": "stack"})
    assert err.code == -32000
    assert err.message == "Boom"
    assert err.data == {"detail": "stack"}
    assert "JSON-RPC error -32000" in str(err)
    assert "Boom" in str(err)


def test_jsonrpc_error_data_optional() -> None:
    err = JsonRpcError(-1, "x")
    assert err.data is None


def test_protocol_version_constant_matches_spec() -> None:
    assert PROTOCOL_VERSION == "2024-11-05"


def test_method_name_constants() -> None:
    assert INITIALIZE == "initialize"
    assert TOOLS_LIST == "tools/list"
    assert TOOLS_CALL == "tools/call"
    assert RESOURCES_LIST == "resources/list"
    assert RESOURCES_READ == "resources/read"
    assert PROMPTS_LIST == "prompts/list"
    assert PING == "ping"
