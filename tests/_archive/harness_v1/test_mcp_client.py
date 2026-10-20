"""Tests for mullet.harness.mcp.client — handshake + correlation + error handling."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from mullet.harness.mcp.client import MCPClient
from mullet.harness.mcp.protocol import JsonRpcError, PROTOCOL_VERSION

pytestmark = [pytest.mark.harness, pytest.mark.asyncio]


class FakeTransport:
    """In-memory bidirectional transport with scripted responses."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self._inbox: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.started = False
        self.closed = False
        # Optional script: method -> result-dict OR callable(params) -> result-dict
        self._handlers: dict[str, Any] = {}

    def on(self, method: str, response: Any) -> None:
        self._handlers[method] = response

    async def start(self) -> None:
        self.started = True

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)
        method = message.get("method")
        # Notifications (no id) get no response.
        if "id" not in message:
            return
        if method in self._handlers:
            handler = self._handlers[method]
            result = handler(message.get("params", {})) if callable(handler) else handler
            await self._inbox.put(
                {"jsonrpc": "2.0", "id": message["id"], "result": result}
            )

    async def receive(self) -> dict[str, Any] | None:
        return await self._inbox.get()

    async def close(self) -> None:
        self.closed = True
        # Push EOF so reader_task can wake and exit.
        await self._inbox.put(None)

    async def push(self, frame: dict[str, Any] | None) -> None:
        await self._inbox.put(frame)


def _initialize_result() -> dict[str, Any]:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "serverInfo": {"name": "test-server", "version": "0.1"},
        "capabilities": {"tools": {}},
    }


async def test_client_initialize_handshake() -> None:
    t = FakeTransport()
    t.on("initialize", _initialize_result())
    client = MCPClient(t, name="mullet", version="1.0.0")
    await client.connect()
    try:
        assert client.initialized
        assert client.server_info == {"name": "test-server", "version": "0.1"}
        assert client.server_capabilities == {"tools": {}}
        # First send was the initialize request.
        first = t.sent[0]
        assert first["method"] == "initialize"
        assert first["params"]["protocolVersion"] == PROTOCOL_VERSION
        assert first["params"]["clientInfo"] == {"name": "mullet", "version": "1.0.0"}
    finally:
        await client.close()


async def test_client_list_tools_returns_tool_list() -> None:
    t = FakeTransport()
    t.on("initialize", _initialize_result())
    t.on(
        "tools/list",
        {"tools": [{"name": "read_file", "description": "Read a file"}]},
    )
    client = MCPClient(t)
    await client.connect()
    try:
        tools = await client.list_tools()
        assert tools == [{"name": "read_file", "description": "Read a file"}]
    finally:
        await client.close()


async def test_client_call_tool_round_trip() -> None:
    t = FakeTransport()
    t.on("initialize", _initialize_result())
    t.on(
        "tools/call",
        lambda params: {
            "content": [{"type": "text", "text": f"called {params['name']}"}],
            "isError": False,
        },
    )
    client = MCPClient(t)
    await client.connect()
    try:
        result = await client.call_tool("read_file", {"path": "/tmp/a"})
        assert result["isError"] is False
        assert result["content"][0]["text"] == "called read_file"
        # Confirm arguments propagated.
        last_request = [m for m in t.sent if m.get("method") == "tools/call"][-1]
        assert last_request["params"]["arguments"] == {"path": "/tmp/a"}
    finally:
        await client.close()


async def test_client_handles_jsonrpc_error() -> None:
    t = FakeTransport()
    t.on("initialize", _initialize_result())

    # Override send to push back an error frame for tools/list.
    original_send = t.send

    async def send(message: dict[str, Any]) -> None:
        await original_send(message)
        if message.get("method") == "tools/list":
            await t.push(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": {"code": -32601, "message": "Method not found"},
                }
            )

    t.send = send  # type: ignore[assignment]

    client = MCPClient(t)
    await client.connect()
    try:
        with pytest.raises(JsonRpcError) as exc_info:
            await client.list_tools()
        assert exc_info.value.code == -32601
        assert exc_info.value.message == "Method not found"
    finally:
        await client.close()


async def test_client_close_cancels_reader() -> None:
    t = FakeTransport()
    t.on("initialize", _initialize_result())
    client = MCPClient(t)
    await client.connect()
    reader = client._reader_task
    assert reader is not None
    assert not reader.done()
    await client.close()
    assert reader.done()
    assert t.closed is True


async def test_client_correlates_concurrent_requests() -> None:
    """Three concurrent calls; server replies out-of-order; all resolve correctly."""
    t = FakeTransport()
    t.on("initialize", _initialize_result())
    # Don't preregister method handlers; reply manually after capturing all 3 ids.

    client = MCPClient(t, request_timeout=5.0)
    await client.connect()
    try:
        # Pause auto-replies for "tools/call" by leaving handler unset; instead
        # we'll manually push responses to specific ids in reverse order.
        pending_calls = asyncio.gather(
            client.call_tool("a", {}),
            client.call_tool("b", {}),
            client.call_tool("c", {}),
        )

        # Wait until all 3 requests have been sent.
        async def _wait_for_three() -> list[int]:
            while True:
                ids = [
                    m["id"] for m in t.sent
                    if m.get("method") == "tools/call"
                ]
                if len(ids) >= 3:
                    return ids
                await asyncio.sleep(0.01)

        ids = await asyncio.wait_for(_wait_for_three(), timeout=3)
        # Reply out of order: c, a, b.
        await t.push(
            {"jsonrpc": "2.0", "id": ids[2], "result": {"label": "c"}}
        )
        await t.push(
            {"jsonrpc": "2.0", "id": ids[0], "result": {"label": "a"}}
        )
        await t.push(
            {"jsonrpc": "2.0", "id": ids[1], "result": {"label": "b"}}
        )

        a_res, b_res, c_res = await asyncio.wait_for(pending_calls, timeout=5)
        assert a_res == {"label": "a"}
        assert b_res == {"label": "b"}
        assert c_res == {"label": "c"}
    finally:
        await client.close()


async def test_client_list_resources_and_prompts() -> None:
    t = FakeTransport()
    t.on("initialize", _initialize_result())
    t.on("resources/list", {"resources": [{"uri": "file:///x"}]})
    t.on("prompts/list", {"prompts": [{"name": "greet"}]})
    t.on("resources/read", {"contents": [{"text": "hello"}]})

    client = MCPClient(t)
    await client.connect()
    try:
        assert await client.list_resources() == [{"uri": "file:///x"}]
        assert await client.list_prompts() == [{"name": "greet"}]
        read = await client.read_resource("file:///x")
        assert read["contents"][0]["text"] == "hello"
    finally:
        await client.close()


async def test_client_ping_round_trip() -> None:
    t = FakeTransport()
    t.on("initialize", _initialize_result())
    t.on("ping", {})
    client = MCPClient(t)
    await client.connect()
    try:
        await client.ping()
    finally:
        await client.close()
