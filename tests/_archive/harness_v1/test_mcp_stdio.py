"""Tests for mullet.harness.mcp.transports.stdio — subprocess JSON-RPC transport."""

from __future__ import annotations

import asyncio
import sys

import pytest

from mullet.harness.mcp.transports.stdio import StdioTransport

pytestmark = [pytest.mark.harness, pytest.mark.asyncio]


# A tiny echo "MCP server": reads one line of JSON, echoes it back, then exits.
ECHO_SERVER = (
    "import sys, json;"
    "line = sys.stdin.readline();"
    "msg = json.loads(line);"
    "msg['echoed'] = True;"
    "sys.stdout.write(json.dumps(msg) + chr(10));"
    "sys.stdout.flush()"
)

# A server that immediately closes stdout by exiting silently.
CLOSE_SERVER = "import sys; sys.exit(0)"

# A long-running server that just sleeps so we can test close()/terminate().
SLEEP_SERVER = "import time; time.sleep(60)"


async def test_stdio_transport_spawns_subprocess() -> None:
    t = StdioTransport(sys.executable, ["-c", SLEEP_SERVER])
    await t.start()
    try:
        assert t.is_alive
    finally:
        await t.close()
    assert not t.is_alive


async def test_stdio_send_and_receive_round_trip() -> None:
    t = StdioTransport(sys.executable, ["-c", ECHO_SERVER])
    await t.start()
    try:
        await t.send({"hello": "world", "n": 1})
        msg = await asyncio.wait_for(t.receive(), timeout=5)
        assert msg is not None
        assert msg["hello"] == "world"
        assert msg["n"] == 1
        assert msg["echoed"] is True
    finally:
        await t.close()


async def test_stdio_close_terminates_process() -> None:
    t = StdioTransport(sys.executable, ["-c", SLEEP_SERVER])
    await t.start()
    assert t.is_alive
    await t.close()
    assert not t.is_alive


async def test_stdio_handles_eof() -> None:
    t = StdioTransport(sys.executable, ["-c", CLOSE_SERVER])
    await t.start()
    try:
        # Server exits without writing → readline returns b'' → receive returns None.
        msg = await asyncio.wait_for(t.receive(), timeout=5)
        assert msg is None
    finally:
        await t.close()


async def test_stdio_is_alive_lifecycle() -> None:
    t = StdioTransport(sys.executable, ["-c", SLEEP_SERVER])
    assert not t.is_alive
    await t.start()
    assert t.is_alive
    await t.close()
    assert not t.is_alive


async def test_stdio_send_before_start_raises() -> None:
    t = StdioTransport(sys.executable, ["-c", "pass"])
    with pytest.raises(RuntimeError):
        await t.send({"x": 1})


async def test_stdio_receive_before_start_raises() -> None:
    t = StdioTransport(sys.executable, ["-c", "pass"])
    with pytest.raises(RuntimeError):
        await t.receive()


async def test_stdio_close_idempotent() -> None:
    t = StdioTransport(sys.executable, ["-c", SLEEP_SERVER])
    await t.start()
    await t.close()
    # Second close should not raise.
    await t.close()
