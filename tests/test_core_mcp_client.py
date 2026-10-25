"""Tests for mullet/core/tools/mcp_client.py — async and connection paths."""

import asyncio
import sys
import time
import warnings
from concurrent.futures import Future
from contextlib import AsyncExitStack
from unittest.mock import MagicMock, patch

try:
    from unittest.mock import AsyncMock
except ImportError:  # pragma: no cover
    from unittest.mock import MagicMock as AsyncMock


from mullet.core.tools.mcp_client import (
    MCPClient,
    MCPServerConfig,
    MCPTool,
    MCPCallResult,
    Context7Client,
)

warnings.filterwarnings("ignore", message="coroutine .* was never awaited")


# ── Import-error branch (lines 37-38) ──────────────────────────────────────

def test_import_error_sets_mcp_available_false():
    """Cover the except ImportError branch that sets _MCP_AVAILABLE = False.

    Runs in a subprocess to avoid polluting the current interpreter state.
    """
    import subprocess

    code = (
        "import sys\n"
        "for k in list(sys.modules.keys()):\n"
        "    if k == 'mcp' or k.startswith('mcp.') or k == 'mullet.core.tools.mcp_client':\n"
        "        del sys.modules[k]\n"
        "sys.modules['mcp'] = None\n"
        "import mullet.core.tools.mcp_client as mod\n"
        "assert mod._MCP_AVAILABLE is False, f'_MCP_AVAILABLE={mod._MCP_AVAILABLE}'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ── MCPClient background loop / _ensure_loop ───────────────────────────────

def test_ensure_loop_creates_thread():
    client = MCPClient()
    client._ensure_loop()
    assert client._loop is not None
    assert client._thread is not None
    assert client._thread.is_alive()
    client.disconnect_all()


def test_ensure_loop_idempotent():
    client = MCPClient()
    client._ensure_loop()
    loop1 = client._loop
    thread1 = client._thread
    client._ensure_loop()
    assert client._loop is loop1
    assert client._thread is thread1
    client.disconnect_all()


# ── _async_connect (lines 227-258) ─────────────────────────────────────────

class _FakeAsyncCtx:
    """Async context manager compatible with AsyncExitStack.enter_async_context."""

    def __init__(self, read, write):
        self._read = read
        self._write = write

    async def __aenter__(self):
        return (self._read, self._write)

    async def __aexit__(self, *args):
        pass


def test_async_connect_discovers_tools():
    client = MCPClient()

    mock_tool = MagicMock(spec=["name", "description", "inputSchema"])
    mock_tool.name = "read_file"
    mock_tool.description = "Read a file"
    mock_tool.inputSchema = {"type": "object"}

    mock_response = MagicMock()
    mock_response.tools = [mock_tool]

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=mock_response)

    def mock_stdio_client(params):
        return _FakeAsyncCtx(MagicMock(), MagicMock())

    with patch("mullet.core.tools.mcp_client.StdioServerParameters") as _mock_params, \
         patch("mullet.core.tools.mcp_client.stdio_client", side_effect=mock_stdio_client), \
         patch("mullet.core.tools.mcp_client.ClientSession", return_value=mock_session):
        config = MCPServerConfig(name="fs", command="npx", args=["-y", "pkg"])
        tools = asyncio.run(client._async_connect(config))

    assert len(tools) == 1
    assert tools[0].name == "fs__read_file"
    assert "fs__read_file" in client._tools
    assert client._exit_stacks["fs"] is not None

    # Clean up exit stacks so they don't leak across tests.
    for stack in list(client._exit_stacks.values()):
        asyncio.run(stack.aclose())


def test_async_connect_no_inputSchema():
    """Tool without inputSchema gets an empty dict default."""
    client = MCPClient()

    mock_tool = MagicMock(spec=["name", "description"])
    mock_tool.name = "list"
    mock_tool.description = None
    # Deliberately no inputSchema attribute

    mock_response = MagicMock()
    mock_response.tools = [mock_tool]

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=mock_response)

    def mock_stdio_client(params):
        return _FakeAsyncCtx(MagicMock(), MagicMock())

    with patch("mullet.core.tools.mcp_client.stdio_client", side_effect=mock_stdio_client), \
         patch("mullet.core.tools.mcp_client.ClientSession", return_value=mock_session):
        config = MCPServerConfig(name="srv", command="cmd")
        tools = asyncio.run(client._async_connect(config))

    assert tools[0].input_schema == {}
    assert tools[0].description == "MCP tool from srv"

    for stack in list(client._exit_stacks.values()):
        asyncio.run(stack.aclose())


# ── _connect_server (lines 219-223) ────────────────────────────────────────

def test_connect_server():
    client = MCPClient()
    client._ensure_loop()

    mock_tool = MagicMock()
    mock_tool.name = "t"
    mock_tool.description = "desc"
    mock_tool.inputSchema = {}

    mock_response = MagicMock()
    mock_response.tools = [mock_tool]

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=mock_response)

    def mock_stdio_client(params):
        return _FakeAsyncCtx(MagicMock(), MagicMock())

    with patch("mullet.core.tools.mcp_client.stdio_client", side_effect=mock_stdio_client), \
         patch("mullet.core.tools.mcp_client.ClientSession", return_value=mock_session):
        config = MCPServerConfig(name="s", command="c")
        tools = client._connect_server(config)
        assert len(tools) == 1
        assert tools[0].name == "s__t"

    client.disconnect_all()


# ── connect_all (lines 136-147) ────────────────────────────────────────────

def test_connect_all_skips_already_connected():
    client = MCPClient()
    client._connected.add("already")
    client._servers["already"] = MCPServerConfig(name="already", command="c")

    with patch.object(client, "_connect_server") as mock_connect:
        tools = client.connect_all()
        mock_connect.assert_not_called()
        assert tools == []


def test_connect_all_exception_handling():
    client = MCPClient()
    client._servers["bad"] = MCPServerConfig(name="bad", command="c")
    client._servers["good"] = MCPServerConfig(name="good", command="c")

    def side_effect(config):
        if config.name == "bad":
            raise RuntimeError("boom")
        tool = MCPTool(name="good__t", description="d", input_schema={}, server_name="good")
        client._tools[tool.name] = tool
        return [tool]

    with patch.object(client, "_connect_server", side_effect=side_effect):
        tools = client.connect_all()

    assert len(tools) == 1
    assert "bad" not in client._connected
    assert "good" in client._connected


# ── call_tool (lines 157-177) ──────────────────────────────────────────────

def test_call_tool_success():
    client = MCPClient()
    client._ensure_loop()
    client._tools["s__t"] = MCPTool(name="s__t", description="d", input_schema={}, server_name="s")
    client._servers["s"] = MCPServerConfig(name="s", command="c", timeout=5)

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.content = [MagicMock(text="hello")]
    mock_result.isError = False
    mock_session.call_tool = AsyncMock(return_value=mock_result)
    client._sessions["s"] = mock_session

    result = client.call_tool("s__t", {"arg": 1})
    assert result.content == "hello"
    assert not result.is_error
    client.disconnect_all()


def test_call_tool_missing_server_config():
    client = MCPClient()
    client._tools["s__t"] = MCPTool(name="s__t", description="d", input_schema={}, server_name="s")
    # No entry in client._servers
    result = client.call_tool("s__t", {})
    assert result.is_error
    assert "not configured" in result.content


def _make_patched_run_coroutine_threadsafe(future):
    """Return a patched asyncio.run_coroutine_threadsafe that discards the coroutine."""
    def _patched(coro, loop):
        coro.close()
        return future
    return _patched


def test_call_tool_timeout():
    client = MCPClient()
    client._ensure_loop()
    client._tools["s__t"] = MCPTool(name="s__t", description="d", input_schema={}, server_name="s")
    client._servers["s"] = MCPServerConfig(name="s", command="c", timeout=5)

    future = Future()
    future.set_exception(TimeoutError())

    with patch("asyncio.run_coroutine_threadsafe", side_effect=_make_patched_run_coroutine_threadsafe(future)):
        result = client.call_tool("s__t", {})
    assert result.is_error
    assert "timed out" in result.content
    client.disconnect_all()


def test_call_tool_exception():
    client = MCPClient()
    client._ensure_loop()
    client._tools["s__t"] = MCPTool(name="s__t", description="d", input_schema={}, server_name="s")
    client._servers["s"] = MCPServerConfig(name="s", command="c", timeout=5)

    future = Future()
    future.set_exception(RuntimeError("crashed"))

    with patch("asyncio.run_coroutine_threadsafe", side_effect=_make_patched_run_coroutine_threadsafe(future)):
        result = client.call_tool("s__t", {})
    assert result.is_error
    assert "crashed" in result.content
    client.disconnect_all()


# ── _async_call_tool (lines 262-294) ───────────────────────────────────────

def test_async_call_tool_no_session():
    client = MCPClient()
    tool = MCPTool(name="s__t", description="d", input_schema={}, server_name="s")
    result = asyncio.run(client._async_call_tool(tool, {}, 5))
    assert result.is_error
    assert "Not connected" in result.content


def test_async_call_tool_text_content():
    client = MCPClient()
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.content = [MagicMock(text="line1"), MagicMock(text="line2")]
    mock_result.isError = False
    mock_session.call_tool = AsyncMock(return_value=mock_result)
    client._sessions["s"] = mock_session

    tool = MCPTool(name="s__t", description="d", input_schema={}, server_name="s")
    result = asyncio.run(client._async_call_tool(tool, {}, 5))
    assert result.content == "line1\nline2"
    assert not result.is_error


def test_async_call_tool_no_text_attr():
    client = MCPClient()
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.content = [MagicMock()]  # no .text attribute
    mock_result.isError = True
    mock_session.call_tool = AsyncMock(return_value=mock_result)
    client._sessions["s"] = mock_session

    tool = MCPTool(name="s__t", description="d", input_schema={}, server_name="s")
    result = asyncio.run(client._async_call_tool(tool, {}, 5))
    assert result.is_error is True


def test_async_call_tool_async_timeout():
    client = MCPClient()
    mock_session = AsyncMock()

    async def slow(*a, **k):
        await asyncio.sleep(10)

    mock_session.call_tool = slow
    client._sessions["s"] = mock_session

    tool = MCPTool(name="s__t", description="d", input_schema={}, server_name="s")
    result = asyncio.run(client._async_call_tool(tool, {}, 0.1))
    assert result.is_error
    assert "timed out" in result.content


def test_async_call_tool_exception():
    client = MCPClient()
    mock_session = AsyncMock()
    mock_session.call_tool = AsyncMock(side_effect=ValueError("bad arg"))
    client._sessions["s"] = mock_session

    tool = MCPTool(name="s__t", description="d", input_schema={}, server_name="s")
    result = asyncio.run(client._async_call_tool(tool, {}, 5))
    assert result.is_error
    assert "bad arg" in result.content


def test_async_call_tool_no_double_underscore():
    """Tool name without '__' uses the full name as raw_name."""
    client = MCPClient()
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.content = [MagicMock(text="ok")]
    mock_result.isError = False
    mock_session.call_tool = AsyncMock(return_value=mock_result)
    client._sessions["s"] = mock_session

    tool = MCPTool(name="t", description="d", input_schema={}, server_name="s")
    result = asyncio.run(client._async_call_tool(tool, {}, 5))
    assert result.content == "ok"
    mock_session.call_tool.assert_awaited_once_with("t", {})


# ── disconnect_all (lines 187-196, 198) ────────────────────────────────────

def test_disconnect_all_with_running_loop():
    client = MCPClient()
    client._ensure_loop()

    async def make_stack():
        return AsyncExitStack()

    stack = asyncio.run(make_stack())
    client._exit_stacks["s"] = stack
    time.sleep(0.05)

    client.disconnect_all()
    assert client._loop is None
    assert client._thread is None
    assert client._sessions == {}
    assert client._tools == {}


def test_disconnect_all_closes_stacks_exception():
    """Cover the warning log when aclose() raises."""
    client = MCPClient()
    client._ensure_loop()

    async def make_bad_stack():
        stack = AsyncExitStack()
        original = stack.aclose

        async def bad_close():
            await original()
            raise RuntimeError("close fail")

        stack.aclose = bad_close
        return stack

    stack = asyncio.run(make_bad_stack())
    client._exit_stacks["s"] = stack
    time.sleep(0.05)

    # Must not raise
    client.disconnect_all()


def test_disconnect_all_thread_already_dead():
    client = MCPClient()
    client._loop = None
    client._thread = None
    client.disconnect_all()
    assert client._loop is None


# ── Context7Client (lines 330-410) ─────────────────────────────────────────

def test_context7_init():
    c7 = Context7Client()
    assert not c7._connected


def test_context7_ensure_connected_success():
    c7 = Context7Client()
    with patch.object(c7._client, "add_server") as mock_add, \
         patch.object(c7._client, "connect_all", return_value=[]) as mock_connect:
        c7._ensure_connected()
        assert c7._connected
        mock_add.assert_called_once()
        mock_connect.assert_called_once()


def test_context7_ensure_connected_failure():
    c7 = Context7Client()
    with patch.object(c7._client, "add_server") as mock_add, \
         patch.object(c7._client, "connect_all", side_effect=RuntimeError("fail")) as mock_connect:
        c7._ensure_connected()
        assert not c7._connected


def test_context7_resolve_library_id_success_json():
    c7 = Context7Client()
    c7._connected = True
    mock_result = MCPCallResult(content='{"id": "/lib/123"}', is_error=False)
    with patch.object(c7._client, "call_tool", return_value=mock_result):
        lib_id = c7.resolve_library_id("react")
        assert lib_id == "/lib/123"


def test_context7_resolve_library_id_success_libraryId_key():
    c7 = Context7Client()
    c7._connected = True
    mock_result = MCPCallResult(content='{"libraryId": "/lib/456"}', is_error=False)
    with patch.object(c7._client, "call_tool", return_value=mock_result):
        lib_id = c7.resolve_library_id("vue")
        assert lib_id == "/lib/456"


def test_context7_resolve_library_id_success_plain():
    c7 = Context7Client()
    c7._connected = True
    mock_result = MCPCallResult(content="  /libraries/react  ", is_error=False)
    with patch.object(c7._client, "call_tool", return_value=mock_result):
        lib_id = c7.resolve_library_id("react")
        assert lib_id == "/libraries/react"


def test_context7_resolve_library_id_not_connected():
    c7 = Context7Client()
    with patch.object(c7, "_ensure_connected"):
        assert c7.resolve_library_id("react") is None


def test_context7_resolve_library_id_error():
    c7 = Context7Client()
    c7._connected = True
    mock_result = MCPCallResult(content="error", is_error=True)
    with patch.object(c7._client, "call_tool", return_value=mock_result):
        assert c7.resolve_library_id("react") is None


def test_context7_resolve_library_id_bad_json_no_slash():
    c7 = Context7Client()
    c7._connected = True
    mock_result = MCPCallResult(content="notjson", is_error=False)
    with patch.object(c7._client, "call_tool", return_value=mock_result):
        assert c7.resolve_library_id("react") is None


def test_context7_search_docs_success():
    c7 = Context7Client()
    c7._connected = True
    with patch.object(c7, "resolve_library_id", return_value="/lib/123"), \
         patch.object(c7._client, "call_tool", return_value=MCPCallResult(content="docs", is_error=False)):
        docs = c7.search_docs("react", "18", "hooks")
    assert len(docs) == 1
    assert docs[0]["content"] == "docs"
    assert docs[0]["library"] == "react"
    assert docs[0]["library_id"] == "/lib/123"


def test_context7_search_docs_not_connected():
    c7 = Context7Client()
    with patch.object(c7, "_ensure_connected"):
        assert c7.search_docs("react", "18", "hooks") == []


def test_context7_search_docs_no_lib_id():
    c7 = Context7Client()
    c7._connected = True
    with patch.object(c7, "resolve_library_id", return_value=None):
        assert c7.search_docs("react", "18", "hooks") == []


def test_context7_search_docs_error():
    c7 = Context7Client()
    c7._connected = True
    with patch.object(c7, "resolve_library_id", return_value="/lib/123"), \
         patch.object(c7._client, "call_tool", return_value=MCPCallResult(content="err", is_error=True)):
        assert c7.search_docs("react", "18", "hooks") == []


def test_context7_search_docs_empty_content():
    c7 = Context7Client()
    c7._connected = True
    with patch.object(c7, "resolve_library_id", return_value="/lib/123"), \
         patch.object(c7._client, "call_tool", return_value=MCPCallResult(content="", is_error=False)):
        assert c7.search_docs("react", "18", "hooks") == []


def test_context7_get_versions():
    c7 = Context7Client()
    assert c7.get_versions("react") == []


def test_context7_search_libraries_success():
    c7 = Context7Client()
    c7._connected = True
    mock_result = MCPCallResult(content='{"libraryId": "/lib/123"}', is_error=False)
    with patch.object(c7._client, "call_tool", return_value=mock_result):
        assert c7.search_libraries("react") == ["/lib/123"]


def test_context7_search_libraries_not_connected():
    c7 = Context7Client()
    with patch.object(c7, "_ensure_connected"):
        assert c7.search_libraries("react") == []


def test_context7_search_libraries_error():
    c7 = Context7Client()
    c7._connected = True
    mock_result = MCPCallResult(content="err", is_error=True)
    with patch.object(c7._client, "call_tool", return_value=mock_result):
        assert c7.search_libraries("react") == []


def test_context7_search_libraries_plain_text():
    c7 = Context7Client()
    c7._connected = True
    mock_result = MCPCallResult(content="  /lib/123  ", is_error=False)
    with patch.object(c7._client, "call_tool", return_value=mock_result):
        assert c7.search_libraries("react") == ["/lib/123"]


def test_context7_search_libraries_bad_text():
    c7 = Context7Client()
    c7._connected = True
    mock_result = MCPCallResult(content="nope", is_error=False)
    with patch.object(c7._client, "call_tool", return_value=mock_result):
        assert c7.search_libraries("react") == []


def test_context7_disconnect():
    c7 = Context7Client()
    c7._connected = True
    with patch.object(c7._client, "disconnect_all") as mock_disconnect:
        c7.disconnect()
        assert not c7._connected
        mock_disconnect.assert_called_once()


def test_context7_disconnect_when_not_connected():
    c7 = Context7Client()
    c7._connected = False
    c7.disconnect()  # must not raise
    assert not c7._connected


# ---- Cover remaining missing lines ----

def test_mcp_client_available_property():
    from mullet.core.tools.mcp_client import MCPClient, _MCP_AVAILABLE
    client = MCPClient()
    assert client.available == _MCP_AVAILABLE


def test_mcp_client_tools_property():
    from mullet.core.tools.mcp_client import MCPClient, MCPTool
    client = MCPClient()
    assert client.tools == {}
    tool = MCPTool(name="t", description="d", input_schema={}, server_name="s")
    client._tools["s__t"] = tool
    assert "s__t" in client.tools


def test_mcp_client_add_server():
    from mullet.core.tools.mcp_client import MCPClient, MCPServerConfig
    client = MCPClient()
    cfg = MCPServerConfig(name="srv", command="node", args=["x"])
    client.add_server(cfg)
    assert "srv" in client._servers


def test_mcp_client_connect_all_unavailable(monkeypatch):
    from mullet.core.tools import mcp_client as mod
    monkeypatch.setattr(mod, "_MCP_AVAILABLE", False)
    client = mod.MCPClient()
    result = client.connect_all()
    assert result == []


def test_mcp_client_call_tool_unknown():
    from mullet.core.tools.mcp_client import MCPClient
    client = MCPClient()
    result = client.call_tool("nonexistent__tool", {})
    assert result.is_error
    assert "Unknown MCP tool" in result.content


def test_to_mullet_schema_function():
    from mullet.core.tools.mcp_client import to_mullet_schema, MCPTool
    tool = MCPTool(name="t", description="does stuff", input_schema={"type": "object"}, server_name="s")
    schema = to_mullet_schema(tool)
    assert schema["description"] == "does stuff"
    assert schema["parameters"] == {"type": "object"}


def test_to_mullet_schema_null_schema():
    from mullet.core.tools.mcp_client import to_mullet_schema, MCPTool
    tool = MCPTool(name="t", description="d", input_schema=None, server_name="s")
    schema = to_mullet_schema(tool)
    assert schema["parameters"]["type"] == "object"


def test_load_mcp_servers():
    from mullet.core.tools.mcp_client import load_mcp_servers
    config = {
        "srv1": {"command": "node", "args": ["s.js"], "env": {"X": "1"}},
        "srv2": {"command": "python", "timeout": 30},
    }
    servers = load_mcp_servers(config)
    assert len(servers) == 2
    names = {s.name for s in servers}
    assert names == {"srv1", "srv2"}
    srv1 = next(s for s in servers if s.name == "srv1")
    assert srv1.args == ["s.js"]
    assert srv1.env == {"X": "1"}
    srv2 = next(s for s in servers if s.name == "srv2")
    assert srv2.timeout == 30
