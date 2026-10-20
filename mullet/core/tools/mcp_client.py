"""MCP (Model Context Protocol) client — connect to external MCP servers.

Discovers tools from MCP servers via stdio transport and registers them
into Mullet's tool registry so the agent can call them like built-in tools.

Configuration is read from config or passed directly:

    mcp_servers:
      filesystem:
        command: "npx"
        args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        env: {}
        timeout: 120

Requires: pip install mcp (optional dependency).
Ported from Hermes tools/mcp_tool.py, simplified for Mullet.
"""

import asyncio
from contextlib import AsyncExitStack
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Graceful import
_MCP_AVAILABLE = False
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    _MCP_AVAILABLE = True
except ImportError:
    logger.debug("mcp package not installed — MCP support disabled")

# Safe env keys for stdio subprocesses
_SAFE_ENV_KEYS = frozenset({
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM", "SHELL", "TMPDIR",
})

# Strip credentials from error messages
_CREDENTIAL_RE = re.compile(
    r"(?:ghp_[A-Za-z0-9_]{1,255}|sk-[A-Za-z0-9_]{1,255}|Bearer\s+\S+|"
    r"token=[^\s&,;\"']{1,255}|key=[^\s&,;\"']{1,255})"
)

DEFAULT_TIMEOUT = 120
DEFAULT_CONNECT_TIMEOUT = 60
MAX_RETRIES = 3


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    timeout: int = DEFAULT_TIMEOUT
    connect_timeout: int = DEFAULT_CONNECT_TIMEOUT


@dataclass
class MCPTool:
    """An MCP tool discovered from a server."""
    name: str
    description: str
    input_schema: dict
    server_name: str


@dataclass
class MCPCallResult:
    """Result of calling an MCP tool."""
    content: str
    is_error: bool = False


def _sanitize_error(msg: str) -> str:
    """Strip credentials from error messages before returning to LLM."""
    return _CREDENTIAL_RE.sub("[REDACTED]", msg)


def _build_env(config: MCPServerConfig) -> dict[str, str]:
    """Build a safe environment for the stdio subprocess."""
    env = {k: v for k, v in os.environ.items() if k in _SAFE_ENV_KEYS}
    env.update(config.env)
    return env


class MCPClient:
    """Client for connecting to MCP servers via stdio transport.

    Manages a background event loop for async MCP communication.
    Thread-safe — tool calls are scheduled via run_coroutine_threadsafe.
    """

    def __init__(self):
        self._servers: dict[str, MCPServerConfig] = {}
        self._sessions: dict[str, Any] = {}  # name -> ClientSession
        self._exit_stacks: dict[str, AsyncExitStack] = {}  # name -> exit stack holding context managers alive
        self._tools: dict[str, MCPTool] = {}  # qualified_name -> MCPTool
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._connected: set[str] = set()

    @property
    def available(self) -> bool:
        """Check if MCP SDK is installed."""
        return _MCP_AVAILABLE

    @property
    def tools(self) -> dict[str, MCPTool]:
        """All discovered tools across all servers."""
        return dict(self._tools)

    def add_server(self, config: MCPServerConfig):
        """Register a server configuration."""
        with self._lock:
            self._servers[config.name] = config

    def connect_all(self) -> list[MCPTool]:
        """Connect to all registered servers and discover tools.

        Returns list of all discovered tools.
        """
        if not _MCP_AVAILABLE:
            logger.warning("MCP SDK not installed. Run: pip install mcp")
            return []

        self._ensure_loop()

        for name, config in self._servers.items():
            if name not in self._connected:
                try:
                    tools = self._connect_server(config)
                    self._connected.add(name)
                    logger.info("MCP: connected to %s (%d tools)", name, len(tools))
                except Exception as e:
                    logger.error("MCP: failed to connect to %s: %s", name, _sanitize_error(str(e)))

        return list(self._tools.values())

    def call_tool(self, tool_name: str, arguments: dict) -> MCPCallResult:
        """Call an MCP tool by its qualified name (server__tool)."""
        if tool_name not in self._tools:
            return MCPCallResult(
                content=f"Unknown MCP tool: {tool_name}",
                is_error=True,
            )

        tool = self._tools[tool_name]
        config = self._servers.get(tool.server_name)
        if not config:
            return MCPCallResult(
                content=f"MCP server '{tool.server_name}' not configured",
                is_error=True,
            )

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._async_call_tool(tool, arguments, config.timeout),
                self._loop,
            )
            return future.result(timeout=config.timeout + 5)
        except TimeoutError:
            return MCPCallResult(
                content=f"MCP tool '{tool_name}' timed out ({config.timeout}s)",
                is_error=True,
            )
        except Exception as e:
            return MCPCallResult(
                content=_sanitize_error(f"MCP tool error: {e}"),
                is_error=True,
            )

    def disconnect_all(self):
        """Disconnect from all servers and shut down the event loop."""
        with self._lock:
            # Close all exit stacks to properly tear down stdio transports
            if self._loop and self._loop.is_running():
                for stack in list(self._exit_stacks.values()):
                    try:
                        future = asyncio.run_coroutine_threadsafe(
                            stack.aclose(), self._loop
                        )
                        future.result(timeout=5)
                    except Exception as e:
                        logger.warning("mcp_client: Failed to close MCP server exit stack: %s", e)
                self._exit_stacks.clear()
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5)
            self._sessions.clear()
            self._tools.clear()
            self._connected.clear()
            self._loop = None
            self._thread = None

    def _ensure_loop(self):
        """Start the background event loop if not already running."""
        with self._lock:
            if self._loop is None or not self._loop.is_running():
                self._loop = asyncio.new_event_loop()
                self._thread = threading.Thread(
                    target=self._loop.run_forever,
                    daemon=True,
                    name="mcp-event-loop",
                )
                self._thread.start()

    def _connect_server(self, config: MCPServerConfig) -> list[MCPTool]:
        """Connect to a single server and discover its tools."""
        future = asyncio.run_coroutine_threadsafe(
            self._async_connect(config),
            self._loop,
        )
        return future.result(timeout=config.connect_timeout)

    async def _async_connect(self, config: MCPServerConfig) -> list[MCPTool]:
        """Async connect to an MCP server via stdio."""
        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=_build_env(config),
        )

        # Use AsyncExitStack to keep the stdio context manager alive
        stack = AsyncExitStack()
        read_stream, write_stream = await stack.enter_async_context(stdio_client(server_params))
        self._exit_stacks[config.name] = stack

        session = ClientSession(read_stream, write_stream)
        await session.initialize()

        self._sessions[config.name] = session

        # Discover tools
        tools_response = await session.list_tools()
        discovered: list[MCPTool] = []

        for tool in tools_response.tools:
            qualified_name = f"{config.name}__{tool.name}"
            mcp_tool = MCPTool(
                name=qualified_name,
                description=tool.description or f"MCP tool from {config.name}",
                input_schema=tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                server_name=config.name,
            )
            self._tools[qualified_name] = mcp_tool
            discovered.append(mcp_tool)

        return discovered

    async def _async_call_tool(self, tool: MCPTool, arguments: dict, timeout: int) -> MCPCallResult:
        """Call a tool on its MCP server."""
        session = self._sessions.get(tool.server_name)
        if not session:
            return MCPCallResult(
                content=f"Not connected to server '{tool.server_name}'",
                is_error=True,
            )

        # Strip server prefix from tool name
        raw_name = tool.name.split("__", 1)[1] if "__" in tool.name else tool.name

        try:
            result = await asyncio.wait_for(
                session.call_tool(raw_name, arguments),
                timeout=timeout,
            )

            # Extract text content from result
            content_parts = []
            for item in (result.content or []):
                if hasattr(item, 'text'):
                    content_parts.append(item.text)

            return MCPCallResult(
                content="\n".join(content_parts) if content_parts else str(result),
                is_error=getattr(result, 'isError', False),
            )
        except asyncio.TimeoutError:
            return MCPCallResult(
                content=f"Tool call timed out after {timeout}s",
                is_error=True,
            )
        except Exception as e:
            return MCPCallResult(
                content=_sanitize_error(str(e)),
                is_error=True,
            )


def to_mullet_schema(mcp_tool: MCPTool) -> dict:
    """Convert an MCP tool to Mullet tool registry schema format."""
    return {
        "description": mcp_tool.description,
        "parameters": mcp_tool.input_schema or {
            "type": "object",
            "properties": {},
        },
    }


def load_mcp_servers(config: dict) -> list[MCPServerConfig]:
    """Load MCP server configs from a dict (e.g. from config.yaml)."""
    servers = []
    for name, spec in config.items():
        servers.append(MCPServerConfig(
            name=name,
            command=spec.get("command", ""),
            args=spec.get("args", []),
            env=spec.get("env", {}),
            timeout=spec.get("timeout", DEFAULT_TIMEOUT),
            connect_timeout=spec.get("connect_timeout", DEFAULT_CONNECT_TIMEOUT),
        ))
    return servers


class Context7Client:
    """Context7 MCP client — version-aware library documentation."""

    def __init__(self):
        self._client = MCPClient()
        self._connected = False
        self._lock = threading.Lock()

    def _ensure_connected(self) -> None:
        with self._lock:
            if not self._connected:
                try:
                    self._client.add_server(MCPServerConfig(
                        name="context7",
                        command="npx",
                        args=["-y", "@upstash/context7-mcp@latest"],
                        timeout=30,
                        connect_timeout=45,
                    ))
                    self._client.connect_all()
                    self._connected = True
                except Exception as e:
                    logger.warning("Context7: failed to connect: %s", _sanitize_error(str(e)))
                    self._connected = False

    def resolve_library_id(self, library_name: str) -> str | None:
        """Resolve a library name to a Context7 library ID."""
        self._ensure_connected()
        if not self._connected:
            return None
        result = self._client.call_tool("context7__resolve-library-id", {"libraryName": library_name})
        if result.is_error:
            return None
        # Parse the returned text — Context7 returns JSON or plain ID
        try:
            data = json.loads(result.content)
            return data.get("id") or data.get("libraryId")
        except (json.JSONDecodeError, AttributeError):
            # Might be plain text
            text = result.content.strip()
            return text if text.startswith("/") else None

    def search_docs(self, library: str, version: str, query: str) -> list[dict]:
        """Get docs for a library at a specific topic."""
        self._ensure_connected()
        if not self._connected:
            return []
        lib_id = self.resolve_library_id(library)
        if not lib_id:
            return []
        result = self._client.call_tool("context7__get-library-docs", {
            "context7CompatibleLibraryID": lib_id,
            "topic": query,
            "tokens": 4000,
        })
        if result.is_error or not result.content:
            return []
        return [{"content": result.content, "library": library, "library_id": lib_id}]

    def get_versions(self, library: str) -> list[str]:
        """Context7 doesn't expose version listing directly — return empty."""
        return []

    def search_libraries(self, query: str) -> list[str]:
        """Resolve approximate library names via Context7."""
        self._ensure_connected()
        if not self._connected:
            return []
        result = self._client.call_tool("context7__resolve-library-id", {"libraryName": query})
        if result.is_error:
            return []
        lib_id = None
        try:
            data = json.loads(result.content)
            lib_id = data.get("id") or data.get("libraryId")
        except (json.JSONDecodeError, AttributeError):
            text = result.content.strip()
            lib_id = text if text.startswith("/") else None
        return [lib_id] if lib_id else []

    def disconnect(self) -> None:
        """Disconnect from the Context7 MCP server."""
        if self._connected:
            self._client.disconnect_all()
            self._connected = False
