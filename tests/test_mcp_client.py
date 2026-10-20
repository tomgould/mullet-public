"""Tests for the MCP client module."""

from unittest.mock import patch

from mullet.core.tools.mcp_client import (
    MCPServerConfig,
    MCPTool,
    MCPCallResult,
    MCPClient,
    _sanitize_error,
    _build_env,
    to_mullet_schema,
    load_mcp_servers,
)


# ── MCPServerConfig ─────────────────────────────────────────────────────────

def test_server_config_defaults():
    c = MCPServerConfig(name="test", command="echo")
    assert c.args == []
    assert c.env == {}
    assert c.timeout == 120
    assert c.connect_timeout == 60


def test_server_config_custom():
    c = MCPServerConfig(
        name="fs", command="npx",
        args=["-y", "@mcp/server-filesystem", "/tmp"],
        env={"MY_VAR": "hello"},
        timeout=30,
    )
    assert c.command == "npx"
    assert len(c.args) == 3
    assert c.env["MY_VAR"] == "hello"


# ── MCPTool ──────────────────────────────────────────────────────────────────

def test_mcp_tool_dataclass():
    t = MCPTool(
        name="fs__read", description="Read a file",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        server_name="fs",
    )
    assert t.name == "fs__read"
    assert t.server_name == "fs"


# ── MCPCallResult ────────────────────────────────────────────────────────────

def test_call_result_success():
    r = MCPCallResult(content="file contents here")
    assert not r.is_error


def test_call_result_error():
    r = MCPCallResult(content="not found", is_error=True)
    assert r.is_error


# ── _sanitize_error ──────────────────────────────────────────────────────────

def test_sanitize_github_pat():
    msg = "auth failed with ghp_abc123def456"
    assert "ghp_" not in _sanitize_error(msg)
    assert "[REDACTED]" in _sanitize_error(msg)


def test_sanitize_openai_key():
    msg = "error: sk-test123abc456"
    assert "sk-" not in _sanitize_error(msg)


def test_sanitize_bearer_token():
    msg = "failed with Bearer eyJhbGciOiJI..."
    assert "eyJ" not in _sanitize_error(msg)


def test_sanitize_no_creds():
    msg = "normal error message"
    assert _sanitize_error(msg) == msg


# ── _build_env ───────────────────────────────────────────────────────────────

def test_build_env_safe_keys():
    config = MCPServerConfig(name="test", command="echo")
    env = _build_env(config)
    assert "PATH" in env
    # Should not include random env vars
    assert "HISTFILE" not in env


def test_build_env_custom_vars():
    config = MCPServerConfig(name="test", command="echo", env={"MY_API_KEY": "secret"})
    env = _build_env(config)
    assert env["MY_API_KEY"] == "secret"


# ── to_mullet_schema ────────────────────────────────────────────────────────

def test_to_mullet_schema():
    tool = MCPTool(
        name="fs__read", description="Read file",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        server_name="fs",
    )
    schema = to_mullet_schema(tool)
    assert schema["description"] == "Read file"
    assert schema["parameters"]["type"] == "object"
    assert "path" in schema["parameters"]["properties"]


def test_to_mullet_schema_empty():
    tool = MCPTool(name="t", description="test", input_schema={}, server_name="s")
    schema = to_mullet_schema(tool)
    assert schema["parameters"]["type"] == "object"


# ── load_mcp_servers ─────────────────────────────────────────────────────────

def test_load_mcp_servers():
    config = {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@mcp/server-filesystem", "/tmp"],
            "env": {"API_KEY": "test"},
            "timeout": 30,
        },
        "github": {
            "command": "npx",
            "args": ["-y", "@mcp/server-github"],
        },
    }
    servers = load_mcp_servers(config)
    assert len(servers) == 2
    fs = next(s for s in servers if s.name == "filesystem")
    assert fs.command == "npx"
    assert fs.timeout == 30
    gh = next(s for s in servers if s.name == "github")
    assert gh.timeout == 120  # default


def test_load_mcp_servers_empty():
    assert load_mcp_servers({}) == []


# ── MCPClient ────────────────────────────────────────────────────────────────

def test_client_available():
    client = MCPClient()
    # Should be True if mcp package installed, False otherwise
    assert isinstance(client.available, bool)


def test_client_add_server():
    client = MCPClient()
    client.add_server(MCPServerConfig(name="test", command="echo"))
    assert "test" in client._servers


def test_client_tools_empty():
    client = MCPClient()
    assert client.tools == {}


def test_client_call_unknown_tool():
    client = MCPClient()
    result = client.call_tool("nonexistent__tool", {})
    assert result.is_error
    assert "Unknown MCP tool" in result.content


def test_client_disconnect_clean():
    client = MCPClient()
    client.disconnect_all()  # Should not raise even when not connected
    assert client.tools == {}


@patch("mullet.core.tools.mcp_client._MCP_AVAILABLE", False)
def test_client_connect_without_sdk():
    client = MCPClient()
    client.add_server(MCPServerConfig(name="test", command="echo"))
    tools = client.connect_all()
    assert tools == []


def test_mcp_import_error_path():
    """Lines 37-38: ImportError when mcp package missing sets _MCP_AVAILABLE=False."""
    import sys
    import importlib
    # Force reimport with mcp blocked
    with patch.dict(sys.modules, {"mcp": None, "mcp.client": None, "mcp.client.stdio": None}):
        if "mullet.core.tools.mcp_client" in sys.modules:
            del sys.modules["mullet.core.tools.mcp_client"]
        mod = importlib.import_module("mullet.core.tools.mcp_client")
        assert mod._MCP_AVAILABLE is False
    # Restore original module
    if "mullet.core.tools.mcp_client" in sys.modules:
        del sys.modules["mullet.core.tools.mcp_client"]
    importlib.import_module("mullet.core.tools.mcp_client")
