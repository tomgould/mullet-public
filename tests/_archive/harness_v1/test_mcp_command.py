"""Tests for mullet.harness.commands.mcp — /mcp slash command dispatcher."""

from __future__ import annotations

import pytest

from mullet.harness.commands.mcp import handle_mcp_command
from mullet.harness.mcp.manager import MCPManager

pytestmark = pytest.mark.harness


def test_handle_mcp_help_or_no_args() -> None:
    mgr = MCPManager(config_paths=[])
    out = handle_mcp_command("", manager=mgr)
    assert "Usage" in out
    assert "/mcp list" in out


def test_handle_mcp_help_alias() -> None:
    mgr = MCPManager(config_paths=[])
    out = handle_mcp_command("help", manager=mgr)
    assert "Usage" in out


def test_handle_mcp_list_no_servers_returns_friendly_message() -> None:
    mgr = MCPManager(config_paths=[])
    out = handle_mcp_command("list", manager=mgr)
    assert "No MCP servers" in out


def test_handle_mcp_tools_no_tools_returns_message() -> None:
    mgr = MCPManager(config_paths=[])
    out = handle_mcp_command("tools", manager=mgr)
    assert "No tools" in out


def test_handle_mcp_tools_lists_qualified_names() -> None:
    mgr = MCPManager(config_paths=[])
    mgr.tool_index["fs.read_file"] = ("fs", {"name": "read_file", "description": "Read a file"})
    out = handle_mcp_command("tools", manager=mgr)
    assert "fs.read_file" in out
    assert "Read a file" in out


def test_handle_mcp_disconnect_unknown_returns_error() -> None:
    mgr = MCPManager(config_paths=[])
    out = handle_mcp_command("disconnect ghost", manager=mgr)
    assert "ghost" in out
    assert "not connected" in out.lower()


def test_handle_mcp_disconnect_missing_arg() -> None:
    mgr = MCPManager(config_paths=[])
    out = handle_mcp_command("disconnect", manager=mgr)
    assert "Usage" in out


def test_handle_mcp_connect_missing_args() -> None:
    mgr = MCPManager(config_paths=[])
    out = handle_mcp_command("connect myserver", manager=mgr)
    assert "Usage" in out


def test_handle_mcp_unknown_subcommand() -> None:
    mgr = MCPManager(config_paths=[])
    out = handle_mcp_command("frobnicate", manager=mgr)
    assert "Unknown" in out
    assert "Usage" in out


def test_handle_mcp_call_missing_args() -> None:
    mgr = MCPManager(config_paths=[])
    out = handle_mcp_command("call", manager=mgr)
    assert "Usage" in out


def test_handle_mcp_call_invalid_json() -> None:
    mgr = MCPManager(config_paths=[])
    mgr.tool_index["fs.read_file"] = ("fs", {"name": "read_file"})
    out = handle_mcp_command("call fs.read_file {not-json}", manager=mgr)
    assert "Invalid JSON" in out


def test_handle_mcp_call_unknown_tool() -> None:
    mgr = MCPManager(config_paths=[])
    out = handle_mcp_command('call ghost.nope {}', manager=mgr)
    # Surfaces the underlying KeyError message text.
    assert "ghost.nope" in out


def test_handle_mcp_list_shows_connected_servers() -> None:
    mgr = MCPManager(config_paths=[])

    class FakeTransport:
        pass

    class FakeClient:
        initialized = True

        def __init__(self) -> None:
            self.transport = FakeTransport()

    mgr.clients["fs"] = FakeClient()  # type: ignore[assignment]
    mgr.tool_index["fs.read_file"] = ("fs", {"name": "read_file"})

    out = handle_mcp_command("list", manager=mgr)
    assert "fs" in out
    assert "1 tool" in out
