"""Unit tests for ToolDispatcher."""

from __future__ import annotations

import json

import pytest

from mullet.core.tools.registry import ToolRegistry
from mullet.harness.engine.tool_dispatch import ToolDispatcher
from mullet.harness.hooks.registry import Block, HookRegistry
from mullet.harness.hooks.types import PostToolUsePayload, PreToolUsePayload


@pytest.fixture
def fresh_registry():
    return ToolRegistry()


@pytest.fixture
def dispatcher(fresh_registry):
    return ToolDispatcher(registry=fresh_registry)


class TestToolDiscovery:
    def test_dispatcher_has_known_tools(self):
        d = ToolDispatcher()
        # Core tools that have no external deps should always be discovered
        assert d.has_tool("file_ops")
        assert d.has_tool("terminal")
        assert d.has_tool("web_search")
        assert d.has_tool("fetch_page")
        assert d.has_tool("code_exec")
        assert d.has_tool("git")

    def test_dispatcher_list_tools_returns_names(self, dispatcher):
        dispatcher._registry.register(
            name="mock_tool",
            func=lambda x: str(x),
            schema={"description": "Mock", "parameters": {"type": "object", "properties": {}}},
            toolset="test",
        )
        names = dispatcher.list_tools()
        assert "mock_tool" in names

    def test_dispatcher_list_schemas_returns_openai_format(self, dispatcher):
        dispatcher._registry.register(
            name="mock_tool",
            func=lambda x: str(x),
            schema={
                "description": "A mock tool",
                "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
            },
            toolset="test",
        )
        schemas = dispatcher.list_schemas()
        assert len(schemas) >= 1
        schema = next(s for s in schemas if s["function"]["name"] == "mock_tool")
        assert schema["type"] == "function"
        assert "function" in schema
        assert schema["function"]["name"] == "mock_tool"
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]

    def test_get_schema_known(self, dispatcher):
        dispatcher._registry.register(
            name="mock_tool",
            func=lambda x: str(x),
            schema={
                "description": "A mock tool",
                "parameters": {"type": "object", "properties": {}},
            },
            toolset="test",
        )
        schema = dispatcher.get_schema("mock_tool")
        assert schema is not None
        assert schema["function"]["name"] == "mock_tool"

    def test_get_schema_unknown_returns_none(self, dispatcher):
        assert dispatcher.get_schema("nonexistent_tool_xyz") is None


class TestToolExecution:
    @pytest.mark.asyncio
    async def test_execute_simple_tool(self, dispatcher):
        dispatcher._registry.register(
            name="echo",
            func=lambda message: f"echo: {message}",
            schema={
                "description": "Echo tool",
                "parameters": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                },
            },
            toolset="test",
        )
        result = await dispatcher.execute(
            "echo",
            {"message": "hello"},
            session_id="sess-1",
            tool_call_id="tc-1",
        )
        assert result["output"] == "echo: hello"
        assert result["error"] is None
        assert result["blocked_reason"] is None

    @pytest.mark.asyncio
    async def test_execute_fires_pre_and_post_tool_use_hooks(self, dispatcher):
        pre_fired = []
        post_fired = []

        def pre_hook(payload):
            pre_fired.append(payload)
            return None

        def post_hook(payload):
            post_fired.append(payload)
            return None

        hooks = HookRegistry()
        hooks.register("PreToolUse", pre_hook)
        hooks.register("PostToolUse", post_hook)

        dispatcher._registry.register(
            name="echo",
            func=lambda message: f"echo: {message}",
            schema={
                "description": "Echo tool",
                "parameters": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                },
            },
            toolset="test",
        )

        await dispatcher.execute(
            "echo",
            {"message": "hi"},
            session_id="sess-1",
            tool_call_id="tc-1",
            hooks=hooks,
        )

        assert len(pre_fired) == 1
        assert isinstance(pre_fired[0], PreToolUsePayload)
        assert pre_fired[0].tool_name == "echo"
        assert pre_fired[0].tool_args == {"message": "hi"}

        assert len(post_fired) == 1
        assert isinstance(post_fired[0], PostToolUsePayload)
        assert post_fired[0].tool_name == "echo"
        assert post_fired[0].tool_result == "echo: hi"
        assert post_fired[0].tool_error is None

    @pytest.mark.asyncio
    async def test_execute_pre_tool_use_block_short_circuits(self, dispatcher):
        executed = []

        def tool_fn(message):
            executed.append(message)
            return "ran"

        dispatcher._registry.register(
            name="echo",
            func=tool_fn,
            schema={
                "description": "Echo tool",
                "parameters": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                },
            },
            toolset="test",
        )

        hooks = HookRegistry()
        hooks.register("PreToolUse", lambda p: Block("permission denied"))

        result = await dispatcher.execute(
            "echo",
            {"message": "hi"},
            session_id="sess-1",
            tool_call_id="tc-1",
            hooks=hooks,
        )

        assert result["blocked_reason"] == "permission denied"
        assert result["output"] == ""
        assert result["error"] is None
        assert len(executed) == 0

    @pytest.mark.asyncio
    async def test_execute_handles_tool_exception_returns_error(self, dispatcher):
        def boom(**kwargs):
            raise RuntimeError("kaboom")

        dispatcher._registry.register(
            name="boom",
            func=boom,
            schema={
                "description": "Boom tool",
                "parameters": {"type": "object", "properties": {}},
            },
            toolset="test",
        )

        result = await dispatcher.execute(
            "boom",
            {},
            session_id="sess-1",
            tool_call_id="tc-1",
        )

        assert result["error"] == "kaboom"
        assert "error" in json.loads(result["output"])
        assert result["blocked_reason"] is None
        assert result["metadata"]["exception_type"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_returns_error(self, dispatcher):
        result = await dispatcher.execute(
            "no_such_tool_12345",
            {"x": 1},
            session_id="sess-1",
            tool_call_id="tc-1",
        )

        assert result["error"] == "Unknown tool: no_such_tool_12345"
        assert "error" in json.loads(result["output"])
        assert result["blocked_reason"] is None
