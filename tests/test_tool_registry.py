import json
from unittest.mock import MagicMock, patch


from mullet.core.tools.registry import ToolRegistry


class TestRegister:
    def test_register_adds_tool(self):
        reg = ToolRegistry()
        func = lambda x: str(x)
        reg.register("tool_a", func, {"description": "A tool"}, toolset="test")
        assert "tool_a" in reg._tools
        assert reg._tools["tool_a"].name == "tool_a"
        assert reg._tools["tool_a"].toolset == "test"

    def test_register_default_toolset(self):
        reg = ToolRegistry()
        func = lambda: "ok"
        reg.register("tool_b", func, {})
        assert "default" in reg._toolsets
        assert "tool_b" in reg._toolsets["default"]


class TestGetSchemas:
    def test_empty_registry(self):
        reg = ToolRegistry()
        assert reg.get_schemas() == []

    def test_single_schema(self):
        reg = ToolRegistry()
        reg.register("t1", lambda: "", {"description": "d1", "parameters": {"type": "object"}}, toolset="ts1")
        schemas = reg.get_schemas(["ts1"])
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "t1"

    def test_multiple_toolsets(self):
        reg = ToolRegistry()
        reg.register("t1", lambda: "", {}, toolset="a")
        reg.register("t2", lambda: "", {}, toolset="b")
        schemas = reg.get_schemas(["a", "b"])
        names = [s["function"]["name"] for s in schemas]
        assert sorted(names) == ["t1", "t2"]

    def test_default_toolsets(self):
        reg = ToolRegistry()
        reg.register("t1", lambda: "", {}, toolset="x")
        schemas = reg.get_schemas()
        assert len(schemas) == 1


class TestDispatch:
    def test_unknown_tool(self):
        reg = ToolRegistry()
        result = reg.dispatch("missing", "{}")
        assert json.loads(result)["error"] == "Unknown tool: missing"

    def test_dispatch_with_dict_args(self):
        reg = ToolRegistry()
        mock_func = MagicMock(return_value="result")
        reg.register("add", mock_func, {})
        result = reg.dispatch("add", {"a": 1, "b": 2})
        mock_func.assert_called_once_with(a=1, b=2, task_id="")
        assert result == "result"

    def test_dispatch_with_str_args(self):
        reg = ToolRegistry()
        mock_func = MagicMock(return_value={"ok": True})
        reg.register("calc", mock_func, {})
        result = reg.dispatch("calc", '{"x": 10}')
        mock_func.assert_called_once_with(x=10, task_id="")
        assert json.loads(result) == {"ok": True}

    def test_dispatch_non_string_result(self):
        reg = ToolRegistry()
        mock_func = MagicMock(return_value={"data": 42})
        reg.register("get", mock_func, {})
        result = reg.dispatch("get", "{}")
        assert json.loads(result) == {"data": 42}

    def test_dispatch_func_without_task_id(self):
        reg = ToolRegistry()
        def old_func(a=0):
            return str(a)
        reg.register("old", old_func, {})
        result = reg.dispatch("old", '{"a": 5}')
        assert result == "5"

    def test_dispatch_exception(self):
        reg = ToolRegistry()
        def bad_func():
            raise RuntimeError("boom")
        reg.register("bad", bad_func, {})
        result = reg.dispatch("bad", "{}")
        assert "boom" in json.loads(result)["error"]


class TestGetSchemasForProfile:
    def test_default_profile_returns_all(self):
        reg = ToolRegistry()
        reg.register("t1", lambda: "", {}, toolset="a")
        schemas = reg.get_schemas_for_profile("default")
        assert len(schemas) == 1

    def test_unknown_profile_returns_all(self):
        reg = ToolRegistry()
        reg.register("t1", lambda: "", {}, toolset="a")
        schemas = reg.get_schemas_for_profile("nonexistent")
        assert len(schemas) == 1

    @patch("random.random")
    def test_profile_filters_by_weight(self, mock_random):
        mock_random.return_value = 0.5
        reg = ToolRegistry()
        reg.register("t1", lambda: "", {}, toolset="web")
        reg.register("t2", lambda: "", {}, toolset="terminal")
        reg.register_profile("custom", {"web": 0.6, "terminal": 0.4})
        schemas = reg.get_schemas_for_profile("custom")
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "t1"

    @patch("random.random")
    def test_profile_excludes_low_weight(self, mock_random):
        mock_random.return_value = 0.9
        reg = ToolRegistry()
        reg.register("t1", lambda: "", {}, toolset="web")
        reg.register_profile("custom", {"web": 0.1})
        schemas = reg.get_schemas_for_profile("custom")
        assert schemas == []


class TestListMethods:
    def test_list_tools(self):
        reg = ToolRegistry()
        reg.register("a", lambda: "", {})
        reg.register("b", lambda: "", {})
        assert sorted(reg.list_tools()) == ["a", "b"]

    def test_list_toolsets(self):
        reg = ToolRegistry()
        reg.register("a", lambda: "", {}, toolset="ts1")
        reg.register("b", lambda: "", {}, toolset="ts2")
        assert sorted(reg.list_toolsets()) == ["ts1", "ts2"]

    def test_list_profiles(self):
        reg = ToolRegistry()
        profiles = reg.list_profiles()
        assert "default" in profiles
        assert "research" in profiles
        assert "coding" in profiles
        assert "minimal" in profiles

    def test_register_profile(self):
        reg = ToolRegistry()
        reg.register_profile("custom", {"web": 0.5})
        assert "custom" in reg.list_profiles()
