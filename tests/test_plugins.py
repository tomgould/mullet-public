"""Tests for the plugin system."""

from mullet.core.plugins import discover_plugins, list_plugins, load_plugins, register_tool
from mullet.core.tools.registry import ToolRegistry


def test_discover_no_plugins():
    """No mullet plugins installed — should return empty."""
    plugins = discover_plugins()
    # Might be 0 or more depending on installed packages
    assert isinstance(plugins, list)


def test_list_plugins_empty():
    """No plugins loaded yet."""
    # Reset state
    from mullet.core import plugins
    plugins._loaded_plugins.clear()
    assert list_plugins() == []


def test_load_plugins_no_crash():
    """load_plugins should not crash even with no plugins."""
    registry = ToolRegistry()
    loaded = load_plugins(tool_registry=registry)
    assert loaded >= 0  # 0 is fine if no plugins installed


def test_register_tool_helper():
    """register_tool helper adds to registry."""
    registry = ToolRegistry()
    register_tool(
        registry,
        name="test_plugin_tool",
        func=lambda **_: "ok",
        schema={"description": "Test plugin tool"},
        toolset="plugin",
    )
    assert "test_plugin_tool" in registry.list_tools()
    assert "plugin" in registry.list_toolsets()


def test_directory_plugin(tmp_path, monkeypatch):
    """Directory plugins should be loaded from .mullet/plugins/."""
    from mullet.core import plugins
    plugins._loaded_plugins.clear()

    # Create a plugin file
    plugin_dir = tmp_path / ".mullet" / "plugins"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "hello_plugin.py").write_text('''
def register(tool_registry=None, **kwargs):
    if tool_registry:
        tool_registry.register(
            name="hello_world",
            func=lambda **kw: "Hello from plugin!",
            schema={"description": "Test", "parameters": {"type": "object", "properties": {}}},
            toolset="plugin",
        )
''')

    monkeypatch.chdir(tmp_path)
    registry = ToolRegistry()
    loaded = load_plugins(tool_registry=registry)
    assert loaded >= 1
    assert "hello_world" in registry.list_tools()


def test_directory_plugin_skips_underscore(tmp_path, monkeypatch):
    """Files starting with _ should be skipped."""
    from mullet.core import plugins
    plugins._loaded_plugins.clear()

    plugin_dir = tmp_path / ".mullet" / "plugins"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "_internal.py").write_text("def register(**kw): pass")

    monkeypatch.chdir(tmp_path)
    registry = ToolRegistry()
    loaded = load_plugins(tool_registry=registry)
    assert loaded == 0


def test_directory_plugin_bad_file(tmp_path, monkeypatch):
    """Broken plugin files should not crash loading."""
    from mullet.core import plugins
    plugins._loaded_plugins.clear()

    plugin_dir = tmp_path / ".mullet" / "plugins"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "broken.py").write_text("raise RuntimeError('broken')")

    monkeypatch.chdir(tmp_path)
    registry = ToolRegistry()
    loaded = load_plugins(tool_registry=registry)
    assert loaded == 0  # Broken plugin skipped


# ---------------------------------------------------------------------------
# Extension points — register / get
# ---------------------------------------------------------------------------

from mullet.core.plugins import register_extension_point, get_extension_point


class TestExtensionPoints:
    def setup_method(self):
        """Clear shared extension point state before each test."""
        from mullet.core import plugins
        plugins._extension_points.clear()

    def test_register_and_get(self):
        class MyInterface:
            pass
        register_extension_point("my_ext", MyInterface)
        result = get_extension_point("my_ext")
        assert result is MyInterface

    def test_get_unknown_returns_none(self):
        assert get_extension_point("does_not_exist") is None

    def test_overwrite_registration(self):
        class OldInterface:
            pass
        class NewInterface:
            pass
        register_extension_point("my_ext", OldInterface)
        register_extension_point("my_ext", NewInterface)
        assert get_extension_point("my_ext") is NewInterface


# ---------------------------------------------------------------------------
# scan_plugin_dirs
# ---------------------------------------------------------------------------

from mullet.core.plugins import scan_plugin_dirs


class TestScanPluginDirs:
    def test_nonexistent_dir_returns_empty(self, tmp_path):
        result = scan_plugin_dirs([str(tmp_path / "nonexistent")])
        assert result == []

    def test_empty_dir_returns_empty(self, tmp_path):
        result = scan_plugin_dirs([str(tmp_path)])
        assert result == []

    def test_finds_python_file(self, tmp_path):
        (tmp_path / "my_plugin.py").write_text("pass")
        result = scan_plugin_dirs([str(tmp_path)])
        assert any("my_plugin.py" in r for r in result)

    def test_skips_underscore_files(self, tmp_path):
        (tmp_path / "_internal.py").write_text("pass")
        result = scan_plugin_dirs([str(tmp_path)])
        assert result == []

    def test_broken_file_skipped_gracefully(self, tmp_path):
        (tmp_path / "broken.py").write_text("raise RuntimeError('boom')")
        result = scan_plugin_dirs([str(tmp_path)])
        assert result == []


# ---------------------------------------------------------------------------
# register_provider
# ---------------------------------------------------------------------------

from mullet.core.plugins import register_provider


class TestRegisterProvider:
    def test_register_adds_to_registry_providers(self):
        registry = type("Registry", (), {"providers": {}})()
        provider = object()
        register_provider(registry, "my_provider", provider)
        assert registry.providers["my_provider"] is provider

    def test_no_providers_attr_does_not_raise(self):
        registry = object()  # no .providers attribute
        register_provider(registry, "x", object())  # must not raise
