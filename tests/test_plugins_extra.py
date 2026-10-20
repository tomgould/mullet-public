"""Extra tests for mullet.core.plugins covering uncovered branches."""

from unittest.mock import MagicMock, patch

import pytest

from mullet.core import plugins as plugins_mod
from mullet.core.plugins import (
    _load_directory_plugins,
    discover_plugins,
    load_plugins,
    per_client_plugin_config,
)
from mullet.core.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# discover_plugins — older Python dict-style entry_points
# ---------------------------------------------------------------------------

class FakeEP:
    def __init__(self, name, value):
        self.name = name
        self.value = value


class FakeSelectableGroups:
    def __init__(self, eps):
        self._eps = eps

    def select(self, *, group):
        return self._eps.get(group, [])


def test_discover_plugins_dict_style_entry_points():
    """Python <3.12 returns a dict from entry_points()."""
    fake_ep = FakeEP("demo", "demo_mod:register")

    with patch("importlib.metadata.entry_points") as mock_eps:
        mock_eps.return_value = {"mullet.plugins": [fake_ep]}
        result = discover_plugins()

    assert len(result) == 1
    assert result[0]["name"] == "demo"


def test_discover_plugins_unknown_eps_type():
    """If eps is neither SelectableGroups nor dict, return empty list."""
    with patch("importlib.metadata.entry_points") as mock_eps:
        mock_eps.return_value = 42  # unexpected type
        result = discover_plugins()

    assert result == []


def test_discover_plugins_exception():
    """If entry_points() itself raises, log and return empty list."""
    with patch("importlib.metadata.entry_points", side_effect=ImportError("no metadata")):
        result = discover_plugins()
    assert result == []


# ---------------------------------------------------------------------------
# load_plugins — entry point loading success / failure
# ---------------------------------------------------------------------------

def test_load_plugins_entry_point_success(monkeypatch, tmp_path):
    """A valid entry-point plugin is loaded and registered."""
    plugins_mod._loaded_plugins.clear()

    register_fn = MagicMock()
    fake_ep = MagicMock()
    fake_ep.load.return_value = register_fn

    with patch("mullet.core.plugins.discover_plugins", return_value=[{
        "name": "my_plugin",
        "module": "my_mod",
        "entry_point": fake_ep,
    }]):
        registry = ToolRegistry()
        loaded = load_plugins(tool_registry=registry)

    assert loaded == 1
    register_fn.assert_called_once_with(
        tool_registry=registry,
        provider_registry=None,
    )


def test_load_plugins_entry_point_not_callable(monkeypatch, tmp_path):
    """If entry_point.load() returns a non-callable, skip it."""
    plugins_mod._loaded_plugins.clear()

    fake_ep = MagicMock()
    fake_ep.load.return_value = "not_callable"

    with patch("mullet.core.plugins.discover_plugins", return_value=[{
        "name": "bad_plugin",
        "module": "bad_mod",
        "entry_point": fake_ep,
    }]):
        loaded = load_plugins()

    assert loaded == 0


def test_load_plugins_entry_point_exception(monkeypatch, tmp_path):
    """If entry_point.load() raises, log and continue."""
    plugins_mod._loaded_plugins.clear()

    fake_ep = MagicMock()
    fake_ep.load.side_effect = RuntimeError("broken")

    with patch("mullet.core.plugins.discover_plugins", return_value=[{
        "name": "broken_plugin",
        "module": "broken_mod",
        "entry_point": fake_ep,
    }]):
        loaded = load_plugins()

    assert loaded == 0


# ---------------------------------------------------------------------------
# _load_directory_plugins — skip already loaded, missing spec/loader
# ---------------------------------------------------------------------------

def test_directory_plugin_skips_already_loaded(monkeypatch, tmp_path):
    """If name already in _loaded_plugins, skip the file."""
    plugins_mod._loaded_plugins.clear()
    plugins_mod._loaded_plugins["duplicate"] = {"name": "duplicate"}

    plugin_dir = tmp_path / ".mullet" / "plugins"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "duplicate.py").write_text("def register(**kw): pass")

    monkeypatch.chdir(tmp_path)
    loaded = _load_directory_plugins()
    assert loaded == 0


def test_directory_plugin_missing_spec_loader(monkeypatch, tmp_path):
    """If spec_from_file_location returns None or no loader, skip."""
    plugins_mod._loaded_plugins.clear()

    plugin_dir = tmp_path / ".mullet" / "plugins"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "nospec.py").write_text("def register(**kw): pass")

    monkeypatch.chdir(tmp_path)

    with patch("importlib.util.spec_from_file_location", return_value=None):
        loaded = _load_directory_plugins()

    assert loaded == 0


# ---------------------------------------------------------------------------
# per_client_plugin_config
# ---------------------------------------------------------------------------

def test_per_client_plugin_config_missing_files(monkeypatch, tmp_path):
    """No config files present => empty dict."""
    with patch("mullet.core.env.get_pools_dir", return_value=tmp_path):
        cfg = per_client_plugin_config("nonexistent_client")
    assert cfg == {}


def test_per_client_plugin_config_from_config_yaml(monkeypatch, tmp_path):
    """Load plugins section from config.yaml."""
    client_dir = tmp_path / "myclient"
    client_dir.mkdir()
    (client_dir / "config.yaml").write_text("plugins:\n  docker:\n    enabled: true\n")

    with patch("mullet.core.env.get_pools_dir", return_value=tmp_path):
        cfg = per_client_plugin_config("myclient")

    assert cfg == {"docker": {"enabled": True}}


def test_per_client_plugin_config_from_client_yaml(monkeypatch, tmp_path):
    """Fallback to client.yaml when config.yaml is missing."""
    client_dir = tmp_path / "myclient"
    client_dir.mkdir()
    (client_dir / "client.yaml").write_text("plugins:\n  git:\n    enabled: false\n")

    with patch("mullet.core.env.get_pools_dir", return_value=tmp_path):
        cfg = per_client_plugin_config("myclient")

    assert cfg == {"git": {"enabled": False}}


def test_per_client_plugin_config_bad_yaml(monkeypatch, tmp_path):
    """Broken YAML returns empty dict."""
    client_dir = tmp_path / "myclient"
    client_dir.mkdir()
    (client_dir / "config.yaml").write_text("{invalid yaml")

    with patch("mullet.core.env.get_pools_dir", return_value=tmp_path):
        cfg = per_client_plugin_config("myclient")

    assert cfg == {}


# ---------------------------------------------------------------------------
# Additional coverage for missing lines when running only this file
# ---------------------------------------------------------------------------

from mullet.core.plugins import (
    list_plugins,
    register_extension_point,
    get_extension_point,
    register_provider,
    register_tool,
    scan_plugin_dirs,
)


class FakeSelectableGroups2:
    def __init__(self, eps):
        self._eps = eps

    def select(self, *, group):
        return self._eps.get(group, [])


def test_discover_plugins_selectable_groups():
    """Python 3.12+ returns SelectableGroups with .select (line 40)."""
    fake_ep = FakeEP("demo", "demo_mod:register")
    with patch("importlib.metadata.entry_points") as mock_eps:
        mock_eps.return_value = FakeSelectableGroups({"mullet.plugins": [fake_ep]})
        result = discover_plugins()
    assert len(result) == 1
    assert result[0]["name"] == "demo"


def test_load_directory_plugins_skips_underscore_files(monkeypatch, tmp_path):
    """Files starting with _ are skipped in _load_directory_plugins (line 111)."""
    plugins_mod._loaded_plugins.clear()
    plugin_dir = tmp_path / ".mullet" / "plugins"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "_hidden.py").write_text("def register(**kw): pass")
    monkeypatch.chdir(tmp_path)
    loaded = _load_directory_plugins()
    assert loaded == 0


def test_load_directory_plugins_loads_valid_plugin(monkeypatch, tmp_path):
    """A valid directory plugin is loaded and registered (lines 121-133)."""
    plugins_mod._loaded_plugins.clear()
    plugin_dir = tmp_path / ".mullet" / "plugins"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "valid.py").write_text('''
def register(tool_registry=None, **kwargs):
    if tool_registry:
        tool_registry.register(
            name="valid_tool",
            func=lambda **kw: "ok",
            schema={"description": "Valid"},
            toolset="plugin",
        )
''')
    monkeypatch.chdir(tmp_path)
    registry = ToolRegistry()
    loaded = _load_directory_plugins(tool_registry=registry)
    assert loaded == 1
    assert "valid_tool" in registry.list_tools()


def test_list_plugins_after_loading(monkeypatch, tmp_path):
    """list_plugins reflects loaded directory plugins (line 140)."""
    plugins_mod._loaded_plugins.clear()
    plugin_dir = tmp_path / ".mullet" / "plugins"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "listed.py").write_text("def register(**kw): pass")
    monkeypatch.chdir(tmp_path)
    _load_directory_plugins()
    plugins = list_plugins()
    names = [p["name"] for p in plugins]
    assert "listed" in names


def test_register_tool_helper_line_150():
    """register_tool helper delegates to registry (line 150)."""
    registry = ToolRegistry()
    register_tool(registry, "helper_tool", lambda **_: "yep", {"description": "Helper"})
    assert "helper_tool" in registry.list_tools()


def test_register_provider_helper_lines_155_156():
    """register_provider adds to registry.providers (lines 155-156)."""
    registry = type("Registry", (), {"providers": {}})()
    provider = object()
    register_provider(registry, "p1", provider)
    assert registry.providers["p1"] is provider


def test_register_and_get_extension_point_lines_168_173():
    """Extension point registration and retrieval (lines 168, 173)."""
    plugins_mod._extension_points.clear()
    class Interface:
        pass
    register_extension_point("iface", Interface)
    assert get_extension_point("iface") is Interface


def test_scan_plugin_dirs_finds_valid_file(tmp_path):
    """scan_plugin_dirs discovers .py files (lines 178-201)."""
    (tmp_path / "plugin.py").write_text("pass")
    result = scan_plugin_dirs([str(tmp_path)])
    assert any("plugin.py" in r for r in result)


def test_scan_plugin_dirs_skips_underscore(tmp_path):
    """scan_plugin_dirs ignores underscore files."""
    (tmp_path / "_skip.py").write_text("pass")
    result = scan_plugin_dirs([str(tmp_path)])
    assert result == []


def test_scan_plugin_dirs_skips_broken_file(tmp_path):
    """scan_plugin_dirs silently skips broken plugins."""
    (tmp_path / "broken.py").write_text("raise RuntimeError('boom')")
    result = scan_plugin_dirs([str(tmp_path)])
    assert result == []


def test_scan_plugin_dirs_nonexistent_dir():
    """scan_plugin_dirs returns empty for missing directories."""
    result = scan_plugin_dirs(["/nonexistent/path/12345"])
    assert result == []


def test_load_directory_plugins_exception_during_load(monkeypatch, tmp_path):
    """Exception during module exec/register is caught (lines 132-133)."""
    plugins_mod._loaded_plugins.clear()
    plugin_dir = tmp_path / ".mullet" / "plugins"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "bad_register.py").write_text('''
def register(tool_registry=None, **kwargs):
    raise RuntimeError("bad register")
''')
    monkeypatch.chdir(tmp_path)
    loaded = _load_directory_plugins()
    assert loaded == 0
