
import sys
import pytest
from unittest.mock import MagicMock, patch

from mullet.knowledge.engine.plugins.loader import discover_plugins, load_plugin, load_plugins
from mullet.knowledge.engine.plugins.registry import PluginRegistry
from mullet.knowledge.engine.plugins.interfaces import BaseEmbedder, BaseChunker

# --- PluginRegistry Tests ---

class TestPluginRegistry:
    def test_init(self):
        registry = PluginRegistry()
        assert registry.list_all() == {
            "embedders": [],
            "chunkers": [],
            "parsers": [],
            "rerankers": [],
            "llm_providers": [],
            "merge_strategies": [],
        }

    def test_register_and_get_embedder(self):
        registry = PluginRegistry()

        class MyEmbedder(BaseEmbedder):
            name = "my_embedder"

        registry.register_embedder(MyEmbedder)
        assert registry.get_embedder("my_embedder") == MyEmbedder
        assert "my_embedder" in registry.list_embedders()

    def test_register_and_get_chunker(self):
        registry = PluginRegistry()

        class MyChunker(BaseChunker):
            name = "my_chunker"

        registry.register_chunker(MyChunker)
        assert registry.get_chunker("my_chunker") == MyChunker
        assert "my_chunker" in registry.list_chunkers()

    def test_register_without_name_raises_error(self):
        registry = PluginRegistry()

        class BadEmbedder(BaseEmbedder):
            pass

        with pytest.raises(ValueError, match="must define a 'name' attribute"):
            registry.register_embedder(BadEmbedder)

    def test_override_plugin_warns(self, caplog):
        registry = PluginRegistry()

        class EmbedderV1(BaseEmbedder):
            name = "test_embedder"

        class EmbedderV2(BaseEmbedder):
            name = "test_embedder"

        registry.register_embedder(EmbedderV1)
        with caplog.at_level("WARNING"):
            registry.register_embedder(EmbedderV2)
            assert "Overriding embedder plugin 'test_embedder': EmbedderV1 -> EmbedderV2" in caplog.text
        assert registry.get_embedder("test_embedder") == EmbedderV2

    def test_list_all(self):
        registry = PluginRegistry()

        class E1(BaseEmbedder):
            name = "e1"

        class E2(BaseEmbedder):
            name = "e2"

        class C1(BaseChunker):
            name = "c1"

        registry.register_embedder(E1)
        registry.register_embedder(E2)
        registry.register_chunker(C1)

        result = registry.list_all()
        assert result["embedders"] == ["e1", "e2"]
        assert result["chunkers"] == ["c1"]
        assert result["parsers"] == []

# --- PluginLoader Tests ---

def test_discover_plugins_empty_dir(tmp_path):
    assert discover_plugins(tmp_path) == []

def test_discover_plugins_py_file(tmp_path):
    (tmp_path / "my_plugin.py").touch()
    (tmp_path / "__init__.py").touch() # Should be ignored
    (tmp_path / ".hidden_plugin.py").touch() # Should be ignored
    assert discover_plugins(tmp_path) == [tmp_path / "my_plugin.py"]

def test_discover_plugins_package(tmp_path):
    plugin_dir = tmp_path / "my_package"
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").touch()
    (tmp_path / "another_file.txt").touch() # Should be ignored
    assert discover_plugins(tmp_path) == [plugin_dir]

def test_discover_plugins_mixed(tmp_path):
    plugin_file = tmp_path / "file_plugin.py"
    plugin_file.touch()

    package_dir = tmp_path / "package_plugin"
    package_dir.mkdir()
    (package_dir / "__init__.py").touch()

    (tmp_path / "_internal.py").touch() # Ignored
    (tmp_path / ".dotfile").touch() # Ignored

    expected = sorted([plugin_file, package_dir])
    assert discover_plugins(tmp_path) == expected

# patch.dict keeps sys.modules intact and removes the plugin module afterwards.
# Replacing sys.modules with {} (as these tests used to) made patch() re-import
# a fresh importlib.util on Python 3.11+, so the loader kept the real one.
@patch("importlib.util.spec_from_file_location")
@patch("importlib.util.module_from_spec")
@patch.dict("sys.modules")
def test_load_plugin_success_file(mock_module_from_spec, mock_spec_from_file_location, tmp_path):
    plugin_path = tmp_path / "my_plugin.py"
    plugin_path.write_text("def register(registry): registry.registered = True")

    mock_spec = MagicMock()
    mock_spec.loader = MagicMock()
    mock_spec_from_file_location.return_value = mock_spec

    mock_module = MagicMock()
    mock_module.register = MagicMock()
    mock_module_from_spec.return_value = mock_module

    registry = MagicMock(spec=PluginRegistry)
    registry.registered = False

    assert load_plugin(plugin_path, registry) is True
    mock_spec_from_file_location.assert_called_once_with(
        "trove_plugin_my_plugin", plugin_path
    )
    mock_spec.loader.exec_module.assert_called_once_with(mock_module)
    mock_module.register.assert_called_once_with(registry)
    assert sys.modules["trove_plugin_my_plugin"] == mock_module

# patch.dict keeps sys.modules intact and removes the plugin module afterwards.
# Replacing sys.modules with {} (as these tests used to) made patch() re-import
# a fresh importlib.util on Python 3.11+, so the loader kept the real one.
@patch("importlib.util.spec_from_file_location")
@patch("importlib.util.module_from_spec")
@patch.dict("sys.modules")
def test_load_plugin_success_package(mock_module_from_spec, mock_spec_from_file_location, tmp_path):
    plugin_dir = tmp_path / "my_package"
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text("def register(registry): registry.registered = True")

    mock_spec = MagicMock()
    mock_spec.loader = MagicMock()
    mock_spec_from_file_location.return_value = mock_spec

    mock_module = MagicMock()
    mock_module.register = MagicMock()
    mock_module_from_spec.return_value = mock_module

    registry = MagicMock(spec=PluginRegistry)
    registry.registered = False

    assert load_plugin(plugin_dir, registry) is True
    mock_spec_from_file_location.assert_called_once_with(
        "trove_plugin_my_package", plugin_dir / "__init__.py", submodule_search_locations=[str(plugin_dir)]
    )
    mock_spec.loader.exec_module.assert_called_once_with(mock_module)
    mock_module.register.assert_called_once_with(registry)
    assert sys.modules["trove_plugin_my_package"] == mock_module

@patch("importlib.util.spec_from_file_location", return_value=None)
def test_load_plugin_no_spec(mock_spec_from_file_location, tmp_path, caplog):
    plugin_path = tmp_path / "no_spec.py"
    plugin_path.touch()
    registry = MagicMock(spec=PluginRegistry)
    with caplog.at_level("WARNING"):
        assert load_plugin(plugin_path, registry) is False
        assert "Could not create module spec" in caplog.text

@patch("importlib.util.spec_from_file_location")
@patch("importlib.util.module_from_spec")
def test_load_plugin_no_register_function(mock_module_from_spec, mock_spec_from_file_location, tmp_path, caplog):
    plugin_path = tmp_path / "no_register.py"
    plugin_path.write_text("class MyClass: pass")

    mock_spec = MagicMock()
    mock_spec.loader = MagicMock()
    mock_spec_from_file_location.return_value = mock_spec

    mock_module = MagicMock(spec=object)
    mock_module_from_spec.return_value = mock_module

    registry = MagicMock(spec=PluginRegistry)

    with caplog.at_level("WARNING"):
        assert load_plugin(plugin_path, registry) is False
        assert "has no register() function" in caplog.text


@patch("importlib.util.spec_from_file_location")
@patch("importlib.util.module_from_spec")
def test_load_plugin_import_error(mock_module_from_spec, mock_spec_from_file_location, tmp_path, caplog):
    plugin_path = tmp_path / "bad_import.py"
    plugin_path.write_text("raise ImportError('Test error')")

    mock_spec = MagicMock()
    mock_spec.loader = MagicMock()
    mock_spec_from_file_location.return_value = mock_spec

    mock_module_from_spec.side_effect = ImportError("Test error")

    registry = MagicMock(spec=PluginRegistry)

    with caplog.at_level("ERROR"):
        assert load_plugin(plugin_path, registry) is False
        assert "Failed to import plugin" in caplog.text

@patch("mullet.knowledge.engine.plugins.loader.discover_plugins", return_value=[])
def test_load_plugins_no_plugins_found(mock_discover_plugins, tmp_path):
    registry = PluginRegistry()
    result_registry = load_plugins(tmp_path, registry)
    assert result_registry is registry
    assert registry.list_all()["embedders"] == []

@patch("mullet.knowledge.engine.plugins.loader.load_plugin")
def test_load_plugins_success_and_failure(mock_load_plugin, tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "plugin_a.py").touch()
    (plugin_dir / "plugin_b.py").touch()

    # Simulate plugin_a loading successfully, plugin_b failing
    mock_load_plugin.side_effect = [True, False]

    registry = PluginRegistry()
    with patch("mullet.knowledge.engine.plugins.loader.discover_plugins", return_value=[
        plugin_dir / "plugin_a.py",
        plugin_dir / "plugin_b.py",
    ]):
        result_registry = load_plugins(plugin_dir, registry)

    assert result_registry is registry
    assert mock_load_plugin.call_count == 2
    mock_load_plugin.assert_any_call(plugin_dir / "plugin_a.py", registry)
    mock_load_plugin.assert_any_call(plugin_dir / "plugin_b.py", registry)


# ---------------------------------------------------------------------------
# Missing-line gap tests
# ---------------------------------------------------------------------------

def test_discover_plugins_nonexistent_dir(tmp_path):
    """Line 35: discover_plugins returns [] when dir doesn't exist."""
    result = discover_plugins(tmp_path / "nonexistent")
    assert result == []


def test_load_plugin_invalid_path(tmp_path):
    """Lines 72-73: path is neither file nor dir → returns False."""
    fake_path = tmp_path / "doesnotexist.py"
    registry = PluginRegistry()
    # The path doesn't exist at all — is_file() and is_dir() both False
    result = load_plugin(fake_path, registry)
    assert result is False


def test_load_plugin_register_raises_returns_false(tmp_path):
    """Lines 101-103: register() raises → returns False."""
    plugin_file = tmp_path / "bad_register.py"
    plugin_file.write_text("def register(registry): raise ValueError('boom')")
    registry = PluginRegistry()
    result = load_plugin(plugin_file, registry)
    assert result is False


def test_load_plugins_creates_registry_when_none(tmp_path):
    """Line 120: load_plugins creates a new PluginRegistry when none given."""
    result = load_plugins(tmp_path / "nonexistent", registry=None)
    assert isinstance(result, PluginRegistry)


def test_load_plugins_default_plugins_dir_when_none(tmp_path):
    """Line 123: load_plugins uses Path('plugins') when plugins_dir is None."""
    # Since Path('plugins') doesn't exist in CWD during tests, it returns early
    result = load_plugins(plugins_dir=None, registry=PluginRegistry())
    assert isinstance(result, PluginRegistry)


def test_load_plugins_nonexistent_dir_returns_registry(tmp_path):
    """Lines 126-127: load_plugins returns registry early when dir doesn't exist."""
    registry = PluginRegistry()
    result = load_plugins(tmp_path / "missing", registry=registry)
    assert result is registry
