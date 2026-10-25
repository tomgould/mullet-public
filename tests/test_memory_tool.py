import pytest
import os
import json
import time
from unittest.mock import patch

from mullet.core.tools.memory import MemoryStore, memory_handler, MEMORY_SCHEMA, register

# Fixture for a temporary memory store instance
@pytest.fixture
def temp_memory_store():
    test_dir = '/tmp/mullet_test_memory'
    if os.path.exists(test_dir):
        import shutil
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)
    store = MemoryStore(storage_dir=test_dir)
    yield store
    if os.path.exists(test_dir):
        import shutil
        shutil.rmtree(test_dir)

    # Fixture for a clean memory.json file for handler tests
@pytest.fixture
def clean_handler_memory_file():
    test_dir = '/tmp/mullet_test_handler_memory'
    if os.path.exists(test_dir):
        import shutil
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)

    # Create a real MemoryStore instance and patch the global _memory_store
    test_store = MemoryStore(storage_dir=test_dir)
    with patch('mullet.core.tools.memory._memory_store', test_store):
        yield

    # Clean up after the test
    if os.path.exists(test_dir):
        import shutil
        shutil.rmtree(test_dir)

class TestMemoryStore:

    def test_init_creates_directory_and_file(self, temp_memory_store):
        assert os.path.exists(temp_memory_store.storage_dir)
        assert os.path.exists(temp_memory_store.storage_file)

    def test_set_and_get(self, temp_memory_store):
        temp_memory_store.set("test_key", "test_value")
        assert temp_memory_store.get("test_key") == "test_value"

    def test_get_non_existent_key(self, temp_memory_store):
        assert temp_memory_store.get("non_existent") is None

    def test_set_with_ttl_expires(self, temp_memory_store):
        temp_memory_store.set("ttl_key", "ttl_value", ttl=1)
        time.sleep(1.5)
        assert temp_memory_store.get("ttl_key") is None

    def test_set_with_ttl_not_expired(self, temp_memory_store):
        temp_memory_store.set("ttl_key", "ttl_value", ttl=2)
        time.sleep(0.5)
        assert temp_memory_store.get("ttl_key") == "ttl_value"

    def test_delete_existing_key(self, temp_memory_store):
        temp_memory_store.set("del_key", "del_value")
        assert temp_memory_store.delete("del_key") is True
        assert temp_memory_store.get("del_key") is None

    def test_delete_non_existent_key(self, temp_memory_store):
        assert temp_memory_store.delete("non_existent_del") is False

    def test_search_by_key(self, temp_memory_store):
        temp_memory_store.set("search_key_1", "value_a")
        temp_memory_store.set("another_key", "value_b")
        results = temp_memory_store.search("search")
        assert len(results) == 1
        assert results[0]['key'] == "search_key_1"

    def test_search_by_value(self, temp_memory_store):
        temp_memory_store.set("key_c", "search_value_1")
        temp_memory_store.set("key_d", "another_value")
        results = temp_memory_store.search("search_value")
        assert len(results) == 1
        assert results[0]['value'] == "search_value_1"

    def test_list_keys(self, temp_memory_store):
        temp_memory_store.set("key1", "val1")
        temp_memory_store.set("key2", "val2")
        keys = temp_memory_store.list_keys()
        assert set(keys) == {"key1", "key2"}

    def test_list_keys_excludes_expired(self, temp_memory_store):
        temp_memory_store.set("key_exp", "val_exp", ttl=1)
        temp_memory_store.set("key_non_exp", "val_non_exp")
        time.sleep(1.5)
        keys = temp_memory_store.list_keys()
        assert set(keys) == {"key_non_exp"}

    def test_clear(self, temp_memory_store):
        temp_memory_store.set("key1", "val1")
        temp_memory_store.set("key2", "val2")
        count = temp_memory_store.clear()
        assert count == 2
        assert len(temp_memory_store.list_keys()) == 0

    def test_persistence_across_instances(self, temp_memory_store):
        temp_memory_store.set("p_key", "p_value")
        temp_memory_store._save() # Ensure it's saved
        
        # Create a new instance pointing to the same storage_dir
        new_store = MemoryStore(storage_dir=temp_memory_store.storage_dir)
        assert new_store.get("p_key") == "p_value"

class TestMemoryHandler:

    def test_memory_handler_set(self, clean_handler_memory_file):
        result = memory_handler(action="set", key="h_key", value="h_value")
        assert result == "Key 'h_key' set successfully."
        # Verify via handler get
        assert memory_handler(action="get", key="h_key") == "h_value"

    def test_memory_handler_get(self, clean_handler_memory_file):
        memory_handler(action="set", key="h_get_key", value="h_get_value")
        result = memory_handler(action="get", key="h_get_key")
        assert result == "h_get_value"

    def test_memory_handler_get_non_existent(self, clean_handler_memory_file):
        result = memory_handler(action="get", key="non_existent_h")
        assert result == "Key 'non_existent_h' not found or expired."

    def test_memory_handler_delete(self, clean_handler_memory_file):
        memory_handler(action="set", key="h_del_key", value="h_del_value")
        result = memory_handler(action="delete", key="h_del_key")
        assert result == "Key 'h_del_key' deleted successfully."

    def test_memory_handler_search(self, clean_handler_memory_file):
        memory_handler(action="set", key="h_search_key", value="search_val")
        memory_handler(action="set", key="other_key", value="other_val")
        result = memory_handler(action="search", query="search")
        parsed_result = json.loads(result)
        assert len(parsed_result) >= 1

    def test_memory_handler_list_keys(self, clean_handler_memory_file):
        memory_handler(action="set", key="h_list_key1", value="val1")
        memory_handler(action="set", key="h_list_key2", value="val2")
        result = memory_handler(action="list_keys")
        parsed_result = json.loads(result)
        assert "h_list_key1" in parsed_result
        assert "h_list_key2" in parsed_result

    def test_memory_handler_clear(self, clean_handler_memory_file):
        memory_handler(action="set", key="h_clear_key1", value="val1")
        result = memory_handler(action="clear")
        assert "Cleared" in result

    def test_memory_handler_set_missing_key_or_value(self, clean_handler_memory_file):
        result = memory_handler(action="set", value="no_key")
        assert "Error: 'key' and 'value' are required" in result
        result = memory_handler(action="set", key="no_value")
        assert "Error: 'key' and 'value' are required" in result

    def test_memory_handler_get_missing_key(self, clean_handler_memory_file):
        result = memory_handler(action="get")
        assert "Error: 'key' is required for 'get' action." in result

    def test_memory_handler_delete_missing_key(self, clean_handler_memory_file):
        result = memory_handler(action="delete")
        assert "Error: 'key' is required for 'delete' action." in result

    def test_memory_handler_search_missing_query(self, clean_handler_memory_file):
        result = memory_handler(action="search")
        assert "Error: 'query' is required for 'search' action." in result

    def test_memory_handler_unknown_action(self, clean_handler_memory_file):
        result = memory_handler(action="unknown_action")
        assert "Error: Unknown action 'unknown_action'." in result

    def test_register_function(self):
        class MockRegistry:
            def __init__(self):
                self.registered_tools = {}

            def register(self, name, func, schema, toolset="default"):
                self.registered_tools[name] = {"schema": schema, "func": func, "toolset": toolset}

        mock_registry = MockRegistry()
        register(mock_registry)
        assert "memory" in mock_registry.registered_tools
        assert mock_registry.registered_tools["memory"]["schema"] == MEMORY_SCHEMA
        assert mock_registry.registered_tools["memory"]["func"] == memory_handler

class TestMemoryMissingLines:
    def test_load_corrupted_json(self, tmp_path):
        from mullet.core.tools.memory import MemoryStore
        memory_file = tmp_path / "memory.json"
        memory_file.write_text("not valid json {")
        store = MemoryStore(storage_dir=str(tmp_path))
        assert store._memory == {}

    def test_memory_handler_delete_not_found(self, clean_handler_memory_file):
        result = memory_handler(action="delete", key="nonexistent_key")
        assert result == "Key 'nonexistent_key' not found."

    def test_main_block(self):
        import runpy
        import sys
        import os
        import shutil
        test_dir = '/tmp/mullet_memory_test'
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        mod_name = 'mullet.core.tools.memory'
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        runpy.run_module(mod_name, run_name='__main__')
        assert os.path.exists(test_dir)
        shutil.rmtree(test_dir)


class TestMemoryLocation:
    def test_import_writes_nothing_to_the_working_directory(self, tmp_path, monkeypatch):
        import importlib
        import mullet.core.tools.memory as memory_mod
        monkeypatch.chdir(tmp_path)
        importlib.reload(memory_mod)
        assert list(tmp_path.iterdir()) == []

    def test_default_store_lives_in_the_data_dir(self, tmp_path, monkeypatch):
        from mullet.core import paths
        monkeypatch.chdir(tmp_path)
        store = MemoryStore()
        assert store.storage_dir == str(paths.data_dir() / "memory")
        assert not (tmp_path / ".mullet").exists()

    def test_handler_creates_the_default_store_on_first_use(self, monkeypatch):
        import mullet.core.tools.memory as memory_mod
        monkeypatch.setattr(memory_mod, "_memory_store", None)
        assert memory_mod.memory_handler(action="list_keys") == "[]"
        assert memory_mod._memory_store is not None
