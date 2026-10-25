import json
import pytest
from unittest.mock import Mock

from mullet.core.tools import shared_mem
from mullet.core.orchestration.shared_memory import SharedMemory as BaseSharedMemory


@pytest.fixture
def mock_registry():
    return Mock()

@pytest.fixture
def mock_shared_memory(tmp_path):
    original_shared_memory = shared_mem.shared_memory
    # Create a new SharedMemory instance with the temporary path
    test_instance = BaseSharedMemory(base_dir=str(tmp_path))
    shared_mem.shared_memory = test_instance
    yield test_instance
    # Restore the original SharedMemory instance
    shared_mem.shared_memory = original_shared_memory


def test_shared_mem_schema_has_required_action_field():
    assert "action" in shared_mem.SHARED_MEM_SCHEMA["parameters"]["properties"]
    # The 'action' field is not explicitly in 'required' because it has a default,
    # but it is conceptually required for the tool to do anything.
    # We can check if it's an enum, indicating it's a primary control.
    assert "enum" in shared_mem.SHARED_MEM_SCHEMA["parameters"]["properties"]["action"]


def test_shared_mem_register_with_mock_registry(mock_registry):
    shared_mem.register(mock_registry)
    mock_registry.register.assert_called_once_with(
        name="shared_mem",
        func=shared_mem.shared_mem_handler,
        schema=shared_mem.SHARED_MEM_SCHEMA,
        toolset="memory",
    )


def test_shared_mem_set_and_get_roundtrip(mock_shared_memory):
    key = "test_key"
    value = "test_value"
    namespace = "test_namespace"

    # Test set
    set_result = shared_mem.shared_mem_handler(action="set", key=key, value=value, namespace=namespace)
    set_data = json.loads(set_result)
    assert set_data["status"] == "success"
    assert set_data["action"] == "set"
    assert set_data["key"] == key
    assert set_data["namespace"] == namespace

    # Test get
    get_result = shared_mem.shared_mem_handler(action="get", key=key, namespace=namespace)
    get_data = json.loads(get_result)
    assert get_data["status"] == "success"
    assert get_data["action"] == "get"
    assert get_data["key"] == key
    assert get_data["namespace"] == namespace
    assert get_data["value"] == value


def test_shared_mem_list_keys(mock_shared_memory):
    namespace = "list_test_namespace"
    shared_mem.shared_mem_handler(action="set", key="key1", value="val1", namespace=namespace)
    shared_mem.shared_mem_handler(action="set", key="key2", value="val2", namespace=namespace)

    list_result = shared_mem.shared_mem_handler(action="list_keys", namespace=namespace)
    list_data = json.loads(list_result)
    assert list_data["status"] == "success"
    assert list_data["action"] == "list_keys"
    assert list_data["namespace"] == namespace
    assert "key1" in list_data["keys"]
    assert "key2" in list_data["keys"]


def test_shared_mem_set_missing_key_returns_error(mock_shared_memory):
    result = json.loads(shared_mem.shared_mem_handler(action="set", value="v"))
    assert "error" in result
    assert "required" in result["error"].lower() or "key" in result["error"].lower()


def test_shared_mem_set_missing_value_returns_error(mock_shared_memory):
    result = json.loads(shared_mem.shared_mem_handler(action="set", key="k"))
    assert "error" in result


def test_shared_mem_get_missing_key_returns_error(mock_shared_memory):
    result = json.loads(shared_mem.shared_mem_handler(action="get"))
    assert "error" in result
    assert "key" in result["error"].lower()


def test_shared_mem_delete_missing_key_returns_error(mock_shared_memory):
    result = json.loads(shared_mem.shared_mem_handler(action="delete"))
    assert "error" in result
    assert "key" in result["error"].lower()


def test_shared_mem_list_namespaces(mock_shared_memory):
    shared_mem.shared_mem_handler(action="set", key="k", value="v", namespace="ns_a")
    shared_mem.shared_mem_handler(action="set", key="k", value="v", namespace="ns_b")
    result = json.loads(shared_mem.shared_mem_handler(action="list_namespaces"))
    assert result["status"] == "success"
    assert "ns_a" in result["namespaces"]
    assert "ns_b" in result["namespaces"]


def test_shared_mem_invalid_action_returns_error(mock_shared_memory):
    result = json.loads(shared_mem.shared_mem_handler(action="frobnicate"))
    assert "error" in result
    assert "frobnicate" in result["error"]


def test_shared_mem_handler_exception_returns_error(mock_shared_memory):
    from unittest.mock import patch
    with patch.object(mock_shared_memory, "list_keys", side_effect=RuntimeError("boom")):
        result = json.loads(shared_mem.shared_mem_handler(action="list_keys"))
    assert "error" in result
    assert "boom" in result["error"]


def test_shared_mem_delete(mock_shared_memory):
    key = "delete_key"
    value = "delete_value"
    namespace = "delete_namespace"

    shared_mem.shared_mem_handler(action="set", key=key, value=value, namespace=namespace)
    
    # Verify it's there
    get_result_before_delete = shared_mem.shared_mem_handler(action="get", key=key, namespace=namespace)
    get_data_before_delete = json.loads(get_result_before_delete)
    assert get_data_before_delete["value"] == value

    # Test delete
    delete_result = shared_mem.shared_mem_handler(action="delete", key=key, namespace=namespace)
    delete_data = json.loads(delete_result)
    assert delete_data["status"] == "success"
    assert delete_data["action"] == "delete"
    assert delete_data["key"] == key
    assert delete_data["namespace"] == namespace
    assert delete_data["deleted"] is True

    # Verify it's gone
    get_result_after_delete = shared_mem.shared_mem_handler(action="get", key=key, namespace=namespace)
    get_data_after_delete = json.loads(get_result_after_delete)
    assert get_data_after_delete["value"] is None


def test_import_writes_nothing_to_the_working_directory(tmp_path, monkeypatch):
    import importlib
    monkeypatch.chdir(tmp_path)
    importlib.reload(shared_mem)
    assert list(tmp_path.iterdir()) == []
    assert shared_mem.shared_memory is None


def test_default_store_lives_in_the_state_dir(tmp_path, monkeypatch):
    from mullet.core import paths
    monkeypatch.chdir(tmp_path)
    store = BaseSharedMemory()
    assert store.base_dir == (paths.state_dir() / "shared-memory").resolve()
    assert not (tmp_path / ".mullet").exists()
