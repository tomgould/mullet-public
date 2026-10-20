"""Tests for mullet.core.orchestration.shared_memory — cross-agent state."""

import time
import pytest
from mullet.core.orchestration.shared_memory import SharedMemory


@pytest.fixture
def sm(tmp_path):
    return SharedMemory(str(tmp_path))


def test_set_and_get(sm):
    sm.set("key1", "value1")
    assert sm.get("key1") == "value1"


def test_get_not_found(sm):
    assert sm.get("nonexistent") is None


def test_set_complex_value(sm):
    sm.set("data", {"nested": [1, 2, 3], "flag": True})
    val = sm.get("data")
    assert val == {"nested": [1, 2, 3], "flag": True}


def test_namespace_isolation(sm):
    sm.set("key", "ns1-value", namespace="ns1")
    sm.set("key", "ns2-value", namespace="ns2")
    assert sm.get("key", namespace="ns1") == "ns1-value"
    assert sm.get("key", namespace="ns2") == "ns2-value"
    assert sm.get("key") is None  # default namespace has nothing


def test_ttl_expiry(sm):
    sm.set("temp", "expires", ttl=1)
    assert sm.get("temp") == "expires"
    time.sleep(1.1)
    assert sm.get("temp") is None


def test_delete(sm):
    sm.set("gone", "soon")
    assert sm.delete("gone") is True
    assert sm.get("gone") is None


def test_delete_not_found(sm):
    assert sm.delete("nope") is False


def test_list_keys(sm):
    sm.set("a", 1)
    sm.set("b", 2)
    sm.set("c", 3)
    keys = sm.list_keys()
    assert sorted(keys) == ["a", "b", "c"]


def test_list_keys_empty_namespace(sm):
    assert sm.list_keys(namespace="empty") == []


def test_list_namespaces(sm):
    sm.set("k", "v", namespace="alpha")
    sm.set("k", "v", namespace="beta")
    ns = sm.list_namespaces()
    assert "alpha" in ns
    assert "beta" in ns


def test_clear(sm):
    sm.set("a", 1, namespace="team")
    sm.set("b", 2, namespace="team")
    count = sm.clear(namespace="team")
    assert count == 2
    assert sm.list_keys(namespace="team") == []


def test_clear_empty(sm):
    assert sm.clear(namespace="empty") == 0


def test_overwrite(sm):
    sm.set("key", "old")
    sm.set("key", "new")
    assert sm.get("key") == "new"


def test_sanitize_name(sm):
    sm.set("bad/key.name", "value", namespace="bad/ns.name")
    assert sm.get("bad/key.name", namespace="bad/ns.name") == "value"
