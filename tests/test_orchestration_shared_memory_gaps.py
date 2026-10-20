"""Tests for shared_memory.py missing coverage lines."""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from mullet.core.orchestration.shared_memory import SharedMemory


@pytest.fixture
def sm(tmp_path):
    return SharedMemory(str(tmp_path))


class TestSharedMemoryMissingLines:
    def test_set_exception_cleanup(self, sm, tmp_path):
        """Lines 60-65: exception during set cleans up temp file and re-raises."""
        sm.set("key", "value")
        file_path = sm._get_file_path("key", "default")
        with patch(
            "mullet.core.orchestration.shared_memory.os.replace",
            side_effect=OSError("disk full"),
        ):
            with pytest.raises(OSError, match="disk full"):
                sm.set("key", "value")
        # temp file should have been cleaned up
        assert len(list(file_path.parent.glob(".*.tmp"))) == 0

    def test_set_exception_cleanup_unlink_fails(self, sm):
        """Lines 63-64: OSError during temp-file cleanup is swallowed."""
        file_path = sm._get_file_path("key", "default")
        with patch(
            "mullet.core.orchestration.shared_memory.os.replace",
            side_effect=OSError("disk full"),
        ):
            with patch(
                "mullet.core.orchestration.shared_memory.os.unlink",
                side_effect=OSError("cant delete"),
            ):
                with pytest.raises(OSError, match="disk full"):
                    sm.set("key", "value")
        # Even if unlink fails, no exception leaks from cleanup
        assert len(list(file_path.parent.glob(".*.tmp"))) <= 1

    def test_get_corrupted_json(self, sm):
        """Lines 76-78: corrupted JSON file is deleted and returns None."""
        sm.set("key", "value")
        file_path = sm._get_file_path("key", "default")
        file_path.write_text("not json")
        assert sm.get("key") is None
        assert not file_path.exists()

    def test_list_keys_expired_file(self, sm):
        """Lines 111-114: expired file is deleted during list_keys."""
        sm.set("key", "value", ttl=1)
        time.sleep(1.1)
        keys = sm.list_keys()
        assert "key" not in keys
        file_path = sm._get_file_path("key", "default")
        assert not file_path.exists()

    def test_list_keys_corrupted_file(self, sm):
        """Lines 116-118: corrupted file is deleted during list_keys."""
        ns_dir = sm._get_namespace_dir("default")
        ns_dir.mkdir(parents=True, exist_ok=True)
        bad_file = ns_dir / "bad.json"
        bad_file.write_text("not json")
        keys = sm.list_keys()
        assert "bad" not in keys
        assert not bad_file.exists()

    def test_list_keys_missing_file(self, sm):
        """Lines 116-118: FileNotFoundError during list_keys is handled."""
        ns_dir = sm._get_namespace_dir("default")
        ns_dir.mkdir(parents=True, exist_ok=True)
        with patch(
            "mullet.core.orchestration.shared_memory.Path.glob",
            return_value=[ns_dir / "ghost.json"],
        ):
            keys = sm.list_keys()
        assert keys == []

    def test_list_namespaces_no_base_dir(self, sm):
        """Line 124: if base_dir is not a dir, return []."""
        sm.base_dir.rmdir()
        sm.base_dir.write_text("not a dir")
        assert sm.list_namespaces() == []

    def test_clear_oserror_on_file(self, sm, monkeypatch):
        """Lines 138-139: OSError on file deletion during clear is logged."""
        sm.set("a", 1, namespace="team")
        file_path = sm._get_file_path("a", "team")
        original_unlink = Path.unlink

        def bad_unlink(self, missing_ok=False):
            if str(self) == str(file_path):
                raise OSError("permission")
            return original_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", bad_unlink)
        with patch(
            "mullet.core.orchestration.shared_memory.logger.warning"
        ) as mock_warn:
            count = sm.clear(namespace="team")
        assert count == 0
        assert any(
            "Failed to delete file" in str(c) for c in mock_warn.call_args_list
        )

    def test_clear_oserror_on_rmdir(self, sm, monkeypatch):
        """Lines 144-145: OSError on namespace rmdir during clear is logged."""
        sm.set("a", 1, namespace="team")
        ns_dir = sm._get_namespace_dir("team")
        original_rmdir = Path.rmdir

        def bad_rmdir(self):
            if str(self) == str(ns_dir):
                raise OSError("not empty")
            return original_rmdir(self)

        monkeypatch.setattr(Path, "rmdir", bad_rmdir)
        with patch(
            "mullet.core.orchestration.shared_memory.logger.warning"
        ) as mock_warn:
            count = sm.clear(namespace="team")
        assert count == 1
        mock_warn.assert_called_once()

    # --- Additional happy-path & edge-case coverage ---

    def test_get_missing_key(self, sm):
        """Line 71: get() returns None when key does not exist."""
        assert sm.get("nonexistent") is None

    def test_get_happy_path_not_expired(self, sm):
        """Lines 80-86: get() returns value for existing, non-expired key."""
        sm.set("key", "value", ttl=3600)
        assert sm.get("key") == "value"

    def test_get_expired_key(self, sm):
        """Lines 84-85: get() deletes and returns None for expired key."""
        sm.set("key", "value", ttl=1)
        time.sleep(1.1)
        assert sm.get("key") is None
        file_path = sm._get_file_path("key", "default")
        assert not file_path.exists()

    def test_delete_missing_key(self, sm):
        """Line 94: delete() returns False when key does not exist."""
        assert sm.delete("nonexistent") is False

    def test_list_keys_empty_namespace(self, sm):
        """Line 100: list_keys() returns [] for missing namespace."""
        assert sm.list_keys(namespace="empty_ns") == []

    def test_list_keys_appends_valid_key(self, sm):
        """Line 115: list_keys() appends non-expired key."""
        sm.set("alpha", 1)
        sm.set("beta", 2)
        keys = sm.list_keys()
        assert sorted(keys) == ["alpha", "beta"]

    def test_list_namespaces_normal(self, sm):
        """Line 125: list_namespaces() returns existing namespaces."""
        sm.set("k", "v", namespace="ns1")
        sm.set("k", "v", namespace="ns2")
        namespaces = sm.list_namespaces()
        assert "ns1" in namespaces
        assert "ns2" in namespaces

    def test_clear_empty_namespace(self, sm):
        """Line 131: clear() returns 0 for non-existent namespace."""
        assert sm.clear(namespace="empty_ns") == 0
