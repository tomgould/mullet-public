"""Tests for interaction_log pure helpers and CheckpointManager."""

import os
from pathlib import Path

from mullet.core.logging.interaction_log import _extract_preview, _extract_last_user_message
from mullet.core.agent.checkpoint import CheckpointManager


# ---------------------------------------------------------------------------
# _extract_preview
# ---------------------------------------------------------------------------

class TestExtractPreview:
    def test_short_text_returned_as_is(self):
        assert _extract_preview("hello") == "hello"

    def test_none_returns_empty_string(self):
        assert _extract_preview(None) == ""

    def test_exact_length_not_truncated(self):
        text = "a" * 500
        result = _extract_preview(text, max_length=500)
        assert result == text

    def test_over_length_truncated_with_ellipsis(self):
        text = "a" * 600
        result = _extract_preview(text, max_length=500)
        assert result.endswith("...")
        assert len(result) == 500

    def test_custom_max_length(self):
        result = _extract_preview("hello world", max_length=5)
        assert len(result) == 5
        assert result.endswith("...")

    def test_returns_string(self):
        assert isinstance(_extract_preview("test"), str)


# ---------------------------------------------------------------------------
# _extract_last_user_message
# ---------------------------------------------------------------------------

class TestExtractLastUserMessage:
    def test_empty_messages_returns_empty(self):
        assert _extract_last_user_message([]) == ""

    def test_single_user_message(self):
        messages = [{"role": "user", "content": "Hello!"}]
        result = _extract_last_user_message(messages)
        assert result == "Hello!"

    def test_returns_last_user_message(self):
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Reply"},
            {"role": "user", "content": "Second"},
        ]
        result = _extract_last_user_message(messages)
        assert result == "Second"

    def test_skips_assistant_messages(self):
        messages = [
            {"role": "assistant", "content": "Assistant text"},
        ]
        result = _extract_last_user_message(messages)
        assert result == ""

    def test_content_as_list_of_text_blocks(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello from list"}]}
        ]
        result = _extract_last_user_message(messages)
        assert result == "Hello from list"

    def test_content_as_mixed_list(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First part"},
                    {"type": "image", "url": "img.png"},
                    {"type": "text", "text": "Second part"},
                ]
            }
        ]
        result = _extract_last_user_message(messages)
        assert "First part" in result
        assert "Second part" in result

    def test_content_as_string_in_list(self):
        messages = [{"role": "user", "content": ["plain string"]}]
        result = _extract_last_user_message(messages)
        assert "plain string" in result

    def test_no_user_messages_returns_empty(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "reply"},
        ]
        assert _extract_last_user_message(messages) == ""


# ---------------------------------------------------------------------------
# CheckpointManager — create, list, cleanup
# ---------------------------------------------------------------------------

class TestCheckpointManagerInit:
    def test_creates_base_dir(self, tmp_path):
        base = str(tmp_path / "checkpoints")
        mgr = CheckpointManager(base_dir=base)
        assert os.path.isdir(base)

    def test_default_max_snapshots(self, tmp_path):
        mgr = CheckpointManager(base_dir=str(tmp_path / "ckpts"))
        assert mgr.max_snapshots == 5

    def test_custom_max_snapshots(self, tmp_path):
        mgr = CheckpointManager(base_dir=str(tmp_path / "ckpts"), max_snapshots=3)
        assert mgr.max_snapshots == 3


class TestCheckpointManagerCreateSnapshot:
    def test_create_returns_timestamp_string(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        (work / "file.txt").write_text("content")
        mgr = CheckpointManager(base_dir=str(tmp_path / "ckpts"))
        ts = mgr.create_snapshot(str(work))
        assert isinstance(ts, str)
        assert len(ts) == 14  # YYYYMMDDHHmmss

    def test_create_makes_snapshot_dir(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        (work / "file.txt").write_text("content")
        mgr = CheckpointManager(base_dir=str(tmp_path / "ckpts"))
        ts = mgr.create_snapshot(str(work))
        snapshot_dir = Path(mgr.base_dir) / ts
        assert snapshot_dir.is_dir()

    def test_create_writes_metadata(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        (work / "f.txt").write_text("x")
        mgr = CheckpointManager(base_dir=str(tmp_path / "ckpts"))
        ts = mgr.create_snapshot(str(work), label="my-label")
        meta = Path(mgr.base_dir) / ts / "metadata.txt"
        assert meta.exists()
        content = meta.read_text()
        assert "my-label" in content


class TestCheckpointManagerListSnapshots:
    def test_empty_returns_empty_list(self, tmp_path):
        mgr = CheckpointManager(base_dir=str(tmp_path / "ckpts"))
        assert mgr.list_snapshots() == []

    def test_returns_list_of_dicts(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        (work / "f.txt").write_text("x")
        mgr = CheckpointManager(base_dir=str(tmp_path / "ckpts"))
        mgr.create_snapshot(str(work))
        snapshots = mgr.list_snapshots()
        assert isinstance(snapshots, list)
        assert isinstance(snapshots[0], dict)

    def test_snapshot_has_expected_keys(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        (work / "f.txt").write_text("x")
        mgr = CheckpointManager(base_dir=str(tmp_path / "ckpts"))
        mgr.create_snapshot(str(work))
        s = mgr.list_snapshots()[0]
        assert "id" in s
        assert "label" in s
        assert "work_dir" in s

    def test_label_stored_correctly(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        (work / "f.txt").write_text("x")
        mgr = CheckpointManager(base_dir=str(tmp_path / "ckpts"))
        mgr.create_snapshot(str(work), label="test-label")
        s = mgr.list_snapshots()[0]
        assert s["label"] == "test-label"


class TestCheckpointManagerCleanup:
    def test_cleanup_removes_old_snapshots(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        (work / "f.txt").write_text("x")
        mgr = CheckpointManager(base_dir=str(tmp_path / "ckpts"), max_snapshots=2)
        for _ in range(3):
            mgr.create_snapshot(str(work))
        snapshots = mgr.list_snapshots()
        assert len(snapshots) <= 2
