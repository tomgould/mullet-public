"""Tests for mullet.core.agent.checkpoint.CheckpointManager — filesystem operations."""

import time
import pytest
from pathlib import Path

from mullet.core.agent.checkpoint import CheckpointManager


@pytest.fixture
def checkpoints_dir(tmp_path):
    return str(tmp_path / "checkpoints")


@pytest.fixture
def work_dir(tmp_path):
    d = tmp_path / "workdir"
    d.mkdir()
    (d / "file_a.txt").write_text("content A")
    (d / "file_b.txt").write_text("content B")
    return str(d)


@pytest.fixture
def mgr(checkpoints_dir):
    return CheckpointManager(base_dir=checkpoints_dir, max_snapshots=5)


# ---------------------------------------------------------------------------
# create_snapshot
# ---------------------------------------------------------------------------

class TestCreateSnapshot:
    def test_returns_timestamp_string(self, mgr, work_dir):
        ts = mgr.create_snapshot(work_dir)
        assert isinstance(ts, str)
        assert len(ts) == 14  # YYYYMMDDHHmmss

    def test_creates_snapshot_directory(self, mgr, work_dir):
        ts = mgr.create_snapshot(work_dir)
        snapshot_path = Path(mgr.base_dir) / ts
        assert snapshot_path.is_dir()

    def test_copies_files_from_work_dir(self, mgr, work_dir):
        ts = mgr.create_snapshot(work_dir)
        snapshot_path = Path(mgr.base_dir) / ts
        assert (snapshot_path / "file_a.txt").exists()
        assert (snapshot_path / "file_a.txt").read_text() == "content A"

    def test_metadata_file_created(self, mgr, work_dir):
        ts = mgr.create_snapshot(work_dir, label="my-label")
        metadata = (Path(mgr.base_dir) / ts / "metadata.txt").read_text()
        assert "my-label" in metadata
        assert work_dir in metadata

    def test_empty_label_stored(self, mgr, work_dir):
        ts = mgr.create_snapshot(work_dir)
        metadata = (Path(mgr.base_dir) / ts / "metadata.txt").read_text()
        assert "label:" in metadata

    def test_multiple_snapshots_unique_ids(self, mgr, work_dir):
        ts1 = mgr.create_snapshot(work_dir)
        time.sleep(1.1)
        ts2 = mgr.create_snapshot(work_dir)
        assert ts1 != ts2


# ---------------------------------------------------------------------------
# list_snapshots
# ---------------------------------------------------------------------------

class TestListSnapshots:
    def test_empty_returns_empty_list(self, mgr):
        assert mgr.list_snapshots() == []

    def test_returns_list_of_dicts(self, mgr, work_dir):
        mgr.create_snapshot(work_dir, label="test")
        snapshots = mgr.list_snapshots()
        assert isinstance(snapshots, list)
        assert isinstance(snapshots[0], dict)

    def test_snapshot_has_required_keys(self, mgr, work_dir):
        mgr.create_snapshot(work_dir, label="test")
        s = mgr.list_snapshots()[0]
        assert "id" in s
        assert "label" in s
        assert "work_dir" in s

    def test_label_preserved(self, mgr, work_dir):
        mgr.create_snapshot(work_dir, label="my-label")
        s = mgr.list_snapshots()[0]
        assert s["label"] == "my-label"

    def test_work_dir_preserved(self, mgr, work_dir):
        mgr.create_snapshot(work_dir, label="x")
        s = mgr.list_snapshots()[0]
        assert s["work_dir"] == work_dir

    def test_sorted_newest_first(self, mgr, work_dir):
        mgr.create_snapshot(work_dir, label="first")
        time.sleep(1.1)
        mgr.create_snapshot(work_dir, label="second")
        snapshots = mgr.list_snapshots()
        assert snapshots[0]["label"] == "second"


# ---------------------------------------------------------------------------
# restore_snapshot
# ---------------------------------------------------------------------------

class TestRestoreSnapshot:
    def test_restores_file_content(self, mgr, work_dir):
        ts = mgr.create_snapshot(work_dir, label="backup")
        # Modify the work file
        Path(work_dir, "file_a.txt").write_text("modified")
        mgr.restore_snapshot(ts)
        assert Path(work_dir, "file_a.txt").read_text() == "content A"

    def test_nonexistent_snapshot_raises(self, mgr):
        with pytest.raises(ValueError, match="not found"):
            mgr.restore_snapshot("19991231235959")

    def test_snapshot_without_work_dir_metadata_raises(self, mgr, tmp_path):
        # Create a snapshot dir with no metadata
        fake_id = "20000101000000"
        (Path(mgr.base_dir) / fake_id).mkdir(parents=True)
        with pytest.raises(ValueError, match="work_dir"):
            mgr.restore_snapshot(fake_id)


# ---------------------------------------------------------------------------
# _cleanup_old_snapshots
# ---------------------------------------------------------------------------

class TestCleanupOldSnapshots:
    def test_cleanup_respects_max_snapshots(self, work_dir, checkpoints_dir):
        mgr = CheckpointManager(base_dir=checkpoints_dir, max_snapshots=3)
        for i in range(5):
            mgr.create_snapshot(work_dir, label=f"snap-{i}")
            time.sleep(1.1)
        snapshots = mgr.list_snapshots()
        assert len(snapshots) == 3

    def test_oldest_removed(self, work_dir, checkpoints_dir):
        mgr = CheckpointManager(base_dir=checkpoints_dir, max_snapshots=2)
        ts1 = mgr.create_snapshot(work_dir, label="first")
        time.sleep(1.1)
        ts2 = mgr.create_snapshot(work_dir, label="second")
        time.sleep(1.1)
        ts3 = mgr.create_snapshot(work_dir, label="third")
        ids = {s["id"] for s in mgr.list_snapshots()}
        assert ts1 not in ids  # oldest removed
        assert ts2 in ids      # second-most-recent kept
        assert ts3 in ids      # newest kept


# ---------------------------------------------------------------------------
# ensure_checkpoint
# ---------------------------------------------------------------------------

class TestEnsureCheckpoint:
    def test_creates_snapshot_when_none_exists(self, mgr, work_dir):
        snap_id = mgr.ensure_checkpoint(work_dir, label="init")
        assert snap_id in {s["id"] for s in mgr.list_snapshots()}

    def test_reuses_recent_snapshot(self, mgr, work_dir):
        ts1 = mgr.ensure_checkpoint(work_dir, label="init", recent_threshold_seconds=60)
        ts2 = mgr.ensure_checkpoint(work_dir, label="init", recent_threshold_seconds=60)
        assert ts1 == ts2

    def test_creates_new_when_snapshot_expired(self, mgr, work_dir):
        ts1 = mgr.ensure_checkpoint(work_dir, label="init", recent_threshold_seconds=0)
        time.sleep(1.1)
        ts2 = mgr.ensure_checkpoint(work_dir, label="init", recent_threshold_seconds=0)
        assert ts1 != ts2
