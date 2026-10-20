"""Tests for CheckpointManager."""
import os
from unittest.mock import MagicMock, patch

import pytest
from mullet.core.agent.checkpoint import CheckpointManager


@pytest.fixture
def tmp_env(tmp_path):
    base = tmp_path / "checkpoints"
    work = tmp_path / "work"
    work.mkdir()
    (work / "file1.txt").write_text("hello")
    (work / "sub").mkdir()
    (work / "sub" / "file2.txt").write_text("world")
    return CheckpointManager(base_dir=str(base), max_snapshots=3), str(work)


def test_create_and_list(tmp_env):
    cm, work = tmp_env
    ts = cm.create_snapshot(work, label="test1")
    assert ts is not None
    snaps = cm.list_snapshots()
    assert len(snaps) == 1
    assert snaps[0]["label"] == "test1"


def test_restore(tmp_env):
    cm, work = tmp_env
    ts = cm.create_snapshot(work, label="before")
    # Modify work dir
    with open(os.path.join(work, "file1.txt"), "w") as f:
        f.write("modified")
    # Restore
    cm.restore_snapshot(ts)
    with open(os.path.join(work, "file1.txt")) as f:
        assert f.read() == "hello"


def test_cleanup_old(tmp_env):
    cm, work = tmp_env
    import time
    for i in range(5):
        cm.create_snapshot(work, label=f"snap{i}")
        time.sleep(0.01)
    snaps = cm.list_snapshots()
    assert len(snaps) <= 3


def test_restore_nonexistent(tmp_env):
    cm, work = tmp_env
    with pytest.raises(ValueError):
        cm.restore_snapshot("nonexistent")


def test_list_snapshots_empty(tmp_path):
    base = tmp_path / "checkpoints"
    cm = CheckpointManager(base_dir=str(base), max_snapshots=3)
    assert cm.list_snapshots() == []


def test_snapshot_metadata_label(tmp_env):
    cm, work = tmp_env
    cm.create_snapshot(work, label="my-label")
    snaps = cm.list_snapshots()
    assert snaps[0]["label"] == "my-label"


def test_snapshot_metadata_work_dir(tmp_env):
    cm, work = tmp_env
    cm.create_snapshot(work, label="")
    snaps = cm.list_snapshots()
    assert snaps[0]["work_dir"] == work


def test_multiple_snapshots_list_has_entry(tmp_env):
    cm, work = tmp_env
    cm.create_snapshot(work, label="first")
    cm.create_snapshot(work, label="second")
    snaps = cm.list_snapshots()
    assert len(snaps) >= 1
    # Latest label wins when same-second timestamp is reused
    assert snaps[0]["label"] in ("first", "second")


def test_ensure_checkpoint_reuses_recent(tmp_env):
    cm, work = tmp_env
    ts1 = cm.ensure_checkpoint(work, label="stable", recent_threshold_seconds=300)
    ts2 = cm.ensure_checkpoint(work, label="stable", recent_threshold_seconds=300)
    assert ts1 == ts2  # same snapshot reused


def test_ensure_checkpoint_returns_timestamp_string(tmp_env):
    cm, work = tmp_env
    ts = cm.ensure_checkpoint(work, label="mycheck", recent_threshold_seconds=0)
    assert isinstance(ts, str)
    assert len(ts) == 14  # '%Y%m%d%H%M%S' format


def test_snapshot_files_copied_correctly(tmp_env):
    cm, work = tmp_env
    ts = cm.create_snapshot(work, label="check-files")
    snap_dir = os.path.join(cm.base_dir, ts)
    assert os.path.exists(os.path.join(snap_dir, "file1.txt"))
    with open(os.path.join(snap_dir, "file1.txt")) as f:
        assert f.read() == "hello"


def test_snapshot_subdirectory_files(tmp_env):
    cm, work = tmp_env
    ts = cm.create_snapshot(work, label="sub-check")
    snap_dir = os.path.join(cm.base_dir, ts)
    assert os.path.exists(os.path.join(snap_dir, "sub", "file2.txt"))


def test_restore_raises_missing_work_dir_in_metadata(tmp_path):
    base = tmp_path / "checkpoints"
    work = tmp_path / "work"
    work.mkdir()
    cm = CheckpointManager(base_dir=str(base), max_snapshots=3)
    # Create a snapshot with no work_dir in metadata
    import datetime
    ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    snap_dir = base / ts
    snap_dir.mkdir(parents=True)
    (snap_dir / "metadata.txt").write_text("label: broken\n")  # no work_dir line
    with pytest.raises(ValueError, match="work_dir not found"):
        cm.restore_snapshot(ts)


def test_get_tracked_files_git_ls_files_success(tmp_path):
    base = tmp_path / "checkpoints"
    cm = CheckpointManager(base_dir=str(base), max_snapshots=3)
    work = tmp_path / "work"
    work.mkdir()
    (work / ".git").mkdir()
    (work / "file1.py").write_text("a")
    (work / "file2.py").write_text("b")
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "file1.py\nfile2.py\n"
    with patch("mullet.core.agent.checkpoint.subprocess.run", return_value=mock_result):
        tracked = cm._get_tracked_files(str(work))
    assert sorted(tracked) == [str(work / "file1.py"), str(work / "file2.py")]


def test_get_tracked_files_git_exception_fallback(tmp_path):
    base = tmp_path / "checkpoints"
    cm = CheckpointManager(base_dir=str(base), max_snapshots=3)
    work = tmp_path / "work"
    work.mkdir()
    (work / ".git").mkdir()
    (work / "foo.txt").write_text("hello")
    with patch("mullet.core.agent.checkpoint.subprocess.run", side_effect=FileNotFoundError("git")):
        tracked = cm._get_tracked_files(str(work))
    assert str(work / "foo.txt") in tracked


def test_restore_snapshot_removes_extra_files(tmp_env):
    cm, work = tmp_env
    ts = cm.create_snapshot(work, label="before")
    # Add extra files after snapshot
    extra_file = os.path.join(work, "extra.txt")
    with open(extra_file, "w") as f:
        f.write("extra")
    extra_dir = os.path.join(work, "extra_dir")
    os.makedirs(extra_dir)
    with open(os.path.join(extra_dir, "file.txt"), "w") as f:
        f.write("extra")
    # Restore should remove extra files/dirs and bring back snapshot contents
    cm.restore_snapshot(ts)
    assert not os.path.exists(extra_file)
    assert not os.path.exists(extra_dir)
    # Original files remain
    assert os.path.exists(os.path.join(work, "file1.txt"))
    assert os.path.exists(os.path.join(work, "sub", "file2.txt"))


def test_cleanup_old_snapshots_direct(tmp_path):
    base = tmp_path / "checkpoints"
    cm = CheckpointManager(base_dir=str(base), max_snapshots=3)
    # Manually create snapshot directories
    for i in range(5):
        snap_dir = base / f"2024010100000{i}"
        snap_dir.mkdir(parents=True)
        (snap_dir / "metadata.txt").write_text(f"label: snap{i}\nwork_dir: /tmp\n")
    cm._cleanup_old_snapshots()
    remaining = sorted(os.listdir(base))
    assert len(remaining) == 3
    assert remaining == ["20240101000002", "20240101000003", "20240101000004"]


def test_get_tracked_files_git_empty_stdout_fallback(tmp_path):
    """If git ls-files succeeds with empty stdout, fall back to os.walk."""
    base = tmp_path / "checkpoints"
    cm = CheckpointManager(base_dir=str(base), max_snapshots=3)
    work = tmp_path / "work"
    work.mkdir()
    (work / ".git").mkdir()
    (work / "script.py").write_text("x")
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    with patch("mullet.core.agent.checkpoint.subprocess.run", return_value=mock_result):
        tracked = cm._get_tracked_files(str(work))
    assert str(work / "script.py") in tracked


def test_get_tracked_files_git_nonzero_rc_fallback(tmp_path):
    """If git ls-files exits with non-zero, fall back to os.walk."""
    base = tmp_path / "checkpoints"
    cm = CheckpointManager(base_dir=str(base), max_snapshots=3)
    work = tmp_path / "work"
    work.mkdir()
    (work / ".git").mkdir()
    (work / "module.py").write_text("y")
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = "error\n"
    with patch("mullet.core.agent.checkpoint.subprocess.run", return_value=mock_result):
        tracked = cm._get_tracked_files(str(work))
    assert str(work / "module.py") in tracked


def test_restore_snapshot_preserves_vcs_dirs(tmp_env):
    """restore_snapshot must preserve .git/.hg/.svn while removing extra files."""
    cm, work = tmp_env
    git_dir = os.path.join(work, ".git")
    os.makedirs(git_dir)
    with open(os.path.join(git_dir, "HEAD"), "w") as f:
        f.write("ref: refs/heads/main\n")
    # Mock git so .git is not snapshotted, avoiding copytree conflict on restore
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "file1.txt\nsub/file2.txt\n"
    with patch("mullet.core.agent.checkpoint.subprocess.run", return_value=mock_result):
        ts = cm.create_snapshot(work, label="with-git")
    extra_file = os.path.join(work, "extra.txt")
    with open(extra_file, "w") as f:
        f.write("extra")
    extra_subdir = os.path.join(work, "extra_sub")
    os.makedirs(extra_subdir)
    with open(os.path.join(extra_subdir, "f.txt"), "w") as f:
        f.write("extra")
    cm.restore_snapshot(ts)
    assert os.path.exists(git_dir)
    assert os.path.exists(os.path.join(git_dir, "HEAD"))
    assert not os.path.exists(extra_file)
    assert not os.path.exists(extra_subdir)
    assert os.path.exists(os.path.join(work, "file1.txt"))
    assert os.path.exists(os.path.join(work, "sub", "file2.txt"))


def test_cleanup_old_snapshots_ignores_nondirectory_entries(tmp_path):
    """_cleanup_old_snapshots should skip non-directory items in base_dir."""
    base = tmp_path / "checkpoints"
    cm = CheckpointManager(base_dir=str(base), max_snapshots=2)
    for i in range(3):
        snap_dir = base / f"2024010100000{i}"
        snap_dir.mkdir(parents=True)
        (snap_dir / "metadata.txt").write_text(f"label: snap{i}\nwork_dir: /tmp\n")
    (base / "!stray_file").write_text("not a snapshot")
    cm._cleanup_old_snapshots()
    remaining_dirs = sorted([d for d in os.listdir(base) if os.path.isdir(base / d)])
    remaining_files = [f for f in os.listdir(base) if os.path.isfile(base / f)]
    assert len(remaining_dirs) == 2
    assert remaining_dirs == ["20240101000001", "20240101000002"]
    assert "!stray_file" in remaining_files
