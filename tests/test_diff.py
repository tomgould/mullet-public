"""Tests for file change tracking."""

from mullet.core.agent.diff import ChangeTracker


def test_no_changes(tmp_path):
    tracker = ChangeTracker(watch_dirs=[str(tmp_path)])
    (tmp_path / "file.txt").write_text("hello")
    tracker.snapshot_before()
    # No changes
    changes = tracker.snapshot_after()
    assert len(changes) == 0


def test_created_file(tmp_path):
    tracker = ChangeTracker(watch_dirs=[str(tmp_path)])
    tracker.snapshot_before()

    (tmp_path / "new.txt").write_text("created")
    changes = tracker.snapshot_after()

    assert len(changes) == 1
    assert changes[0].change_type == "created"
    assert "new.txt" in changes[0].path


def test_modified_file(tmp_path):
    (tmp_path / "file.txt").write_text("before")
    tracker = ChangeTracker(watch_dirs=[str(tmp_path)])
    tracker.snapshot_before()

    (tmp_path / "file.txt").write_text("after — different content")
    changes = tracker.snapshot_after()

    assert len(changes) == 1
    assert changes[0].change_type == "modified"


def test_deleted_file(tmp_path):
    f = tmp_path / "doomed.txt"
    f.write_text("goodbye")
    tracker = ChangeTracker(watch_dirs=[str(tmp_path)])
    tracker.snapshot_before()

    f.unlink()
    changes = tracker.snapshot_after()

    assert len(changes) == 1
    assert changes[0].change_type == "deleted"


def test_format_changes(tmp_path):
    tracker = ChangeTracker(watch_dirs=[str(tmp_path)])
    tracker.snapshot_before()
    (tmp_path / "new.py").write_text("print('hello')")
    tracker.snapshot_after()

    output = tracker.format_changes()
    assert "new.py" in output
    assert "1" in output  # "1" file changed


def test_no_changes_message(tmp_path):
    tracker = ChangeTracker(watch_dirs=[str(tmp_path)])
    tracker.snapshot_before()
    tracker.snapshot_after()
    assert "No files changed" in tracker.format_changes()


def test_extension_filter(tmp_path):
    tracker = ChangeTracker(watch_dirs=[str(tmp_path)], extensions=[".py"])
    tracker.snapshot_before()
    (tmp_path / "code.py").write_text("x = 1")
    (tmp_path / "data.json").write_text("{}")
    changes = tracker.snapshot_after()
    assert len(changes) == 1  # Only .py file
    assert "code.py" in changes[0].path
