import hashlib
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from mullet.knowledge.code.incremental import (
    CodeIndexState,
    FileState,
    compute_chunk_id,
)


def test_file_state_defaults():
    fs = FileState(file_hash="abc", mtime=1.0)
    assert fs.chunk_ids == []


def test_compute_chunk_id_deterministic():
    chunk = {
        "metadata": {"source_file": "s.py", "symbol_name": "foo", "symbol_type": "func", "line_start": 10},
        "content": "def foo(): pass",
    }
    id1 = compute_chunk_id(chunk)
    id2 = compute_chunk_id(chunk)
    assert id1 == id2
    assert len(id1) == 12


def test_compute_chunk_id_unique():
    chunk1 = {
        "metadata": {"source_file": "a.py", "symbol_name": "foo", "symbol_type": "func", "line_start": 1},
        "content": "def foo(): pass",
    }
    chunk2 = {
        "metadata": {"source_file": "a.py", "symbol_name": "bar", "symbol_type": "func", "line_start": 1},
        "content": "def bar(): pass",
    }
    assert compute_chunk_id(chunk1) != compute_chunk_id(chunk2)


def test_is_file_changed_new_file():
    with patch.object(CodeIndexState, "_load_state"):
        c = CodeIndexState(Path("/fake"))
        mock_path = MagicMock()
        mock_path.__str__.return_value = "new.py"
        assert c.is_file_changed(mock_path) is True


def test_is_file_changed_same_mtime():
    with patch.object(CodeIndexState, "_load_state"):
        c = CodeIndexState(Path("/fake"))
        c.state = {"same.py": FileState(file_hash="h", mtime=123.0)}
        mock_path = MagicMock()
        mock_path.__str__.return_value = "same.py"
        mock_path.stat.return_value = MagicMock(st_mtime=123.0)
        assert c.is_file_changed(mock_path) is False


def test_is_file_changed_hash_differs():
    with patch.object(CodeIndexState, "_load_state"):
        c = CodeIndexState(Path("/fake"))
        c.state = {"mod.py": FileState(file_hash="oldhash", mtime=100.0)}
        mock_path = MagicMock()
        mock_path.__str__.return_value = "mod.py"
        mock_path.stat.return_value = MagicMock(st_mtime=200.0)
        mock_path.read_bytes.return_value = b"new content"
        assert c.is_file_changed(mock_path) is True


def test_remove_file_returns_chunk_ids():
    with patch.object(CodeIndexState, "_load_state"):
        c = CodeIndexState(Path("/fake"))
        c.state = {"path.py": FileState(file_hash="h", mtime=1.0, chunk_ids=["c1", "c2"])}
        assert c.remove_file("path.py") == ["c1", "c2"]
        assert "path.py" not in c.state


def test_get_all_chunk_ids():
    with patch.object(CodeIndexState, "_load_state"):
        c = CodeIndexState(Path("/fake"))
        c.state = {
            "a.py": FileState(file_hash="h1", mtime=1.0, chunk_ids=["c1", "c2"]),
            "b.py": FileState(file_hash="h2", mtime=2.0, chunk_ids=["c3"]),
        }
        assert sorted(c.get_all_chunk_ids()) == ["c1", "c2", "c3"]


def test_load_state_success(tmp_path):
    state_file = tmp_path / "code_index_state.json"
    state_file.write_text(json.dumps({
        "a.py": {"file_hash": "h1", "mtime": 1.0, "chunk_ids": ["c1"]},
        "b.py": {"file_hash": "h2", "mtime": 2.0},
    }))
    c = CodeIndexState(tmp_path)
    assert len(c.state) == 2
    assert c.state["a.py"] == FileState(file_hash="h1", mtime=1.0, chunk_ids=["c1"])
    assert c.state["b.py"] == FileState(file_hash="h2", mtime=2.0, chunk_ids=[])


def test_load_state_large_file(tmp_path, caplog):
    state_file = tmp_path / "code_index_state.json"
    data = {"x.py": {"file_hash": "h" * 32, "mtime": 1.0, "chunk_ids": ["c" * 100] * 20000}}
    state_file.write_text(json.dumps(data))
    with caplog.at_level(logging.INFO):
        c = CodeIndexState(tmp_path)
    assert "Loading code index state" in caplog.text
    assert len(c.state) == 1


def test_load_state_corrupt(tmp_path, caplog):
    state_file = tmp_path / "code_index_state.json"
    state_file.write_text("not json")
    with caplog.at_level(logging.WARNING):
        c = CodeIndexState(tmp_path)
    assert c.state == {}
    assert "Failed to load code index state" in caplog.text


def test_load_state_no_file(tmp_path):
    c = CodeIndexState(tmp_path)
    assert c.state == {}


def test_save_state_success(tmp_path):
    c = CodeIndexState(tmp_path)
    c.state = {"a.py": FileState(file_hash="h1", mtime=1.0, chunk_ids=["c1"])}
    c.save_state()
    assert c.state_file.exists()
    data = json.loads(c.state_file.read_text())
    assert data["a.py"]["file_hash"] == "h1"
    assert data["a.py"]["chunk_ids"] == ["c1"]


def test_save_state_failure(tmp_path, caplog):
    c = CodeIndexState(tmp_path)
    c.state = {"a.py": FileState(file_hash="h1", mtime=1.0)}
    with patch.object(Path, "mkdir", side_effect=OSError("fail")):
        with caplog.at_level(logging.WARNING):
            c.save_state()
    assert "Failed to save code index state" in caplog.text


def test_clear(tmp_path):
    c = CodeIndexState(tmp_path)
    c.state = {"a.py": FileState(file_hash="h1", mtime=1.0)}
    c.save_state()
    assert c.state_file.exists()
    c.clear()
    assert c.state == {}
    assert not c.state_file.exists()


def test_clear_no_file(tmp_path):
    c = CodeIndexState(tmp_path)
    c.state = {"a.py": FileState(file_hash="h1", mtime=1.0)}
    c.clear()
    assert c.state == {}


def test_get_file_hash_success(tmp_path):
    c = CodeIndexState(tmp_path)
    f = tmp_path / "test.py"
    f.write_text("hello")
    h = c.get_file_hash(f)
    assert len(h) == 16
    assert h == hashlib.sha256(b"hello").hexdigest()[:16]


def test_get_file_hash_failure(tmp_path):
    c = CodeIndexState(tmp_path)
    with patch.object(Path, "read_bytes", side_effect=OSError("fail")):
        h = c.get_file_hash(tmp_path / "missing.py")
    assert h == ""


def test_is_file_changed_hash_same_mtime_differs():
    with patch.object(CodeIndexState, "_load_state"):
        c = CodeIndexState(Path("/fake"))
        content = b"same"
        file_hash = hashlib.sha256(content).hexdigest()[:16]
        c.state = {"mod.py": FileState(file_hash=file_hash, mtime=100.0)}
        mock_path = MagicMock()
        mock_path.__str__.return_value = "mod.py"
        mock_path.stat.return_value = MagicMock(st_mtime=200.0)
        mock_path.read_bytes.return_value = content
        assert c.is_file_changed(mock_path) is False


def test_is_file_changed_oserror():
    with patch.object(CodeIndexState, "_load_state"):
        c = CodeIndexState(Path("/fake"))
        c.state = {"mod.py": FileState(file_hash="oldhash", mtime=100.0)}
        mock_path = MagicMock()
        mock_path.__str__.return_value = "mod.py"
        mock_path.stat.side_effect = OSError("fail")
        assert c.is_file_changed(mock_path) is True


def test_get_changes_sequential():
    with patch.object(CodeIndexState, "_load_state"):
        c = CodeIndexState(Path("/fake"))
        c.state = {
            "same.py": FileState(file_hash="h", mtime=1.0),
            "old.py": FileState(file_hash="old", mtime=1.0),
        }
        same = MagicMock()
        same.__str__.return_value = "same.py"
        same.stat.return_value = MagicMock(st_mtime=1.0)
        new_f = MagicMock()
        new_f.__str__.return_value = "new.py"
        new_f.stat.return_value = MagicMock(st_mtime=1.0)
        new_f.read_bytes.return_value = b"new"
        mod = MagicMock()
        mod.__str__.return_value = "old.py"
        mod.stat.return_value = MagicMock(st_mtime=2.0)
        mod.read_bytes.return_value = b"modified"
        current = [same, new_f, mod]
        new_or_mod, unchanged, deleted = c.get_changes(current, workers=1)
        assert sorted([str(f) for f in new_or_mod]) == ["new.py", "old.py"]
        assert [str(f) for f in unchanged] == ["same.py"]
        assert deleted == []


def test_get_changes_deleted():
    with patch.object(CodeIndexState, "_load_state"):
        c = CodeIndexState(Path("/fake"))
        c.state = {"old.py": FileState(file_hash="h", mtime=1.0)}
        current = []
        new_or_mod, unchanged, deleted = c.get_changes(current, workers=1)
        assert new_or_mod == []
        assert unchanged == []
        assert deleted == ["old.py"]


def test_get_changes_default_workers():
    with patch.object(CodeIndexState, "_load_state"):
        c = CodeIndexState(Path("/fake"))
        c.state = {"same.py": FileState(file_hash="h", mtime=1.0)}
        same = MagicMock()
        same.__str__.return_value = "same.py"
        same.stat.return_value = MagicMock(st_mtime=1.0)
        new_or_mod, unchanged, deleted = c.get_changes([same], workers=0)
        assert new_or_mod == []
        assert [str(f) for f in unchanged] == ["same.py"]
        assert deleted == []


class _FakePath:
    def __init__(self, name, mtime=1.0, content=b"content"):
        self._name = name
        self._mtime = mtime
        self._content = content

    def __str__(self):
        return self._name

    def stat(self):
        class S:
            pass
        s = S()
        s.st_mtime = self._mtime
        return s

    def read_bytes(self):
        return self._content


def test_get_changes_parallel():
    with patch.object(CodeIndexState, "_load_state"):
        c = CodeIndexState(Path("/fake"))
        c.state = {}
        files = [_FakePath(f"f{i}.py") for i in range(101)]
        new_or_mod, unchanged, deleted = c.get_changes(files, workers=2)
        assert len(new_or_mod) == 101
        assert unchanged == []
        assert deleted == []


def test_get_changes_parallel_with_unchanged():
    with patch.object(CodeIndexState, "_load_state"):
        c = CodeIndexState(Path("/fake"))
        c.state = {f"f{i}.py": FileState(file_hash="h", mtime=1.0) for i in range(50)}
        files = [_FakePath(f"f{i}.py", mtime=1.0) for i in range(50)] + [_FakePath(f"f{i}.py", mtime=2.0) for i in range(50, 101)]
        new_or_mod, unchanged, deleted = c.get_changes(files, workers=2)
        assert len(unchanged) == 50
        assert len(new_or_mod) == 51
        assert deleted == []


def test_get_changes_future_exception():
    with patch.object(CodeIndexState, "_load_state"):
        c = CodeIndexState(Path("/fake"))
        c.state = {}
        files = [_FakePath(f"f{i}.py") for i in range(101)]
        with patch.object(c, "is_file_changed", side_effect=RuntimeError("boom")):
            new_or_mod, unchanged, deleted = c.get_changes(files, workers=2)
        assert len(new_or_mod) == 101
        assert unchanged == []
        assert deleted == []


def test_get_changes_massive_parallel(caplog):
    with patch.object(CodeIndexState, "_load_state"):
        c = CodeIndexState(Path("/fake"))
        c.state = {}
        files = [_FakePath(f"f{i}.py") for i in range(10001)]
        with caplog.at_level(logging.INFO):
            c.get_changes(files, workers=2)
        assert "10000/10001 files checked" in caplog.text


def test_update_file_success(tmp_path):
    c = CodeIndexState(tmp_path)
    f = tmp_path / "a.py"
    f.write_text("hello")
    c.update_file(f, ["c1", "c2"])
    key = str(f)
    assert key in c.state
    assert c.state[key].chunk_ids == ["c1", "c2"]
    assert c.state[key].file_hash == hashlib.sha256(b"hello").hexdigest()[:16]


def test_update_file_failure(caplog):
    with patch.object(CodeIndexState, "_load_state"):
        c = CodeIndexState(Path("/fake"))
        mock_path = MagicMock()
        mock_path.__str__.return_value = "a.py"
        mock_path.stat.side_effect = OSError("fail")
        with caplog.at_level(logging.WARNING):
            c.update_file(mock_path, ["c1"])
        assert "Failed to update state for a.py" in caplog.text


def test_remove_file_missing():
    with patch.object(CodeIndexState, "_load_state"):
        c = CodeIndexState(Path("/fake"))
        assert c.remove_file("missing.py") == []


def test_get_chunk_ids_for_file():
    with patch.object(CodeIndexState, "_load_state"):
        c = CodeIndexState(Path("/fake"))
        c.state = {"a.py": FileState(file_hash="h", mtime=1.0, chunk_ids=["c1"])}
        assert c.get_chunk_ids_for_file("a.py") == ["c1"]
        assert c.get_chunk_ids_for_file("missing.py") == []
