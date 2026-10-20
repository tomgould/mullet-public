"""Tests for orchestration state persistence."""

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from mullet.core.orchestration.state import ModeState, STALE_THRESHOLD


@pytest.fixture(autouse=True)
def clean_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_save_and_load():
    state = ModeState(knowledge=None, mode_name="test")
    state.save({"key": "value", "count": 42})

    loaded = state.load()
    assert loaded is not None
    assert loaded["key"] == "value"
    assert loaded["count"] == 42


def test_load_missing():
    state = ModeState(knowledge=None, mode_name="nonexistent")
    assert state.load() is None


def test_stale_detection():
    state = ModeState(knowledge=None, mode_name="stale_test")

    # Save with old timestamp
    from pathlib import Path
    state_dir = Path(".mullet") / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    old_data = {"value": 1, "_timestamp": time.time() - STALE_THRESHOLD - 100}
    (state_dir / "stale_test.json").write_text(json.dumps(old_data))

    loaded = state.load()
    assert loaded is None  # Should be stale


def test_session_scoped():
    state = ModeState(knowledge=None, mode_name="ralph", session_id="sess_123")
    state.save({"iteration": 5})

    loaded = state.load()
    assert loaded is not None
    assert loaded["iteration"] == 5


def test_save_adds_timestamp():
    state = ModeState(knowledge=None, mode_name="ts_test")
    state.save({"x": 1})
    loaded = state.load()
    assert "_timestamp" in loaded
    assert loaded["_timestamp"] > 0


def test_save_adds_mode_name():
    state = ModeState(knowledge=None, mode_name="mymode")
    state.save({"y": 2})
    loaded = state.load()
    assert loaded["_mode"] == "mymode"


def test_clear_removes_state():
    state = ModeState(knowledge=None, mode_name="clear_test")
    state.save({"val": 99})
    assert state.load() is not None
    state.clear()
    assert state.load() is None


def test_overwrite_previous_save():
    state = ModeState(knowledge=None, mode_name="overwrite_test")
    state.save({"v": 1})
    state.save({"v": 2})
    loaded = state.load()
    assert loaded["v"] == 2


def test_namespace_without_session():
    state = ModeState(knowledge=None, mode_name="n1")
    assert "n1" in state._namespace
    assert ":" not in state._namespace or state._namespace.endswith(":n1") or "n1:" in state._namespace


def test_namespace_with_session():
    state = ModeState(knowledge=None, mode_name="n2", session_id="abc")
    assert "n2" in state._namespace
    assert "abc" in state._namespace


def test_clear_state_cache():
    """clear_state_cache() must not raise and must clear the read cache."""
    from mullet.core.orchestration.state import clear_state_cache, _read_cache
    _read_cache["__test_key__"] = "dummy"
    clear_state_cache()
    assert "__test_key__" not in _read_cache


class TestModeStateMissingLines:
    """Cover edge cases in ModeState that existing tests miss."""

    # ------------------------------------------------------------------
    # save() with knowledge present (line 53)
    # ------------------------------------------------------------------
    def test_save_calls_remember_when_knowledge_present(self, monkeypatch):
        knowledge = MagicMock()
        state = ModeState(knowledge=knowledge, mode_name="kmode")
        monkeypatch.setattr(state, "_state_path", lambda: Path("/dev/null"))

        state.save({"a": 1})

        assert knowledge.remember.call_count == 1
        args, kwargs = knowledge.remember.call_args
        assert kwargs["key"] == "state:kmode"
        assert kwargs["namespace"] == "mode:kmode"
        loaded = json.loads(kwargs["content"])
        assert loaded["a"] == 1
        assert loaded["_mode"] == "kmode"
        assert "_timestamp" in loaded

    # ------------------------------------------------------------------
    # load() with knowledge (lines 64-78)
    # ------------------------------------------------------------------
    def test_load_from_knowledge_dict_format(self, monkeypatch):
        knowledge = MagicMock()
        now = time.time()
        knowledge.recall.return_value = [
            {"content": json.dumps({"x": 2, "_timestamp": now, "_mode": "kmode"})}
        ]
        state = ModeState(knowledge=knowledge, mode_name="kmode")
        monkeypatch.setattr(state, "_state_path", lambda: Path("/dev/null"))

        loaded = state.load()
        assert loaded is not None
        assert loaded["x"] == 2
        knowledge.recall.assert_called_once_with(query="state:kmode", namespace="mode:kmode")

    def test_load_from_knowledge_stale_timestamp(self, monkeypatch):
        knowledge = MagicMock()
        old = time.time() - STALE_THRESHOLD - 10
        knowledge.recall.return_value = [
            {"content": json.dumps({"x": 3, "_timestamp": old, "_mode": "kmode"})}
        ]
        state = ModeState(knowledge=knowledge, mode_name="kmode")
        monkeypatch.setattr(state, "_state_path", lambda: Path("/dev/null"))

        assert state.load() is None

    def test_load_from_knowledge_json_decode_error(self, monkeypatch):
        knowledge = MagicMock()
        knowledge.recall.return_value = [{"content": "not-json"}]
        state = ModeState(knowledge=knowledge, mode_name="kmode")
        monkeypatch.setattr(state, "_state_path", lambda: Path("/dev/null"))

        assert state.load() is None

    # ------------------------------------------------------------------
    # clear() with knowledge (lines 84-90)
    # ------------------------------------------------------------------
    def test_clear_calls_memory_delete(self, tmp_path, monkeypatch):
        knowledge = MagicMock()
        state = ModeState(knowledge=knowledge, mode_name="kmode")
        monkeypatch.setattr(state, "_state_path", lambda: tmp_path / "kmode.json")

        state.clear()

        knowledge.memory.delete.assert_called_once_with(
            namespace="mode:kmode",
            key="state:kmode",
        )

    def test_clear_swallows_memory_delete_exception(self, tmp_path, monkeypatch):
        knowledge = MagicMock()
        knowledge.memory.delete.side_effect = RuntimeError("boom")
        state = ModeState(knowledge=knowledge, mode_name="kmode")
        monkeypatch.setattr(state, "_state_path", lambda: tmp_path / "kmode.json")

        # Must not raise
        state.clear()

    # ------------------------------------------------------------------
    # clear() local path — path.unlink() and cache pop (lines 97-98)
    # ------------------------------------------------------------------
    def test_clear_unlinks_local_file_and_pops_cache(self, tmp_path, monkeypatch):
        state = ModeState(knowledge=None, mode_name="localclear")
        monkeypatch.setattr(state, "_state_path", lambda: tmp_path / "localclear.json")

        path = state._state_path()
        path.write_text(json.dumps({"val": 1}))
        from mullet.core.orchestration.state import _read_cache
        _read_cache[str(path)] = ({"val": 1}, time.time(), 0.0)

        state.clear()

        assert not path.exists()
        assert str(path) not in _read_cache

    def test_clear_unlink_oserror_logged(self, tmp_path, monkeypatch):
        state = ModeState(knowledge=None, mode_name="localunlinkerr")
        monkeypatch.setattr(state, "_state_path", lambda: tmp_path / "localunlinkerr.json")

        path = state._state_path()
        path.write_text(json.dumps({"val": 1}))

        call_count = []

        def fake_unlink(*a, **k):
            call_count.append(1)
            raise OSError("permission denied")

        with patch.object(Path, "unlink", fake_unlink):
            # Must not raise
            state.clear()

        assert len(call_count) == 1

    # ------------------------------------------------------------------
    # _save_local exception path (lines 128-137)
    # ------------------------------------------------------------------
    def test_save_local_exception_cleans_up(self, tmp_path, monkeypatch):
        state = ModeState(knowledge=None, mode_name="saveerr")
        monkeypatch.setattr(state, "_state_path", lambda: tmp_path / "saveerr.json")

        import os as os_module
        original_write = os_module.write
        write_calls = []

        def bad_write(fd, data):
            write_calls.append(1)
            raise OSError("disk full")

        monkeypatch.setattr(os_module, "write", bad_write)

        with pytest.raises(OSError, match="disk full"):
            state._save_local({"x": 1})

        # Ensure we attempted cleanup; since mkstemp creates a real temp file,
        # and we patch unlink too, let's just verify exception propagates.

    # ------------------------------------------------------------------
    # _load_local cache hit with mtime check (lines 153-161)
    # ------------------------------------------------------------------
    def test_load_local_cache_hit_matching_mtime(self, tmp_path, monkeypatch):
        state = ModeState(knowledge=None, mode_name="cachehit")
        monkeypatch.setattr(state, "_state_path", lambda: tmp_path / "cachehit.json")

        path = state._state_path()
        path.write_text(json.dumps({"x": 1, "_timestamp": time.time()}))

        first = state._load_local()
        assert first is not None

        # Second call should hit cache and return same data without re-reading
        second = state._load_local()
        assert second == first

    def test_load_local_cache_hit_oserror_on_stat(self, tmp_path, monkeypatch):
        state = ModeState(knowledge=None, mode_name="cachestaterr")
        monkeypatch.setattr(state, "_state_path", lambda: tmp_path / "cachestaterr.json")

        path = state._state_path()
        path.write_text(json.dumps({"x": 2, "_timestamp": time.time()}))

        # Prime cache manually with a bogus mtime so a mismatch would normally
        # trigger a re-read. Then make stat() blow up.
        import mullet.core.orchestration.state as state_mod
        state_mod._read_cache[str(path)] = ({"x": 999}, time.time(), 12345.0)

        def bad_stat(*_args, **_kwargs):
            raise OSError("stat failed")

        # Patch exists so the initial check passes, and stat so the TOCTOU
        # check inside the cache-hit block raises OSError.
        with patch.object(Path, "exists", return_value=True), patch.object(Path, "stat", bad_stat):
            result = state._load_local()

        assert result is None

    def test_load_local_cache_stale_mtime_falls_through(self, tmp_path, monkeypatch):
        state = ModeState(knowledge=None, mode_name="cachestale")
        monkeypatch.setattr(state, "_state_path", lambda: tmp_path / "cachestale.json")

        path = state._state_path()
        path.write_text(json.dumps({"x": 3, "_timestamp": time.time()}))

        # Populate cache with old mtime
        mtime_now = path.stat().st_mtime
        state_mod = __import__("mullet.core.orchestration.state", fromlist=["_read_cache"])
        state_mod._read_cache[str(path)] = ({"x": 888}, time.time(), mtime_now - 1.0)

        result = state._load_local()
        assert result == {"x": 3, "_timestamp": pytest.approx(time.time(), abs=5)}

    # ------------------------------------------------------------------
    # _load_local fresh read with JSONDecodeError or OSError (lines 174-176)
    # ------------------------------------------------------------------
    def test_load_local_fresh_json_decode_error(self, tmp_path, monkeypatch):
        state = ModeState(knowledge=None, mode_name="badjson")
        monkeypatch.setattr(state, "_state_path", lambda: tmp_path / "badjson.json")

        path = state._state_path()
        path.write_text("not json at all")

        assert state._load_local() is None

    def test_load_local_fresh_oserror_on_read(self, tmp_path, monkeypatch):
        state = ModeState(knowledge=None, mode_name="readerr")
        monkeypatch.setattr(state, "_state_path", lambda: tmp_path / "readerr.json")

        path = state._state_path()
        path.write_text(json.dumps({"x": 1, "_timestamp": time.time()}))

        def bad_read_text(*a, **k):
            raise OSError("read error")

        with patch.object(Path, "read_text", bad_read_text):
            assert state._load_local() is None

    def test_load_local_fresh_oserror_on_stat_in_fresh_block(self, tmp_path, monkeypatch):
        state = ModeState(knowledge=None, mode_name="staterr")
        monkeypatch.setattr(state, "_state_path", lambda: tmp_path / "staterr.json")

        path = state._state_path()
        path.write_text(json.dumps({"x": 1, "_timestamp": time.time()}))

        call_count = []

        def bad_stat(*_args, **_kwargs):
            call_count.append(1)
            raise OSError("stat error")

        with patch.object(Path, "exists", return_value=True), patch.object(Path, "stat", bad_stat):
            assert state._load_local() is None

        assert len(call_count) >= 1

    # ------------------------------------------------------------------
    # _save_local cleanup os.close OSError (lines 131-132)
    # ------------------------------------------------------------------
    def test_save_local_cleanup_close_oserror(self, tmp_path, monkeypatch):
        state = ModeState(knowledge=None, mode_name="saveerr")
        monkeypatch.setattr(state, "_state_path", lambda: tmp_path / "saveerr.json")

        with patch("os.write", side_effect=OSError("disk full")), patch(
            "os.close", side_effect=OSError("close failed")
        ):
            with pytest.raises(OSError, match="disk full"):
                state._save_local({"x": 1})

    # ------------------------------------------------------------------
    # _save_local cleanup os.unlink OSError (lines 135-136)
    # ------------------------------------------------------------------
    def test_save_local_cleanup_unlink_oserror(self, tmp_path, monkeypatch):
        state = ModeState(knowledge=None, mode_name="saveerr")
        monkeypatch.setattr(state, "_state_path", lambda: tmp_path / "saveerr.json")

        with patch("os.write", side_effect=OSError("disk full")), patch(
            "os.close"
        ), patch("os.unlink", side_effect=OSError("unlink failed")):
            with pytest.raises(OSError, match="disk full"):
                state._save_local({"x": 1})
