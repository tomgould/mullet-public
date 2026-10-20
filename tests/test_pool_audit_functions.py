"""Tests for audit_pools and format_audit_report."""

from pathlib import Path
import os
import sqlite3
import unittest.mock
import concurrent.futures
import yaml

from mullet.cli.pool_audit import (
    _dir_size_mb,
    _read_yaml_fields,
    _get_doc_entries,
    _get_vector_info,
    _get_graph_info,
    _auditer_slug_for,
    _classify_sources,
    _DocEntry,
)


def _make_pool(tmp_path: Path, name: str) -> Path:
    """Create a minimal pool directory structure."""
    pool_dir = tmp_path / name
    pool_dir.mkdir()
    config = {
        "name": name,
        "machine_name": name.lower(),
        "embedding": {"model": "BAAI/bge-base-en-v1.5"},
    }
    (pool_dir / "config.yaml").write_text(yaml.dump(config))
    return pool_dir


def test_audit_pools_empty_dir(tmp_path):
    from mullet.cli.pool_audit import audit_pools
    result = audit_pools(tmp_path, verbose=False)
    assert result == []


def test_audit_pools_nonexistent_dir(tmp_path):
    from mullet.cli.pool_audit import audit_pools
    result = audit_pools(tmp_path / "nonexistent", verbose=False)
    assert result == []


def test_audit_pools_returns_one_audit_per_pool(tmp_path):
    from mullet.cli.pool_audit import audit_pools
    _make_pool(tmp_path, "pool_a")
    _make_pool(tmp_path, "pool_b")
    result = audit_pools(tmp_path, verbose=False)
    assert len(result) == 2
    names = {a.name for a in result}
    assert "pool_a" in names
    assert "pool_b" in names


def test_audit_pools_pool_without_yaml_included(tmp_path):
    from mullet.cli.pool_audit import audit_pools
    (tmp_path / "raw_pool").mkdir()
    result = audit_pools(tmp_path, verbose=False)
    assert len(result) == 1
    assert result[0].has_client_yaml is False


def test_format_audit_report_produces_string(tmp_path):
    from mullet.cli.pool_audit import audit_pools, format_audit_report
    _make_pool(tmp_path, "mypool")
    audits = audit_pools(tmp_path, verbose=False)
    report = format_audit_report(audits)
    assert isinstance(report, str)
    assert "mypool" in report


# ---------------------------------------------------------------------------
# _dir_size_mb
# ---------------------------------------------------------------------------


def test_dir_size_mb_oserror_on_getsize(tmp_path, monkeypatch):
    d = tmp_path / "dir"
    d.mkdir()
    (d / "a.txt").write_text("a")
    (d / "b.txt").write_text("b")

    orig_getsize = os.path.getsize

    def fake_getsize(path):
        if "b.txt" in str(path):
            raise OSError("fail")
        return orig_getsize(path)

    monkeypatch.setattr(os.path, "getsize", fake_getsize)
    result = _dir_size_mb(d)
    assert result >= 0


def test_dir_size_mb_exception_on_walk(tmp_path, monkeypatch):
    d = tmp_path / "dir"
    d.mkdir()

    def fake_walk(*args, **kwargs):
        raise RuntimeError("walk fail")

    monkeypatch.setattr(os, "walk", fake_walk)
    result = _dir_size_mb(d)
    assert result == 0.0


# ---------------------------------------------------------------------------
# _read_yaml_fields
# ---------------------------------------------------------------------------


def test_read_yaml_fields_nonexistent_path(tmp_path):
    result = _read_yaml_fields(tmp_path / "missing.yaml")
    assert result == ("", "")


def test_read_yaml_fields_bad_yaml(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("{{not yaml")
    result = _read_yaml_fields(bad)
    assert result == ("", "")


# ---------------------------------------------------------------------------
# _get_doc_entries
# ---------------------------------------------------------------------------


def test_get_doc_entries_readlink_exception(tmp_path, monkeypatch):
    docs = tmp_path / "documents"
    docs.mkdir()
    target = tmp_path / "real"
    target.mkdir()
    link = docs / "link"
    link.symlink_to(target)

    monkeypatch.setattr(os, "readlink", lambda p: (_ for _ in ()).throw(RuntimeError("fail")))
    entries = _get_doc_entries(docs)
    assert len(entries) == 1
    assert entries[0].target == "?"
    assert entries[0].target_exists is False


def test_get_doc_entries_iterdir_exception(tmp_path, monkeypatch):
    docs = tmp_path / "documents"
    docs.mkdir()

    def fake_iterdir(self):
        raise PermissionError("no access")

    monkeypatch.setattr(Path, "iterdir", fake_iterdir)
    entries = _get_doc_entries(docs)
    assert entries == []


# ---------------------------------------------------------------------------
# _get_vector_info
# ---------------------------------------------------------------------------


def test_get_vector_info_chromadb_fallback_client(tmp_path):
    index_path = tmp_path / "index"
    index_path.mkdir()

    call_count = [0]

    def fake_persistent_client(*args, **kwargs):
        call_count[0] += 1
        if kwargs.get("settings"):
            raise RuntimeError("settings fail")
        mock_coll = type("MockColl", (), {"count": lambda self: 0})()
        return type("MockClient", (), {"list_collections": lambda self: [mock_coll]})()

    with unittest.mock.patch("chromadb.PersistentClient", fake_persistent_client):
        result = _get_vector_info(index_path)
    assert result.error is None
    assert result.chunk_count == 0
    assert call_count[0] == 2


def test_get_vector_info_timeout(tmp_path):
    index_path = tmp_path / "index"
    index_path.mkdir()

    mock_future = unittest.mock.MagicMock()
    mock_future.result.side_effect = concurrent.futures.TimeoutError()

    with unittest.mock.patch("concurrent.futures.ThreadPoolExecutor") as mock_ex:
        mock_instance = mock_ex.return_value
        mock_instance.__enter__ = lambda s: s
        mock_instance.__exit__ = lambda *args: None
        mock_instance.submit.return_value = mock_future

        result = _get_vector_info(index_path, timeout=1.0)
    assert result.error is not None
    assert "timed out" in result.error


def test_get_vector_info_import_error(tmp_path):
    index_path = tmp_path / "index"
    index_path.mkdir()

    mock_future = unittest.mock.MagicMock()
    mock_future.result.side_effect = ImportError("no chromadb")

    with unittest.mock.patch("concurrent.futures.ThreadPoolExecutor") as mock_ex:
        mock_instance = mock_ex.return_value
        mock_instance.__enter__ = lambda s: s
        mock_instance.__exit__ = lambda *args: None
        mock_instance.submit.return_value = mock_future

        result = _get_vector_info(index_path)
    assert result.error == "chromadb not installed"


def test_get_vector_info_generic_exception(tmp_path):
    index_path = tmp_path / "index"
    index_path.mkdir()

    mock_future = unittest.mock.MagicMock()
    mock_future.result.side_effect = RuntimeError("boom")

    with unittest.mock.patch("concurrent.futures.ThreadPoolExecutor") as mock_ex:
        mock_instance = mock_ex.return_value
        mock_instance.__enter__ = lambda s: s
        mock_instance.__exit__ = lambda *args: None
        mock_instance.submit.return_value = mock_future

        result = _get_vector_info(index_path)
    assert result.error == "boom"


# ---------------------------------------------------------------------------
# _get_graph_info
# ---------------------------------------------------------------------------


def test_get_graph_info_empty_db(tmp_path):
    index_path = tmp_path / "index"
    index_path.mkdir()
    db_path = index_path / "knowledge_graph.db"
    conn = sqlite3.connect(str(db_path))
    conn.close()
    result = _get_graph_info(index_path)
    assert result.built is True
    assert result.entity_count == 0
    assert result.relationship_count == 0
    assert result.db_size_mb >= 0
    assert result.error is not None
    assert "no such table" in result.error.lower()


# ---------------------------------------------------------------------------
# _auditer_slug_for
# ---------------------------------------------------------------------------


def test_auditer_slug_for_none():
    assert _auditer_slug_for(None) is None


def test_auditer_slug_for_question_mark():
    assert _auditer_slug_for("?") is None


def test_auditer_slug_for_with_auditer_data(tmp_path, monkeypatch):
    auditer_data = tmp_path / "github_auditer" / "data"
    auditer_data.mkdir(parents=True)
    target = auditer_data / "my-slug" / "file.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x")
    monkeypatch.setattr("mullet.cli.pool_audit._AUDITER_DATA", auditer_data)
    result = _auditer_slug_for(str(target))
    assert result == "my-slug"


def test_auditer_slug_for_textual_fallback():
    target = "/some/path/github_auditer/data/my-slug/file.txt"
    result = _auditer_slug_for(target)
    assert result == "my-slug"


def test_auditer_slug_for_resolve_exception(monkeypatch, tmp_path):
    monkeypatch.setattr("mullet.cli.pool_audit._AUDITER_DATA", tmp_path)

    def fake_resolve(self, strict=False):
        raise RuntimeError("resolve fail")

    monkeypatch.setattr(Path, "resolve", fake_resolve)
    result = _auditer_slug_for("/some/path")
    assert result is None


# ---------------------------------------------------------------------------
# _classify_sources
# ---------------------------------------------------------------------------


def test_classify_sources_auditer_manager():
    target = "/some/path/github_auditer/data/my-slug/docs"
    manager, cmds = _classify_sources(target, [])
    assert manager == "auditer"
    assert any("auditer checkout --account my-slug" in c for c in cmds)


def test_classify_sources_manual_manager(tmp_path):
    target = str(tmp_path / "manual_docs")
    manager, cmds = _classify_sources(target, [])
    assert manager == "manual"
    assert any(f"git -C {Path(target).resolve()} pull" in c for c in cmds)


def test_classify_sources_mixed_manager(tmp_path):
    auditer_target = "/some/path/github_auditer/data/my-slug/docs"
    manual_target = str(tmp_path / "manual_docs")
    entries = [
        _DocEntry("auditer_link", True, auditer_target, True),
        _DocEntry("manual_link", True, manual_target, True),
    ]
    manager, cmds = _classify_sources(None, entries)
    assert manager == "mixed"
    assert any("auditer checkout" in c for c in cmds)
    assert any("git -C" in c for c in cmds)


def test_classify_sources_from_doc_entries(tmp_path):
    """When docs_symlink_target is None, targets are gathered from doc_entries."""
    target = "/some/path/github_auditer/data/my-slug/docs"
    entries = [
        _DocEntry("auditer_link", True, target, True),
    ]
    manager, cmds = _classify_sources(None, entries)
    assert manager == "auditer"
    assert any("auditer checkout --account my-slug" in c for c in cmds)
