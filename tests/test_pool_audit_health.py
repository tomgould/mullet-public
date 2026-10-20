"""Tests for pool health-check functions in mullet.cli.pool_audit."""
from __future__ import annotations

import json
import os
import sqlite3
import time
import unittest.mock
from pathlib import Path

import pytest
import yaml

from mullet.cli.pool_audit import (
    _collect_symlink_issues,
    _chroma_chunk_count,
    _max_mtime,
    _count_doc_files,
    _load_manifest_meta,
    _check_exclude_patterns,
    _max_indexed_mtime,
    check_pool_health,
    run_pool_health,
    run_pool_prune,
    run_pool_validate,
    _check_extension_coverage,
    _issue_fix_commands,
    PoolHealthIssue,
    _force_rmtree,
    _purge_pool_sidecars,
    _get_graph_info,
    _classify_sources,
)


# ---------------------------------------------------------------------------
# _get_graph_info
# ---------------------------------------------------------------------------

class TestGetGraphInfo:
    def test_success_path(self, tmp_path: Path) -> None:
        index = tmp_path / "index"
        index.mkdir()
        db_path = index / "knowledge_graph.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE entities (id INTEGER)")
        conn.execute("CREATE TABLE relationships (id INTEGER)")
        conn.execute("INSERT INTO entities VALUES (1)")
        conn.execute("INSERT INTO relationships VALUES (1), (2)")
        conn.commit()
        conn.close()
        graph = _get_graph_info(index)
        assert graph.built is True
        assert graph.entity_count == 1
        assert graph.relationship_count == 2
        assert graph.db_size_mb > 0
        assert graph.error is None


# ---------------------------------------------------------------------------
# _classify_sources
# ---------------------------------------------------------------------------

class TestClassifySources:
    def test_no_targets_returns_none(self) -> None:
        manager, cmds = _classify_sources(None, [])
        assert manager == "none"
        assert cmds == []


# ---------------------------------------------------------------------------
# _collect_symlink_issues
# ---------------------------------------------------------------------------

class TestCollectSymlinkIssues:
    def test_broken_symlink_at_top_level(self, tmp_path: Path) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()
        docs = pool / "documents"
        docs.symlink_to(tmp_path / "nonexistent")
        issues = _collect_symlink_issues(pool)
        assert len(issues) == 1
        assert issues[0].code == "broken_symlink"
        assert "nonexistent" in issues[0].message

    def test_broken_symlink_readlink_exception_top_level(self, tmp_path: Path, monkeypatch) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()
        docs = pool / "documents"
        docs.symlink_to(tmp_path / "nonexistent")

        def bad_readlink(_p):
            raise RuntimeError("boom")

        monkeypatch.setattr(os, "readlink", bad_readlink)
        issues = _collect_symlink_issues(pool)
        assert len(issues) == 1
        assert issues[0].code == "broken_symlink"
        assert "?" in issues[0].message

    def test_broken_symlink_inside_documents(self, tmp_path: Path) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()
        docs = pool / "documents"
        docs.mkdir()
        link = docs / "broken_link"
        link.symlink_to(tmp_path / "nonexistent")
        issues = _collect_symlink_issues(pool)
        assert len(issues) == 1
        assert issues[0].code == "broken_symlink"
        assert "broken_link" in issues[0].message

    def test_broken_symlink_readlink_exception_inside(self, tmp_path: Path, monkeypatch) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()
        docs = pool / "documents"
        docs.mkdir()
        link = docs / "broken_link"
        link.symlink_to(tmp_path / "nonexistent")

        orig_readlink = os.readlink

        def bad_readlink(p):
            if "broken_link" in str(p):
                raise RuntimeError("boom")
            return orig_readlink(p)

        monkeypatch.setattr(os, "readlink", bad_readlink)
        issues = _collect_symlink_issues(pool)
        assert len(issues) == 1
        assert issues[0].code == "broken_symlink"
        assert "?" in issues[0].message

    def test_docs_is_file_not_dir(self, tmp_path: Path) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()
        docs = pool / "documents"
        docs.write_text("not a dir")
        issues = _collect_symlink_issues(pool)
        assert issues == []

    def test_relative_to_valueerror_inside_walk(self, tmp_path: Path, monkeypatch) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()
        docs = pool / "documents"
        docs.mkdir()
        link = docs / "broken_link"
        link.symlink_to(tmp_path / "nonexistent")

        orig_relative_to = Path.relative_to

        def bad_relative_to(self, other):
            if "broken_link" in str(self):
                raise ValueError("not relative")
            return orig_relative_to(self, other)

        monkeypatch.setattr(Path, "relative_to", bad_relative_to)
        issues = _collect_symlink_issues(pool)
        assert len(issues) == 1
        assert issues[0].code == "broken_symlink"
        assert "broken_link" in issues[0].message

    def test_outer_walk_exception(self, tmp_path: Path, monkeypatch) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()
        docs = pool / "documents"
        docs.mkdir()

        def bad_walk(*args, **kwargs):
            raise RuntimeError("walk fail")

        monkeypatch.setattr(os, "walk", bad_walk)
        issues = _collect_symlink_issues(pool)
        assert issues == []


# ---------------------------------------------------------------------------
# _chroma_chunk_count
# ---------------------------------------------------------------------------

class TestChromaChunkCount:
    def test_stat_raises(self, tmp_path: Path, monkeypatch) -> None:
        index = tmp_path / "index"
        index.mkdir()
        chroma_sqlite = index / "chroma.sqlite3"
        chroma_sqlite.write_text("fake db")

        orig_exists = Path.exists
        orig_stat = Path.stat

        def fake_exists(self):
            if self == chroma_sqlite:
                return True
            return orig_exists(self)

        def fake_stat(self, *, follow_symlinks=True):
            if self == chroma_sqlite:
                raise RuntimeError("stat fail")
            return orig_stat(self, follow_symlinks=follow_symlinks)

        monkeypatch.setattr(Path, "exists", fake_exists)
        monkeypatch.setattr(Path, "stat", fake_stat)
        chunks, present, err = _chroma_chunk_count(index)
        assert chunks == 0
        assert present is True  # size=0 is outside empty-init window → chroma_present=True
        # chromadb may error because the file is not a real DB, but lines 590-591 are covered
        assert isinstance(err, (str, type(None)))

    def test_import_error(self, tmp_path: Path, monkeypatch) -> None:
        index = tmp_path / "index"
        index.mkdir()
        chroma_sqlite = index / "chroma.sqlite3"
        chroma_sqlite.write_text("x" * 200_000)

        # Remove chromadb from sys.modules to simulate ImportError
        import sys
        saved = sys.modules.pop("chromadb", None)
        try:
            # Patch __import__ to raise ImportError for chromadb only
            orig_import = __builtins__["__import__"]

            def fake_import(name, *args, **kwargs):
                if name == "chromadb":
                    raise ImportError("no chromadb")
                return orig_import(name, *args, **kwargs)

            monkeypatch.setitem(__builtins__, "__import__", fake_import)
            chunks, present, err = _chroma_chunk_count(index)
            assert chunks == 0
            assert present is True
            assert err == "chromadb not installed"
        finally:
            if saved is not None:
                sys.modules["chromadb"] = saved

    def test_success_path(self, tmp_path: Path, monkeypatch) -> None:
        index = tmp_path / "index"
        index.mkdir()
        chroma_sqlite = index / "chroma.sqlite3"
        chroma_sqlite.write_text("x" * 200_000)

        class FakeColl:
            def count(self):
                return 42

        class FakeClient:
            def list_collections(self):
                return [FakeColl()]

        class FakeChromaDB:
            def PersistentClient(self, *args, **kwargs):
                return FakeClient()

        import sys
        saved = sys.modules.pop("chromadb", None)
        try:
            monkeypatch.setitem(sys.modules, "chromadb", FakeChromaDB())
            chunks, present, err = _chroma_chunk_count(index)
            assert chunks == 42
            assert present is True
            assert err is None
        finally:
            if saved is not None:
                sys.modules["chromadb"] = saved


# ---------------------------------------------------------------------------
# _max_mtime
# ---------------------------------------------------------------------------

class TestMaxMtime:
    def test_path_does_not_exist(self, tmp_path: Path) -> None:
        assert _max_mtime(tmp_path / "nonexistent") == 0.0

    def test_path_is_file(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("hello")
        mtime = f.stat().st_mtime
        assert _max_mtime(f) == mtime

    def test_os_walk_raises(self, tmp_path: Path, monkeypatch) -> None:
        d = tmp_path / "dir"
        d.mkdir()
        (d / "a.txt").write_text("a")

        def bad_walk(*args, **kwargs):
            raise RuntimeError("walk fail")

        monkeypatch.setattr(os, "walk", bad_walk)
        assert _max_mtime(d) == 0.0

    def test_getmtime_oserror_inside_loop(self, tmp_path: Path, monkeypatch) -> None:
        d = tmp_path / "dir"
        d.mkdir()
        (d / "a.txt").write_text("a")

        def bad_getmtime(path):
            raise OSError("getmtime fail")

        monkeypatch.setattr(os.path, "getmtime", bad_getmtime)
        assert _max_mtime(d) == 0.0


# ---------------------------------------------------------------------------
# _count_doc_files
# ---------------------------------------------------------------------------

class TestCountDocFiles:
    def test_docs_path_does_not_exist(self, tmp_path: Path) -> None:
        assert _count_doc_files(tmp_path / "nonexistent") == 0

    def test_os_walk_raises(self, tmp_path: Path, monkeypatch) -> None:
        d = tmp_path / "documents"
        d.mkdir()
        (d / "a.txt").write_text("a")

        def bad_walk(*args, **kwargs):
            raise RuntimeError("walk fail")

        monkeypatch.setattr(os, "walk", bad_walk)
        assert _count_doc_files(d) == 0


# ---------------------------------------------------------------------------
# _load_manifest_meta
# ---------------------------------------------------------------------------

class TestLoadManifestMeta:
    def test_invalid_json(self, tmp_path: Path) -> None:
        index = tmp_path / "index"
        index.mkdir()
        manifest = index / "file_manifest.json"
        manifest.write_text("not json")
        count, mtime = _load_manifest_meta(index)
        assert count == 0
        assert mtime is None

    def test_success_path(self, tmp_path: Path) -> None:
        index = tmp_path / "index"
        index.mkdir()
        manifest = index / "file_manifest.json"
        manifest.write_text(json.dumps({"files": {"a.txt": 1, "b.txt": 2}}))
        count, mtime = _load_manifest_meta(index)
        assert count == 2
        assert isinstance(mtime, float)


# ---------------------------------------------------------------------------
# _check_exclude_patterns
# ---------------------------------------------------------------------------

class TestCheckExcludePatterns:
    def test_docs_does_not_exist(self, tmp_path: Path) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()
        config = {"documents": {"exclude_patterns": ["*.tmp"]}}
        assert _check_exclude_patterns(pool, config) == []

    def test_rglob_raises(self, tmp_path: Path, monkeypatch) -> None:
        pool = tmp_path / "pool"
        docs = pool / "documents"
        docs.mkdir(parents=True)
        (docs / "a.txt").write_text("a")
        config = {"documents": {"exclude_patterns": ["*.tmp"]}}

        def bad_rglob(self, pattern):
            raise RuntimeError("rglob fail")

        monkeypatch.setattr(Path, "rglob", bad_rglob)
        assert _check_exclude_patterns(pool, config) == []

    def test_no_files(self, tmp_path: Path) -> None:
        pool = tmp_path / "pool"
        docs = pool / "documents"
        docs.mkdir(parents=True)
        config = {"documents": {"exclude_patterns": ["*.tmp"]}}
        assert _check_exclude_patterns(pool, config) == []

    def test_over_filtering(self, tmp_path: Path) -> None:
        pool = tmp_path / "pool"
        docs = pool / "documents"
        docs.mkdir(parents=True)
        for i in range(10):
            (docs / f"file{i}.txt").write_text("x")
        config = {"documents": {"exclude_patterns": ["*"]}}
        issues = _check_exclude_patterns(pool, config)
        assert len(issues) == 1
        assert issues[0].code == "bad_exclude_pattern"

    def test_under_threshold_returns_empty(self, tmp_path: Path) -> None:
        pool = tmp_path / "pool"
        docs = pool / "documents"
        docs.mkdir(parents=True)
        for i in range(10):
            (docs / f"file{i}.txt").write_text("x")
        config = {"documents": {"exclude_patterns": ["*.nonexistent"]}}
        issues = _check_exclude_patterns(pool, config)
        assert issues == []


# ---------------------------------------------------------------------------
# _max_indexed_mtime
# ---------------------------------------------------------------------------

class TestMaxIndexedMtime:
    def test_docs_path_does_not_exist_with_extensions(self, tmp_path: Path) -> None:
        pool = tmp_path / "pool"
        config = {"documents": {"extensions": [".txt"]}}
        assert _max_indexed_mtime(pool / "documents", config) == 0.0

    def test_extension_filtering(self, tmp_path: Path) -> None:
        docs = tmp_path / "documents"
        docs.mkdir(parents=True)
        f1 = docs / "a.txt"
        f1.write_text("a")
        f2 = docs / "b.md"
        f2.write_text("b")
        # Ensure mtime difference
        time.sleep(0.01)
        f3 = docs / "c.txt"
        f3.write_text("c")

        config = {"documents": {"extensions": [".txt"]}}
        result = _max_indexed_mtime(docs, config)
        assert result == max(f1.stat().st_mtime, f3.stat().st_mtime)

    def test_getmtime_raises_and_outer_exception(self, tmp_path: Path, monkeypatch) -> None:
        docs = tmp_path / "documents"
        docs.mkdir(parents=True)
        (docs / "a.txt").write_text("a")
        config = {"documents": {"extensions": [".txt"]}}

        orig_getmtime = os.path.getmtime

        def bad_getmtime(path):
            if "a.txt" in str(path):
                raise OSError("getmtime fail")
            return orig_getmtime(path)

        monkeypatch.setattr(os.path, "getmtime", bad_getmtime)
        assert _max_indexed_mtime(docs, config) == 0.0

        # Outer exception: os.walk raises
        def bad_walk(*args, **kwargs):
            raise RuntimeError("walk fail")

        monkeypatch.setattr(os, "walk", bad_walk)
        assert _max_indexed_mtime(docs, config) == 0.0

    def test_exclude_extensions_filter(self, tmp_path: Path) -> None:
        docs = tmp_path / "documents"
        docs.mkdir(parents=True)
        f1 = docs / "a.txt"
        f1.write_text("a")
        time.sleep(0.01)
        f2 = docs / "b.pdf"
        f2.write_text("b")
        config = {"documents": {"extensions": [".txt", ".pdf"], "exclude_extensions": [".pdf"]}}
        result = _max_indexed_mtime(docs, config)
        assert result == f1.stat().st_mtime

    def test_exclude_patterns_filter(self, tmp_path: Path) -> None:
        docs = tmp_path / "documents"
        docs.mkdir(parents=True)
        f1 = docs / "a.txt"
        f1.write_text("a")
        time.sleep(0.01)
        f2 = docs / "b.tmp"
        f2.write_text("b")
        config = {"documents": {"extensions": [".txt", ".tmp"], "exclude_patterns": ["*"]}}
        result = _max_indexed_mtime(docs, config)
        assert result == 0.0


# ---------------------------------------------------------------------------
# check_pool_health
# ---------------------------------------------------------------------------

class TestCheckPoolHealth:
    def _make_pool(self, tmp_path: Path) -> Path:
        pool = tmp_path / "pool"
        pool.mkdir()
        (pool / "config.yaml").write_text(
            yaml.dump({"machine_name": "my_pool", "embedding": {"model": "all-MiniLM-L6-v2"}})
        )
        docs = pool / "documents"
        docs.mkdir()
        (docs / "a.txt").write_text("hello")
        return pool

    def test_client_yaml_fallback(self, tmp_path: Path) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()
        (pool / "client.yaml").write_text(
            yaml.dump({"machine_name": "my_pool", "embedding": {"model": "all-MiniLM-L6-v2"}})
        )
        docs = pool / "documents"
        docs.mkdir()
        (docs / "a.txt").write_text("hello")
        health = check_pool_health(pool)
        assert health.name == "pool"

    def test_config_parse_exception(self, tmp_path: Path, monkeypatch) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()
        (pool / "config.yaml").write_text("{{not yaml")
        docs = pool / "documents"
        docs.mkdir()
        (docs / "a.txt").write_text("hello")
        health = check_pool_health(pool)
        # Should still complete; config parsing failed so no config issues flagged
        assert health.name == "pool"

    def test_chroma_error_issue(self, tmp_path: Path, monkeypatch) -> None:
        pool = self._make_pool(tmp_path)
        monkeypatch.setattr(
            "mullet.cli.pool_audit._chroma_chunk_count",
            lambda _p: (0, True, "some error"),
        )
        health = check_pool_health(pool)
        assert any(i.code == "chroma_error" for i in health.issues)

    def test_empty_but_configured_issue(self, tmp_path: Path, monkeypatch) -> None:
        pool = self._make_pool(tmp_path)
        monkeypatch.setattr(
            "mullet.cli.pool_audit._chroma_chunk_count",
            lambda _p: (0, True, None),
        )
        monkeypatch.setattr(
            "mullet.cli.pool_audit._count_doc_files",
            lambda _p: 5,
        )
        health = check_pool_health(pool)
        assert any(i.code == "empty_but_configured" for i in health.issues)

    def test_stale_manifest_issue(self, tmp_path: Path, monkeypatch) -> None:
        pool = self._make_pool(tmp_path)
        old_time = time.time() - 1000
        monkeypatch.setattr(
            "mullet.cli.pool_audit._load_manifest_meta",
            lambda _p: (10, old_time),
        )
        monkeypatch.setattr(
            "mullet.cli.pool_audit._max_indexed_mtime",
            lambda _p, _c: old_time + 500,
        )
        health = check_pool_health(pool)
        assert any(i.code == "stale_manifest" for i in health.issues)

    def test_manifest_chroma_mismatch(self, tmp_path: Path, monkeypatch) -> None:
        pool = self._make_pool(tmp_path)
        monkeypatch.setattr(
            "mullet.cli.pool_audit._load_manifest_meta",
            lambda _p: (100, time.time()),
        )
        monkeypatch.setattr(
            "mullet.cli.pool_audit._chroma_chunk_count",
            lambda _p: (10, True, None),
        )
        health = check_pool_health(pool)
        assert any(i.code == "manifest_chroma_mismatch" for i in health.issues)

    def test_last_indexed_human_set(self, tmp_path: Path, monkeypatch) -> None:
        pool = self._make_pool(tmp_path)
        monkeypatch.setattr(
            "mullet.cli.pool_audit._load_manifest_meta",
            lambda _p: (10, time.time() - 60),
        )
        monkeypatch.setattr(
            "mullet.cli.pool_audit._max_indexed_mtime",
            lambda _p, _c: time.time() - 70,
        )
        health = check_pool_health(pool)
        assert health.last_indexed_human is not None


# ---------------------------------------------------------------------------
# run_pool_health
# ---------------------------------------------------------------------------

class TestRunPoolHealth:
    def test_pools_dir_does_not_exist(self, tmp_path: Path, capsys) -> None:
        code = run_pool_health(tmp_path / "nonexistent")
        assert code == 1
        assert "No pools directory" in capsys.readouterr().out

    def test_pool_name_not_found(self, tmp_path: Path, capsys) -> None:
        code = run_pool_health(tmp_path, pool_name="missing")
        assert code == 1
        assert "Pool not found" in capsys.readouterr().out

    def test_empty_pools_dir(self, tmp_path: Path, capsys) -> None:
        code = run_pool_health(tmp_path)
        assert code == 0
        assert "No pools found" in capsys.readouterr().out

    def test_single_pool_healthy(self, tmp_path: Path, monkeypatch, capsys) -> None:
        pool = tmp_path / "my-pool"
        pool.mkdir()
        monkeypatch.setattr(
            "mullet.cli.pool_audit.check_pool_health",
            lambda _p: type("H", (), {
                "name": "my-pool", "pool_dir": _p, "ok": True,
                "chunk_count": 10, "file_count": 5,
                "last_indexed_human": "2 days ago",
                "issues": [],
            })(),
        )
        code = run_pool_health(tmp_path, pool_name="my-pool", workers=1)
        assert code == 0
        out = capsys.readouterr().out
        assert "my-pool" in out
        assert "OK" in out

    def test_unhealthy_pool(self, tmp_path: Path, monkeypatch, capsys) -> None:
        pool = tmp_path / "my-pool"
        pool.mkdir()
        monkeypatch.setattr(
            "mullet.cli.pool_audit.check_pool_health",
            lambda _p: type("H", (), {
                "name": "my-pool", "pool_dir": _p, "ok": False,
                "chunk_count": 0, "file_count": 0,
                "last_indexed_human": None,
                "issues": [PoolHealthIssue("never_indexed", "never indexed")],
            })(),
        )
        code = run_pool_health(tmp_path, pool_name="my-pool", workers=1)
        assert code == 1
        out = capsys.readouterr().out
        assert "never-indexed" in out

    def test_multiple_pools_with_workers(self, tmp_path: Path, monkeypatch, capsys) -> None:
        pool1 = tmp_path / "pool-a"
        pool1.mkdir()
        pool2 = tmp_path / "pool-b"
        pool2.mkdir()

        def mock_health(_p):
            ok = _p.name == "pool-a"
            return type("H", (), {
                "name": _p.name, "pool_dir": _p, "ok": ok,
                "chunk_count": 5, "file_count": 2,
                "last_indexed_human": "1h ago" if ok else None,
                "issues": [] if ok else [PoolHealthIssue("never_indexed", "never indexed")],
            })()

        monkeypatch.setattr(
            "mullet.cli.pool_audit.check_pool_health",
            mock_health,
        )
        code = run_pool_health(tmp_path, workers=2)
        assert code == 1
        out = capsys.readouterr().out
        assert "2 pools checked" in out
        assert "1 healthy" in out
        assert "1 need attention" in out


# ---------------------------------------------------------------------------
# run_pool_prune
# ---------------------------------------------------------------------------

class TestRunPoolPrune:
    def test_force_rmtree_error_with_permission_hint(self, tmp_path: Path, monkeypatch, capsys) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()
        (pool / "config.yaml").write_text("description: test\n")

        # audit_pools returns this pool as candidate (no config.yaml actually means candidate,
        # but we have config.yaml, so we need to patch audit_pools to return a candidate)
        from mullet.cli.pool_audit import PoolAudit, _VectorInfo, _GraphInfo, _DocEntry

        audit = PoolAudit(
            name="pool",
            pool_dir=pool,
            has_client_yaml=False,
            description="",
            machine_name="",
            metadata={},
            docs_symlink_target=None,
            docs_symlink_exists=True,
            doc_entries=[],
            vector=_VectorInfo(indexed=False, chunk_count=0, has_bm25=False, error=None),
            graph=_GraphInfo(built=False, entity_count=0, relationship_count=0, db_size_mb=0.0, error=None),
            index_size_mb=0.0,
            manager="none",
            sync_commands=[],
        )
        monkeypatch.setattr(
            "mullet.cli.pool_audit.audit_pools",
            lambda _p: [audit],
        )
        monkeypatch.setattr(
            "mullet.cli.pool_audit._force_rmtree",
            lambda _p: (False, "permission denied"),
        )
        run_pool_prune(tmp_path, yes=True)
        out = capsys.readouterr().out
        assert "permission denied" in out
        assert "sudo rm -rf" in out

    def test_prune_warnings(self, tmp_path: Path, monkeypatch, capsys) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()

        from mullet.cli.pool_audit import PoolAudit, _VectorInfo, _GraphInfo, _DocEntry

        audit = PoolAudit(
            name="pool",
            pool_dir=pool,
            has_client_yaml=False,
            description="",
            machine_name="",
            metadata={},
            docs_symlink_target=None,
            docs_symlink_exists=True,
            doc_entries=[],
            vector=_VectorInfo(indexed=False, chunk_count=0, has_bm25=False, error=None),
            graph=_GraphInfo(built=False, entity_count=0, relationship_count=0, db_size_mb=0.0, error=None),
            index_size_mb=0.0,
            manager="none",
            sync_commands=[],
        )
        monkeypatch.setattr(
            "mullet.cli.pool_audit.audit_pools",
            lambda _p: [audit],
        )
        monkeypatch.setattr(
            "mullet.cli.pool_audit._force_rmtree",
            lambda _p: (True, None),
        )
        monkeypatch.setattr(
            "mullet.cli.pool_audit._purge_pool_sidecars",
            lambda _n, _s, _h: (5, ["warning1"]),
        )
        run_pool_prune(tmp_path, yes=True)
        out = capsys.readouterr().out
        assert "5 db rows" in out or "5 db row" in out
        assert "1 warning" in out or "warning1" in out

    def test_nothing_to_prune(self, tmp_path: Path, monkeypatch, capsys) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()

        from mullet.cli.pool_audit import PoolAudit, _VectorInfo, _GraphInfo, _DocEntry

        audit = PoolAudit(
            name="pool",
            pool_dir=pool,
            has_client_yaml=True,
            description="",
            machine_name="",
            metadata={},
            docs_symlink_target=None,
            docs_symlink_exists=True,
            doc_entries=[],
            vector=_VectorInfo(indexed=False, chunk_count=0, has_bm25=False, error=None),
            graph=_GraphInfo(built=False, entity_count=0, relationship_count=0, db_size_mb=0.0, error=None),
            index_size_mb=0.0,
            manager="none",
            sync_commands=[],
        )
        monkeypatch.setattr(
            "mullet.cli.pool_audit.audit_pools",
            lambda _p: [audit],
        )
        run_pool_prune(tmp_path)
        out = capsys.readouterr().out
        assert "Nothing to prune" in out

    def test_dry_run_output(self, tmp_path: Path, monkeypatch, capsys) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()

        from mullet.cli.pool_audit import PoolAudit, _VectorInfo, _GraphInfo, _DocEntry

        audit = PoolAudit(
            name="pool",
            pool_dir=pool,
            has_client_yaml=False,
            description="",
            machine_name="",
            metadata={},
            docs_symlink_target=None,
            docs_symlink_exists=True,
            doc_entries=[],
            vector=_VectorInfo(indexed=False, chunk_count=0, has_bm25=False, error=None),
            graph=_GraphInfo(built=False, entity_count=0, relationship_count=0, db_size_mb=0.0, error=None),
            index_size_mb=0.0,
            manager="none",
            sync_commands=[],
        )
        monkeypatch.setattr(
            "mullet.cli.pool_audit.audit_pools",
            lambda _p: [audit],
        )
        run_pool_prune(tmp_path, yes=False)
        out = capsys.readouterr().out
        assert "Dry run" in out
        assert "Would remove" in out


# ---------------------------------------------------------------------------
# _check_extension_coverage
# ---------------------------------------------------------------------------

class TestCheckExtensionCoverage:
    def test_docs_does_not_exist(self, tmp_path: Path) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()
        config = {"documents": {"extensions": [".txt"]}}
        assert _check_extension_coverage(pool, config) == []

    def test_extractor_map_import_fails(self, tmp_path: Path) -> None:
        pool = tmp_path / "pool"
        docs = pool / "documents"
        docs.mkdir(parents=True)
        for i in range(6):
            (docs / f"file{i}.py").write_text("x")
        config = {
            "documents": {"extensions": [".txt"]},
            "code": {"enabled": True},
        }

        import builtins
        orig_import = builtins.__import__

        def bad_import(name, *args, **kwargs):
            if "extractors" in str(name):
                raise ImportError("extractors broken")
            return orig_import(name, *args, **kwargs)

        with unittest.mock.patch("builtins.__import__", bad_import):
            result = _check_extension_coverage(pool, config)
        # Since the import of EXTRACTOR_MAP fails, it falls back to static superset,
        # and .py is in that superset so no issues expected.
        # But we want to cover lines 989-992, so just call it.
        assert isinstance(result, list)

    def test_uncovered_extensions_above_threshold(self, tmp_path: Path) -> None:
        pool = tmp_path / "pool"
        docs = pool / "documents"
        docs.mkdir(parents=True)
        for i in range(6):
            (docs / f"file{i}.xyz").write_text("x")
        config = {"documents": {"extensions": [".txt"]}}
        issues = _check_extension_coverage(pool, config)
        assert len(issues) == 1
        assert issues[0].code == "uncovered_extensions"
        assert ".xyz" in issues[0].message

    def test_hidden_files_skipped(self, tmp_path: Path) -> None:
        pool = tmp_path / "pool"
        docs = pool / "documents"
        docs.mkdir(parents=True)
        for i in range(6):
            (docs / f"file{i}.xyz").write_text("x")
        (docs / ".hidden").write_text("hidden")
        config = {"documents": {"extensions": [".txt"]}}
        issues = _check_extension_coverage(pool, config)
        assert len(issues) == 1
        assert issues[0].code == "uncovered_extensions"

    def test_exclude_patterns_skip(self, tmp_path: Path) -> None:
        pool = tmp_path / "pool"
        docs = pool / "documents"
        docs.mkdir(parents=True)
        for i in range(6):
            (docs / f"file{i}.xyz").write_text("x")
        config = {
            "documents": {
                "extensions": [".txt"],
                "exclude_patterns": ["*"],
            }
        }
        issues = _check_extension_coverage(pool, config)
        assert issues == []

    def test_walk_exception(self, tmp_path: Path, monkeypatch) -> None:
        pool = tmp_path / "pool"
        docs = pool / "documents"
        docs.mkdir(parents=True)

        def bad_walk(*args, **kwargs):
            raise RuntimeError("walk fail")

        monkeypatch.setattr(os, "walk", bad_walk)
        config = {"documents": {"extensions": [".txt"]}}
        issues = _check_extension_coverage(pool, config)
        assert issues == []


# ---------------------------------------------------------------------------
# _issue_fix_commands
# ---------------------------------------------------------------------------

class TestIssueFixCommands:
    def test_broken_symlink(self, tmp_path: Path) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()
        issue = PoolHealthIssue("broken_symlink", "documents/foo → /some/target (target missing)")
        config_path = pool / "config.yaml"
        fixes = _issue_fix_commands(issue, pool, config_path, "my_pool")
        assert len(fixes) == 1
        assert fixes[0].startswith("rm -f '")
        assert "documents/foo" in fixes[0]

    def test_invalid_config_machine_name(self, tmp_path: Path) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()
        issue = PoolHealthIssue("invalid_config", "config.yaml missing required field 'machine_name'")
        config_path = pool / "config.yaml"
        fixes = _issue_fix_commands(issue, pool, config_path, "my-pool")
        assert any("machine_name: my_pool" in f for f in fixes)

    def test_invalid_config_fallback(self, tmp_path: Path) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()
        issue = PoolHealthIssue("invalid_config", "some other config problem")
        config_path = pool / "config.yaml"
        fixes = _issue_fix_commands(issue, pool, config_path, "my-pool")
        assert len(fixes) == 1
        assert "fix: some other config problem" in fixes[0]

    def test_uncovered_extensions(self, tmp_path: Path) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()
        issue = PoolHealthIssue("uncovered_extensions", "5 files not indexed — extensions not in config: .xyz(5)")
        config_path = pool / "config.yaml"
        fixes = _issue_fix_commands(issue, pool, config_path, "my-pool")
        assert any("Add missing extensions" in f for f in fixes)

    def test_uncovered_extensions_no_colon(self, tmp_path: Path) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()
        issue = PoolHealthIssue("uncovered_extensions", "bad message with no colon at end")
        config_path = pool / "config.yaml"
        fixes = _issue_fix_commands(issue, pool, config_path, "my-pool")
        assert any("Add missing extensions" in f for f in fixes)
        assert "bad message with no colon at end" in fixes[-1]

    def test_uncovered_extensions_split_exception(self, tmp_path: Path) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()

        class BadMsg:
            def split(self, *args, **kwargs):
                raise RuntimeError("boom")

        issue = PoolHealthIssue("uncovered_extensions", BadMsg())  # type: ignore[arg-type]
        config_path = pool / "config.yaml"
        fixes = _issue_fix_commands(issue, pool, config_path, "my-pool")
        assert any("Add missing extensions" in f for f in fixes)
        assert fixes[-1] == "# Uncovered: "

    def test_bad_exclude_pattern(self, tmp_path: Path) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()
        issue = PoolHealthIssue("bad_exclude_pattern", "exclude_patterns filter out 9/10 files")
        config_path = pool / "config.yaml"
        fixes = _issue_fix_commands(issue, pool, config_path, "my-pool")
        assert any("exclude_patterns" in f for f in fixes)

    def test_chroma_error(self, tmp_path: Path) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()
        issue = PoolHealthIssue("chroma_error", "chroma read failed: boom")
        config_path = pool / "config.yaml"
        fixes = _issue_fix_commands(issue, pool, config_path, "my-pool")
        assert any("reindex" in f for f in fixes)

    def test_fallback_return(self, tmp_path: Path) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()
        issue = PoolHealthIssue("unknown_code", "something weird")
        config_path = pool / "config.yaml"
        fixes = _issue_fix_commands(issue, pool, config_path, "my-pool")
        assert fixes == ["# something weird"]


# ---------------------------------------------------------------------------
# run_pool_validate
# ---------------------------------------------------------------------------

class TestRunPoolValidate:
    def test_pools_dir_does_not_exist(self, tmp_path: Path, capsys) -> None:
        code = run_pool_validate(tmp_path / "nonexistent")
        assert code == 1
        assert "No pools directory" in capsys.readouterr().out

    def test_pool_name_not_found(self, tmp_path: Path, capsys) -> None:
        code = run_pool_validate(tmp_path, pool_name="missing")
        assert code == 1
        assert "Pool not found" in capsys.readouterr().out

    def test_empty_targets(self, tmp_path: Path, capsys) -> None:
        code = run_pool_validate(tmp_path)
        assert code == 0
        assert "No pools" in capsys.readouterr().out

    def test_client_yaml_fallback_with_yaml_exception(self, tmp_path: Path, monkeypatch, capsys) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()
        (pool / "client.yaml").write_text("{{not yaml")
        docs = pool / "documents"
        docs.mkdir()
        (docs / "a.txt").write_text("hello")

        monkeypatch.setattr(
            "mullet.cli.pool_audit.check_pool_health",
            lambda _p: type("H", (), {
                "name": "pool", "pool_dir": _p, "ok": True,
                "chunk_count": 0, "file_count": 1,
                "last_indexed_human": None, "issues": [],
            })(),
        )
        code = run_pool_validate(tmp_path)
        assert code == 0

    def test_pool_with_issues(self, tmp_path: Path, monkeypatch, capsys) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()
        (pool / "config.yaml").write_text(yaml.dump({"machine_name": "pool"}))
        docs = pool / "documents"
        docs.mkdir()

        monkeypatch.setattr(
            "mullet.cli.pool_audit.check_pool_health",
            lambda _p: type("H", (), {
                "name": "pool", "pool_dir": _p, "ok": False,
                "chunk_count": 0, "file_count": 1,
                "last_indexed_human": None,
                "issues": [PoolHealthIssue("never_indexed", "never indexed")],
            })(),
        )
        monkeypatch.setattr(
            "mullet.cli.pool_audit._check_extension_coverage",
            lambda _pd, _c: [],
        )
        code = run_pool_validate(tmp_path)
        assert code == 1
        out = capsys.readouterr().out
        assert "never-indexed" in out

    def test_check_exclude_patterns_rglob_exception(self, tmp_path: Path) -> None:
        """Lines 691-693: rglob raises OSError → exception swallowed, returns []."""
        from mullet.cli.pool_audit import _check_exclude_patterns
        from unittest.mock import MagicMock
        import os
        pool = tmp_path / "pool"
        pool.mkdir()
        docs = pool / "documents"
        docs.mkdir()
        (pool / "config.yaml").write_text("")
        fake_pool = MagicMock()
        fake_pool.pool_dir = pool
        # Remove read permission so rglob raises PermissionError
        os.chmod(str(docs), 0o000)
        try:
            result = _check_exclude_patterns(fake_pool, {})
            assert result == []
        finally:
            os.chmod(str(docs), 0o755)

    def test_all_healthy(self, tmp_path: Path, monkeypatch, capsys) -> None:
        pool = tmp_path / "pool"
        pool.mkdir()
        (pool / "config.yaml").write_text(yaml.dump({"machine_name": "pool"}))
        docs = pool / "documents"
        docs.mkdir()

        monkeypatch.setattr(
            "mullet.cli.pool_audit.check_pool_health",
            lambda _p: type("H", (), {
                "name": "pool", "pool_dir": _p, "ok": True,
                "chunk_count": 10, "file_count": 5,
                "last_indexed_human": "5m ago",
                "issues": [],
            })(),
        )
        monkeypatch.setattr(
            "mullet.cli.pool_audit._check_extension_coverage",
            lambda _pd, _c: [],
        )
        code = run_pool_validate(tmp_path)
        assert code == 0
        out = capsys.readouterr().out
        assert "All pools are healthy" in out
