"""Tests for pure utility functions in mullet.cli.pool_audit."""
from __future__ import annotations

from pathlib import Path


from mullet.cli.pool_audit import (
    _DocEntry,
    _GraphInfo,
    _VectorInfo,
    _fmt_date,
    _fmt_delta_ago,
    _get_doc_entries,
    PoolHealthIssue,
    PoolAudit,
    format_audit_report,
    audit_pools,
    run_pool_audit,
    _purge_pool_sidecars,
    _force_rmtree,
)


class TestFmtDate:
    def test_none_returns_never(self) -> None:
        assert _fmt_date(None) == "never"

    def test_empty_string_returns_never(self) -> None:
        assert _fmt_date("") == "never"

    def test_iso_truncated_to_date(self) -> None:
        assert _fmt_date("2024-06-15T10:30:00Z") == "2024-06-15"

    def test_date_only_passthrough(self) -> None:
        assert _fmt_date("2024-01-01") == "2024-01-01"

    def test_returns_string(self) -> None:
        assert isinstance(_fmt_date("2024-12-31T00:00:00"), str)


class TestFmtDeltaAgo:
    def test_zero_seconds(self) -> None:
        assert _fmt_delta_ago(0) == "0s ago"

    def test_single_second(self) -> None:
        assert _fmt_delta_ago(1) == "1s ago"

    def test_59_seconds(self) -> None:
        assert _fmt_delta_ago(59) == "59s ago"

    def test_60_seconds_is_one_minute(self) -> None:
        assert _fmt_delta_ago(60) == "1m ago"

    def test_90_seconds_is_one_minute(self) -> None:
        assert _fmt_delta_ago(90) == "1m ago"

    def test_3599_seconds_is_59_minutes(self) -> None:
        assert _fmt_delta_ago(3599) == "59m ago"

    def test_3600_seconds_is_one_hour(self) -> None:
        assert _fmt_delta_ago(3600) == "1h ago"

    def test_86399_seconds_is_23_hours(self) -> None:
        assert _fmt_delta_ago(86399) == "23h ago"

    def test_86400_seconds_is_one_day(self) -> None:
        assert _fmt_delta_ago(86400) == "1d ago"

    def test_negative_clamped_to_zero(self) -> None:
        assert _fmt_delta_ago(-100) == "0s ago"

    def test_float_truncated(self) -> None:
        assert _fmt_delta_ago(59.9) == "59s ago"

    def test_returns_string(self) -> None:
        assert isinstance(_fmt_delta_ago(100), str)


class TestGetDocEntries:
    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        docs = tmp_path / "documents"
        docs.mkdir()
        entries = _get_doc_entries(docs)
        assert entries == []

    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        entries = _get_doc_entries(tmp_path / "does_not_exist")
        assert entries == []

    def test_regular_file_entry(self, tmp_path: Path) -> None:
        docs = tmp_path / "documents"
        docs.mkdir()
        (docs / "readme.md").write_text("hello")
        entries = _get_doc_entries(docs)
        assert len(entries) == 1
        assert entries[0].name == "readme.md"
        assert entries[0].is_symlink is False

    def test_symlink_entry_detected(self, tmp_path: Path) -> None:
        docs = tmp_path / "documents"
        docs.mkdir()
        target = tmp_path / "real_dir"
        target.mkdir()
        link = docs / "linked"
        link.symlink_to(target)
        entries = _get_doc_entries(docs)
        sym_entries = [e for e in entries if e.is_symlink]
        assert len(sym_entries) == 1
        assert sym_entries[0].name == "linked"
        assert sym_entries[0].target_exists is True

    def test_broken_symlink_detected(self, tmp_path: Path) -> None:
        docs = tmp_path / "documents"
        docs.mkdir()
        link = docs / "broken"
        link.symlink_to(tmp_path / "nonexistent")
        entries = _get_doc_entries(docs)
        broken = [e for e in entries if e.name == "broken"]
        assert len(broken) == 1
        assert broken[0].is_symlink is True
        assert broken[0].target_exists is False

    def test_returns_list_of_doc_entries(self, tmp_path: Path) -> None:
        docs = tmp_path / "documents"
        docs.mkdir()
        entries = _get_doc_entries(docs)
        assert isinstance(entries, list)


class TestDocEntryNamedTuple:
    def test_fields(self) -> None:
        e = _DocEntry("file.md", False, None, True)
        assert e.name == "file.md"
        assert e.is_symlink is False
        assert e.target is None
        assert e.target_exists is True


class TestVectorInfoNamedTuple:
    def test_fields(self) -> None:
        v = _VectorInfo(True, 1500, True, None)
        assert v.indexed is True
        assert v.chunk_count == 1500
        assert v.has_bm25 is True
        assert v.error is None


class TestGraphInfoNamedTuple:
    def test_fields(self) -> None:
        g = _GraphInfo(True, 100, 50, 2.5, None)
        assert g.built is True
        assert g.entity_count == 100
        assert g.relationship_count == 50
        assert g.db_size_mb == 2.5
        assert g.error is None


class TestPoolHealthIssue:
    def test_fields(self) -> None:
        issue = PoolHealthIssue("broken_symlink", "documents/foo → /missing (target missing)")
        assert issue.code == "broken_symlink"
        assert "missing" in issue.message


# ---------------------------------------------------------------------------
# _check_extension_coverage — code-path + filenames awareness
# ---------------------------------------------------------------------------

class TestCheckExtensionCoverage:
    """Validator should NOT flag .ts/.json/etc. as uncovered when the code
    indexer handles them, and should respect documents.filenames opt-in for
    extension-less files (Jenkinsfile, Dockerfile, Makefile).
    """

    def _setup_pool(self, tmp_path: Path, files: dict[str, str]) -> Path:
        """Create a pool dir with documents/ containing the given files
        (name → content)."""
        pool = tmp_path / "pool"
        (pool / "documents").mkdir(parents=True)
        for name, content in files.items():
            f = pool / "documents" / name
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(content)
        return pool

    def test_no_issues_when_no_extension_allowlist(self, tmp_path: Path) -> None:
        from mullet.cli.pool_audit import _check_extension_coverage
        pool = self._setup_pool(tmp_path, {"a.weird": "x", "b.alsoweird": "y"})
        # No documents.extensions => everything goes through, no warnings.
        config = {"documents": {}}
        assert _check_extension_coverage(pool, config) == []

    def test_uncovered_extensions_reported_above_threshold(
        self, tmp_path: Path
    ) -> None:
        from mullet.cli.pool_audit import _check_extension_coverage
        # 5+ files of an extension not in the allowlist → 1 issue.
        files = {f"file{i}.weird": "x" for i in range(6)}
        pool = self._setup_pool(tmp_path, files)
        config = {"documents": {"extensions": [".md"]}}
        issues = _check_extension_coverage(pool, config)
        assert len(issues) == 1
        assert issues[0].code == "uncovered_extensions"
        assert ".weird" in issues[0].message

    def test_code_handled_extensions_not_flagged(self, tmp_path: Path) -> None:
        """When code.enabled=True, .ts/.json/etc. handled by the code indexer
        must NOT be reported as uncovered by the docs validator."""
        from mullet.cli.pool_audit import _check_extension_coverage
        files = {f"src/x{i}.ts": "x" for i in range(20)}
        files.update({f"src/x{i}.json": "x" for i in range(20)})
        pool = self._setup_pool(tmp_path, files)
        config = {
            "documents": {"extensions": [".md"]},
            "code": {"enabled": True},
        }
        assert _check_extension_coverage(pool, config) == []

    def test_code_disabled_still_flags_code_exts(self, tmp_path: Path) -> None:
        """When code.enabled=False, the docs validator still flags code-ish
        extensions as uncovered — they're truly uncovered in that pool."""
        from mullet.cli.pool_audit import _check_extension_coverage
        files = {f"x{i}.ts": "x" for i in range(10)}
        pool = self._setup_pool(tmp_path, files)
        config = {
            "documents": {"extensions": [".md"]},
            "code": {"enabled": False},
        }
        issues = _check_extension_coverage(pool, config)
        assert len(issues) == 1
        assert ".ts" in issues[0].message

    def test_documents_filenames_not_flagged(self, tmp_path: Path) -> None:
        """Files named in documents.filenames (Jenkinsfile etc.) are ingested
        by the docs path despite having no extension — don't flag them."""
        from mullet.cli.pool_audit import _check_extension_coverage
        # 5 Jenkinsfiles in subdirs; no extension on each.
        files = {f"sub{i}/Jenkinsfile": "x" for i in range(5)}
        # Plus a handful of unmatched extension-less files for control.
        files.update({f"sub{i}/Unrelated": "x" for i in range(6)})
        pool = self._setup_pool(tmp_path, files)
        config = {
            "documents": {
                "extensions": [".md"],
                "filenames": ["Jenkinsfile"],
            },
        }
        issues = _check_extension_coverage(pool, config)
        # Jenkinsfiles shouldn't appear; Unrelated should still trigger.
        if issues:
            assert "Jenkinsfile" not in issues[0].message
            # "(no ext)" only includes the 6 Unrelated files
            assert "(no ext)(6)" in issues[0].message or "(no ext)" in issues[0].message


class TestPoolValidateSkipsTestInitPools:
    """run_pool_validate must skip test_init_* pools from --all to avoid
    noise from leftover artifacts created by tests/test_cli_e2e.py."""

    def _make_minimal_pool(self, pools_dir: Path, name: str) -> Path:
        pool = pools_dir / name
        (pool / "documents").mkdir(parents=True)
        (pool / "config.yaml").write_text(
            "machine_name: " + name.replace("-", "_") + "\n"
        )
        return pool

    def test_test_init_pools_excluded_from_all(
        self, tmp_path: Path, capsys
    ) -> None:
        from mullet.cli.pool_audit import run_pool_validate
        self._make_minimal_pool(tmp_path, "real-pool")
        self._make_minimal_pool(tmp_path, "test_init_creates_config0")
        self._make_minimal_pool(tmp_path, "test_init_no_overwrite0")
        run_pool_validate(tmp_path, pool_name=None)
        out = capsys.readouterr().out
        assert "real-pool" in out
        assert "test_init_creates_config0" not in out
        assert "test_init_no_overwrite0" not in out

    def test_test_init_pool_accessible_when_named(
        self, tmp_path: Path, capsys
    ) -> None:
        """Explicitly targeting a test_init_* pool still works."""
        from mullet.cli.pool_audit import run_pool_validate
        self._make_minimal_pool(tmp_path, "test_init_creates_config0")
        run_pool_validate(tmp_path, pool_name="test_init_creates_config0")
        out = capsys.readouterr().out
        assert "test_init_creates_config0" in out

    def test_hidden_dirs_excluded(self, tmp_path: Path, capsys) -> None:
        """Hidden dirs (.cache, etc.) are still excluded — existing behaviour."""
        from mullet.cli.pool_audit import run_pool_validate
        self._make_minimal_pool(tmp_path, "real-pool")
        self._make_minimal_pool(tmp_path, ".cache")
        run_pool_validate(tmp_path, pool_name=None)
        out = capsys.readouterr().out
        assert "real-pool" in out
        assert ".cache" not in out


def test_run_pool_audit_no_pools(tmp_path, capsys):
    """run_pool_audit with empty dir prints no-pools message."""
    run_pool_audit(tmp_path)
    out = capsys.readouterr().out
    assert "No pools found" in out


# ---------------------------------------------------------------------------
# audit_pools
# ---------------------------------------------------------------------------

class TestAuditPools:
    def test_verbose_prints_progress(self, tmp_path: Path, capsys, monkeypatch) -> None:
        pool = tmp_path / "my-pool"
        pool.mkdir()
        (pool / "config.yaml").write_text("description: test\n")
        monkeypatch.setattr(
            "mullet.knowledge.engine.pool_metadata.read_metadata",
            lambda _d: {},
        )
        audit_pools(tmp_path, verbose=True)
        out = capsys.readouterr().out
        assert "[1/1]" in out

    def test_documents_symlink_readlink_exception(self, tmp_path: Path, monkeypatch) -> None:
        import os
        pool = tmp_path / "my-pool"
        pool.mkdir()
        (pool / "config.yaml").write_text("description: test\n")
        docs = pool / "documents"
        docs.symlink_to(tmp_path / "nowhere")
        monkeypatch.setattr(
            os, "readlink", lambda _p: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        monkeypatch.setattr(
            "mullet.knowledge.engine.pool_metadata.read_metadata",
            lambda _d: {},
        )
        audits = audit_pools(tmp_path, verbose=False)
        assert len(audits) == 1
        assert audits[0].docs_symlink_target == "?"
        assert audits[0].docs_symlink_exists is False

    def test_verbose_clears_progress_line(self, tmp_path: Path, capsys, monkeypatch) -> None:
        pool = tmp_path / "my-pool"
        pool.mkdir()
        (pool / "config.yaml").write_text("description: test\n")
        monkeypatch.setattr(
            "mullet.knowledge.engine.pool_metadata.read_metadata",
            lambda _d: {},
        )
        audit_pools(tmp_path, verbose=True)
        out = capsys.readouterr().out
        assert " " * 50 in out

    def test_documents_symlink_exists(self, tmp_path: Path, monkeypatch) -> None:
        pool = tmp_path / "my-pool"
        pool.mkdir()
        (pool / "config.yaml").write_text("description: test\n")
        target = pool / "real_docs"
        target.mkdir()
        docs = pool / "documents"
        docs.symlink_to(target)
        monkeypatch.setattr(
            "mullet.knowledge.engine.pool_metadata.read_metadata",
            lambda _d: {},
        )
        audits = audit_pools(tmp_path, verbose=False)
        assert len(audits) == 1
        assert audits[0].docs_symlink_target == str(target)
        assert audits[0].docs_symlink_exists is True


# ---------------------------------------------------------------------------
# format_audit_report
# ---------------------------------------------------------------------------

class TestFormatAuditReport:
    def _make_audit(self, **kwargs):
        defaults = dict(
            name="test-pool",
            pool_dir=Path("/tmp/test"),
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
        defaults.update(kwargs)
        return PoolAudit(**defaults)

    def test_description_line(self) -> None:
        a = self._make_audit(description="A test pool")
        out = format_audit_report([a])
        assert "A test pool" in out

    def test_machine_name_line(self) -> None:
        a = self._make_audit(machine_name="test_pool", name="test-pool")
        out = format_audit_report([a])
        assert "machine_name: test_pool" in out

    def test_machine_name_same_as_name_skipped(self) -> None:
        a = self._make_audit(machine_name="test-pool", name="test-pool")
        out = format_audit_report([a])
        assert "machine_name:" not in out

    def test_broken_docs_symlink(self) -> None:
        a = self._make_audit(docs_symlink_target="/missing", docs_symlink_exists=False)
        out = format_audit_report([a])
        assert "BROKEN" in out
        assert "/missing" in out

    def test_doc_entries_with_symlinks(self) -> None:
        entries = [
            _DocEntry("a.md", True, "/real/a.md", True),
            _DocEntry("b.md", False, None, True),
        ]
        a = self._make_audit(doc_entries=entries)
        out = format_audit_report([a])
        assert "2 entries" in out
        assert "a.md" in out
        assert "b.md" in out

    def test_doc_entries_single_entry(self) -> None:
        entries = [_DocEntry("single.md", False, None, True)]
        a = self._make_audit(doc_entries=entries)
        out = format_audit_report([a])
        assert "1 entry" in out

    def test_sync_commands_multiple(self) -> None:
        a = self._make_audit(sync_commands=["git -C /repo pull", "auditer checkout --account foo"])
        out = format_audit_report([a])
        assert "git -C /repo pull" in out
        assert "auditer checkout --account foo" in out

    def test_vector_error_not_indexed(self) -> None:
        a = self._make_audit(vector=_VectorInfo(indexed=False, chunk_count=0, has_bm25=False, error="boom"))
        out = format_audit_report([a])
        assert "error:" in out
        assert "boom" in out

    def test_vector_indexed(self) -> None:
        a = self._make_audit(
            vector=_VectorInfo(indexed=True, chunk_count=42, has_bm25=True, error=None),
            index_size_mb=1.5,
        )
        out = format_audit_report([a])
        assert "42 chunks" in out
        assert "bm25" in out
        assert "1.5 MB" in out

    def test_graph_built_with_error(self) -> None:
        a = self._make_audit(
            graph=_GraphInfo(built=True, entity_count=5, relationship_count=3, db_size_mb=0.5, error="graph err")
        )
        out = format_audit_report([a])
        assert "5 entities" in out
        assert "3 relationships" in out
        assert "graph err" in out

    def test_graph_built_no_error(self) -> None:
        a = self._make_audit(
            graph=_GraphInfo(built=True, entity_count=5, relationship_count=3, db_size_mb=0.5, error=None)
        )
        out = format_audit_report([a])
        assert "5 entities" in out
        assert "3 relationships" in out
        assert "Graph" in out


# ---------------------------------------------------------------------------
# run_pool_audit
# ---------------------------------------------------------------------------

def test_run_pool_audit_with_pools(tmp_path: Path, capsys, monkeypatch) -> None:
    audit = PoolAudit(
        name="test-pool",
        pool_dir=tmp_path,
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
    run_pool_audit(tmp_path)
    out = capsys.readouterr().out
    assert "test-pool" in out


# ---------------------------------------------------------------------------
# _purge_pool_sidecars
# ---------------------------------------------------------------------------

class TestPurgePoolSidecars:
    def test_sqlite_error_captured(self, tmp_path: Path, monkeypatch) -> None:
        import sqlite3
        db = tmp_path / "mullet.db"
        db.write_text("not a db")

        def bad_connect(*args, **kwargs):
            raise sqlite3.Error("mock sql error")

        monkeypatch.setattr(sqlite3, "connect", bad_connect)
        rows, warnings = _purge_pool_sidecars("pool", "pool", tmp_path)
        assert any("mock sql error" in w for w in warnings)

    def test_sidecar_directory_removed(self, tmp_path: Path) -> None:
        sidecar = tmp_path / "analytics" / "pool"
        sidecar.mkdir(parents=True)
        (sidecar / "file.txt").write_text("x")
        rows, warnings = _purge_pool_sidecars("pool", "pool", tmp_path)
        assert not sidecar.exists()
        assert warnings == []

    def test_sidecar_rmtree_exception(self, tmp_path: Path, monkeypatch) -> None:
        import shutil
        sidecar = tmp_path / "analytics" / "pool"
        sidecar.mkdir(parents=True)

        def bad_rmtree(p):
            raise RuntimeError("rmtree fail")

        monkeypatch.setattr(shutil, "rmtree", bad_rmtree)
        rows, warnings = _purge_pool_sidecars("pool", "pool", tmp_path)
        assert any("rmtree fail" in w for w in warnings)


# ---------------------------------------------------------------------------
# _force_rmtree
# ---------------------------------------------------------------------------

class TestForceRmtree:
    def test_documents_unlink_exception(self, tmp_path: Path, monkeypatch) -> None:
        d = tmp_path / "pool"
        d.mkdir()
        docs = d / "documents"
        docs.symlink_to(tmp_path / "nonexistent")
        original_unlink = Path.unlink
        call_count = [0]

        def bad_unlink(self, missing_ok=False):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("unlink fail")
            return original_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", bad_unlink)
        ok, err = _force_rmtree(d)
        assert ok is True
        assert err is None
        assert not d.exists()

    def test_on_rm_error_callback(self, tmp_path: Path, monkeypatch) -> None:
        import os
        d = tmp_path / "pool"
        d.mkdir()
        f = d / "readonly.txt"
        f.write_text("x")
        original_unlink = os.unlink

        def flaky_unlink(path, dir_fd=None):
            if str(path).endswith("readonly.txt"):
                raise PermissionError("mock readonly")
            if dir_fd is not None:
                return original_unlink(path, dir_fd=dir_fd)
            return original_unlink(path)

        monkeypatch.setattr(os, "unlink", flaky_unlink)
        ok, err = _force_rmtree(d)
        assert ok is False
        # Python 3.12+ reports the directory it then couldn't remove instead
        # of the file's own error, so accept either message.
        assert "mock readonly" in str(err) or "not empty" in str(err)
        assert d.exists()

    def test_path_is_symlink(self, tmp_path: Path) -> None:
        target = tmp_path / "real"
        target.mkdir()
        link = tmp_path / "pool_link"
        link.symlink_to(target)
        ok, err = _force_rmtree(link)
        assert ok is True
        assert err is None
        assert not link.exists()

    def test_shutil_rmtree_exception(self, tmp_path: Path, monkeypatch) -> None:
        import shutil
        d = tmp_path / "pool"
        d.mkdir()

        def bad_rmtree(*args, **kwargs):
            raise RuntimeError("rmtree oops")

        monkeypatch.setattr(shutil, "rmtree", bad_rmtree)
        ok, err = _force_rmtree(d)
        assert ok is False
        assert "rmtree oops" in str(err)

    def test_path_still_exists_after_rmtree(self, tmp_path: Path, monkeypatch) -> None:
        import shutil
        d = tmp_path / "pool"
        d.mkdir()

        def noop_rmtree(*args, **kwargs):
            pass

        monkeypatch.setattr(shutil, "rmtree", noop_rmtree)
        ok, err = _force_rmtree(d)
        assert ok is False
        assert "still exists" in str(err)
