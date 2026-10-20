"""Tests for ``mullet pool health`` diagnostic."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import yaml

from mullet.cli.pool_audit import (
    check_pool_health,
    run_pool_health,
)


def _make_pool(
    pools_dir: Path,
    slug: str,
    *,
    with_config: bool = True,
    doc_files: int = 0,
    chunk_count: int | None = None,
    manifest_files: int | None = None,
    manifest_stale: bool = False,
    broken_symlink: bool = False,
    never_indexed: bool = False,
) -> Path:
    """Build a pool directory with knobs for each health condition.

    chunk_count=None → no chroma.sqlite3 at all
    chunk_count=0 + never_indexed=True → stock-empty 168KB init (no real chunks)
    chunk_count=n (> 0) → write a file-manifest AND a fake chroma.sqlite3 > 200KB
                          so _chroma_chunk_count's "empty-init" heuristic returns
                          chroma_present=True. chunk_count itself is simulated via
                          the fake data path described below.
    manifest_files=N → write file_manifest.json listing N files
    manifest_stale=True → set the manifest mtime to 3 days in the past AFTER
                          touching doc files (so docs mtime > manifest mtime)
    broken_symlink=True → add a dangling symlink inside documents/
    """
    pool = pools_dir / slug
    pool.mkdir(parents=True)
    docs = pool / "documents"
    docs.mkdir()
    index = pool / "index"
    index.mkdir()

    if with_config:
        (pool / "config.yaml").write_text(
            yaml.safe_dump({
                "name": slug.replace("_", " ").title(),
                "machine_name": slug,
                "embedding": {"model": "BAAI/bge-base-en-v1.5"},
            })
        )

    for i in range(doc_files):
        (docs / f"doc_{i}.md").write_text(f"# doc {i}\nsome content\n")

    if broken_symlink:
        (docs / "broken.md").symlink_to(pools_dir / "nonexistent_target.md")

    if never_indexed:
        # Simulate stock empty chroma init (~168KB)
        (index / "chroma.sqlite3").write_bytes(b"\x00" * (168 * 1024))
    elif chunk_count is not None:
        # Write a fake chroma.sqlite3 that exceeds the empty-init window
        (index / "chroma.sqlite3").write_bytes(b"\x00" * (300 * 1024))

    if manifest_files is not None:
        manifest_data = {
            "files": {
                f"/fake/path/{i}.md": {"mtime": 1.0, "size": 100}
                for i in range(manifest_files)
            },
            "indexed_at": "2026-04-23T00:00:00",
            "client": slug,
        }
        manifest_path = index / "file_manifest.json"
        manifest_path.write_text(json.dumps(manifest_data))
        if manifest_stale:
            # Manifest 3d old; docs mtime = now
            old = time.time() - 3 * 86400
            os.utime(manifest_path, (old, old))
            for f in docs.rglob("*"):
                if f.is_file():
                    now = time.time()
                    os.utime(f, (now, now))

    return pool


class _FakeCollection:
    def __init__(self, count_value: int):
        self._count = count_value

    def count(self) -> int:
        return self._count


class _FakeChromaClient:
    def __init__(self, collections):
        self._collections = collections

    def list_collections(self):
        return self._collections


def _patch_chroma(monkeypatch, count_value: int):
    """Make _chroma_chunk_count return the requested count for fake-chroma pools."""
    import chromadb

    def _fake_client(path):
        return _FakeChromaClient([_FakeCollection(count_value)])

    monkeypatch.setattr(chromadb, "PersistentClient", _fake_client)


# ── Happy path ────────────────────────────────────────────────────────────


def test_clean_pool_is_ok(tmp_path, monkeypatch):
    _make_pool(
        tmp_path,
        "healthy_pool",
        doc_files=10,
        chunk_count=20,
        manifest_files=10,
    )
    _patch_chroma(monkeypatch, 20)
    report = check_pool_health(tmp_path / "healthy_pool")
    assert report.ok, f"expected OK, got issues: {report.issues}"
    assert report.chunk_count == 20
    assert report.file_count == 10
    assert report.last_indexed_human is not None


# ── (a) empty-but-configured ─────────────────────────────────────────────


def test_empty_but_configured(tmp_path, monkeypatch):
    # config + docs with files, chroma present but zero chunks
    _make_pool(
        tmp_path,
        "empty_pool",
        doc_files=5,
        chunk_count=0,  # triggers the "present but empty" path
    )
    _patch_chroma(monkeypatch, 0)
    report = check_pool_health(tmp_path / "empty_pool")
    codes = [i.code for i in report.issues]
    assert "empty_but_configured" in codes
    assert not report.ok


# ── (b) stale manifest ────────────────────────────────────────────────────


def test_stale_manifest_detected(tmp_path, monkeypatch):
    _make_pool(
        tmp_path,
        "stale_pool",
        doc_files=3,
        chunk_count=10,
        manifest_files=3,
        manifest_stale=True,
    )
    _patch_chroma(monkeypatch, 10)
    report = check_pool_health(tmp_path / "stale_pool")
    codes = [i.code for i in report.issues]
    assert "stale_manifest" in codes
    assert not report.ok


# ── (c) manifest/chroma mismatch ─────────────────────────────────────────


def test_manifest_chroma_mismatch(tmp_path, monkeypatch):
    # manifest has 100 files but chroma has only 10 chunks → 10 < 50
    _make_pool(
        tmp_path,
        "mismatch_pool",
        doc_files=100,
        chunk_count=10,
        manifest_files=100,
    )
    _patch_chroma(monkeypatch, 10)
    report = check_pool_health(tmp_path / "mismatch_pool")
    codes = [i.code for i in report.issues]
    assert "manifest_chroma_mismatch" in codes
    assert not report.ok


# ── (d) broken symlinks ───────────────────────────────────────────────────


def test_broken_symlink_detected(tmp_path, monkeypatch):
    _make_pool(
        tmp_path,
        "broken_pool",
        doc_files=2,
        chunk_count=10,
        manifest_files=2,
        broken_symlink=True,
    )
    _patch_chroma(monkeypatch, 10)
    report = check_pool_health(tmp_path / "broken_pool")
    codes = [i.code for i in report.issues]
    assert "broken_symlink" in codes
    assert any("broken.md" in i.message for i in report.issues)
    assert not report.ok


# ── (e) never-indexed ─────────────────────────────────────────────────────


def test_never_indexed_detected(tmp_path):
    # Config present; no chroma.sqlite3 file at all
    _make_pool(
        tmp_path,
        "fresh_pool",
        doc_files=5,
        chunk_count=None,
    )
    report = check_pool_health(tmp_path / "fresh_pool")
    codes = [i.code for i in report.issues]
    assert "never_indexed" in codes
    assert not report.ok


def test_never_indexed_detected_when_stock_empty(tmp_path):
    # Config present; chroma.sqlite3 exists but is the ~168KB empty init
    _make_pool(
        tmp_path,
        "stock_empty_pool",
        doc_files=5,
        chunk_count=0,
        never_indexed=True,
    )
    report = check_pool_health(tmp_path / "stock_empty_pool")
    codes = [i.code for i in report.issues]
    assert "never_indexed" in codes


# ── CLI runner + exit code ────────────────────────────────────────────────


def test_run_pool_health_exits_zero_when_all_clean(tmp_path, monkeypatch, capsys):
    _make_pool(
        tmp_path,
        "clean_pool",
        doc_files=2,
        chunk_count=10,
        manifest_files=2,
    )
    _patch_chroma(monkeypatch, 10)
    rc = run_pool_health(tmp_path)
    out = capsys.readouterr().out
    assert rc == 0
    assert "clean_pool" in out
    assert "OK" in out


def test_run_pool_health_exits_one_on_issue(tmp_path, monkeypatch, capsys):
    _make_pool(tmp_path, "fresh_pool", doc_files=5, chunk_count=None)
    rc = run_pool_health(tmp_path)
    out = capsys.readouterr().out
    assert rc == 1
    assert "fresh_pool" in out
    assert "never-indexed" in out or "never_indexed" in out


def test_run_pool_health_single_pool_filter(tmp_path, monkeypatch, capsys):
    _make_pool(tmp_path, "a_pool", doc_files=2, chunk_count=10, manifest_files=2)
    _make_pool(tmp_path, "b_pool", doc_files=5, chunk_count=None)
    _patch_chroma(monkeypatch, 10)
    rc = run_pool_health(tmp_path, pool_name="a_pool")
    out = capsys.readouterr().out
    assert rc == 0
    assert "a_pool" in out
    assert "b_pool" not in out


# ── Config validation ─────────────────────────────────────────────────────


def _make_pool_custom_config(pools_dir: Path, slug: str, config: dict) -> Path:
    pool = pools_dir / slug
    pool.mkdir(parents=True)
    (pool / "documents").mkdir()
    (pool / "index").mkdir()
    (pool / "config.yaml").write_text(yaml.safe_dump(config))
    return pool


def test_config_missing_machine_name(tmp_path):
    _make_pool_custom_config(tmp_path, "no_mn", {"name": "No MN"})
    report = check_pool_health(tmp_path / "no_mn")
    codes = [i.code for i in report.issues]
    assert "invalid_config" in codes
    assert not report.ok


def test_config_missing_embedding_model(tmp_path):
    _make_pool_custom_config(
        tmp_path,
        "no_embed",
        {"name": "No Embed", "machine_name": "no_embed", "embedding": {}},
    )
    report = check_pool_health(tmp_path / "no_embed")
    codes = [i.code for i in report.issues]
    assert "invalid_config" in codes
    assert not report.ok


def test_config_valid_fields_no_flag(tmp_path, monkeypatch):
    _make_pool_custom_config(
        tmp_path,
        "good_cfg",
        {
            "name": "Good",
            "machine_name": "good_cfg",
            "embedding": {"model": "BAAI/bge-base-en-v1.5"},
        },
    )
    pool_dir = tmp_path / "good_cfg"
    (pool_dir / "index" / "chroma.sqlite3").write_bytes(b"\x00" * (300 * 1024))
    _patch_chroma(monkeypatch, 5)
    report = check_pool_health(pool_dir)
    codes = [i.code for i in report.issues]
    assert "invalid_config" not in codes


# ── Exclude pattern validation ────────────────────────────────────────────


def _make_pool_with_excludes(
    pools_dir: Path,
    slug: str,
    doc_files: int,
    exclude_patterns: list[str],
) -> Path:
    pool = pools_dir / slug
    pool.mkdir(parents=True)
    docs = pool / "documents"
    docs.mkdir()
    (pool / "index").mkdir()
    for i in range(doc_files):
        (docs / f"doc_{i}.md").write_text(f"# doc {i}")
    config = {
        "name": slug,
        "machine_name": slug,
        "embedding": {"model": "BAAI/bge-base-en-v1.5"},
        "documents": {"exclude_patterns": exclude_patterns},
    }
    (pool / "config.yaml").write_text(yaml.safe_dump(config))
    return pool


def test_exclude_pattern_excludes_all_files_flagged(tmp_path):
    _make_pool_with_excludes(
        tmp_path, "over_exclude", doc_files=5, exclude_patterns=["**/*.md"]
    )
    report = check_pool_health(tmp_path / "over_exclude")
    codes = [i.code for i in report.issues]
    assert "bad_exclude_pattern" in codes
    assert not report.ok


def test_exclude_pattern_partial_exclusion_ok(tmp_path, monkeypatch):
    pool_dir = _make_pool_with_excludes(
        tmp_path,
        "partial_exclude",
        doc_files=10,
        exclude_patterns=["**/doc_9.md"],
    )
    (pool_dir / "index" / "chroma.sqlite3").write_bytes(b"\x00" * (300 * 1024))
    _patch_chroma(monkeypatch, 9)
    report = check_pool_health(pool_dir)
    codes = [i.code for i in report.issues]
    assert "bad_exclude_pattern" not in codes


# ── Summary line ──────────────────────────────────────────────────────────


def test_run_pool_health_prints_summary(tmp_path, monkeypatch, capsys):
    _make_pool(tmp_path, "p1", doc_files=2, chunk_count=10, manifest_files=2)
    _make_pool(tmp_path, "p2", doc_files=3, chunk_count=None)
    _patch_chroma(monkeypatch, 10)
    run_pool_health(tmp_path)
    out = capsys.readouterr().out
    assert "2 pools" in out or "pools checked" in out
