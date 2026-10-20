"""Tests for mullet/core/knowledge/ingest.py."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from mullet.core.knowledge.ingest import (
    FileManifest,
    IngestResult,
    chunk_text,
    discover_files,
    ingest_directory,
    validate_chunk,
)


class TestFileManifest:
    def test_is_changed_detects_change(self):
        m = FileManifest()
        assert m.is_changed("a.py", "hello") is True
        m.record("a.py", "hello")
        assert m.is_changed("a.py", "hello") is False
        assert m.is_changed("a.py", "world") is True

    def test_save_and_load(self, tmp_path: Path):
        m = FileManifest()
        m.record("a.py", "hello")
        path = tmp_path / "manifest.json"
        m.save(path)
        assert path.exists()
        loaded = FileManifest.load(path)
        assert loaded.entries == m.entries

    def test_load_missing_returns_empty(self, tmp_path: Path):
        path = tmp_path / "missing.json"
        loaded = FileManifest.load(path)
        assert loaded.entries == {}

    def test_load_json_decode_error(self, tmp_path: Path):
        """Covers lines 72-73: JSONDecodeError branch."""
        path = tmp_path / "bad_manifest.json"
        path.write_text("not json")
        loaded = FileManifest.load(path)
        assert loaded.entries == {}

    def test_load_oserror(self, tmp_path: Path):
        """Covers lines 72-73: OSError branch."""
        path = tmp_path / "locked_manifest.json"
        path.write_text("{}")
        with patch.object(Path, "read_text", side_effect=OSError("read failed")):
            loaded = FileManifest.load(path)
        assert loaded.entries == {}


class TestDiscoverFiles:
    def test_discovers_matching_files(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.txt").write_text("y")
        files = list(discover_files(tmp_path))
        assert len(files) == 2

    def test_skips_dirs(self, tmp_path: Path):
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "a.py").write_text("x")
        files = list(discover_files(tmp_path))
        assert not any("node_modules" in str(f) for f in files)

    def test_skips_oversized_files(self, tmp_path: Path):
        big = tmp_path / "big.py"
        big.write_text("x" * 600_000)
        files = list(discover_files(tmp_path))
        assert big not in files

    def test_skips_wrong_extensions(self, tmp_path: Path):
        (tmp_path / "a.exe").write_text("x")
        files = list(discover_files(tmp_path))
        assert not any(f.name == "a.exe" for f in files)

    def test_makefile_and_dockerfile_by_name(self, tmp_path: Path):
        (tmp_path / "Makefile").write_text("all:")
        (tmp_path / "Dockerfile").write_text("FROM python")
        files = list(discover_files(tmp_path))
        assert any(f.name == "Makefile" for f in files)
        assert any(f.name == "Dockerfile" for f in files)

    def test_oserror_on_stat_logs_and_skips(self, tmp_path: Path, caplog):
        """Covers lines 119-121: OSError on stat."""
        (tmp_path / "a.py").write_text("x")
        with patch.object(Path, "stat", side_effect=OSError("stat failed")):
            files = list(discover_files(tmp_path))
        assert files == []
        assert "failed to stat file" in caplog.text


class TestChunkText:
    def test_empty_text(self):
        assert chunk_text("") == []

    def test_short_text(self):
        assert chunk_text("hi") == []

    def test_single_paragraph(self):
        text = "This is a paragraph with enough characters to pass validation. " * 2
        chunks = chunk_text(text, max_chars=2000)
        assert len(chunks) == 1

    def test_multiple_paragraphs(self):
        p1 = "Paragraph one has enough text to be valid. " * 3
        p2 = "Paragraph two also has enough text to be valid. " * 3
        text = f"{p1}\n\n{p2}"
        chunks = chunk_text(text, max_chars=150)
        assert len(chunks) >= 2

    def test_empty_paragraphs_stripped(self):
        """Covers line 139: not paragraphs, and line 147: not para."""
        text = "   \n\n   \n\n"
        assert chunk_text(text) == []

    def test_paragraph_split_and_continue(self):
        """Covers line 153 (append current) and line 162 (current = para)."""
        p1 = "A" * 100
        p2 = "B" * 100
        p3 = "C" * 100
        text = f"{p1}\n\n{p2}\n\n{p3}"
        chunks = chunk_text(text, max_chars=250, overlap=10)
        assert len(chunks) >= 2

    def test_hard_split_long_paragraph(self):
        long_para = "X" * 3000
        chunks = chunk_text(long_para, max_chars=500, overlap=50)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 500

    def test_validate_chunk_rejects_short(self):
        assert validate_chunk("ab") is False

    def test_validate_chunk_rejects_low_alpha(self):
        assert validate_chunk("!!!!!!" * 20) is False

    def test_validate_chunk_accepts_good(self):
        assert validate_chunk("This is a valid chunk with enough alphanumeric content.") is True

    def test_chunk_text_not_paragraphs(self):
        """Covers line 139: empty paragraphs list from re.split."""
        with patch("mullet.core.knowledge.ingest.re.split", return_value=[]):
            chunks = chunk_text("a" * 100)
        assert chunks == []

    def test_chunk_text_empty_para_continue(self):
        """Covers line 147: empty paragraph after strip."""
        with patch(
            "mullet.core.knowledge.ingest.re.split",
            return_value=["valid paragraph with enough chars to pass validation." * 3, "   ", "another valid paragraph with enough chars." * 3],
        ):
            chunks = chunk_text("x" * 100)
        assert len(chunks) >= 1


class TestIngestDirectory:
    def _valid_content(self) -> str:
        return "def foo():\n    pass\n" + "# comment line with more text to reach minimum chunk size.\n" * 5

    def test_ingests_files(self, tmp_path: Path):
        (tmp_path / "a.py").write_text(self._valid_content())
        result = ingest_directory(tmp_path)
        assert result.files_scanned == 1
        assert result.files_ingested == 1
        assert result.chunks_created >= 1
        assert result.duration_s >= 0

    def test_incremental_skips_unchanged(self, tmp_path: Path):
        (tmp_path / "a.py").write_text(self._valid_content())
        r1 = ingest_directory(tmp_path, incremental=True)
        r2 = ingest_directory(tmp_path, incremental=True)
        assert r2.files_unchanged == 1
        assert r2.files_ingested == 0

    def test_non_incremental_no_manifest(self, tmp_path: Path):
        (tmp_path / "a.py").write_text(self._valid_content())
        result = ingest_directory(tmp_path, incremental=False)
        assert result.files_ingested == 1
        manifest_path = tmp_path / ".mullet" / "ingest_manifest.json"
        assert not manifest_path.exists()

    def test_skips_binary_or_unreadable_file(self, tmp_path: Path):
        """Covers lines 215-218: OSError/UnicodeDecodeError on read_text."""
        bad = tmp_path / "bad.py"
        bad.write_text("def foo(): pass")
        with patch.object(Path, "read_text", side_effect=OSError("read failed")):
            result = ingest_directory(tmp_path)
        assert result.files_skipped == 1
        assert "read failed" in result.errors[0]

    def test_skips_file_with_no_chunks(self, tmp_path: Path):
        """Covers lines 230-231: no chunks produced."""
        (tmp_path / "empty.py").write_text("\n\n\n\n")
        result = ingest_directory(tmp_path)
        assert result.files_skipped == 1
        assert result.files_ingested == 0

    def test_unicode_decode_error_on_read(self, tmp_path: Path):
        """Covers lines 215-218: UnicodeDecodeError branch."""
        (tmp_path / "bad.py").write_text("x")
        exc = UnicodeDecodeError("utf-8", b"", 0, 1, "boom")
        with patch.object(Path, "read_text", side_effect=exc):
            result = ingest_directory(tmp_path)
        assert result.files_skipped == 1
        assert any("boom" in e for e in result.errors)
