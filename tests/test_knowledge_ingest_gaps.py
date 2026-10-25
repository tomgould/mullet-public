"""Gap coverage for mullet/core/knowledge/ingest.py.

Targets lines reported as missing: 72-73, 119-121, 139, 147, 153, 162,
215-218, 230-231.
"""

from pathlib import Path
from unittest.mock import patch


from mullet.core.knowledge.ingest import (
    FileManifest,
    chunk_text,
    discover_files,
    ingest_directory,
)


class TestFileManifestGaps:
    def test_load_json_decode_error_returns_empty(self, tmp_path: Path):
        """Covers lines 72-73: JSONDecodeError branch."""
        path = tmp_path / "bad.json"
        path.write_text("not json")
        manifest = FileManifest.load(path)
        assert manifest.entries == {}

    def test_load_oserror_returns_empty(self, tmp_path: Path):
        """Covers lines 72-73: OSError branch."""
        path = tmp_path / "locked.json"
        path.write_text("{}")
        with patch.object(Path, "read_text", side_effect=OSError("read failed")):
            manifest = FileManifest.load(path)
        assert manifest.entries == {}


class TestDiscoverFilesGaps:
    def test_oserror_on_stat_logs_warning_and_skips(self, tmp_path: Path, caplog):
        """Covers lines 119-121: OSError on stat."""
        (tmp_path / "a.py").write_text("x")
        with patch.object(Path, "stat", side_effect=OSError("stat failed")):
            files = list(discover_files(tmp_path))
        assert files == []
        assert "failed to stat file" in caplog.text


class TestChunkTextGaps:
    def test_empty_paragraphs_list_returns_empty(self):
        """Covers line 139: not paragraphs after split."""
        with patch("mullet.core.knowledge.ingest.re.split", return_value=[]):
            chunks = chunk_text("a" * 100)
        assert chunks == []

    def test_empty_paragraph_after_strip_is_skipped(self):
        """Covers line 147: empty paragraph after strip."""
        with patch(
            "mullet.core.knowledge.ingest.re.split",
            return_value=[
                "valid paragraph with enough chars to pass validation." * 3,
                "   ",
                "another valid paragraph with enough chars." * 3,
            ],
        ):
            chunks = chunk_text("x" * 100)
        assert len(chunks) >= 1

    def test_appends_current_and_resets(self):
        """Covers line 153 (append current) and line 162 (current = para)."""
        p1 = "A" * 50
        p2 = "B" * 60
        text = f"{p1}\n\n{p2}"
        chunks = chunk_text(text, max_chars=100, overlap=10)
        assert len(chunks) >= 1
        assert any("A" * 40 in c for c in chunks)


class TestIngestDirectoryGaps:
    def _valid_content(self) -> str:
        return (
            "def foo():\n    pass\n"
            + "# comment line with more text to reach minimum chunk size.\n" * 5
        )

    def test_oserror_on_read_text_skips_file(self, tmp_path: Path):
        """Covers lines 215-218: OSError on read_text."""
        (tmp_path / "a.py").write_text(self._valid_content())
        with patch.object(Path, "read_text", side_effect=OSError("read failed")):
            result = ingest_directory(tmp_path)
        assert result.files_skipped == 1
        assert any("read failed" in e for e in result.errors)

    def test_unicode_decode_error_on_read_text_skips_file(self, tmp_path: Path):
        """Covers lines 215-218: UnicodeDecodeError branch."""
        (tmp_path / "a.py").write_text("x")
        exc = UnicodeDecodeError("utf-8", b"", 0, 1, "boom")
        with patch.object(Path, "read_text", side_effect=exc):
            result = ingest_directory(tmp_path)
        assert result.files_skipped == 1
        assert any("boom" in e for e in result.errors)

    def test_no_chunks_produced_skips_file(self, tmp_path: Path):
        """Covers lines 230-231: no chunks produced."""
        (tmp_path / "empty.py").write_text("\n\n\n\n")
        result = ingest_directory(tmp_path)
        assert result.files_skipped == 1
        assert result.files_ingested == 0
