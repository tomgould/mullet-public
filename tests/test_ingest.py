"""Tests for the document ingestion pipeline."""


from mullet.core.knowledge.ingest import (
    discover_files,
    chunk_text,
    validate_chunk,
    ingest_directory,
    FileManifest,
)


# --- FileManifest ---

def test_manifest_empty():
    m = FileManifest()
    assert m.is_changed("file.py", "content") is True


def test_manifest_record_and_check():
    m = FileManifest()
    m.record("file.py", "hello world")
    assert m.is_changed("file.py", "hello world") is False
    assert m.is_changed("file.py", "changed content") is True


def test_manifest_save_and_load(tmp_path):
    m = FileManifest()
    m.record("a.py", "content a")
    m.record("b.py", "content b")
    path = tmp_path / "manifest.json"
    m.save(path)

    loaded = FileManifest.load(path)
    assert loaded.is_changed("a.py", "content a") is False
    assert loaded.is_changed("b.py", "content b") is False
    assert loaded.is_changed("c.py", "new file") is True


def test_manifest_load_missing(tmp_path):
    m = FileManifest.load(tmp_path / "nonexistent.json")
    assert m.entries == {}


# --- discover_files ---

def test_discover_files(tmp_path):
    (tmp_path / "a.py").write_text("print('hello')")
    (tmp_path / "b.js").write_text("console.log('hi')")
    (tmp_path / "c.bin").write_bytes(b"\x00\x01\x02")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "d.js").write_text("skip me")

    files = list(discover_files(tmp_path))
    names = {f.name for f in files}
    assert "a.py" in names
    assert "b.js" in names
    assert "c.bin" not in names  # Not in extensions
    assert "d.js" not in names  # In skip dir


def test_discover_files_max_size(tmp_path):
    (tmp_path / "big.py").write_text("x" * 600_000)
    (tmp_path / "small.py").write_text("y" * 100)

    files = list(discover_files(tmp_path))
    names = {f.name for f in files}
    assert "small.py" in names
    assert "big.py" not in names


# --- chunk_text ---

def test_chunk_text_simple():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = chunk_text(text, max_chars=200)
    assert len(chunks) >= 1


def test_chunk_text_long():
    text = "Word " * 500  # 2500 chars
    chunks = chunk_text(text, max_chars=500)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= 500 + 100  # Allow overlap


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_too_short():
    assert chunk_text("hi") == []  # Below MIN_CHUNK_CHARS


# --- validate_chunk ---

def test_validate_good_chunk():
    assert validate_chunk("This is a valid chunk with enough alphanumeric content to pass validation checks.")


def test_validate_too_short():
    assert validate_chunk("hi") is False


def test_validate_low_alpha():
    assert validate_chunk("...---...---...---...---...---...---...---...---...---...---") is False


# --- ingest_directory ---

def test_ingest_directory(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        "def hello():\n    return 'world'\n\n"
        "def goodbye():\n    return 'farewell'\n\n"
        "# This is a long enough chunk to pass validation\n"
        "# with sufficient alphanumeric content for the test\n"
    )
    (tmp_path / "src" / "utils.py").write_text(
        "import os\nimport sys\n\n"
        "def helper():\n    pass\n\n"
        "# Another file with enough content to be valid\n"
        "# and pass the minimum character requirements\n"
    )

    result = ingest_directory(tmp_path, incremental=False)
    assert result.files_scanned >= 2
    assert result.files_ingested >= 1
    assert result.chunks_created >= 1


def test_ingest_incremental(tmp_path):
    (tmp_path / "test.py").write_text(
        "# First version of the file\n"
        "def main():\n    print('hello world')\n\n"
        "# Enough content to pass chunk validation\n"
    )

    # First run ingests
    r1 = ingest_directory(tmp_path, incremental=True)
    assert r1.files_ingested >= 1

    # Second run skips unchanged
    r2 = ingest_directory(tmp_path, incremental=True)
    assert r2.files_unchanged >= 1
    assert r2.files_ingested == 0


def test_ingest_empty_directory(tmp_path):
    result = ingest_directory(tmp_path)
    assert result.files_scanned == 0
    assert result.files_ingested == 0
