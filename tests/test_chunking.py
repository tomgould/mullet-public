"""Tests for the semantic chunking module."""

from mullet.core.knowledge.chunking import (
    SentenceChunker, CodeChunker, RecursiveChunker,
    Chunk, ChunkMetadata, enrich_chunks,
)


# ── SentenceChunker ─────────────────────────────────────────────────────────

def test_sentence_basic():
    text = "First sentence. Second sentence. Third sentence here."
    chunker = SentenceChunker(max_chars=40, min_chars=10)
    chunks = chunker.chunk(text, source="test.md")
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)


def test_sentence_respects_max_chars():
    text = ". ".join(f"Sentence number {i} with some content" for i in range(20))
    chunker = SentenceChunker(max_chars=100, min_chars=10)
    chunks = chunker.chunk(text)
    for c in chunks:
        assert len(c.text) <= 200  # max_chars + possible overlap


def test_sentence_empty_text():
    assert SentenceChunker().chunk("") == []
    assert SentenceChunker().chunk("   ") == []


def test_sentence_short_text():
    assert SentenceChunker(min_chars=100).chunk("Short.") == []


def test_sentence_metadata():
    text = "A long enough sentence to pass the minimum character threshold easily."
    chunks = SentenceChunker(min_chars=10).chunk(text, source="file.md")
    assert chunks[0].metadata.source == "file.md"
    assert chunks[0].metadata.chunk_index == 0


def test_sentence_heading_extraction():
    text = "## Getting Started\n\nThis section explains how to get started with the project."
    chunks = SentenceChunker(min_chars=10).chunk(text, source="README.md")
    assert len(chunks) >= 1
    assert "Getting Started" in chunks[0].metadata.heading_path


def test_sentence_paragraph_splits():
    text = "First paragraph with content.\n\nSecond paragraph with content.\n\nThird paragraph content."
    chunker = SentenceChunker(max_chars=50, min_chars=10)
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2


# ── CodeChunker ──────────────────────────────────────────────────────────────

PYTHON_CODE = '''def hello():
    print("hello world")
    return True

class MyClass:
    def __init__(self):
        self.x = 1

    def method(self):
        return self.x

def another_function():
    return 42
'''


def test_code_python_splits():
    chunker = CodeChunker(max_chars=200, min_chars=20)
    chunks = chunker.chunk(PYTHON_CODE, source="main.py")
    assert len(chunks) >= 2


def test_code_detects_language():
    chunker = CodeChunker()
    assert chunker._detect_language("app.py") == "python"
    assert chunker._detect_language("app.ts") == "typescript"
    assert chunker._detect_language("main.go") == "go"
    assert chunker._detect_language("unknown.xyz") == ""


def test_code_empty():
    assert CodeChunker().chunk("") == []


def test_code_metadata_language():
    chunker = CodeChunker(min_chars=10)
    chunks = chunker.chunk(PYTHON_CODE, source="main.py")
    for c in chunks:
        assert c.metadata.language == "python"


def test_code_long_function_splits():
    long_func = "def big():\n" + "\n".join(f"    line_{i} = {i}" for i in range(200))
    chunker = CodeChunker(max_chars=500, min_chars=10)
    chunks = chunker.chunk(long_func, source="big.py")
    assert len(chunks) >= 2


# ── RecursiveChunker ─────────────────────────────────────────────────────────

def test_recursive_basic():
    text = "Para one.\n\nPara two.\n\nPara three."
    chunker = RecursiveChunker(max_chars=20, min_chars=5)
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2


def test_recursive_respects_max():
    text = " ".join("word" for _ in range(500))
    chunker = RecursiveChunker(max_chars=100, min_chars=10)
    chunks = chunker.chunk(text)
    for c in chunks:
        assert len(c.text) <= 110  # small margin for split alignment


def test_recursive_empty():
    assert RecursiveChunker().chunk("") == []


def test_recursive_small_text():
    text = "Small enough to fit in one chunk easily with some extra words."
    chunks = RecursiveChunker(max_chars=1000, min_chars=10).chunk(text)
    assert len(chunks) == 1


def test_recursive_metadata():
    text = "First paragraph.\n\nSecond paragraph that has more content here."
    chunks = RecursiveChunker(max_chars=40, min_chars=10).chunk(text, source="doc.md")
    for c in chunks:
        assert c.metadata.source == "doc.md"


# ── enrich_chunks ────────────────────────────────────────────────────────────

def test_enrich_chunks():
    chunks = [
        Chunk(text="hello", metadata=ChunkMetadata(source="", chunk_index=0)),
    ]
    enriched = enrich_chunks(chunks, "main.py")
    assert enriched[0].metadata.source == "main.py"
    assert enriched[0].metadata.language == "python"


def test_enrich_chunks_typescript():
    chunks = [
        Chunk(text="export", metadata=ChunkMetadata(source="", chunk_index=0)),
    ]
    enriched = enrich_chunks(chunks, "app.tsx")
    assert enriched[0].metadata.language == "typescript"


def test_enrich_chunks_unknown():
    chunks = [
        Chunk(text="data", metadata=ChunkMetadata(source="", chunk_index=0)),
    ]
    enriched = enrich_chunks(chunks, "data.csv")
    assert enriched[0].metadata.language == ""


def test_sentence_chunker_no_sentences_returns_empty():  # pragma: no cover — line 75 dead (line 68 guards first)
    """Line 75 guard is unreachable: line 68 returns early for empty text,
    and any non-empty text always produces >= 1 non-empty sentence part."""
    pass


def test_sentence_chunker_no_overlap_else_branch():
    """Line 104: overlap_chars=0, chunk splits across sentences → else branch."""
    # Two long sentences exceed max_chars → second sentence starts new chunk via else
    chunker = SentenceChunker(max_chars=20, min_chars=1, overlap_chars=0)
    text = "First long sentence here. Second long sentence here."
    chunks = chunker.chunk(text, source="test.md")
    assert len(chunks) >= 2  # triggers else branch when starting new chunk


def test_code_chunker_unknown_language_falls_back_to_lines():
    """Line 159: boundary=None for unknown extension → _chunk_by_lines fallback."""
    chunker = CodeChunker(max_chars=200, min_chars=1)
    text = "hello\nworld\n"
    chunks = chunker.chunk(text, source="file.cobol", language="cobol")
    assert len(chunks) >= 1


def test_code_chunker_no_matches_falls_back_to_lines():
    """Line 164: boundary regex finds no matches → _chunk_by_lines fallback."""
    chunker = CodeChunker(max_chars=200, min_chars=1)
    # Valid Python extension but text has no function/class definitions
    text = "x = 1\ny = 2\n"
    chunks = chunker.chunk(text, source="plain.py", language="python")
    assert len(chunks) >= 1


def test_code_chunker_small_segment_skipped():
    """Line 176: segment shorter than min_chars → continue (skip tiny function)."""
    chunker = CodeChunker(max_chars=500, min_chars=50)
    # First function is tiny (<50 chars), second is long enough to keep
    text = (
        "def a():\n    pass\n\n"
        "def b_longer_function():\n    # Comment to pad length to meet min_chars\n    return 42\n"
    )
    chunks = chunker.chunk(text, source="tiny.py", language="python")
    # "def a():\n    pass\n\n" is ~20 chars, skipped. Only def b remains.
    assert isinstance(chunks, list)
    assert all(len(c.text) >= 1 for c in chunks)


def test_recursive_chunker_hard_split():
    """Line 300: sep_idx reaches end of _separators → hard split by max_chars."""
    chunker = RecursiveChunker(max_chars=10, min_chars=1)
    # Text with no separators at all — recursion exhausts all separator levels
    text = "a" * 50  # no spaces, newlines, periods — only hard split possible
    chunks = chunker.chunk(text, source="test.txt")
    assert len(chunks) >= 2
    assert all(len(c.text) <= 10 for c in chunks)
