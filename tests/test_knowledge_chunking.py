"""Tests for the knowledge engine chunking module.

Tests recursive chunking (no external deps) and chunking config.
Semantic chunking requires embedder — tested separately.
"""

import numpy as np
from unittest.mock import MagicMock, patch

from mullet.knowledge.engine.chunking import (
    ChunkingConfig,
    _split_to_sentences,
    chunk_by_recursive,
    chunk_by_semantic,
    _recursive_split,
    _apply_overlap,
)


# --- ChunkingConfig ---

def test_chunking_config_defaults():
    cfg = ChunkingConfig()
    assert cfg.strategy == "title"
    assert cfg.max_chunk_size == 1500
    assert cfg.overlap == 100
    assert cfg.semantic_threshold == 0.5
    assert cfg.min_chunk_size == 100
    assert len(cfg.separators) == 4


def test_chunking_config_custom():
    cfg = ChunkingConfig(strategy="recursive", max_chunk_size=500, overlap=50)
    assert cfg.strategy == "recursive"
    assert cfg.max_chunk_size == 500
    assert cfg.overlap == 50


# --- _split_to_sentences ---

def test_split_sentences_basic():
    text = "Hello world. How are you? I'm fine!"
    sentences = _split_to_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "Hello world."
    assert sentences[1] == "How are you?"
    assert sentences[2] == "I'm fine!"


def test_split_sentences_empty():
    assert _split_to_sentences("") == []
    assert _split_to_sentences("   ") == []


def test_split_sentences_single():
    sentences = _split_to_sentences("Just one sentence")
    assert len(sentences) == 1
    assert sentences[0] == "Just one sentence"


# --- chunk_by_recursive ---

def test_recursive_empty():
    assert chunk_by_recursive("") == []
    assert chunk_by_recursive("  ") == []


def test_recursive_short_text():
    text = "Short text"
    chunks = chunk_by_recursive(text, max_chars=100)
    assert chunks == [text]


def test_recursive_paragraph_split():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = chunk_by_recursive(text, max_chars=30, overlap=0)
    assert len(chunks) >= 2
    assert "First paragraph." in chunks[0]


def test_recursive_line_split():
    text = "Line one\nLine two\nLine three\nLine four"
    chunks = chunk_by_recursive(text, max_chars=20, overlap=0)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= 20


def test_recursive_hard_split():
    """Text with no separators gets hard-split at max_chars."""
    text = "a" * 100
    chunks = chunk_by_recursive(text, max_chars=30, overlap=0)
    assert len(chunks) == 4  # 30 + 30 + 30 + 10
    assert all(len(c) <= 30 for c in chunks)


def test_recursive_with_overlap():
    text = "Part one.\n\nPart two.\n\nPart three."
    chunks = chunk_by_recursive(text, max_chars=20, overlap=5)
    # Overlapped chunks should contain text from previous chunk
    assert len(chunks) >= 2
    if len(chunks) > 1:
        # Second chunk should have overlap from first
        assert len(chunks[1]) >= 5


def test_recursive_respects_max_chars():
    text = "Word " * 200  # 1000 chars
    chunks = chunk_by_recursive(text, max_chars=100, overlap=0)
    for chunk in chunks:
        assert len(chunk) <= 100


def test_recursive_sentence_split():
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    chunks = chunk_by_recursive(text, max_chars=40, overlap=0)
    assert len(chunks) >= 2


# --- _recursive_split ---

def test_recursive_split_no_separators():
    """Exhausted separators → hard split."""
    chunks = _recursive_split("abcdefghij", max_chars=3, separators=[], sep_index=0)
    assert chunks == ["abc", "def", "ghi", "j"]


def test_recursive_split_separator_doesnt_help():
    """Separator not found → tries next."""
    chunks = _recursive_split("hello world", max_chars=5, separators=["\n\n", " "], sep_index=0)
    assert len(chunks) >= 2


# --- _apply_overlap ---

def test_apply_overlap_basic():
    chunks = ["Hello world", "Second chunk", "Third chunk"]
    result = _apply_overlap(chunks, overlap=5)
    assert result[0] == "Hello world"
    assert result[1].startswith("world")  # Last 5 chars of prev
    assert "Second chunk" in result[1]


def test_apply_overlap_zero():
    chunks = ["A", "B", "C"]
    result = _apply_overlap(chunks, overlap=0)
    assert result == chunks


def test_apply_overlap_single_chunk():
    chunks = ["Only one"]
    result = _apply_overlap(chunks, overlap=5)
    assert result == chunks


def test_apply_overlap_max_chars():
    chunks = ["First chunk here", "Second chunk"]
    result = _apply_overlap(chunks, overlap=5, max_chars=20)
    # First chunk is untouched, overlapped chunks are trimmed
    assert result[0] == "First chunk here"
    assert len(result[1]) <= 20


# --- chunk_by_semantic ---

def test_chunk_by_semantic_empty_elements():
    embedder = MagicMock()
    assert chunk_by_semantic([], embedder) == []
    embedder.encode.assert_not_called()


def test_chunk_by_semantic_empty_string_elements():
    embedder = MagicMock()
    assert chunk_by_semantic(["", "   "], embedder) == []
    embedder.encode.assert_not_called()


def test_chunk_by_semantic_single_sentence():
    embedder = MagicMock()
    result = chunk_by_semantic(["Only one sentence."], embedder)
    embedder.encode.assert_not_called()
    assert result == ["Only one sentence."]


def test_chunk_by_semantic_below_threshold_splits():
    embedder = MagicMock()
    embedder.encode.return_value = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    ]
    result = chunk_by_semantic(
        ["First sentence.", "Second sentence."],
        embedder,
        max_chars=16,
        min_chars=1,
    )
    assert len(result) == 2
    assert "First sentence." in result[0]
    assert "Second sentence." in result[1]


def test_chunk_by_semantic_above_threshold_merges():
    embedder = MagicMock()
    embedder.encode.return_value = [
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
    ]
    result = chunk_by_semantic(["First sentence.", "Second sentence."], embedder)
    assert len(result) == 1
    assert "First sentence." in result[0]
    assert "Second sentence." in result[0]


def test_chunk_by_semantic_holds_inference_lock():
    """Regression: chunk_by_semantic must serialize .encode() through
    ModelCache.inference_lock() — this call site runs once per document during
    ingestion (far more often than create_embeddings' once-per-batch call),
    so it was the main driver of the concurrent-encode hang. See
    model_cache.py's inference_lock() docstring for why."""
    embedder = MagicMock()
    embedder.encode.return_value = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]
    mock_lock = MagicMock()
    mock_cache = MagicMock()
    mock_cache.inference_lock.return_value = mock_lock

    with patch(
        "mullet.knowledge.engine.model_cache.get_model_cache",
        return_value=mock_cache,
    ):
        chunk_by_semantic(["First sentence.", "Second sentence."], embedder)

    mock_cache.inference_lock.assert_called_once()
    mock_lock.__enter__.assert_called_once()
    mock_lock.__exit__.assert_called_once()


def test_chunk_by_semantic_respects_max_chars():
    embedder = MagicMock()
    embedder.encode.return_value = [
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
    ]
    result = chunk_by_semantic(
        ["Sentence one here. Sentence two here."],
        embedder,
        max_chars=15,
        min_chars=1,
    )
    assert all(len(c) <= 15 for c in result)


def test_chunk_by_semantic_min_chars_merges_short_final():
    embedder = MagicMock()
    embedder.encode.return_value = [
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
    ]
    result = chunk_by_semantic(
        ["This is a long sentence one. Short."],
        embedder,
        max_chars=20,
        min_chars=20,
    )
    assert len(result) == 1
    assert "This is a long sente" in result[0]
    assert "Short." in result[0]


def test_chunk_by_semantic_flushes_current_before_hard_split():
    """When a single group exceeds max_chars and current is non-empty,
    current should be flushed before hard-splitting the group."""
    embedder = MagicMock()
    embedder.encode.return_value = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    ]
    result = chunk_by_semantic(
        ["Short one.", "This is a very long sentence that exceeds max chars."],
        embedder,
        max_chars=20,
        min_chars=1,
    )
    assert "Short one." in result[0]
    assert all(len(c) <= 20 for c in result)


# --- _recursive_split uncovered lines ---

def test_recursive_split_empty_stripped_text():
    assert _recursive_split("   ", 100, ["\n\n", "\n", ". ", " "], 0) == []


def test_recursive_split_short_text():
    assert _recursive_split("hi", 100, ["\n\n", "\n", ". ", " "], 0) == ["hi"]


def test_recursive_split_skips_empty_piece():
    chunks = _recursive_split("a\n\n\n\nb", 5, ["\n\n", "\n", ". ", " "], 0)
    assert chunks == ["a\n\nb"]


def test_recursive_split_large_piece_recurses():
    chunks = _recursive_split("hello world", 4, [" "], 0)
    assert chunks == ["hell", "o", "worl", "d"]


# --- _apply_overlap uncovered line ---

def test_apply_overlap_trims_to_max_chars():
    result = _apply_overlap(["abcde", "fghij"], overlap=3, max_chars=6)
    assert result[0] == "abcde"
    assert result[1] == "cde fg"
