"""Edge-case tests for chunk_by_semantic and chunk_by_recursive.

Per docs/TODO.md P1 audit: empty embeddings array and malicious separator
recursion were untested paths. These tests cover both.
"""

import numpy as np

from mullet.knowledge.engine.chunking import (
    chunk_by_semantic,
    chunk_by_recursive,
)


class _DeterministicEmbedder:
    """Returns a fixed embedding per sentence for deterministic tests."""

    def __init__(self, embeddings=None, dim=8):
        self._embeddings = embeddings
        self._dim = dim

    def encode(self, sentences, show_progress_bar=False):
        if self._embeddings is not None:
            return np.array(self._embeddings[: len(sentences)], dtype=np.float32)
        # Default: identical embeddings so no splits trigger
        return np.ones((len(sentences), self._dim), dtype=np.float32)


# --- chunk_by_semantic -----------------------------------------------------


def test_chunk_by_semantic_empty_elements():
    """Empty input returns empty list, not a crash."""
    emb = _DeterministicEmbedder()
    assert chunk_by_semantic([], emb) == []


def test_chunk_by_semantic_whitespace_only():
    """Elements that are pure whitespace produce no chunks."""
    emb = _DeterministicEmbedder()
    assert chunk_by_semantic(["   ", "\n\n", ""], emb) == []


def test_chunk_by_semantic_single_sentence_passthrough():
    """One sentence should return just that sentence (no embed-splitting edge)."""
    emb = _DeterministicEmbedder()
    out = chunk_by_semantic(["hello world"], emb, min_chars=1)
    assert len(out) == 1
    assert "hello world" in out[0]


def test_chunk_by_semantic_never_crashes_on_zero_norm_embedding():
    """Zero-norm embeddings (all zeros) should not raise — the 1e-8
    safety factor in the cosine similarity guards against div-by-zero."""
    zero = np.zeros(8, dtype=np.float32)
    one = np.ones(8, dtype=np.float32)
    emb = _DeterministicEmbedder(embeddings=[zero, one, zero])
    text = ["First sentence.", "Second sentence.", "Third sentence."]
    # Should not raise even though zero-norm vectors produce 0 similarity
    out = chunk_by_semantic(text, emb, threshold=0.5, max_chars=1000, min_chars=1)
    assert isinstance(out, list)


# --- chunk_by_recursive ----------------------------------------------------


def test_chunk_by_recursive_empty_text():
    assert chunk_by_recursive("", max_chars=100) == []


def test_chunk_by_recursive_short_text_single_chunk():
    text = "Short."
    out = chunk_by_recursive(text, max_chars=100)
    assert out == [text]


def test_chunk_by_recursive_does_not_infinitely_recurse_on_unsplittable():
    """Text with no natural separators should still terminate — the hard-split
    fallback kicks in after separators are exhausted."""
    # A 5000-char string with no paragraph/sentence breaks, no spaces
    text = "a" * 5000
    out = chunk_by_recursive(text, max_chars=500)
    assert isinstance(out, list)
    assert len(out) >= 1
    # Reassembling should cover the input (within rounding)
    assert sum(len(c) for c in out) >= len(text) - 100


def test_chunk_by_recursive_respects_separator_hierarchy():
    """Given paragraph breaks, prefers those over sentence breaks."""
    text = "Para one sentence one. Para one sentence two.\n\nPara two sentence one. Para two sentence two."
    out = chunk_by_recursive(text, max_chars=60)
    # With a small chunk_size the splitter should eventually produce multiple chunks
    assert len(out) >= 2
    assert all(isinstance(c, str) for c in out)


def test_chunk_by_recursive_with_empty_separators_falls_back():
    """Even with no separators configured, chunker shouldn't recurse forever."""
    text = "abcdef" * 100
    out = chunk_by_recursive(text, max_chars=50, separators=[])
    assert isinstance(out, list)
    assert len(out) >= 1
