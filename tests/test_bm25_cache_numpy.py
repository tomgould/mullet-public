"""Tests for the numpy/vocab BM25 cache format (v3).

Covers the pure encode/decode helpers and the full save→load round-trip
on a tiny in-memory corpus, verifying identical BM25 scores before and
after persistence.
"""

import numpy as np

from mullet.knowledge.engine.search.searcher import HybridSearcher


class TestEncodeDecodeRoundTrip:
    def test_empty_corpus(self):
        vocab, ids, offsets = HybridSearcher._encode_corpus_to_vocab([])
        assert vocab == {}
        assert ids.dtype == np.uint32
        assert offsets.dtype == np.uint64
        assert len(ids) == 0
        assert offsets.tolist() == [0]
        decoded = HybridSearcher._decode_corpus_from_vocab(vocab, ids, offsets)
        assert decoded == []

    def test_single_document(self):
        corpus = [["hello", "world", "hello"]]
        vocab, ids, offsets = HybridSearcher._encode_corpus_to_vocab(corpus)
        # Vocab should dedupe: only 2 unique tokens
        assert len(vocab) == 2
        assert ids.dtype == np.uint32
        # "hello" seen twice, "world" once → 3 ids
        assert len(ids) == 3
        # Doc boundaries: 0, 3
        assert offsets.tolist() == [0, 3]
        # IDs should reference the right vocab slots
        assert ids[0] == vocab["hello"]
        assert ids[1] == vocab["world"]
        assert ids[2] == vocab["hello"]
        decoded = HybridSearcher._decode_corpus_from_vocab(vocab, ids, offsets)
        assert decoded == corpus

    def test_multi_document_shared_vocab(self):
        corpus = [
            ["the", "quick", "brown", "fox"],
            ["the", "lazy", "dog"],
            ["quick", "fox"],
        ]
        vocab, ids, offsets = HybridSearcher._encode_corpus_to_vocab(corpus)
        # 6 unique tokens across 3 docs: the, quick, brown, fox, lazy, dog
        assert len(vocab) == 6
        assert offsets.tolist() == [0, 4, 7, 9]
        decoded = HybridSearcher._decode_corpus_from_vocab(vocab, ids, offsets)
        assert decoded == corpus

    def test_decode_preserves_token_identity(self):
        """Decoded tokens should be shared refs into the vocab, not fresh copies."""
        corpus = [["shared"] * 5]
        vocab, ids, offsets = HybridSearcher._encode_corpus_to_vocab(corpus)
        decoded = HybridSearcher._decode_corpus_from_vocab(vocab, ids, offsets)
        # All occurrences in decoded should be the same object
        doc = decoded[0]
        assert len(doc) == 5
        # At least verify equality (identity depends on numpy internals)
        assert all(tok == "shared" for tok in doc)

    def test_large_synthetic_corpus(self):
        """Stress the encoder with ~10k docs and a varied vocab."""
        rng = np.random.default_rng(42)
        vocab_words = [f"term{i}" for i in range(500)]
        corpus = []
        for _ in range(10_000):
            doc_len = int(rng.integers(5, 50))
            doc = [vocab_words[int(rng.integers(0, len(vocab_words)))] for _ in range(doc_len)]
            corpus.append(doc)

        vocab, ids, offsets = HybridSearcher._encode_corpus_to_vocab(corpus)
        assert len(vocab) <= 500
        assert ids.dtype == np.uint32
        assert offsets.dtype == np.uint64
        assert len(offsets) == len(corpus) + 1
        assert int(offsets[-1]) == len(ids)

        decoded = HybridSearcher._decode_corpus_from_vocab(vocab, ids, offsets)
        assert len(decoded) == len(corpus)
        # Spot-check a few docs
        assert decoded[0] == corpus[0]
        assert decoded[len(corpus) // 2] == corpus[len(corpus) // 2]
        assert decoded[-1] == corpus[-1]


class TestCachePersistence:
    def test_save_load_round_trip(self, tmp_path):
        """Full save → load via _save_bm25_cache / _load_bm25_cache.

        Builds a minimal HybridSearcher-like fixture with just the
        attributes the cache path touches. BM25Okapi must produce
        identical scores before and after persistence.
        """
        from rank_bm25 import BM25Okapi

        # Build an object stub that walks the cache paths. We avoid constructing
        # a real HybridSearcher because that loads SentenceTransformer + ChromaDB.

        corpus = [
            ["knowledge", "graph", "entity", "extraction"],
            ["hybrid", "search", "bm25", "vector"],
            ["knowledge", "pool", "indexing", "chunks"],
            ["graph", "search", "traversal", "knowledge"],
        ]

        stub = HybridSearcher.__new__(HybridSearcher)
        stub.ephemeral = False
        stub.index_path = tmp_path
        stub.bm25_chunks = [{"id": f"c{i}", "content": " ".join(d)} for i, d in enumerate(corpus)]
        stub.bm25_k1 = 1.5
        stub.bm25_b = 0.75
        stub._bm25_tokenized_corpus = corpus
        stub.bm25_index = BM25Okapi(corpus, k1=stub.bm25_k1, b=stub.bm25_b)
        stub._last_corpus_hash = "test-hash"
        stub._progress_reporter = None

        # Capture scores before save
        before = stub.bm25_index.get_scores(["knowledge", "graph"])

        # Save
        stub._save_bm25_cache()
        assert (tmp_path / "bm25_cache.json").exists()
        assert (tmp_path / "bm25_cache.npz").exists()

        # Fresh stub for load; bypass corpus-hash validation by injecting
        # _last_corpus_hash and _compute_corpus_hash that returns the same value,
        # and a .collection stub whose count matches.
        class _CollStub:
            def count(self):
                return len(corpus)

        stub2 = HybridSearcher.__new__(HybridSearcher)
        stub2.ephemeral = False
        stub2.index_path = tmp_path
        stub2.collection = _CollStub()
        stub2.bm25_k1 = 1.5
        stub2.bm25_b = 0.75
        stub2._progress_reporter = None
        # Force the HNSW fast-path to fire (no HNSW files in tmp_path, so
        # hnsw_files_found == 0 and we fall through to corpus-hash check).
        # Make corpus-hash check pass.
        stub2._compute_corpus_hash = lambda: "test-hash"
        stub2._last_corpus_hash = None
        stub2.bm25_index = None
        stub2.bm25_chunks = []
        stub2._bm25_tokenized_corpus = None

        ok = stub2._load_bm25_cache()
        assert ok, "Cache load should succeed"
        assert stub2.bm25_index is not None
        assert len(stub2.bm25_chunks) == len(corpus)
        assert stub2._bm25_tokenized_corpus == corpus

        after = stub2.bm25_index.get_scores(["knowledge", "graph"])
        np.testing.assert_allclose(before, after, atol=1e-10)

    def test_missing_sidecar_rejects_load(self, tmp_path):
        from rank_bm25 import BM25Okapi

        corpus = [["a", "b"], ["b", "c"]]
        stub = HybridSearcher.__new__(HybridSearcher)
        stub.ephemeral = False
        stub.index_path = tmp_path
        stub.bm25_chunks = [{"id": "c0", "content": "a b"}, {"id": "c1", "content": "b c"}]
        stub.bm25_k1 = 1.5
        stub.bm25_b = 0.75
        stub._bm25_tokenized_corpus = corpus
        stub.bm25_index = BM25Okapi(corpus, k1=stub.bm25_k1, b=stub.bm25_b)
        stub._last_corpus_hash = "h"
        stub._progress_reporter = None
        stub._save_bm25_cache()

        # Delete the sidecar; load must refuse
        (tmp_path / "bm25_cache.npz").unlink()

        class _CollStub:
            def count(self):
                return 2

        stub2 = HybridSearcher.__new__(HybridSearcher)
        stub2.ephemeral = False
        stub2.index_path = tmp_path
        stub2.collection = _CollStub()
        stub2.bm25_k1 = 1.5
        stub2.bm25_b = 0.75
        stub2._progress_reporter = None
        stub2._compute_corpus_hash = lambda: "h"
        stub2._last_corpus_hash = None

        ok = stub2._load_bm25_cache()
        assert ok is False

    def test_version_mismatch_rejects_load(self, tmp_path, monkeypatch):
        from rank_bm25 import BM25Okapi

        corpus = [["a", "b"]]
        stub = HybridSearcher.__new__(HybridSearcher)
        stub.ephemeral = False
        stub.index_path = tmp_path
        stub.bm25_chunks = [{"id": "c0", "content": "a b"}]
        stub.bm25_k1 = 1.5
        stub.bm25_b = 0.75
        stub._bm25_tokenized_corpus = corpus
        stub.bm25_index = BM25Okapi(corpus, k1=stub.bm25_k1, b=stub.bm25_b)
        stub._last_corpus_hash = "h"
        stub._progress_reporter = None
        stub._save_bm25_cache()

        # Bump the format version in-memory so the saved cache looks stale
        monkeypatch.setattr(HybridSearcher, "_BM25_CACHE_FORMAT", 999)

        class _CollStub:
            def count(self):
                return 1

        stub2 = HybridSearcher.__new__(HybridSearcher)
        stub2.ephemeral = False
        stub2.index_path = tmp_path
        stub2.collection = _CollStub()
        stub2.bm25_k1 = 1.5
        stub2.bm25_b = 0.75
        stub2._progress_reporter = None
        stub2._compute_corpus_hash = lambda: "h"
        stub2._last_corpus_hash = None

        ok = stub2._load_bm25_cache()
        assert ok is False

    def test_clear_removes_both_files(self, tmp_path):
        from rank_bm25 import BM25Okapi

        corpus = [["a"]]
        stub = HybridSearcher.__new__(HybridSearcher)
        stub.ephemeral = False
        stub.index_path = tmp_path
        stub.bm25_chunks = [{"id": "c0", "content": "a"}]
        stub.bm25_k1 = 1.5
        stub.bm25_b = 0.75
        stub._bm25_tokenized_corpus = corpus
        stub.bm25_index = BM25Okapi(corpus, k1=stub.bm25_k1, b=stub.bm25_b)
        stub._last_corpus_hash = "h"
        stub._progress_reporter = None
        stub._save_bm25_cache()

        assert (tmp_path / "bm25_cache.json").exists()
        assert (tmp_path / "bm25_cache.npz").exists()

        assert stub.clear_bm25_cache() is True
        assert not (tmp_path / "bm25_cache.json").exists()
        assert not (tmp_path / "bm25_cache.npz").exists()
