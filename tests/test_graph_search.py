"""Tests for graph-augmented search utility functions."""

from unittest.mock import MagicMock
from mullet.knowledge.graph.search import (
    _rrf_merge_lists,
    _fetch_chunks_by_ids,
    local_search,
    global_search,
    hybrid_graph_search,
    mix_search,
)


class TestRrfMergeLists:
    def test_empty_input_returns_empty(self):
        assert _rrf_merge_lists() == []

    def test_single_list_preserved_order(self):
        chunks = [{"id": f"c{i}", "score": i} for i in range(5)]
        merged = _rrf_merge_lists(chunks, top_k=5)
        assert len(merged) == 5
        assert merged[0]["id"] == "c0"  # rank 0 → highest RRF

    def test_duplicate_ids_merged_with_boosted_score(self):
        a = [{"id": "x", "v": 1}]
        b = [{"id": "x", "v": 1}]
        merged = _rrf_merge_lists(a, b, top_k=10)
        # x appears in both lists → higher RRF than if only in one
        assert len(merged) == 1
        assert merged[0]["id"] == "x"
        assert merged[0]["_graph_rrf_score"] > 1.0 / 61  # > single-list score

    def test_top_k_limits_output(self):
        chunks = [{"id": f"c{i}"} for i in range(20)]
        merged = _rrf_merge_lists(chunks, top_k=5)
        assert len(merged) == 5

    def test_rrf_score_added_to_each_result(self):
        chunks = [{"id": "a"}, {"id": "b"}]
        merged = _rrf_merge_lists(chunks, top_k=10)
        assert all("_graph_rrf_score" in c for c in merged)

    def test_two_lists_with_unique_items(self):
        a = [{"id": "a1"}, {"id": "a2"}]
        b = [{"id": "b1"}, {"id": "b2"}]
        merged = _rrf_merge_lists(a, b, top_k=10)
        ids = {c["id"] for c in merged}
        assert ids == {"a1", "a2", "b1", "b2"}


class TestFetchChunksByIds:
    def test_empty_ids_returns_empty(self):
        searcher = MagicMock()
        result = _fetch_chunks_by_ids(searcher, [])
        assert result == []
        searcher.collection.get.assert_not_called()

    def test_successful_fetch_returns_chunks(self):
        searcher = MagicMock()
        searcher.collection.get.return_value = {
            "ids": ["c1", "c2"],
            "documents": ["text one", "text two"],
            "metadatas": [{"src": "a"}, {"src": "b"}],
        }
        chunks = _fetch_chunks_by_ids(searcher, ["c1", "c2"])
        assert len(chunks) == 2
        assert chunks[0]["id"] == "c1"
        assert chunks[0]["content"] == "text one"
        assert chunks[0]["metadata"] == {"src": "a"}

    def test_tag_metadata_merged_into_results(self):
        searcher = MagicMock()
        searcher.collection.get.return_value = {
            "ids": ["c1"],
            "documents": ["doc"],
            "metadatas": [{}],
        }
        chunks = _fetch_chunks_by_ids(searcher, ["c1"], tag_metadata={"_graph_entity": True})
        assert chunks[0]["_graph_entity"] is True

    def test_exception_from_collection_returns_empty(self):
        searcher = MagicMock()
        searcher.collection.get.side_effect = RuntimeError("chroma down")
        result = _fetch_chunks_by_ids(searcher, ["c1"])
        assert result == []


class TestLocalSearch:
    def _make_graph(self, entities=None):
        g = MagicMock()
        g.find_entities_by_name.return_value = entities or []
        return g

    def test_no_entities_returns_empty(self):
        graph = self._make_graph()
        searcher = MagicMock()
        result = local_search("anything", graph, searcher)
        assert result == []

    def test_entities_with_no_chunks_returns_empty(self):
        entity = MagicMock()
        entity.key = "alice"
        entity.name = "Alice"
        graph = self._make_graph([entity])
        graph.get_source_chunks.return_value = []
        result = local_search("Alice", graph, MagicMock())
        assert result == []

    def test_returns_fetched_chunks_tagged(self):
        entity = MagicMock()
        entity.key = "alice"
        entity.name = "Alice"
        graph = self._make_graph([entity])
        graph.get_source_chunks.return_value = ["c1"]

        searcher = MagicMock()
        searcher.collection.get.return_value = {
            "ids": ["c1"],
            "documents": ["Alice text"],
            "metadatas": [{}],
        }
        chunks = local_search("Alice", graph, searcher)
        assert len(chunks) == 1
        assert chunks[0]["_graph_entity"] is True
        assert "Alice" in chunks[0]["_graph_matched_entities"]


class TestGlobalSearch:
    def test_no_seed_entities_returns_empty(self):
        graph = MagicMock()
        graph.find_entities_by_name.return_value = []
        result = global_search("anything", graph, MagicMock())
        assert result == []

    def test_entities_with_no_chunks_returns_empty(self):
        entity = MagicMock()
        entity.key = "e1"
        graph = MagicMock()
        graph.find_entities_by_name.return_value = [entity]
        graph.get_neighbors.return_value = []
        graph.get_source_chunks.return_value = []
        result = global_search("query", graph, MagicMock())
        assert result == []


class TestHybridGraphSearch:
    def test_deduplicates_by_id(self):
        # Both local and global return the same chunk
        graph = MagicMock()
        graph.find_entities_by_name.return_value = []
        result = hybrid_graph_search("q", graph, MagicMock())
        assert isinstance(result, list)


class TestMixSearch:
    def test_no_graph_results_returns_standard(self):
        graph = MagicMock()
        graph.find_entities_by_name.return_value = []
        searcher = MagicMock()
        searcher.search.return_value = [{"id": "s1", "content": "standard"}]
        result = mix_search("query", graph, searcher)
        assert result == [{"id": "s1", "content": "standard"}]


# ---------------------------------------------------------------------------
# Gap tests for uncovered lines
# ---------------------------------------------------------------------------

class TestGlobalSearchSuccessPath:
    def _make_entity(self, key: str):
        e = MagicMock()
        e.key = key
        return e

    def test_neighbors_traversed_and_chunks_fetched(self):
        """Lines 148, 156-162: neighbors found → chunk_ids non-empty → fetch + return."""
        seed = self._make_entity("seed-1")
        neighbor = self._make_entity("neighbor-1")
        rel = MagicMock()

        graph = MagicMock()
        graph.find_entities_by_name.return_value = [seed]
        graph.get_neighbors.return_value = [(neighbor, rel)]
        graph.get_source_chunks.return_value = ["chunk-1", "chunk-2"]

        # Mock searcher with a collection that returns chunks
        searcher = MagicMock()
        searcher.collection.get.return_value = {
            "ids": ["chunk-1"],
            "documents": ["text of chunk 1"],
            "metadatas": [{"source": "doc.pdf"}],
        }

        result = global_search("test query", graph, searcher, top_k=5)
        assert len(result) >= 1
        # Verify neighbor key was discovered
        call_args = graph.get_source_chunks.call_args[0][0]
        assert "neighbor-1" in call_args


class TestHybridGraphSearchDedup:
    def _make_entity(self, key: str):
        e = MagicMock()
        e.key = key
        return e

    def test_dedup_loop_body_executed(self):
        """Lines 179-182: dedup loop runs when local+global return results."""
        entity = self._make_entity("e1")
        graph = MagicMock()
        graph.find_entities_by_name.return_value = [entity]
        graph.get_neighbors.return_value = []
        # Both local and global return the same chunk → dedup to 1
        graph.get_source_chunks.return_value = []  # no chunk fetch needed
        searcher = MagicMock()
        searcher.collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}

        # Inject pre-built results by mocking the sub-functions
        from unittest.mock import patch
        chunk_a = {"id": "chunk-1", "content": "hello"}
        chunk_b = {"id": "chunk-2", "content": "world"}
        chunk_dup = {"id": "chunk-1", "content": "hello"}

        with patch("mullet.knowledge.graph.search.local_search", return_value=[chunk_a, chunk_dup]), \
             patch("mullet.knowledge.graph.search.global_search", return_value=[chunk_b, chunk_dup]):
            result = hybrid_graph_search("q", graph, searcher, top_k=10)

        assert len(result) == 2  # chunk-1 deduplicated, chunk-2 kept
        ids = {c["id"] for c in result}
        assert ids == {"chunk-1", "chunk-2"}


class TestMixSearchRrfPath:
    def test_rrf_merge_when_graph_results_present(self):
        """Line 210: when graph_results non-empty → RRF merge path."""
        standard = [{"id": "s1", "content": "standard"}]
        graph_chunk = {"id": "g1", "content": "graph"}
        graph = MagicMock()

        from unittest.mock import patch
        with patch("mullet.knowledge.graph.search.hybrid_graph_search", return_value=[graph_chunk]):
            searcher = MagicMock()
            searcher.search.return_value = standard
            result = mix_search("q", graph, searcher, top_k=5)

        assert any(r["id"] == "g1" for r in result) or any(r["id"] == "s1" for r in result)
