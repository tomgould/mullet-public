"""Comprehensive tests for knowledge graph extractor and search modules."""

from unittest.mock import MagicMock


from mullet.knowledge.graph.extractor import (
    GraphExtractor,
    _build_batches_by_budget,
    _is_garbage_response,
)
from mullet.knowledge.graph.models import Entity
from mullet.knowledge.graph.search import (
    _fetch_chunks_by_ids,
    _rrf_merge_lists,
    global_search,
    hybrid_graph_search,
    local_search,
    mix_search,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_extractor(llm_fn=None):
    """Create GraphExtractor with a mock LLM function."""
    if llm_fn is None:
        llm_fn = lambda msgs, sys, max_tok: '{"entities": [], "relationships": []}'
    return GraphExtractor(get_llm_response=llm_fn)


def _entity(name, entity_type="person"):
    """Build a real Entity dataclass for use in graph mocks."""
    return Entity(name=name, entity_type=entity_type)


# ===========================================================================
# _is_garbage_response
# ===========================================================================

class TestIsGarbageResponse:
    def test_short_response_always_false(self):
        # < GARBAGE_MIN_LENGTH (200)
        is_garbage, reason = _is_garbage_response("short")
        assert is_garbage is False
        assert reason == ""

        is_garbage2, _ = _is_garbage_response("")
        assert is_garbage2 is False

    def test_special_tokens_garbage(self):
        # >10 SPECIAL/UNK/PAD/unused tokens and >200 chars
        payload = "<SPECIAL_0> " * 20
        is_garbage, reason = _is_garbage_response(payload)
        assert is_garbage is True
        assert "special tokens" in reason

    def test_low_char_diversity_garbage(self):
        # >100 non-whitespace chars, <5 unique
        payload = "??????????" * 20
        is_garbage, reason = _is_garbage_response(payload)
        assert is_garbage is True
        assert "char diversity too low" in reason

    def test_single_char_dominance_garbage(self):
        # >100 non-ws chars, one char >80%
        payload = "a" * 195 + "bcdef"
        is_garbage, reason = _is_garbage_response(payload)
        assert is_garbage is True
        assert "'a'" in reason
        assert "%" in reason

    def test_low_token_diversity_garbage(self):
        # >50 tokens, unique/total < 0.05
        payload = "abcdefgh12 " * 60
        is_garbage, reason = _is_garbage_response(payload)
        assert is_garbage is True
        assert "low token diversity" in reason

    def test_normal_response_not_garbage(self):
        payload = (
            "This is a perfectly reasonable LLM response with lots of different words "
            "and enough length to pass the minimum threshold. It should not trigger any "
            "garbage detection heuristics because the character diversity is high and the "
            "token diversity is also perfectly fine."
        )
        is_garbage, reason = _is_garbage_response(payload)
        assert is_garbage is False
        assert reason == ""


# ===========================================================================
# _build_batches_by_budget
# ===========================================================================

class TestBuildBatchesByBudget:
    def _chunk(self, cid, content="x"):
        return {"id": cid, "content": content}

    def test_empty_input(self):
        assert _build_batches_by_budget([], 1000) == []

    def test_single_chunk_fits(self):
        chunks = [self._chunk("c1", "hello world")]
        batches = _build_batches_by_budget(chunks, 10000)
        assert len(batches) == 1
        assert batches[0][0]["id"] == "c1"

    def test_chunks_split_into_multiple_batches(self):
        # Large content forces split
        chunks = [
            {"id": f"c{i}", "content": "x" * 500}
            for i in range(3)
        ]
        batches = _build_batches_by_budget(chunks, 600)
        assert len(batches) == 3

    def test_empty_content_chunks_pass_through(self):
        # Empty-content chunks go into current batch without counting budget
        chunks = [
            {"id": "c1", "content": "some text here"},
            {"id": "blank", "content": "   "},
            {"id": "c2", "content": "more text"},
        ]
        batches = _build_batches_by_budget(chunks, 10000)
        assert len(batches) == 1
        ids = [c["id"] for c in batches[0]]
        assert ids == ["c1", "blank", "c2"]

    def test_budget_smaller_than_one_chunk(self):
        # Even when budget is tiny, each chunk still ends up in a batch
        chunks = [
            {"id": "c1", "content": "x" * 1000},
        ]
        batches = _build_batches_by_budget(chunks, 10)
        assert len(batches) == 1
        assert len(batches[0]) == 1


# ===========================================================================
# GraphExtractor.extract_from_chunk
# ===========================================================================

class TestExtractFromChunk:
    def test_empty_content_returns_empty(self):
        ext = _make_extractor()
        ents, rels = ext.extract_from_chunk({"id": "c1", "content": "   "})
        assert ents == [] and rels == []

    def test_success_with_valid_json(self):
        def good_llm(msgs, sys, max_tok):
            return '{"entities": [{"name": "Alice", "type": "person", "description": "test"}], "relationships": []}'
        ext = _make_extractor(good_llm)
        ents, rels = ext.extract_from_chunk({"id": "c1", "content": "Alice is here"})
        assert len(ents) == 1
        assert ents[0].name == "Alice"
        assert ents[0].entity_type == "person"

    def test_llm_exception_returns_empty(self):
        def boom(*a):
            raise RuntimeError("timeout")
        ext = _make_extractor(boom)
        ents, rels = ext.extract_from_chunk({"id": "c1", "content": "some text"})
        assert ents == [] and rels == []


# ===========================================================================
# GraphExtractor._parse_response
# ===========================================================================

class TestParseResponse:
    def test_valid_json(self):
        ext = _make_extractor()
        response = '{"entities": [{"name": "Alice", "type": "person", "description": "d"}], "relationships": [{"source": "Alice", "target": "Bob", "type": "knows", "weight": 0.8}]}'
        entities, rels = ext._parse_response(response, "chunk-1")
        assert len(entities) == 1
        assert entities[0].name == "Alice"
        assert len(rels) == 1
        assert rels[0].source_entity == "alice::person"
        assert rels[0].weight == 0.8

    def test_code_fence_stripped(self):
        ext = _make_extractor()
        response = '```json\n{"entities": [{"name": "Bob", "type": "person"}], "relationships": []}\n```'
        entities, rels = ext._parse_response(response, "chunk-1")
        assert len(entities) == 1
        assert entities[0].name == "Bob"

    def test_think_tag_stripped(self):
        ext = _make_extractor()
        response = '<think>reasoning...</think>\n{"entities": [{"name": "Alice", "type": "person"}], "relationships": []}'
        entities, rels = ext._parse_response(response, "chunk-1")
        assert len(entities) == 1

    def test_invalid_json_returns_empty(self):
        ext = _make_extractor()
        entities, rels = ext._parse_response("this is not json at all", "chunk-1")
        assert entities == [] and rels == []

    def test_unknown_entity_type_filtered(self):
        ext = _make_extractor()
        response = '{"entities": [{"name": "X", "type": "spacecraft", "description": ""}], "relationships": []}'
        entities, rels = ext._parse_response(response, "chunk-1")
        assert entities == []

    def test_relationships_reference_known_entities(self):
        ext = _make_extractor()
        response = '{"entities": [{"name": "Alice", "type": "person"}, {"name": "Bob", "type": "person"}], "relationships": [{"source": "Alice", "target": "Bob", "type": "knows", "weight": 0.9}]}'
        entities, rels = ext._parse_response(response, "chunk-1")
        assert len(rels) == 1
        assert rels[0].source_entity == "alice::person"
        assert rels[0].target_entity == "bob::person"

    def test_weight_clamped_to_0_1(self):
        ext = _make_extractor()
        response = '{"entities": [{"name": "Alice", "type": "person"}], "relationships": [{"source": "Alice", "target": "Bob", "type": "knows", "weight": 1.5}]}'
        _, rels = ext._parse_response(response, "chunk-1")
        assert rels[0].weight == 1.0

        response2 = '{"entities": [{"name": "Alice", "type": "person"}], "relationships": [{"source": "Alice", "target": "Bob", "type": "knows", "weight": -0.5}]}'
        _, rels2 = ext._parse_response(response2, "chunk-1")
        assert rels2[0].weight == 0.0


# ===========================================================================
# GraphExtractor._call_llm_for_batch
# ===========================================================================

class TestCallLlmForBatch:
    def test_circuit_breaker_closed_skips(self):
        ext = _make_extractor()
        # Trip the breaker (threshold is 3)
        for _ in range(3):
            ext._circuit_breaker.record_failure()
        batch = [{"id": "c1", "content": "some text"}]
        entities, rels, chunk_ids, success = ext._call_llm_for_batch(batch)
        assert success is False
        assert entities == [] and rels == []

    def test_empty_content_batch_returns_true(self):
        ext = _make_extractor()
        batch = [{"id": "c1", "content": "   "}, {"id": "c2", "content": ""}]
        entities, rels, chunk_ids, success = ext._call_llm_for_batch(batch)
        assert success is True
        assert entities == [] and rels == []

    def test_garbage_response_sets_fatal_error(self):
        def garbage_llm(*a):
            return "<SPECIAL_0> " * 20
        ext = _make_extractor(garbage_llm)
        # Pre-warm counters so one more garbage trips the fatal threshold
        ext._batches_processed = 5
        ext._garbage_batch_count = 3
        batch = [{"id": "c1", "content": "some text"}]
        entities, rels, chunk_ids, success = ext._call_llm_for_batch(batch)
        assert success == "garbage"
        assert entities == [] and rels == []
        assert ext._fatal_error is not None
        assert "model is broken" in ext._fatal_error

    def test_error_response_returns_failure(self):
        def error_llm(*a):
            return "Error: connection refused"
        ext = _make_extractor(error_llm)
        batch = [{"id": "c1", "content": "some text"}]
        entities, rels, chunk_ids, success = ext._call_llm_for_batch(batch)
        assert success is False
        assert entities == [] and rels == []

    def test_success(self):
        def good_llm(*a):
            return '{"entities": [{"name": "Alice", "type": "person", "description": "a person"}], "relationships": []}'
        ext = _make_extractor(good_llm)
        batch = [{"id": "c1", "content": "Alice did something"}]
        entities, rels, chunk_ids, success = ext._call_llm_for_batch(batch)
        assert success is True
        assert len(entities) == 1
        assert "c1" in entities[0].source_chunk_ids


# ===========================================================================
# GraphExtractor.extract_from_chunks
# ===========================================================================

class TestExtractFromChunks:
    def test_empty_chunks(self):
        ext = _make_extractor()
        ents, rels = ext.extract_from_chunks([])
        assert ents == [] and rels == []

    def test_single_batch_success(self):
        def good_llm(*a):
            return '{"entities": [{"name": "A", "type": "person"}], "relationships": []}'
        ext = _make_extractor(good_llm)
        chunks = [{"id": "c1", "content": "text"}]
        ents, rels = ext.extract_from_chunks(chunks)
        assert len(ents) == 1
        assert ents[0].name == "A"

    def test_progress_callback_called(self):
        progress_calls = []
        def cb(prog, ents, rels):
            progress_calls.append((prog, len(ents), len(rels)))
        def good_llm(*a):
            return '{"entities": [{"name": "A", "type": "person"}], "relationships": []}'
        ext = _make_extractor(good_llm)
        ext.batch_text_budget = 10
        ext.save_every = 1
        chunks = [{"id": f"c{i}", "content": f"text {i}"} for i in range(3)]
        ents, rels = ext.extract_from_chunks(chunks, progress_callback=cb)
        assert len(progress_calls) >= 1

    def test_fatal_error_aborts(self):
        def garbage_llm(*a):
            return "<SPECIAL_0> " * 20
        ext = _make_extractor(garbage_llm)
        ext.batch_text_budget = 10
        ext.max_workers = 1
        ext._circuit_breaker.failure_threshold = 100
        chunks = [{"id": f"c{i}", "content": f"text {i}"} for i in range(8)]
        ents, rels = ext.extract_from_chunks(chunks)
        assert ext._fatal_error is not None
        assert "model is broken" in ext._fatal_error

    def test_exception_in_future_handled(self):
        def good_llm(*a):
            return '{"entities": [{"name": "A", "type": "person"}], "relationships": []}'
        ext = _make_extractor(good_llm)
        ext.batch_text_budget = 10
        original = ext._process_batch
        def raise_once(batch, idx):
            if idx == 0:
                raise RuntimeError("boom")
            return original(batch, idx)
        ext._process_batch = raise_once
        chunks = [{"id": "c1", "content": "t"}, {"id": "c2", "content": "t"}]
        ents, rels = ext.extract_from_chunks(chunks)
        assert len(ents) == 1


# ===========================================================================
# _fetch_chunks_by_ids
# ===========================================================================

class TestFetchChunksByIds:
    def test_empty_chunk_ids(self):
        searcher = MagicMock()
        result = _fetch_chunks_by_ids(searcher, [])
        assert result == []
        searcher.collection.get.assert_not_called()

    def test_success_with_tag_metadata(self):
        searcher = MagicMock()
        searcher.collection.get.return_value = {
            "ids": ["id1", "id2"],
            "documents": ["doc1", "doc2"],
            "metadatas": [{"a": 1}, {"b": 2}],
        }
        result = _fetch_chunks_by_ids(searcher, ["id1", "id2"], tag_metadata={"_graph_tag": True})
        assert len(result) == 2
        assert result[0]["id"] == "id1"
        assert result[0]["_graph_tag"] is True
        assert result[1]["_graph_tag"] is True

    def test_collection_get_exception_returns_empty(self):
        searcher = MagicMock()
        searcher.collection.get.side_effect = RuntimeError("DB down")
        result = _fetch_chunks_by_ids(searcher, ["id1"])
        assert result == []


# ===========================================================================
# _rrf_merge_lists
# ===========================================================================

class TestRrfMergeLists:
    def test_single_list(self):
        lst = [
            {"id": "a", "content": "A"},
            {"id": "b", "content": "B"},
        ]
        merged = _rrf_merge_lists(lst, top_k=10)
        assert len(merged) == 2
        assert merged[0]["id"] == "a"
        assert "_graph_rrf_score" in merged[0]

    def test_two_lists_with_overlap(self):
        lst1 = [
            {"id": "a", "content": "A1"},
            {"id": "b", "content": "B1"},
        ]
        lst2 = [
            {"id": "b", "content": "B2"},
            {"id": "c", "content": "C"},
        ]
        merged = _rrf_merge_lists(lst1, lst2, top_k=10)
        ids = [m["id"] for m in merged]
        # "b" appears in both lists so has higher RRF score
        assert ids[0] == "b"
        assert set(ids) == {"a", "b", "c"}

    def test_top_k_limit(self):
        lst = [{"id": f"i{i}"} for i in range(10)]
        merged = _rrf_merge_lists(lst, top_k=3)
        assert len(merged) == 3


# ===========================================================================
# local_search
# ===========================================================================

class TestLocalSearch:
    def test_no_entities_found(self):
        graph = MagicMock()
        graph.find_entities_by_name.return_value = []
        searcher = MagicMock()
        result = local_search("query", graph, searcher)
        assert result == []

    def test_entities_but_no_chunk_ids(self):
        graph = MagicMock()
        ent = _entity("Alice")
        graph.find_entities_by_name.return_value = [ent]
        graph.get_source_chunks.return_value = []
        searcher = MagicMock()
        result = local_search("query", graph, searcher)
        assert result == []

    def test_success_with_tagged_chunks(self):
        graph = MagicMock()
        ent = _entity("Alice")
        graph.find_entities_by_name.return_value = [ent]
        graph.get_source_chunks.return_value = ["chunk-1"]

        searcher = MagicMock()
        searcher.collection.get.return_value = {
            "ids": ["chunk-1"],
            "documents": ["content"],
            "metadatas": [{}],
        }

        result = local_search("query", graph, searcher, top_k=5)
        assert len(result) == 1
        assert result[0]["id"] == "chunk-1"
        assert result[0]["_graph_entity"] is True
        assert result[0]["_graph_matched_entities"] == ["Alice"]


# ===========================================================================
# global_search
# ===========================================================================

class TestGlobalSearch:
    def test_no_seed_entities(self):
        graph = MagicMock()
        graph.find_entities_by_name.return_value = []
        searcher = MagicMock()
        result = global_search("query", graph, searcher)
        assert result == []

    def test_success_with_bfs_traversal(self):
        graph = MagicMock()
        seed = _entity("Alice")
        neighbor = _entity("Bob")
        graph.find_entities_by_name.return_value = [seed]
        graph.get_neighbors.return_value = [(neighbor, None)]
        graph.get_source_chunks.return_value = ["chunk-1", "chunk-2"]

        searcher = MagicMock()
        searcher.collection.get.return_value = {
            "ids": ["chunk-1", "chunk-2"],
            "documents": ["d1", "d2"],
            "metadatas": [{}, {}],
        }

        result = global_search("query", graph, searcher, top_k=5)
        assert len(result) == 2
        assert result[0]["_graph_path"] is True

    def test_no_chunks(self):
        graph = MagicMock()
        seed = _entity("Alice")
        graph.find_entities_by_name.return_value = [seed]
        graph.get_neighbors.return_value = []
        graph.get_source_chunks.return_value = []
        searcher = MagicMock()
        result = global_search("query", graph, searcher)
        assert result == []


# ===========================================================================
# hybrid_graph_search
# ===========================================================================

class TestHybridGraphSearch:
    def test_deduplication(self):
        graph = MagicMock()
        ent = _entity("Alice")
        graph.find_entities_by_name.return_value = [ent]
        graph.get_source_chunks.return_value = ["chunk-1", "chunk-2"]
        graph.get_neighbors.return_value = []

        searcher = MagicMock()
        searcher.collection.get.return_value = {
            "ids": ["chunk-1", "chunk-2"],
            "documents": ["d1", "d2"],
            "metadatas": [{}, {}],
        }

        result = hybrid_graph_search("query", graph, searcher, top_k=10)
        ids = [r["id"] for r in result]
        # local and global both return the same chunks; dedup should keep unique
        assert len(ids) == len(set(ids))

    def test_top_k_enforced(self):
        graph = MagicMock()
        ent = _entity("Alice")
        graph.find_entities_by_name.return_value = [ent]
        graph.get_source_chunks.return_value = [f"chunk-{i}" for i in range(20)]
        graph.get_neighbors.return_value = []

        searcher = MagicMock()
        searcher.collection.get.return_value = {
            "ids": [f"chunk-{i}" for i in range(20)],
            "documents": [f"d{i}" for i in range(20)],
            "metadatas": [{} for _ in range(20)],
        }

        result = hybrid_graph_search("query", graph, searcher, top_k=5)
        assert len(result) == 5


# ===========================================================================
# mix_search
# ===========================================================================

class TestMixSearch:
    def test_no_graph_results_returns_standard(self):
        graph = MagicMock()
        graph.find_entities_by_name.return_value = []

        searcher = MagicMock()
        searcher.search.return_value = [
            {"id": "s1", "content": "std1"},
            {"id": "s2", "content": "std2"},
        ]

        result = mix_search("query", graph, searcher, top_k=5)
        assert len(result) == 2
        assert result[0]["id"] == "s1"

    def test_graph_results_rrf_merged(self):
        graph = MagicMock()
        ent = _entity("Alice")
        graph.find_entities_by_name.return_value = [ent]
        graph.get_source_chunks.return_value = ["g1"]
        graph.get_neighbors.return_value = []

        searcher = MagicMock()
        searcher.search.return_value = [
            {"id": "s1", "content": "std1"},
        ]
        searcher.collection.get.return_value = {
            "ids": ["g1"],
            "documents": ["graph1"],
            "metadatas": [{}],
        }

        result = mix_search("query", graph, searcher, top_k=5)
        ids = [r["id"] for r in result]
        assert set(ids) == {"s1", "g1"}
        for r in result:
            assert "_graph_rrf_score" in r
