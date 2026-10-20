"""Tests for GraphExtractor error handling paths."""

from mullet.knowledge.graph.extractor import GraphExtractor, _is_garbage_response, _build_batches_by_budget


def _make_extractor(llm_fn=None):
    """Create GraphExtractor with a mock LLM function."""
    if llm_fn is None:
        llm_fn = lambda msgs, sys, max_tok: '{"entities": [], "relationships": []}'
    return GraphExtractor(get_llm_response=llm_fn)


class TestGarbageDetection:
    def test_garbage_special_prefix(self):
        # Need >200 chars and >10 special tokens
        is_garbage, reason = _is_garbage_response("<SPECIAL_0> " * 20)
        assert is_garbage is True

    def test_valid_json_not_garbage(self):
        is_garbage, _ = _is_garbage_response('{"entities": [], "relationships": []}')
        assert is_garbage is False

    def test_short_response_not_garbage(self):
        # Responses below GARBAGE_MIN_LENGTH (200 chars) are not classified as garbage
        is_garbage, _ = _is_garbage_response("")
        assert is_garbage is False


class TestParseResponse:
    def test_valid_json_parsed(self):
        ext = _make_extractor()
        entities, rels = ext._parse_response(
            '{"entities": [{"name": "Alice", "type": "person", "description": "test"}], "relationships": []}',
            "chunk-1"
        )
        assert len(entities) == 1
        assert entities[0].name == "Alice"

    def test_malformed_json_returns_empty(self):
        ext = _make_extractor()
        entities, rels = ext._parse_response("this is not json at all", "chunk-1")
        assert entities == [] and rels == []

    def test_json_in_markdown_fence_parsed(self):
        ext = _make_extractor()
        response = '```json\n{"entities": [{"name": "Bob", "type": "person"}], "relationships": []}\n```'
        entities, rels = ext._parse_response(response, "chunk-1")
        assert len(entities) == 1

    def test_unknown_entity_type_filtered(self):
        ext = _make_extractor()
        entities, rels = ext._parse_response(
            '{"entities": [{"name": "X", "type": "spacecraft", "description": ""}], "relationships": []}',
            "chunk-1"
        )
        assert entities == []

    def test_empty_entity_name_skipped(self):
        ext = _make_extractor()
        entities, rels = ext._parse_response(
            '{"entities": [{"name": "", "type": "person"}], "relationships": []}',
            "chunk-1"
        )
        assert entities == []


class TestCallLlmForBatch:
    def test_error_prefix_response_returns_empty(self):
        ext = _make_extractor(lambda *a: "Error: connection refused")
        batch = [{"id": "c1", "content": "some text"}]
        entities, rels, chunk_ids, success = ext._call_llm_for_batch(batch)
        assert entities == [] and rels == [] and success is False

    def test_exception_returns_empty(self):
        def boom(*a):
            raise RuntimeError("timeout")
        ext = _make_extractor(boom)
        batch = [{"id": "c1", "content": "some text"}]
        entities, rels, chunk_ids, success = ext._call_llm_for_batch(batch)
        assert success is False and entities == []

    def test_empty_batch_content_skipped(self):
        ext = _make_extractor()
        batch = [{"id": "c1", "content": "   "}]
        entities, rels, chunk_ids, success = ext._call_llm_for_batch(batch)
        assert success is True and entities == []

    def test_valid_response_returns_entities(self):
        def good_llm(*a):
            return '{"entities": [{"name": "Alice", "type": "person", "description": "a person"}], "relationships": []}'
        ext = _make_extractor(good_llm)
        batch = [{"id": "c1", "content": "Alice did something"}]
        entities, rels, chunk_ids, success = ext._call_llm_for_batch(batch)
        assert success is True and len(entities) == 1


class TestBuildBatchesByBudget:
    def _chunk(self, cid, size):
        return {"id": cid, "content": "x" * size}

    def test_empty_input_returns_empty(self):
        assert _build_batches_by_budget([], 1000) == []

    def test_single_chunk_fits_in_one_batch(self):
        chunks = [self._chunk("c1", 50)]
        batches = _build_batches_by_budget(chunks, 10000)
        assert len(batches) == 1
        assert batches[0][0]["id"] == "c1"

    def test_chunks_split_when_budget_exceeded(self):
        # each chunk is 500 chars; budget 600 -> each in its own batch
        chunks = [self._chunk(f"c{i}", 500) for i in range(3)]
        batches = _build_batches_by_budget(chunks, 600)
        assert len(batches) == 3

    def test_chunks_combined_when_budget_allows(self):
        # each chunk is 10 chars; budget 1000 -> all in one batch
        chunks = [self._chunk(f"c{i}", 10) for i in range(5)]
        batches = _build_batches_by_budget(chunks, 1000)
        assert len(batches) == 1
        assert len(batches[0]) == 5

    def test_whitespace_only_chunk_passes_through(self):
        # empty-content chunks are appended to current batch without consuming budget
        chunks = [{"id": "blank", "content": "   "}]
        batches = _build_batches_by_budget(chunks, 100)
        assert len(batches) == 1
        assert batches[0][0]["id"] == "blank"

    def test_all_chunks_end_up_in_some_batch(self):
        chunks = [self._chunk(f"c{i}", 200) for i in range(10)]
        batches = _build_batches_by_budget(chunks, 500)
        total = sum(len(b) for b in batches)
        assert total == 10


# ---- Coverage additions below ----


class TestGarbageDetectionExtended:
    def test_char_diversity_too_low(self):
        payload = "??????????" * 20
        is_garbage, reason = _is_garbage_response(payload)
        assert is_garbage is True
        assert "char diversity too low" in reason

    def test_single_char_dominance(self):
        payload = "a" * 195 + "bcdef"
        is_garbage, reason = _is_garbage_response(payload)
        assert is_garbage is True
        assert "%" in reason

    def test_low_token_diversity(self):
        payload = "abcdefgh12 " * 60
        is_garbage, reason = _is_garbage_response(payload)
        assert is_garbage is True
        assert "low token diversity" in reason


class TestExtractFromChunk:
    def test_empty_content_returns_empty(self):
        ext = _make_extractor()
        ents, rels = ext.extract_from_chunk({"id": "c1", "content": "   "})
        assert ents == [] and rels == []

    def test_missing_content_key(self):
        ext = _make_extractor()
        ents, rels = ext.extract_from_chunk({"id": "c1"})
        assert ents == [] and rels == []

    def test_valid_extraction(self):
        def good_llm(msgs, sys, max_tok):
            return '{"entities": [{"name": "Alice", "type": "person", "description": "test"}], "relationships": []}'
        ext = _make_extractor(good_llm)
        ents, rels = ext.extract_from_chunk({"id": "c1", "content": "Alice is here"})
        assert len(ents) == 1
        assert ents[0].name == "Alice"

    def test_exception_returns_empty(self):
        def boom(*a):
            raise RuntimeError("fail")
        ext = _make_extractor(boom)
        ents, rels = ext.extract_from_chunk({"id": "c1", "content": "some text"})
        assert ents == [] and rels == []


class TestCircuitBreakerAndGarbage:
    def test_circuit_breaker_open_skips_batch(self):
        ext = _make_extractor()
        for _ in range(3):
            ext._circuit_breaker.record_failure()
        batch = [{"id": "c1", "content": "some text"}]
        entities, rels, chunk_ids, success = ext._call_llm_for_batch(batch)
        assert success is False
        assert entities == [] and rels == []

    def test_garbage_response_records_failure(self):
        def garbage_llm(*a):
            return "<SPECIAL_0> " * 20
        ext = _make_extractor(garbage_llm)
        batch = [{"id": "c1", "content": "some text"}]
        entities, rels, chunk_ids, success = ext._call_llm_for_batch(batch)
        assert success == "garbage"
        assert entities == [] and rels == []

    def test_garbage_fatal_error_threshold(self):
        def garbage_llm(*a):
            return "<SPECIAL_0> " * 20
        ext = _make_extractor(garbage_llm)
        ext._batches_processed = 5
        ext._garbage_batch_count = 3
        batch = [{"id": "c1", "content": "some text"}]
        ext._call_llm_for_batch(batch)
        assert ext._fatal_error is not None
        assert "model is broken" in ext._fatal_error


class TestProcessBatch:
    def test_success_on_first_try(self):
        def good_llm(*a):
            return '{"entities": [{"name": "A", "type": "person"}], "relationships": []}'
        ext = _make_extractor(good_llm)
        batch = [{"id": "c1", "content": "text"}]
        ents, rels, ids, num, skipped = ext._process_batch(batch, 0)
        assert len(ents) == 1
        assert skipped == 0

    def test_garbage_skips_without_retry(self):
        def garbage_llm(*a):
            return "<SPECIAL_0> " * 20
        ext = _make_extractor(garbage_llm)
        batch = [{"id": "c1", "content": "text"}]
        ents, rels, ids, num, skipped = ext._process_batch(batch, 0)
        assert ents == []
        assert skipped == 1

    def test_failure_then_success_on_split(self):
        calls = []
        def flaky_llm(*a):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("fail")
            return '{"entities": [{"name": "A", "type": "person"}], "relationships": []}'
        ext = _make_extractor(flaky_llm)
        ext.batch_text_budget = 2000
        batch = [
            {"id": "c1", "content": "x" * 500},
            {"id": "c2", "content": "x" * 500},
            {"id": "c3", "content": "x" * 500},
        ]
        ents, rels, ids, num, skipped = ext._process_batch(batch, 0)
        assert len(ents) == 3
        assert skipped == 0

    def test_failure_budget_too_low_skips(self):
        def always_fail(*a):
            raise RuntimeError("fail")
        ext = _make_extractor(always_fail)
        ext.batch_text_budget = 50
        batch = [{"id": "c1", "content": "t"}, {"id": "c2", "content": "t"}]
        ents, rels, ids, num, skipped = ext._process_batch(batch, 0)
        assert ents == []
        assert skipped == 2

    def test_failure_then_deeper_split_success(self):
        calls = []
        def flaky_llm(*a):
            calls.append(1)
            if len(calls) <= 2:
                raise RuntimeError("fail")
            return '{"entities": [{"name": "A", "type": "person"}], "relationships": []}'
        ext = _make_extractor(flaky_llm)
        ext.batch_text_budget = 4000
        batch = [{"id": f"c{i}", "content": "t"} for i in range(101)]
        ents, rels, ids, num, skipped = ext._process_batch(batch, 0)
        assert len(ents) == 3
        assert skipped == 0

    def test_deeper_split_also_fails(self):
        def always_fail(*a):
            raise RuntimeError("fail")
        ext = _make_extractor(always_fail)
        ext.batch_text_budget = 4000
        batch = [{"id": f"c{i}", "content": "t"} for i in range(101)]
        ents, rels, ids, num, skipped = ext._process_batch(batch, 0)
        assert ents == []
        assert skipped == 101

    def test_mixed_whitespace_chunks(self):
        def good_llm(*a):
            return '{"entities": [{"name": "A", "type": "person"}], "relationships": []}'
        ext = _make_extractor(good_llm)
        batch = [{"id": "c1", "content": "text"}, {"id": "c2", "content": "   "}]
        ents, rels, ids, num, skipped = ext._process_batch(batch, 0)
        assert len(ents) == 1
        assert num == 2
        assert skipped == 0


class TestExtractFromChunks:
    def test_empty_chunks(self):
        ext = _make_extractor()
        ents, rels = ext.extract_from_chunks([])
        assert ents == [] and rels == []

    def test_basic_extraction_with_callback(self):
        progress_calls = []
        def cb(prog, ents, rels):
            progress_calls.append((prog, len(ents), len(rels)))
        def good_llm(*a):
            return '{"entities": [{"name": "A", "type": "person"}], "relationships": []}'
        ext = _make_extractor(good_llm)
        ext.batch_text_budget = 10
        chunks = [{"id": f"c{i}", "content": f"text {i}"} for i in range(6)]
        ents, rels = ext.extract_from_chunks(chunks, progress_callback=cb)
        assert len(ents) == 6
        assert len(progress_calls) >= 2

    def test_inter_batch_sleep(self):
        import time
        def good_llm(*a):
            return '{"entities": [{"name": "A", "type": "person"}], "relationships": []}'
        ext = _make_extractor(good_llm)
        ext.inter_batch_sleep = 0.05
        chunks = [{"id": "c1", "content": "text"}, {"id": "c2", "content": "text"}]
        t0 = time.time()
        ext.extract_from_chunks(chunks)
        assert time.time() - t0 >= 0.04

    def test_fatal_error_aborts_early(self):
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

    def test_circuit_breaker_stats_logged(self):
        call_count = [0]
        def error_llm(*a):
            call_count[0] += 1
            return "Error: boom"
        ext = _make_extractor(error_llm)
        ext.batch_text_budget = 10
        ext.max_workers = 1
        chunks = [{"id": f"c{i}", "content": f"text {i}"} for i in range(5)]
        ext.extract_from_chunks(chunks)
        stats = ext._circuit_breaker.stats
        assert stats["total_rejected"] >= 2
        assert stats["consecutive_failures"] == 3

    def test_future_exception_handled(self):
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


class TestParseResponseExtended:
    def test_json_in_braces_but_invalid_inner(self):
        ext = _make_extractor()
        response = 'prefix { "entities": [broken } suffix'
        entities, rels = ext._parse_response(response, "chunk-1")
        assert entities == [] and rels == []

    def test_think_tags_stripped(self):
        ext = _make_extractor()
        response = '<think>reasoning...</think>\n{"entities": [{"name": "Alice", "type": "person"}], "relationships": []}'
        entities, rels = ext._parse_response(response, "chunk-1")
        assert len(entities) == 1

    def test_relationship_with_known_entity(self):
        ext = _make_extractor()
        response = '{"entities": [{"name": "Alice", "type": "person"}, {"name": "Bob", "type": "person"}], "relationships": [{"source": "Alice", "target": "Bob", "type": "knows", "weight": 0.8}]}'
        entities, rels = ext._parse_response(response, "chunk-1")
        assert len(rels) == 1
        assert rels[0].source_entity == "alice::person"
        assert rels[0].target_entity == "bob::person"

    def test_relationship_with_unknown_entity_fallback(self):
        ext = _make_extractor()
        response = '{"entities": [{"name": "Alice", "type": "person"}], "relationships": [{"source": "Alice", "target": "UnknownEntity", "type": "knows", "weight": 0.8}]}'
        entities, rels = ext._parse_response(response, "chunk-1")
        assert len(rels) == 1
        assert rels[0].target_entity == "unknownentity::concept"

    def test_relationship_missing_fields_skipped(self):
        ext = _make_extractor()
        response = '{"entities": [{"name": "Alice", "type": "person"}], "relationships": [{"source": "Alice", "target": "", "type": "knows"}]}'
        entities, rels = ext._parse_response(response, "chunk-1")
        assert rels == []

    def test_relationship_weight_clamped(self):
        ext = _make_extractor()
        response = '{"entities": [{"name": "Alice", "type": "person"}], "relationships": [{"source": "Alice", "target": "Bob", "type": "knows", "weight": 1.5}]}'
        entities, rels = ext._parse_response(response, "chunk-1")
        assert rels[0].weight == 1.0

        response2 = '{"entities": [{"name": "Alice", "type": "person"}], "relationships": [{"source": "Alice", "target": "Bob", "type": "knows", "weight": -0.5}]}'
        entities2, rels2 = ext._parse_response(response2, "chunk-1")
        assert rels2[0].weight == 0.0


class TestRelationshipSourceChunkIds:
    def test_batch_tags_relationships_with_chunk_ids(self):
        def good_llm(*a):
            return '{"entities": [{"name": "Alice", "type": "person"}, {"name": "Bob", "type": "person"}], "relationships": [{"source": "Alice", "target": "Bob", "type": "knows", "weight": 0.9}]}'
        ext = _make_extractor(good_llm)
        batch = [{"id": "c1", "content": "Alice knows Bob"}]
        entities, rels, chunk_ids, success = ext._call_llm_for_batch(batch)
        assert success is True
        assert len(rels) == 1
        assert "c1" in rels[0].source_chunk_ids
