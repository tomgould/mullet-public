"""Tests for small pure modules in mullet/knowledge/engine/."""
import time
import pytest


# ===========================================================================
# bm25_vocabulary
# ===========================================================================

from mullet.knowledge.engine.bm25_vocabulary import (
    load_bm25_vocabulary,
    save_bm25_vocabulary,
)

class TestBm25Vocabulary:
    def test_load_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            load_bm25_vocabulary("some/path")

    def test_save_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            save_bm25_vocabulary({}, "some/path")


# ===========================================================================
# cache (TTL decorator)
# ===========================================================================

from mullet.knowledge.engine.cache import (
    cached, _make_key, clear_all_caches, _CACHE
)

@pytest.fixture(autouse=True)
def clear_cache():
    """Clear global cache before and after each test."""
    clear_all_caches()
    yield
    clear_all_caches()


class TestCachedDecorator:
    def test_caches_return_value(self):
        call_count = 0

        @cached(ttl=60)
        def add(a, b):
            nonlocal call_count
            call_count += 1
            return a + b

        assert add(1, 2) == 3
        assert add(1, 2) == 3  # cached
        assert call_count == 1

    def test_different_args_cache_separately(self):
        call_count = 0

        @cached(ttl=60)
        def fn(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        assert fn(1) == 2
        assert fn(2) == 4
        assert call_count == 2

    def test_expired_entry_recalculated(self):
        call_count = 0

        @cached(ttl=0.01)
        def fn():
            nonlocal call_count
            call_count += 1
            return "value"

        fn()
        time.sleep(0.05)
        fn()
        assert call_count == 2

    def test_ttl_zero_disables_caching(self):
        call_count = 0

        @cached(ttl=0)
        def fn():
            nonlocal call_count
            call_count += 1
            return "v"

        fn()
        fn()
        assert call_count == 2

    def test_cache_clear_clears_entries(self):
        call_count = 0

        @cached(ttl=60)
        def fn():
            nonlocal call_count
            call_count += 1
            return "x"

        fn()
        fn.cache_clear()
        fn()
        assert call_count == 2

    def test_cache_info_returns_counts(self):
        @cached(ttl=60)
        def fn(x):
            return x

        fn(1)
        info = fn.cache_info()
        assert info["cached"] >= 1
        assert "expired" in info
        assert "total" in info

    def test_kwargs_cached_separately_from_positional(self):
        call_count = 0

        @cached(ttl=60)
        def fn(a, b=0):
            nonlocal call_count
            call_count += 1
            return a + b

        fn(1, b=2)
        fn(1, b=2)  # same args
        assert call_count == 1

    def test_preserves_function_name(self):
        @cached(ttl=60)
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_negative_ttl_disables_caching(self):
        call_count = 0

        @cached(ttl=-1)
        def fn():
            nonlocal call_count
            call_count += 1
            return "x"

        fn()
        fn()
        assert call_count == 2


class TestMakeKey:
    def test_same_args_same_key(self):
        k1 = _make_key("fn", (1, 2), {"x": 3})
        k2 = _make_key("fn", (1, 2), {"x": 3})
        assert k1 == k2

    def test_different_args_different_key(self):
        k1 = _make_key("fn", (1,), {})
        k2 = _make_key("fn", (2,), {})
        assert k1 != k2

    def test_json_fallback_on_non_serializable(self):
        # lambda is not JSON-serializable; should fall back without raising
        key = _make_key("fn", (lambda: None,), {})
        assert key.startswith("fn:")


class TestClearAllCaches:
    def test_clears_all_entries(self):
        _CACHE["test_key"] = (time.time() + 1000, "value")
        clear_all_caches()
        assert len(_CACHE) == 0


# ===========================================================================
# contextualize
# ===========================================================================

from mullet.knowledge.engine.search.contextualize import query_contextualize


class TestQueryContextualize:
    def test_empty_query_returns_as_is(self):
        assert query_contextualize("", ["prior context"]) == ""

    def test_empty_history_returns_query_as_is(self):
        assert query_contextualize("what about tests?", []) == "what about tests?"

    def test_standalone_query_returned_unchanged(self):
        result = query_contextualize(
            "How does Python handle memory?",
            ["What is recursion?"]
        )
        assert result == "How does Python handle memory?"

    def test_followup_what_about_expanded(self):
        result = query_contextualize(
            "what about the tests?",
            ["How does the authentication system work in Django?"]
        )
        assert "authentication" in result
        assert "tests" in result

    def test_followup_how_about_expanded(self):
        result = query_contextualize(
            "how about the database?",
            ["Explain the caching layer in this application"]
        )
        assert "caching" in result

    def test_followup_also_expanded(self):
        result = query_contextualize(
            "also what is the timeout?",
            ["How do I configure the connection pool?"]
        )
        assert "connection pool" in result

    def test_short_question_expanded(self):
        result = query_contextualize(
            "why?",
            ["Explain the benefits of using async/await in Python."]
        )
        assert "async" in result

    def test_previous_followup_skipped_for_context(self):
        """A prior follow-up in history should not be used as context."""
        result = query_contextualize(
            "what about performance?",
            ["also?", "Explain how the query optimizer works in PostgreSQL"]
        )
        assert "query optimizer" in result

    def test_no_substantive_prior_returns_query(self):
        """If all prior queries are short, returns original query."""
        result = query_contextualize("what about it?", ["why?", "ok?"])
        assert result == "what about it?"

    def test_case_insensitive_followup_detection(self):
        result = query_contextualize(
            "WHAT ABOUT the tests?",
            ["How does CI/CD pipeline configuration work?"]
        )
        assert "CI/CD" in result

    def test_that_prefix_expanded(self):
        result = query_contextualize(
            "that is confusing",
            ["Explain how the garbage collector works in Python"]
        )
        assert "garbage collector" in result


# ===========================================================================
# merge
# ===========================================================================

from mullet.knowledge.engine.search.merge import merge_search_results


class TestMergeSearchResults:
    def _chunk(self, chunk_id, rerank=None, combined=None):
        c = {"id": chunk_id, "content": f"content-{chunk_id}"}
        if rerank is not None:
            c["_rerank_score"] = rerank
        if combined is not None:
            c["_combined_score"] = combined
        return c

    def test_empty_inputs_returns_empty(self):
        assert merge_search_results([], []) == []

    def test_main_results_returned(self):
        r = merge_search_results([self._chunk("a", rerank=0.9)], [])
        assert len(r) == 1
        assert r[0]["id"] == "a"

    def test_conv_results_merged(self):
        r = merge_search_results([], [self._chunk("b", rerank=0.8)])
        assert len(r) == 1
        assert r[0]["id"] == "b"

    def test_deduplication_main_takes_priority(self):
        main = [self._chunk("a", rerank=0.9)]
        conv = [self._chunk("a", rerank=0.5)]
        r = merge_search_results(main, conv)
        assert len(r) == 1
        # Main version should be kept (original score)
        assert r[0]["_rerank_score"] == 0.9

    def test_conv_boost_applied_to_rerank_score(self):
        conv = [self._chunk("b", rerank=0.5)]
        r = merge_search_results([], conv, boost_conversation=0.2)
        assert r[0]["_rerank_score"] == pytest.approx(0.7)

    def test_conv_boost_applied_to_combined_score(self):
        conv = [self._chunk("b", combined=0.4)]
        r = merge_search_results([], conv, boost_conversation=0.1)
        assert r[0]["_combined_score"] == pytest.approx(0.5)

    def test_conv_results_marked_from_conversation(self):
        conv = [self._chunk("b", rerank=0.5)]
        r = merge_search_results([], conv)
        assert r[0]["_from_conversation"] is True

    def test_main_results_not_marked_from_conversation(self):
        main = [self._chunk("a", rerank=0.9)]
        r = merge_search_results(main, [])
        assert "_from_conversation" not in r[0]

    def test_sorted_by_rerank_score_descending(self):
        main = [self._chunk("a", rerank=0.5), self._chunk("b", rerank=0.9)]
        r = merge_search_results(main, [])
        assert r[0]["id"] == "b"  # highest first

    def test_sorted_by_combined_score_as_fallback(self):
        main = [self._chunk("a", combined=0.3), self._chunk("b", combined=0.7)]
        r = merge_search_results(main, [])
        assert r[0]["id"] == "b"

    def test_chunks_without_id_skipped(self):
        main = [{"content": "no id here"}]
        r = merge_search_results(main, [])
        assert r == []

    def test_mixed_scores_sorted_correctly(self):
        main = [self._chunk("a", rerank=0.3)]
        conv = [self._chunk("b", rerank=0.5)]
        r = merge_search_results(main, conv, boost_conversation=0.2)
        # b gets 0.5 + 0.2 = 0.7, a stays 0.3
        assert r[0]["id"] == "b"


# ===========================================================================
# search/cache (SearchCache)
# ===========================================================================

from mullet.knowledge.engine.search.cache import SearchCache


class TestSearchCache:
    def test_miss_returns_none(self):
        sc = SearchCache()
        assert sc.get("nonexistent") is None

    def test_put_and_get(self):
        sc = SearchCache()
        sc.put("key", "value")
        assert sc.get("key") == "value"

    def test_expired_entry_returns_none(self):
        sc = SearchCache(ttl=0.01)
        sc.put("key", "value")
        time.sleep(0.05)
        assert sc.get("key") is None

    def test_lru_eviction(self):
        sc = SearchCache(max_size=2)
        sc.put("a", 1)
        sc.put("b", 2)
        sc.put("c", 3)  # evicts "a"
        assert sc.get("a") is None
        assert sc.get("b") == 2
        assert sc.get("c") == 3

    def test_access_updates_lru_order(self):
        sc = SearchCache(max_size=2)
        sc.put("a", 1)
        sc.put("b", 2)
        sc.get("a")  # "a" is now most recently used
        sc.put("c", 3)  # evicts "b" (least recently used)
        assert sc.get("b") is None
        assert sc.get("a") == 1

    def test_clear_empties_cache(self):
        sc = SearchCache()
        sc.put("key", "value")
        sc.clear()
        assert sc.size == 0
        assert sc.get("key") is None

    def test_size_property(self):
        sc = SearchCache()
        assert sc.size == 0
        sc.put("a", 1)
        sc.put("b", 2)
        assert sc.size == 2

    def test_overwrite_existing_key(self):
        sc = SearchCache()
        sc.put("key", "old")
        sc.put("key", "new")
        assert sc.get("key") == "new"
        assert sc.size == 1

    def test_large_cache_does_not_exceed_max_size(self):
        sc = SearchCache(max_size=5)
        for i in range(10):
            sc.put(str(i), i)
        assert sc.size == 5
