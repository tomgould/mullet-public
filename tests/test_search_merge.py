from mullet.knowledge.engine.search.merge import merge_search_results


class TestMergeSearchResults:
    def test_empty_inputs(self):
        assert merge_search_results([], []) == []

    def test_main_only(self):
        main = [{"id": "a", "_rerank_score": 1.0}]
        result = merge_search_results(main, [])
        assert len(result) == 1
        assert result[0]["id"] == "a"

    def test_conv_only(self):
        conv = [{"id": "b", "_rerank_score": 0.8}]
        result = merge_search_results([], conv)
        assert len(result) == 1
        assert result[0]["id"] == "b"
        assert result[0].get("_from_conversation") is True

    def test_deduplication(self):
        main = [{"id": "x", "_rerank_score": 1.0}]
        conv = [{"id": "x", "_rerank_score": 0.5}]
        result = merge_search_results(main, conv)
        assert len(result) == 1
        assert result[0]["_rerank_score"] == 1.0
        assert "_from_conversation" not in result[0]

    def test_conv_boost_applied(self):
        conv = [{"id": "c", "_rerank_score": 0.5}]
        result = merge_search_results([], conv, boost_conversation=0.2)
        assert result[0]["_rerank_score"] == 0.7

    def test_conv_boost_on_combined_score(self):
        conv = [{"id": "c", "_combined_score": 0.4}]
        result = merge_search_results([], conv, boost_conversation=0.1)
        assert result[0]["_combined_score"] == 0.5

    def test_sorting_by_rerank_score(self):
        main = [
            {"id": "low", "_rerank_score": 0.3},
            {"id": "high", "_rerank_score": 0.9},
        ]
        result = merge_search_results(main, [])
        assert [r["id"] for r in result] == ["high", "low"]

    def test_sorting_fallback_to_combined_score(self):
        main = [
            {"id": "low", "_combined_score": 0.2},
            {"id": "high", "_combined_score": 0.8},
        ]
        result = merge_search_results(main, [])
        assert [r["id"] for r in result] == ["high", "low"]

    def test_sorting_default_zero(self):
        main = [
            {"id": "a"},
            {"id": "b", "_rerank_score": 0.5},
        ]
        result = merge_search_results(main, [])
        assert [r["id"] for r in result] == ["b", "a"]

    def test_mixed_main_and_conv(self):
        main = [{"id": "m1", "_rerank_score": 0.9}]
        conv = [{"id": "c1", "_rerank_score": 0.95}]
        result = merge_search_results(main, conv)
        assert [r["id"] for r in result] == ["c1", "m1"]
