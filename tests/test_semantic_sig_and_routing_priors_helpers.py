"""Tests for semantic_sig (_trigram_embed, cosine) and routing_priors pure helpers."""

import json
import time
import pytest
from pathlib import Path

from mullet.core.semantic_sig import _trigram_embed, cosine
from mullet.core.routing_history import make_signature
from mullet.core.routing_priors import pick_better_provider, summarize_priors


# ---------------------------------------------------------------------------
# _trigram_embed
# ---------------------------------------------------------------------------

class TestTrigramEmbed:
    def test_returns_list(self):
        result = _trigram_embed("hello world")
        assert isinstance(result, list)

    def test_length_256(self):
        result = _trigram_embed("hello world")
        assert len(result) == 256

    def test_elements_are_float(self):
        result = _trigram_embed("hello world")
        assert all(isinstance(x, float) for x in result)

    def test_unit_vector(self):
        result = _trigram_embed("hello world")
        norm_sq = sum(x * x for x in result)
        assert abs(norm_sq - 1.0) < 1e-9

    def test_deterministic(self):
        v1 = _trigram_embed("hello world")
        v2 = _trigram_embed("hello world")
        assert v1 == v2

    def test_different_texts_differ(self):
        v1 = _trigram_embed("hello world")
        v2 = _trigram_embed("completely different xyz text")
        assert v1 != v2

    def test_short_text_all_zeros(self):
        result = _trigram_embed("hi")
        assert all(x == 0.0 for x in result)

    def test_empty_text_all_zeros(self):
        result = _trigram_embed("")
        assert all(x == 0.0 for x in result)

    def test_short_text_length_256(self):
        result = _trigram_embed("ab")
        assert len(result) == 256

    def test_empty_text_length_256(self):
        result = _trigram_embed("")
        assert len(result) == 256

    def test_three_char_text_nonzero(self):
        result = _trigram_embed("abc")
        assert any(x != 0.0 for x in result)

    def test_case_matters(self):
        lower = _trigram_embed("hello")
        upper = _trigram_embed("HELLO")
        # Both lowercase the text internally so they should match
        assert lower == upper

    def test_longer_text_normalized(self):
        result = _trigram_embed("the quick brown fox jumps over the lazy dog")
        norm_sq = sum(x * x for x in result)
        assert abs(norm_sq - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# cosine
# ---------------------------------------------------------------------------

class TestCosine:
    def test_returns_float(self):
        assert isinstance(cosine([1.0, 0.0], [1.0, 0.0]), float)

    def test_identical_vectors_returns_one(self):
        assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_returns_zero(self):
        assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_clamped_to_zero(self):
        result = cosine([1.0, 0.0], [-1.0, 0.0])
        assert result == pytest.approx(0.0)

    def test_result_never_negative(self):
        result = cosine([1.0, 0.0], [-1.0, 0.0])
        assert result >= 0.0

    def test_result_never_above_one(self):
        result = cosine([0.6, 0.8], [0.6, 0.8])
        assert result <= 1.0

    def test_empty_lists_return_zero(self):
        assert cosine([], []) == 0.0

    def test_empty_a_returns_zero(self):
        assert cosine([], [1.0]) == 0.0

    def test_empty_b_returns_zero(self):
        assert cosine([1.0], []) == 0.0

    def test_mismatched_length_returns_zero(self):
        assert cosine([1.0, 0.0], [1.0]) == 0.0

    def test_zero_vector_a_returns_zero(self):
        assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_zero_vector_b_returns_zero(self):
        assert cosine([1.0, 0.0], [0.0, 0.0]) == 0.0

    def test_parallel_scaled_vectors(self):
        assert cosine([2.0, 0.0], [5.0, 0.0]) == pytest.approx(1.0)

    def test_trigram_self_similarity_is_one(self):
        v = _trigram_embed("test sentence here")
        assert cosine(v, v) == pytest.approx(1.0)

    def test_trigram_different_texts_below_one(self):
        v1 = _trigram_embed("apple pie recipe")
        v2 = _trigram_embed("database schema migration")
        result = cosine(v1, v2)
        assert result < 1.0

    def test_trigram_similar_texts_higher_than_different(self):
        base = _trigram_embed("python programming language")
        similar = _trigram_embed("python programming tutorial")
        different = _trigram_embed("database schema migration sql")
        sim_similar = cosine(base, similar)
        sim_different = cosine(base, different)
        assert sim_similar > sim_different


# ---------------------------------------------------------------------------
# make_signature (routing_history helper used by priors)
# ---------------------------------------------------------------------------

class TestMakeSignature:
    def test_returns_string(self):
        sig = make_signature("fix the bug", "run", "balanced")
        assert isinstance(sig, str)

    def test_nonempty(self):
        sig = make_signature("fix the bug", "run", "balanced")
        assert len(sig) > 0

    def test_deterministic(self):
        s1 = make_signature("fix the bug", "run", "balanced")
        s2 = make_signature("fix the bug", "run", "balanced")
        assert s1 == s2

    def test_different_tasks_differ(self):
        s1 = make_signature("task A", "run", "balanced")
        s2 = make_signature("task B", "run", "balanced")
        assert s1 != s2

    def test_different_tiers_differ(self):
        s1 = make_signature("task", "run", "balanced")
        s2 = make_signature("task", "run", "premium")
        assert s1 != s2


# ---------------------------------------------------------------------------
# Helpers for building a temp history file
# ---------------------------------------------------------------------------

def _make_history(base: Path, task: str, tier: str, records: list[tuple[str, bool]]) -> None:
    """Write routing history records to base/.mullet/routing_history.jsonl."""
    history_dir = base / ".mullet"
    history_dir.mkdir(exist_ok=True)
    sig = make_signature(task, "run", tier)
    lines = [
        json.dumps({
            "task_signature": sig,
            "provider": provider,
            "success": success,
            "ts": time.time(),
            "tier": tier,
            "task_preview": task,
        })
        for provider, success in records
    ]
    (history_dir / "routing_history.jsonl").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# pick_better_provider
# ---------------------------------------------------------------------------

class TestPickBetterProviderNoHistory:
    def test_returns_tuple(self, tmp_path):
        result = pick_better_provider("task", "balanced", "kimi", ["kimi", "gemini"], base=tmp_path)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_keeps_picked_when_no_history(self, tmp_path):
        provider, reason = pick_better_provider("task", "balanced", "kimi", ["kimi", "gemini"], base=tmp_path)
        assert provider == "kimi"

    def test_reason_is_none_when_no_history(self, tmp_path):
        _, reason = pick_better_provider("task", "balanced", "kimi", ["kimi", "gemini"], base=tmp_path)
        assert reason is None

    def test_single_candidate_no_history(self, tmp_path):
        provider, reason = pick_better_provider("task", "balanced", "kimi", ["kimi"], base=tmp_path)
        assert provider == "kimi"
        assert reason is None


class TestPickBetterProviderWithHistory:
    def test_keeps_picked_when_performing_well(self, tmp_path):
        # kimi succeeds 4/4 → keep it
        _make_history(tmp_path, "fix the bug", "balanced", [("kimi", True)] * 4)
        provider, reason = pick_better_provider(
            "fix the bug", "balanced", "kimi", ["kimi", "gemini"], base=tmp_path
        )
        assert provider == "kimi"
        assert reason is None

    def test_swaps_to_better_candidate(self, tmp_path):
        # kimi fails 3/3, gemini succeeds 3/3
        _make_history(tmp_path, "fix the bug", "balanced", [
            ("kimi", False), ("kimi", False), ("kimi", False),
            ("gemini", True), ("gemini", True), ("gemini", True),
        ])
        provider, reason = pick_better_provider(
            "fix the bug", "balanced", "kimi", ["kimi", "gemini"], base=tmp_path
        )
        assert provider == "gemini"
        assert reason is not None

    def test_reason_contains_rates(self, tmp_path):
        _make_history(tmp_path, "fix the bug", "balanced", [
            ("kimi", False), ("kimi", False), ("kimi", False),
            ("gemini", True), ("gemini", True), ("gemini", True),
        ])
        _, reason = pick_better_provider(
            "fix the bug", "balanced", "kimi", ["kimi", "gemini"], base=tmp_path
        )
        assert "kimi" in reason
        assert "gemini" in reason

    def test_no_swap_when_candidate_not_good_enough(self, tmp_path):
        # kimi fails, gemini only 50% (below good_enough 75%)
        _make_history(tmp_path, "fix the bug", "balanced", [
            ("kimi", False), ("kimi", False), ("kimi", False),
            ("gemini", True), ("gemini", False), ("gemini", False),
        ])
        provider, reason = pick_better_provider(
            "fix the bug", "balanced", "kimi", ["kimi", "gemini"], base=tmp_path
        )
        assert provider == "kimi"
        assert reason is None

    def test_keeps_picked_when_insufficient_samples(self, tmp_path):
        # Only 2 samples for kimi — below default min_samples=3
        _make_history(tmp_path, "fix the bug", "balanced", [
            ("kimi", False), ("kimi", False),
        ])
        provider, _ = pick_better_provider(
            "fix the bug", "balanced", "kimi", ["kimi", "gemini"], base=tmp_path
        )
        assert provider == "kimi"


# ---------------------------------------------------------------------------
# summarize_priors
# ---------------------------------------------------------------------------

class TestSummarizePriors:
    def test_returns_dict(self, tmp_path):
        result = summarize_priors("task", "balanced", base=tmp_path)
        assert isinstance(result, dict)

    def test_empty_history_returns_empty(self, tmp_path):
        result = summarize_priors("task", "balanced", base=tmp_path)
        assert result == {}

    def test_entries_have_tuple_values(self, tmp_path):
        _make_history(tmp_path, "task", "balanced", [("kimi", True)] * 3)
        result = summarize_priors("task", "balanced", base=tmp_path)
        for provider, val in result.items():
            assert isinstance(val, tuple)
            assert len(val) == 2

    def test_sample_count_correct(self, tmp_path):
        _make_history(tmp_path, "task", "balanced", [
            ("kimi", True), ("kimi", True), ("kimi", False),
        ])
        result = summarize_priors("task", "balanced", base=tmp_path)
        assert result["kimi"][1] == 3

    def test_success_rate_correct(self, tmp_path):
        _make_history(tmp_path, "task", "balanced", [
            ("kimi", True), ("kimi", True), ("kimi", False),
        ])
        result = summarize_priors("task", "balanced", base=tmp_path)
        rate, count = result["kimi"]
        assert rate == pytest.approx(2 / 3)
        assert count == 3

    def test_multiple_providers(self, tmp_path):
        _make_history(tmp_path, "task", "balanced", [
            ("kimi", True), ("kimi", True),
            ("gemini", False), ("gemini", False),
        ])
        result = summarize_priors("task", "balanced", base=tmp_path)
        assert "kimi" in result
        assert "gemini" in result

    def test_sorted_by_count_descending(self, tmp_path):
        _make_history(tmp_path, "task", "balanced", [
            ("kimi", True),
            ("gemini", True), ("gemini", True), ("gemini", True),
        ])
        result = summarize_priors("task", "balanced", base=tmp_path)
        counts = [v[1] for v in result.values()]
        assert counts == sorted(counts, reverse=True)

    def test_rate_none_below_min_samples(self, tmp_path):
        _make_history(tmp_path, "task", "balanced", [("kimi", True)])
        result = summarize_priors("task", "balanced", base=tmp_path, min_samples=3)
        rate, count = result["kimi"]
        assert rate is None
        assert count == 1

    def test_different_task_not_included(self, tmp_path):
        _make_history(tmp_path, "task A", "balanced", [("kimi", True)] * 3)
        result = summarize_priors("task B", "balanced", base=tmp_path)
        assert result == {}
