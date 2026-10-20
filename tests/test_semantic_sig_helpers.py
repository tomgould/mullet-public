"""Tests for mullet.core.semantic_sig pure helpers: _trigram_embed and cosine."""

import math
import pytest

from mullet.core.semantic_sig import _trigram_embed, cosine


# ---------------------------------------------------------------------------
# _trigram_embed
# ---------------------------------------------------------------------------

class TestTrigramEmbed:
    def test_returns_list(self):
        result = _trigram_embed("hello world")
        assert isinstance(result, list)

    def test_returns_256_dimensions(self):
        result = _trigram_embed("hello world")
        assert len(result) == 256

    def test_all_floats(self):
        result = _trigram_embed("test text")
        assert all(isinstance(x, float) for x in result)

    def test_unit_vector_for_non_trivial_text(self):
        result = _trigram_embed("the quick brown fox")
        norm = math.sqrt(sum(x * x for x in result))
        assert abs(norm - 1.0) < 1e-9

    def test_short_text_returns_zero_vector(self):
        # Text shorter than 3 chars: no trigrams, returns all-zero
        result = _trigram_embed("ab")
        assert all(x == 0.0 for x in result)

    def test_empty_string_returns_zero_vector(self):
        result = _trigram_embed("")
        assert all(x == 0.0 for x in result)

    def test_single_char_returns_zero_vector(self):
        result = _trigram_embed("x")
        assert all(x == 0.0 for x in result)

    def test_deterministic(self):
        text = "this is a deterministic test"
        r1 = _trigram_embed(text)
        r2 = _trigram_embed(text)
        assert r1 == r2

    def test_case_lowered(self):
        # Lowercased before embedding — same result for upper/lowercase
        r1 = _trigram_embed("Hello World")
        r2 = _trigram_embed("hello world")
        assert r1 == r2

    def test_different_texts_give_different_vectors(self):
        r1 = _trigram_embed("authentication with jwt tokens")
        r2 = _trigram_embed("database schema migration script")
        assert r1 != r2


# ---------------------------------------------------------------------------
# cosine
# ---------------------------------------------------------------------------

class TestCosine:
    def test_identical_vectors_return_one(self):
        v = [1.0, 0.0, 0.0]
        assert cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_clamped_to_zero(self):
        # cosine is clamped to [0, 1], so anti-parallel returns 0
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        result = cosine(a, b)
        assert result == pytest.approx(0.0)

    def test_empty_input_returns_zero(self):
        assert cosine([], []) == 0.0

    def test_length_mismatch_returns_zero(self):
        assert cosine([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0

    def test_zero_vector_returns_zero(self):
        assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_symmetric(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        assert cosine(a, b) == pytest.approx(cosine(b, a))

    def test_similar_texts_high_cosine(self):
        # Two similar texts should have higher cosine than two different ones
        va = _trigram_embed("find the authentication code")
        vb = _trigram_embed("find the authentication module")
        vc = _trigram_embed("database schema migration")
        sim_similar = cosine(va, vb)
        sim_different = cosine(va, vc)
        assert sim_similar > sim_different

    def test_returns_float(self):
        result = cosine([1.0, 0.0], [0.5, 0.5])
        assert isinstance(result, float)

    def test_clamped_to_zero_or_above(self):
        result = cosine([1.0, 0.0], [0.5, 0.5])
        assert result >= 0.0

    def test_clamped_to_one_or_below(self):
        result = cosine([1.0, 0.0], [0.5, 0.5])
        assert result <= 1.0
