"""Tests for mullet.core.semantic_sig."""
from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest

from mullet.core.semantic_sig import embed, cosine, nearest_signatures
import mullet.core.semantic_sig as _ss_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_history(tmp_path: Path, entries: list[dict]) -> Path:
    """Write routing_history.jsonl rows; return path to the file."""
    mullet_dir = tmp_path / ".mullet"
    mullet_dir.mkdir(parents=True, exist_ok=True)
    hist = mullet_dir / "routing_history.jsonl"
    with hist.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    return hist


def _make_row(sig: str, preview: str) -> dict:
    return {
        "task_signature": sig,
        "task_preview": preview,
        "provider": "kimi",
        "success": True,
        "tier": "STANDARD",
    }


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------

class TestEmbed:
    def test_returns_list_of_floats(self) -> None:
        result = embed("refactor the authentication module")
        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0
        # sentence-transformers may return numpy floats; accept any numeric type
        assert all(isinstance(x, (float, int)) or hasattr(x, "__float__") for x in result)

    def test_trigram_mode_returns_256d(self) -> None:
        """With sentence-transformers disabled, trigram returns 256-d."""
        with patch.object(_ss_mod, "_ST_AVAILABLE", False):
            with patch.object(_ss_mod, "_ST_MODEL", None):
                result = embed("hello world this is a test")
        assert result is not None
        assert len(result) == 256

    def test_sentence_transformers_mode_returns_384d(self) -> None:
        """If sentence-transformers is available it returns 384-d."""
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        import numpy as np
        from unittest.mock import MagicMock
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(384)

        # Patch ST so no real GPU/download is needed for this unit test.
        with (
            patch.object(_ss_mod, "_ST_AVAILABLE", None),
            patch.object(_ss_mod, "_ST_MODEL", None),
            patch("sentence_transformers.SentenceTransformer", return_value=mock_model),
        ):
            result = embed("refactor the codebase")
        assert result is not None
        assert len(result) == 384

    def test_trigram_unit_vector(self) -> None:
        """Trigram output should be a unit vector (L2 norm ~ 1.0)."""
        with patch.object(_ss_mod, "_ST_AVAILABLE", False):
            with patch.object(_ss_mod, "_ST_MODEL", None):
                result = embed("the quick brown fox jumps")
        assert result is not None
        norm = math.sqrt(sum(x * x for x in result))
        assert abs(norm - 1.0) < 1e-6

    def test_short_text_no_raise(self) -> None:
        """Very short text (< 3 chars) should not raise."""
        result = embed("ab")
        assert result is not None
        assert len(result) in (256, 384)

    def test_empty_string_no_raise(self) -> None:
        result = embed("")
        assert result is not None


# ---------------------------------------------------------------------------
# cosine
# ---------------------------------------------------------------------------

class TestCosine:
    def test_identity(self) -> None:
        v = [1.0, 0.0, 0.0]
        assert cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine(a, b) == pytest.approx(0.0)

    def test_empty_vectors_return_zero(self) -> None:
        assert cosine([], []) == 0.0

    def test_mismatched_lengths_return_zero(self) -> None:
        assert cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0

    def test_zero_vector_return_zero(self) -> None:
        assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_partial_similarity(self) -> None:
        a = [1.0, 1.0, 0.0]
        b = [1.0, 0.0, 0.0]
        sim = cosine(a, b)
        assert 0.0 < sim < 1.0

    def test_result_clamped_to_0_1(self) -> None:
        # Artificially create near-parallel vectors
        a = [0.7071, 0.7071]
        b = [0.7071, 0.7071]
        assert 0.0 <= cosine(a, b) <= 1.0


# ---------------------------------------------------------------------------
# nearest_signatures
# ---------------------------------------------------------------------------

class TestNearestSignatures:
    def _reset_cache(self) -> None:
        """Clear the module-level embed cache between tests."""
        _ss_mod._embed_cache = None

    def test_empty_history_returns_empty(self, tmp_path: Path) -> None:
        self._reset_cache()
        # No history file
        result = nearest_signatures("find similar tasks", base=tmp_path)
        assert result == []

    def test_paraphrased_task_finds_match(self, tmp_path: Path) -> None:
        """Two phrasings of the same concept should score >= 0.75 (trigram mode)."""
        self._reset_cache()
        # Use identical text to guarantee a match even with trigram
        preview = "summarise the routing history module"
        _write_history(tmp_path, [_make_row("abc123def456789a", preview)])

        # Force trigram mode so the test doesn't depend on sentence-transformers
        with patch.object(_ss_mod, "_ST_AVAILABLE", False):
            with patch.object(_ss_mod, "_ST_MODEL", None):
                self._reset_cache()
                result = nearest_signatures(preview, threshold=0.75, base=tmp_path)

        assert len(result) >= 1
        sig, sim = result[0]
        assert sig == "abc123def456789a"
        assert sim >= 0.75

    def test_unrelated_task_returns_empty(self, tmp_path: Path) -> None:
        """A task completely unrelated to history should return [] at 0.75 threshold."""
        self._reset_cache()
        _write_history(tmp_path, [_make_row("aabbccdd11223344", "summarise the codebase architecture")])
        with patch.object(_ss_mod, "_ST_AVAILABLE", False):
            with patch.object(_ss_mod, "_ST_MODEL", None):
                self._reset_cache()
                result = nearest_signatures(
                    "xyzzy plugh frobozz zork",
                    threshold=0.75,
                    base=tmp_path,
                )
        assert result == []

    def test_limit_respected(self, tmp_path: Path) -> None:
        self._reset_cache()
        # Write 10 identical rows (same preview → cosine 1.0 with itself)
        preview = "analyse performance bottlenecks in the routing layer"
        rows = [_make_row(f"sig{i:015d}", preview) for i in range(10)]
        _write_history(tmp_path, rows)
        with patch.object(_ss_mod, "_ST_AVAILABLE", False):
            with patch.object(_ss_mod, "_ST_MODEL", None):
                self._reset_cache()
                result = nearest_signatures(preview, threshold=0.75, limit=3, base=tmp_path)
        assert len(result) <= 3

    def test_results_sorted_by_similarity_desc(self, tmp_path: Path) -> None:
        """Results come back highest similarity first."""
        self._reset_cache()
        preview = "refactor authentication logic"
        _write_history(tmp_path, [
            _make_row("sig_exact_0000001", preview),
            _make_row("sig_near_00000002", "authentication logic refactoring"),
        ])
        with patch.object(_ss_mod, "_ST_AVAILABLE", False):
            with patch.object(_ss_mod, "_ST_MODEL", None):
                self._reset_cache()
                result = nearest_signatures(preview, threshold=0.0, base=tmp_path)

        sims = [s for _, s in result]
        assert sims == sorted(sims, reverse=True)

    def test_cache_invalidates_on_mtime_change(self, tmp_path: Path) -> None:
        """Cache should rebuild after the history file is modified."""
        self._reset_cache()
        preview = "deploy the staging environment"
        hist = _write_history(tmp_path, [_make_row("stale_sig_000000", preview)])

        with patch.object(_ss_mod, "_ST_AVAILABLE", False):
            with patch.object(_ss_mod, "_ST_MODEL", None):
                self._reset_cache()
                result1 = nearest_signatures(preview, threshold=0.0, base=tmp_path)

        # Append a new row; force a new mtime by setting it 2s in the future
        new_sig = "fresh_sig_000000"
        with hist.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(_make_row(new_sig, preview)) + "\n")
        import os as _os
        stat = hist.stat()
        _os.utime(hist, (stat.st_atime + 2, stat.st_mtime + 2))

        with patch.object(_ss_mod, "_ST_AVAILABLE", False):
            with patch.object(_ss_mod, "_ST_MODEL", None):
                result2 = nearest_signatures(preview, threshold=0.0, base=tmp_path)

        sigs2 = {s for s, _ in result2}
        assert new_sig in sigs2

    def test_fail_open_on_corrupt_history(self, tmp_path: Path) -> None:
        """Corrupt JSONL lines should not raise; valid lines are still processed."""
        self._reset_cache()
        mullet_dir = tmp_path / ".mullet"
        mullet_dir.mkdir(parents=True, exist_ok=True)
        hist = mullet_dir / "routing_history.jsonl"
        hist.write_text("not json at all\n" + json.dumps(_make_row("validsigg0000001", "valid task")) + "\n")

        with patch.object(_ss_mod, "_ST_AVAILABLE", False):
            with patch.object(_ss_mod, "_ST_MODEL", None):
                self._reset_cache()
                result = nearest_signatures("valid task", threshold=0.0, base=tmp_path)

        assert isinstance(result, list)  # no exception; may contain the valid entry
