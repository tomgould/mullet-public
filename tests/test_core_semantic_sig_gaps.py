"""Gap tests for mullet.core.semantic_sig — targets uncovered lines."""

import json
import math
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import mullet.core.semantic_sig as ss_mod
from mullet.core.semantic_sig import embed, cosine, nearest_signatures, _trigram_embed


# ---------------------------------------------------------------------------
# _get_st_model uncovered lines 44-47
# ---------------------------------------------------------------------------

def test_get_st_model_import_failure_sets_unavailable():
    """Lines 44-47: exception during ST load marks _ST_AVAILABLE=False."""
    orig_avail = ss_mod._ST_AVAILABLE
    orig_model = ss_mod._ST_MODEL
    try:
        ss_mod._ST_AVAILABLE = None
        ss_mod._ST_MODEL = None
        # Patch the class so instantiation raises, triggering the except block
        with patch("sentence_transformers.SentenceTransformer", side_effect=RuntimeError("fail")):
            result = ss_mod._get_st_model()
        assert result is None
        assert ss_mod._ST_AVAILABLE is False
    finally:
        ss_mod._ST_AVAILABLE = orig_avail
        ss_mod._ST_MODEL = orig_model


# ---------------------------------------------------------------------------
# embed uncovered lines 93-94 and 99-101
# ---------------------------------------------------------------------------

def test_embed_st_encode_exception_falls_back_to_trigram():
    """Lines 93-94: ST encode raises → falls back to trigram."""
    mock_model = MagicMock()
    mock_model.encode.side_effect = RuntimeError("CUDA OOM")
    with patch.object(ss_mod, "_ST_AVAILABLE", None):
        with patch.object(ss_mod, "_ST_MODEL", mock_model):
            result = embed("hello world")
    assert result is not None
    assert len(result) == 256


def test_embed_trigram_also_fails_returns_none():
    """Lines 99-101: even trigram fallback raises → returns None."""
    with patch.object(ss_mod, "_trigram_embed", side_effect=MemoryError("boom")):
        with patch.object(ss_mod, "_ST_AVAILABLE", False):
            with patch.object(ss_mod, "_ST_MODEL", None):
                result = embed("hello")
    assert result is None


# ---------------------------------------------------------------------------
# _build_cache uncovered lines 146, 157-158
# ---------------------------------------------------------------------------

def test_build_cache_skips_blank_lines():
    """Line 146: blank lines in history file are skipped."""
    tmp = Path("/tmp") / "test_semantic_sig_blank"
    tmp.mkdir(parents=True, exist_ok=True)
    hist = tmp / "routing_history.jsonl"
    hist.write_text(
        '\n'
        + json.dumps({"task_signature": "s1", "task_preview": "hello"}) + '\n'
        + '   \n'
    )
    cache = ss_mod._build_cache(hist)
    assert "s1" in cache.entries


def test_build_cache_oserror_on_read():
    """Lines 157-158: OSError reading history file is logged, returns empty entries."""
    tmp = Path("/tmp") / "test_semantic_sig_oserr"
    tmp.mkdir(parents=True, exist_ok=True)
    hist = tmp / "routing_history.jsonl"
    hist.write_text("{}")
    with patch.object(Path, "read_text", side_effect=OSError("permission denied")):
        cache = ss_mod._build_cache(hist)
    assert cache.entries == {}


# ---------------------------------------------------------------------------
# nearest_signatures uncovered lines 201, 213-215
# ---------------------------------------------------------------------------

def test_nearest_signatures_returns_empty_when_embed_none():
    """Line 201: embed returns None → nearest_signatures returns []."""
    with patch.object(ss_mod, "embed", return_value=None):
        result = nearest_signatures("anything")
    assert result == []


def test_nearest_signatures_exception_returns_empty():
    """Lines 213-215: any exception → returns [] and does not raise."""
    with patch.object(ss_mod, "embed", side_effect=RuntimeError("unexpected")):
        result = nearest_signatures("anything")
    assert result == []


# ---------------------------------------------------------------------------
# Additional coverage helpers to push ≥92%
# ---------------------------------------------------------------------------

def test_cosine_clamps_above_one():
    """Artificial vectors with dot > norm product should be clamped to 1.0."""
    a = [1.0, 1.0]
    b = [1.0, 1.0]
    assert cosine(a, b) == pytest.approx(1.0)


def test_trigram_embed_short_text_all_zeros():
    """Text shorter than 3 chars yields all-zero vector (norm stays 0)."""
    vec = _trigram_embed("ab")
    assert all(v == 0.0 for v in vec)
    assert cosine(vec, vec) == 0.0


def test_cosine_empty_vectors_returns_zero():
    """Line 114: empty vectors return 0.0."""
    assert cosine([], []) == 0.0
    assert cosine([1.0], []) == 0.0
    assert cosine([], [1.0]) == 0.0


def test_build_cache_corrupt_json_line():
    """Lines 155-156: corrupt JSON line is skipped, valid line processed."""
    tmp = Path("/tmp") / "test_semantic_sig_corrupt"
    tmp.mkdir(parents=True, exist_ok=True)
    hist = tmp / "routing_history.jsonl"
    hist.write_text(
        'not json\n'
        + json.dumps({"task_signature": "s1", "task_preview": "hello"}) + '\n'
    )
    cache = ss_mod._build_cache(hist)
    assert "s1" in cache.entries


def test_nearest_signatures_no_history_file():
    """Line 192: history file does not exist → returns []."""
    result = nearest_signatures("anything", base=Path("/tmp/nonexistent_mullet_dir"))
    assert result == []


def test_nearest_signatures_with_matches():
    """Lines 203-211: nearest_signatures scores, sorts, and limits results."""
    tmp = Path("/tmp") / "test_semantic_sig_matches"
    tmp.mkdir(parents=True, exist_ok=True)
    mullet_dir = tmp / ".mullet"
    mullet_dir.mkdir(parents=True, exist_ok=True)
    hist = mullet_dir / "routing_history.jsonl"
    hist.write_text(
        json.dumps({"task_signature": "sig_a", "task_preview": "hello world"}) + '\n'
        + json.dumps({"task_signature": "sig_b", "task_preview": "foo bar baz"}) + '\n'
    )
    ss_mod._embed_cache = None
    result = nearest_signatures("hello world", threshold=0.0, limit=1, base=tmp)
    assert len(result) >= 1
    assert result[0][0] == "sig_a"
    assert result[0][1] >= 0.0
