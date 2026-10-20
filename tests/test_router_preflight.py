"""Tests for mullet/core/router/preflight.py targeting >=85%% coverage."""

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mullet.core.router.complexity import Tier
from mullet.core.router.preflight import (
    PreflightResult,
    _cache_key,
    _heuristic,
    _load_cache,
    _parse_meta_json,
    _save_to_cache,
    _try_gemini_flash,
    _try_local_llm,
    analyze_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.content = content
    return resp


# ---------------------------------------------------------------------------
# Cache isolation fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Redirect the preflight module-level cache to a temp dir and clear in-memory cache."""
    import mullet.core.router.preflight as pf_module

    fake_cache_path = tmp_path / "cache" / "preflight.jsonl"
    monkeypatch.setattr(pf_module, "_CACHE_PATH", fake_cache_path)
    monkeypatch.setattr(pf_module, "_CACHE", {})
    yield


# ---------------------------------------------------------------------------
# _load_cache
# ---------------------------------------------------------------------------


class TestLoadCache:
    def test_no_file_returns_empty(self, tmp_path, monkeypatch):
        """Line 45: _load_cache returns {} when cache file does not exist."""
        import mullet.core.router.preflight as pf

        fake = tmp_path / "no_preflight.jsonl"
        monkeypatch.setattr(pf, "_CACHE_PATH", fake)
        assert _load_cache() == {}

    def test_blank_line_skipped(self, tmp_path, monkeypatch):
        """Line 52: blank lines in cache file are skipped."""
        import mullet.core.router.preflight as pf

        fake = tmp_path / "cache.jsonl"
        entry = {
            "key": "k1",
            "result": {
                "input_tokens": 1,
                "expected_output_tokens": 512,
                "complexity_tier": "LOW",
                "needs_reasoning": False,
                "confidence": 0.9,
                "analyzer": "fast_path",
                "latency_ms": 0,
            },
            "written_at": int(time.time()),
        }
        fake.write_text("\n\n" + json.dumps(entry) + "\n")
        monkeypatch.setattr(pf, "_CACHE_PATH", fake)
        result = _load_cache()
        assert "k1" in result
        assert len(result) == 1

    def test_stale_entry_skipped(self, tmp_path, monkeypatch):
        """Line 56: stale entries are dropped."""
        import mullet.core.router.preflight as pf

        fake = tmp_path / "cache.jsonl"
        entry = {
            "key": "k1",
            "result": {
                "input_tokens": 1,
                "expected_output_tokens": 512,
                "complexity_tier": "LOW",
                "needs_reasoning": False,
                "confidence": 0.9,
                "analyzer": "fast_path",
                "latency_ms": 0,
            },
            "written_at": int(time.time() - 8 * 24 * 3600),
        }
        fake.write_text(json.dumps(entry) + "\n")
        monkeypatch.setattr(pf, "_CACHE_PATH", fake)
        assert "k1" not in _load_cache()

    def test_bad_json_line_skipped(self, tmp_path, monkeypatch):
        """Lines 68-69: bad JSON lines are silently skipped."""
        import mullet.core.router.preflight as pf

        fake = tmp_path / "cache.jsonl"
        entry = {
            "key": "k1",
            "result": {
                "input_tokens": 1,
                "expected_output_tokens": 512,
                "complexity_tier": "LOW",
                "needs_reasoning": False,
                "confidence": 0.9,
                "analyzer": "fast_path",
                "latency_ms": 0,
            },
            "written_at": int(time.time()),
        }
        fake.write_text("{ bad json }\n" + json.dumps(entry) + "\n")
        monkeypatch.setattr(pf, "_CACHE_PATH", fake)
        result = _load_cache()
        assert "k1" in result
        assert len(result) == 1

    def test_file_read_exception_logs_debug(self, tmp_path, monkeypatch, caplog):
        """Lines 70-71: file read exception is logged at DEBUG."""
        import mullet.core.router.preflight as pf

        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.open.side_effect = OSError("boom")
        monkeypatch.setattr(pf, "_CACHE_PATH", mock_path)
        with caplog.at_level("DEBUG", logger="mullet.core.router.preflight"):
            result = _load_cache()
        assert result == {}
        assert "Failed to load preflight cache" in caplog.text

    def test_valid_entry_loaded(self, tmp_path, monkeypatch):
        """Valid cache entries are deserialized with cached=True."""
        import mullet.core.router.preflight as pf

        fake = tmp_path / "cache.jsonl"
        entry = {
            "key": "k1",
            "result": {
                "input_tokens": 1,
                "expected_output_tokens": 512,
                "complexity_tier": "LOW",
                "needs_reasoning": False,
                "confidence": 0.9,
                "analyzer": "fast_path",
                "latency_ms": 0,
            },
            "written_at": int(time.time()),
        }
        fake.write_text(json.dumps(entry) + "\n")
        monkeypatch.setattr(pf, "_CACHE_PATH", fake)
        result = _load_cache()
        assert "k1" in result
        assert result["k1"].cached is True


# ---------------------------------------------------------------------------
# _save_to_cache
# ---------------------------------------------------------------------------


class TestSaveToCache:
    def test_save_appends_and_rewrites_stale(self, tmp_path, monkeypatch):
        """Lines 80-132: _save_to_cache writes, appends, and rewrites stale entries."""
        import mullet.core.router.preflight as pf

        fake = tmp_path / "cache.jsonl"
        monkeypatch.setattr(pf, "_CACHE_PATH", fake)
        monkeypatch.setattr(pf, "_CACHE", {})

        result = PreflightResult(
            input_tokens=10,
            expected_output_tokens=512,
            complexity_tier=Tier.LOW,
            needs_reasoning=False,
            confidence=0.9,
            analyzer="fast_path",
            latency_ms=0,
            cached=False,
        )
        _save_to_cache("key1", result)
        assert fake.exists()

        # Second save should append and trigger stale rewrite
        stale_result = PreflightResult(
            input_tokens=10,
            expected_output_tokens=512,
            complexity_tier=Tier.LOW,
            needs_reasoning=False,
            confidence=0.9,
            analyzer="fast_path",
            latency_ms=0,
            cached=False,
        )
        object.__setattr__(stale_result, "_cached_written_at", 0)
        monkeypatch.setattr(pf, "_CACHE", {"old": stale_result})
        _save_to_cache("key2", result)
        text = fake.read_text()
        assert "key1" in text or "key2" in text

    def test_save_append_exception_logs_debug(self, tmp_path, monkeypatch, caplog):
        """Lines 100-101: exception during append is logged."""
        import mullet.core.router.preflight as pf

        fake = tmp_path / "cache.jsonl"
        monkeypatch.setattr(pf, "_CACHE_PATH", fake)
        monkeypatch.setattr(pf, "_CACHE", {})
        result = PreflightResult(
            input_tokens=10,
            expected_output_tokens=512,
            complexity_tier=Tier.LOW,
            needs_reasoning=False,
            confidence=0.9,
            analyzer="fast_path",
            latency_ms=0,
            cached=False,
        )
        with patch(
            "mullet.core.router.preflight.os.replace", side_effect=OSError("replace failed")
        ):
            with caplog.at_level("DEBUG", logger="mullet.core.router.preflight"):
                _save_to_cache("key1", result)
        assert "Failed to append to preflight cache" in caplog.text

    def test_save_rewrite_exception_logs_debug(self, tmp_path, monkeypatch, caplog):
        """Lines 131-132: exception during rewrite is logged."""
        import mullet.core.router.preflight as pf

        fake = tmp_path / "cache.jsonl"
        monkeypatch.setattr(pf, "_CACHE_PATH", fake)
        stale = MagicMock()
        stale.input_tokens = 1
        stale.expected_output_tokens = 512
        stale.complexity_tier.value = "LOW"
        stale.needs_reasoning = False
        stale.confidence = 0.9
        stale.analyzer = "fast_path"
        stale.latency_ms = 0
        stale._cached_written_at = 0
        monkeypatch.setattr(pf, "_CACHE", {"old": stale})
        result = PreflightResult(
            input_tokens=1,
            expected_output_tokens=512,
            complexity_tier=Tier.LOW,
            needs_reasoning=False,
            confidence=0.9,
            analyzer="fast_path",
            latency_ms=0,
            cached=False,
        )
        with patch(
            "mullet.core.router.preflight.os.replace",
            side_effect=[None, OSError("rewrite failed")],
        ):
            with caplog.at_level("DEBUG", logger="mullet.core.router.preflight"):
                _save_to_cache("key1", result)
        assert "Failed to rewrite preflight cache" in caplog.text


# ---------------------------------------------------------------------------
# _cache_key
# ---------------------------------------------------------------------------


def test_cache_key_deterministic():
    """Lines 136-137: _cache_key is deterministic for same inputs."""
    key1 = _cache_key("task", ["b.py", "a.py"], "agent")
    key2 = _cache_key("task", ["a.py", "b.py"], "agent")
    assert key1 == key2
    assert len(key1) == 64  # sha256 hex


def test_cache_key_empty_files_and_none_agent():
    key = _cache_key("task", [], None)
    assert isinstance(key, str)
    assert len(key) == 64


# ---------------------------------------------------------------------------
# _parse_meta_json
# ---------------------------------------------------------------------------


class TestParseMetaJson:
    def test_valid_json(self):
        text = (
            '{"expected_output_tokens": 1000, "complexity": "STANDARD",'
            ' "needs_reasoning": false, "confidence": 0.9}'
        )
        result = _parse_meta_json(text)
        assert result is not None
        assert result["complexity"] == "STANDARD"
        assert result["confidence"] == 0.9

    def test_no_json_returns_none(self):
        assert _parse_meta_json("no json here") is None

    def test_malformed_json_returns_none(self):
        assert _parse_meta_json("{ bad json }") is None

    def test_json_embedded_in_prose(self):
        text = (
            'Here is the analysis: {"complexity": "ADVANCED",'
            ' "expected_output_tokens": 200, "needs_reasoning": true, "confidence": 0.8}'
        )
        result = _parse_meta_json(text)
        assert result is not None
        assert result["complexity"] == "ADVANCED"
        assert result["needs_reasoning"] is True

    def test_invalid_complexity_defaults_to_standard(self):
        text = '{"complexity": "BOGUS", "expected_output_tokens": 100}'
        result = _parse_meta_json(text)
        assert result["complexity"] == "STANDARD"

    def test_output_tokens_clamped_to_min(self):
        text = '{"expected_output_tokens": 1}'
        result = _parse_meta_json(text)
        assert result["expected_output_tokens"] == 50

    def test_output_tokens_clamped_to_max(self):
        text = '{"expected_output_tokens": 999999}'
        result = _parse_meta_json(text)
        assert result["expected_output_tokens"] == 32000

    def test_output_tokens_non_numeric_defaults_to_512(self):
        text = '{"expected_output_tokens": "lots"}'
        result = _parse_meta_json(text)
        assert result["expected_output_tokens"] == 512

    def test_needs_reasoning_string_true(self):
        text = '{"needs_reasoning": "true"}'
        result = _parse_meta_json(text)
        assert result["needs_reasoning"] is True

    def test_needs_reasoning_string_yes(self):
        text = '{"needs_reasoning": "yes"}'
        result = _parse_meta_json(text)
        assert result["needs_reasoning"] is True

    def test_needs_reasoning_string_false(self):
        text = '{"needs_reasoning": "false"}'
        result = _parse_meta_json(text)
        assert result["needs_reasoning"] is False

    def test_confidence_clamped_below_zero(self):
        text = '{"confidence": -0.5}'
        result = _parse_meta_json(text)
        assert result["confidence"] == 0.0

    def test_confidence_clamped_above_one(self):
        text = '{"confidence": 2.5}'
        result = _parse_meta_json(text)
        assert result["confidence"] == 1.0

    def test_missing_fields_use_defaults(self):
        text = "{}"
        result = _parse_meta_json(text)
        assert result["expected_output_tokens"] == 512
        assert result["complexity"] == "STANDARD"
        assert result["needs_reasoning"] is False
        assert result["confidence"] == 0.5

    def test_needs_reasoning_numeric_one(self):
        text = '{"needs_reasoning": 1}'
        result = _parse_meta_json(text)
        assert result["needs_reasoning"] is True

    def test_needs_reasoning_numeric_zero(self):
        text = '{"needs_reasoning": 0}'
        result = _parse_meta_json(text)
        assert result["needs_reasoning"] is False

    def test_invalid_confidence_type_defaults(self):
        text = '{"confidence": "not-a-number"}'
        result = _parse_meta_json(text)
        assert result["confidence"] == 0.5


# ---------------------------------------------------------------------------
# analyze_prompt
# ---------------------------------------------------------------------------


class TestAnalyzePrompt:
    def test_cache_hit(self, monkeypatch):
        """Lines 196-207: cache hit returns cached result with cached=True."""
        import mullet.core.router.preflight as pf

        cached = PreflightResult(
            input_tokens=10,
            expected_output_tokens=512,
            complexity_tier=Tier.STANDARD,
            needs_reasoning=False,
            confidence=0.9,
            analyzer="local_llm",
            latency_ms=50,
            cached=True,
        )
        monkeypatch.setattr(pf, "_CACHE", {"abc": cached})
        with patch("mullet.core.router.preflight._cache_key", return_value="abc"):
            result = analyze_prompt("any task")
        assert result.cached is True
        assert result.analyzer == "local_llm"

    def test_fast_path(self):
        """Lines 209-221: simple messages take the fast path."""
        result = analyze_prompt("list files")
        assert result.analyzer == "fast_path"
        assert result.complexity_tier == Tier.LOW
        assert result.cached is False

    def test_local_llm_success(self):
        """Line 223: local LLM path succeeds."""
        mock_provider = MagicMock()
        mock_provider.create.return_value = _make_response(
            '{"expected_output_tokens": 500, "complexity": "LOW",'
            ' "needs_reasoning": false, "confidence": 0.9}'
        )
        with patch(
            "mullet.core.providers.local_detection.detect_local_provider",
            return_value=mock_provider,
        ):
            result = analyze_prompt("some task that is not simple " * 30)
        assert result is not None
        assert result.analyzer == "local_llm"
        assert result.complexity_tier == Tier.LOW

    def test_gemini_fallback(self):
        """Lines 224-225: gemini flash is used when local LLM unavailable."""
        gemini_result = PreflightResult(
            input_tokens=20,
            expected_output_tokens=800,
            complexity_tier=Tier.ADVANCED,
            needs_reasoning=False,
            confidence=0.85,
            analyzer="gemini_flash",
            latency_ms=200,
            cached=False,
        )
        with (
            patch(
                "mullet.core.providers.local_detection.detect_local_provider",
                return_value=None,
            ),
            patch(
                "mullet.core.router.preflight._try_gemini_flash",
                return_value=gemini_result,
            ),
        ):
            result = analyze_prompt("some task that is not simple " * 30)
        assert result.analyzer == "gemini_flash"

    def test_heuristic_fallback(self):
        """Lines 226-227: heuristic when both LLMs unavailable."""
        with (
            patch(
                "mullet.core.providers.local_detection.detect_local_provider",
                return_value=None,
            ),
            patch(
                "mullet.core.router.preflight._try_gemini_flash",
                return_value=None,
            ),
        ):
            result = analyze_prompt("some task that is not simple " * 30)
        assert result.analyzer == "heuristic"
        assert result.confidence == 0.5

    def test_economy_mode_skips_gemini(self):
        """Economy mode does not call Gemini."""
        gemini_called = []

        def _boom(task):
            gemini_called.append(True)
            raise AssertionError("Gemini should NOT be called in economy mode")

        with (
            patch(
                "mullet.core.providers.local_detection.detect_local_provider",
                return_value=None,
            ),
            patch(
                "mullet.core.router.preflight._try_gemini_flash",
                side_effect=_boom,
            ),
        ):
            result = analyze_prompt(
                "some task that is not simple " * 30,
                speed_mode="economy",
            )
        assert result.analyzer == "heuristic"
        assert not gemini_called


# ---------------------------------------------------------------------------
# _try_local_llm
# ---------------------------------------------------------------------------


class TestTryLocalLlm:
    def test_provider_none_returns_none(self):
        """Line 237-238: detect_local_provider returns None -> None."""
        with patch(
            "mullet.core.providers.local_detection.detect_local_provider",
            return_value=None,
        ):
            result = _try_local_llm("task")
        assert result is None

    def test_good_json_returns_preflight_result(self):
        """Lines 240-258: valid response from local LLM."""
        mock_provider = MagicMock()
        mock_provider.create.return_value = _make_response(
            '{"expected_output_tokens": 500, "complexity": "LOW",'
            ' "needs_reasoning": false, "confidence": 0.9}'
        )
        with patch(
            "mullet.core.providers.local_detection.detect_local_provider",
            return_value=mock_provider,
        ):
            result = _try_local_llm("task")
        assert result is not None
        assert result.analyzer == "local_llm"
        assert result.complexity_tier == Tier.LOW
        assert result.expected_output_tokens == 500

    def test_exception_returns_none(self, caplog):
        """Lines 259-261: exception in local LLM call -> None."""
        mock_provider = MagicMock()
        mock_provider.create.side_effect = RuntimeError("boom")
        with (
            patch(
                "mullet.core.providers.local_detection.detect_local_provider",
                return_value=mock_provider,
            ),
            caplog.at_level("DEBUG", logger="mullet.core.router.preflight"),
        ):
            result = _try_local_llm("task")
        assert result is None
        assert "Local LLM preflight failed" in caplog.text

    def test_parse_meta_json_none_returns_none(self):
        """Lines 247-248: unparseable JSON response -> None."""
        mock_provider = MagicMock()
        mock_provider.create.return_value = _make_response("no json here")
        with patch(
            "mullet.core.providers.local_detection.detect_local_provider",
            return_value=mock_provider,
        ):
            result = _try_local_llm("task")
        assert result is None

    def test_malformed_json_returns_none(self):
        mock_provider = MagicMock()
        mock_provider.create.return_value = _make_response("{ bad json }")
        with patch(
            "mullet.core.providers.local_detection.detect_local_provider",
            return_value=mock_provider,
        ):
            result = _try_local_llm("task")
        assert result is None


# ---------------------------------------------------------------------------
# _try_gemini_flash
# ---------------------------------------------------------------------------


class TestTryGeminiFlash:
    def test_no_api_key_returns_none(self, monkeypatch):
        """Lines 267-269: no API key -> None."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        result = _try_gemini_flash("task")
        assert result is None

    def test_good_json_returns_preflight_result(self, monkeypatch):
        """Lines 270-290: valid response from Gemini."""
        monkeypatch.setenv("GOOGLE_API_KEY", "fake_key")
        mock_provider = MagicMock()
        mock_provider.create.return_value = _make_response(
            '{"expected_output_tokens": 500, "complexity": "LOW",'
            ' "needs_reasoning": false, "confidence": 0.9}'
        )
        with patch(
            "mullet.core.providers.gemini.GeminiProvider",
            return_value=mock_provider,
        ):
            result = _try_gemini_flash("task")
        assert result is not None
        assert result.analyzer == "gemini_flash"
        assert result.complexity_tier == Tier.LOW
        assert result.expected_output_tokens == 500

    def test_exception_returns_none(self, monkeypatch, caplog):
        """Lines 291-293: exception in Gemini call -> None."""
        monkeypatch.setenv("GOOGLE_API_KEY", "fake_key")
        with (
            patch(
                "mullet.core.providers.gemini.GeminiProvider",
                side_effect=RuntimeError("boom"),
            ),
            caplog.at_level("DEBUG", logger="mullet.core.router.preflight"),
        ):
            result = _try_gemini_flash("task")
        assert result is None
        assert "Gemini flash preflight failed" in caplog.text

    def test_parse_meta_json_none_returns_none(self, monkeypatch):
        """Lines 278-280: unparseable JSON response -> None."""
        monkeypatch.setenv("GOOGLE_API_KEY", "fake_key")
        mock_provider = MagicMock()
        mock_provider.create.return_value = _make_response("no json here")
        with patch(
            "mullet.core.providers.gemini.GeminiProvider",
            return_value=mock_provider,
        ):
            result = _try_gemini_flash("task")
        assert result is None

    def test_gemini_api_key_works(self, monkeypatch):
        """GOOGLE_API_KEY missing but GEMINI_API_KEY present."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        mock_provider = MagicMock()
        mock_provider.create.return_value = _make_response(
            '{"expected_output_tokens": 500, "complexity": "LOW",'
            ' "needs_reasoning": false, "confidence": 0.9}'
        )
        with patch(
            "mullet.core.providers.gemini.GeminiProvider",
            return_value=mock_provider,
        ):
            result = _try_gemini_flash("task")
        assert result is not None
        assert result.analyzer == "gemini_flash"


# ---------------------------------------------------------------------------
# _heuristic
# ---------------------------------------------------------------------------


def test_heuristic():
    """Lines 297-300: heuristic analyzer returns a PreflightResult."""
    result = _heuristic("refactor the auth module", 100)
    assert result.analyzer == "heuristic"
    assert result.confidence == 0.5
    assert isinstance(result.complexity_tier, Tier)
    assert result.cached is False
