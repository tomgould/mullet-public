"""Tests for mullet.core.router.preflight prompt analyzer."""
import dataclasses
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from mullet.core.router.preflight import (
    PreflightResult, analyze_prompt, _load_cache, _save_to_cache, _parse_meta_json, _try_local_llm, _try_gemini_flash,
)
from mullet.core.router.complexity import Tier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(json_str: str) -> MagicMock:
    """Return a mock provider response whose .content is *json_str*."""
    resp = MagicMock()
    resp.content = json_str
    return resp


def _valid_json_response(
    expected_output_tokens: int = 1000,
    complexity: str = "STANDARD",
    needs_reasoning: bool = False,
    confidence: float = 0.9,
) -> str:
    return json.dumps({
        "expected_output_tokens": expected_output_tokens,
        "complexity": complexity,
        "needs_reasoning": needs_reasoning,
        "confidence": confidence,
    })


# ---------------------------------------------------------------------------
# Cache isolation fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Redirect the preflight module-level cache to a temp dir and clear in-memory cache."""
    import mullet.core.router.preflight as pf_module

    fake_cache_path = tmp_path / "cache" / "preflight.jsonl"
    monkeypatch.setattr(pf_module, "_CACHE_PATH", fake_cache_path)
    # Clear the in-memory cache between tests
    monkeypatch.setattr(pf_module, "_CACHE", {})
    yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fast_path_skips_llm():
    """Short simple messages should short-circuit via fast_path without calling any LLM."""
    with patch("mullet.core.router.preflight.is_simple_message", return_value=True):
        result = analyze_prompt("list files")

    assert result.analyzer == "fast_path"
    assert result.complexity_tier == Tier.LOW
    assert result.cached is False


def test_local_llm_happy_path():
    """A reachable local LLM that returns valid JSON produces a local_llm result."""
    mock_provider = MagicMock()
    mock_provider.create.return_value = _make_response(
        _valid_json_response(expected_output_tokens=1000, complexity="STANDARD")
    )

    with (
        patch("mullet.core.router.preflight.is_simple_message", return_value=False),
        patch(
            "mullet.core.providers.local_detection.detect_local_provider",
            return_value=mock_provider,
        ),
        # Also patch at the import site inside _try_local_llm
        patch(
            "mullet.core.router.preflight._try_local_llm",
            return_value=PreflightResult(
                input_tokens=10,
                expected_output_tokens=1000,
                complexity_tier=Tier.STANDARD,
                needs_reasoning=False,
                confidence=0.9,
                analyzer="local_llm",
                latency_ms=50,
                cached=False,
            ),
        ),
    ):
        result = analyze_prompt("refactor the auth module")

    assert result.analyzer == "local_llm"
    assert result.complexity_tier == Tier.STANDARD
    assert result.expected_output_tokens == 1000


def test_local_llm_turns_thinking_off():
    """_try_local_llm asks for reasoning_effort="none" so a reasoning model answers within 200 tokens."""
    mock_provider = MagicMock()
    mock_provider.create.return_value = _make_response(_valid_json_response(complexity="LOW"))

    with patch(
        "mullet.core.providers.local_detection.detect_local_provider",
        return_value=mock_provider,
    ):
        result = _try_local_llm("rename get_user to fetch_user")

    assert result is not None and result.analyzer == "local_llm"
    kwargs = mock_provider.create.call_args.kwargs
    assert kwargs["reasoning_effort"] == "none"
    assert kwargs["max_tokens"] == 200


def test_local_unavailable_falls_back_to_gemini(monkeypatch):
    """When local provider returns None, fall back to Gemini Flash."""
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
        patch("mullet.core.router.preflight.is_simple_message", return_value=False),
        patch("mullet.core.router.preflight._try_local_llm", return_value=None),
        patch("mullet.core.router.preflight._try_gemini_flash", return_value=gemini_result),
    ):
        result = analyze_prompt("design a scalable microservice architecture")

    assert result.analyzer == "gemini_flash"


def test_both_llms_unavailable_falls_back_to_heuristic():
    """When both local and Gemini are unavailable, fall back to heuristic analysis."""
    with (
        patch("mullet.core.router.preflight.is_simple_message", return_value=False),
        patch("mullet.core.router.preflight._try_local_llm", return_value=None),
        patch("mullet.core.router.preflight._try_gemini_flash", return_value=None),
    ):
        result = analyze_prompt("implement a distributed cache with Redis")

    assert result.analyzer == "heuristic"
    assert result.confidence == 0.5


def test_malformed_json_falls_back_to_heuristic():
    """A provider that returns unparseable JSON triggers the heuristic fallback.

    _try_local_llm returns None when JSON parse fails; we simulate this directly.
    """
    with (
        patch("mullet.core.router.preflight.is_simple_message", return_value=False),
        patch("mullet.core.router.preflight._try_local_llm", return_value=None),
        patch("mullet.core.router.preflight._try_gemini_flash", return_value=None),
    ):
        result = analyze_prompt("build a new API endpoint")

    assert result.analyzer == "heuristic"


def test_cache_hit_returns_cached_result():
    """Second call with the same task should return a cached result."""
    llm_call_count = []
    local_result = PreflightResult(
        input_tokens=10,
        expected_output_tokens=512,
        complexity_tier=Tier.STANDARD,
        needs_reasoning=False,
        confidence=0.9,
        analyzer="local_llm",
        latency_ms=50,
        cached=False,
    )

    def _counting_local_llm(task):
        llm_call_count.append(1)
        return local_result

    task = "write a function to parse JSON"

    with (
        patch("mullet.core.router.preflight.is_simple_message", return_value=False),
        patch("mullet.core.router.preflight._try_local_llm", side_effect=_counting_local_llm),
    ):
        first = analyze_prompt(task)
        second = analyze_prompt(task)

    assert first.cached is False
    assert second.cached is True
    # LLM was called only once
    assert len(llm_call_count) == 1


def test_cache_miss_on_different_files():
    """Different file lists should produce different cache keys (fresh LLM call)."""
    llm_call_count = []
    local_result = PreflightResult(
        input_tokens=10,
        expected_output_tokens=512,
        complexity_tier=Tier.STANDARD,
        needs_reasoning=False,
        confidence=0.9,
        analyzer="local_llm",
        latency_ms=50,
        cached=False,
    )

    def _counting_local_llm(task):
        llm_call_count.append(1)
        return local_result

    task = "review the module"

    with (
        patch("mullet.core.router.preflight.is_simple_message", return_value=False),
        patch("mullet.core.router.preflight._try_local_llm", side_effect=_counting_local_llm),
    ):
        first = analyze_prompt(task, files=["a.py"])
        second = analyze_prompt(task, files=["b.py"])

    # Both should trigger the LLM (two different cache keys)
    assert len(llm_call_count) == 2
    assert second.cached is False


def test_economy_mode_skips_gemini_fallback():
    """In economy mode, Gemini is not consulted even when local LLM is unavailable."""
    gemini_called = []

    def _boom(task):
        gemini_called.append(True)
        raise AssertionError("Gemini should NOT be called in economy mode")

    with (
        patch("mullet.core.router.preflight.is_simple_message", return_value=False),
        patch("mullet.core.router.preflight._try_local_llm", return_value=None),
        patch("mullet.core.router.preflight._try_gemini_flash", side_effect=_boom),
    ):
        result = analyze_prompt(
            "implement background job processing",
            speed_mode="economy",
        )

    assert result.analyzer == "heuristic"
    assert not gemini_called


def test_preflight_result_is_frozen():
    """PreflightResult is a frozen dataclass — attribute assignment must raise."""
    result = PreflightResult(
        input_tokens=100,
        expected_output_tokens=512,
        complexity_tier=Tier.STANDARD,
        needs_reasoning=False,
        confidence=0.9,
        analyzer="heuristic",
        latency_ms=0,
        cached=False,
    )
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError, AttributeError)):
        result.analyzer = "local_llm"  # type: ignore[misc]


def test_input_tokens_estimated():
    """analyze_prompt should estimate and store a positive input_tokens count."""
    task = "implement a user authentication system with JWT tokens"

    with (
        patch("mullet.core.router.preflight.is_simple_message", return_value=False),
        patch("mullet.core.router.preflight._try_local_llm", return_value=None),
        patch("mullet.core.router.preflight._try_gemini_flash", return_value=None),
    ):
        result = analyze_prompt(task)

    # Heuristic path still populates input_tokens from estimate_tokens()
    assert result.input_tokens > 0


# ---------------------------------------------------------------------------
# _parse_meta_json
# ---------------------------------------------------------------------------

class TestParseMetaJson:
    def test_valid_json_returns_dict(self):
        text = '{"expected_output_tokens": 1000, "complexity": "STANDARD", "needs_reasoning": false, "confidence": 0.9}'
        result = _parse_meta_json(text)
        assert result is not None
        assert result["complexity"] == "STANDARD"
        assert result["confidence"] == 0.9

    def test_no_json_returns_none(self):
        assert _parse_meta_json("no json here") is None

    def test_malformed_json_returns_none(self):
        assert _parse_meta_json("{ bad json }") is None

    def test_json_embedded_in_prose(self):
        text = 'Here is the analysis: {"complexity": "ADVANCED", "expected_output_tokens": 200, "needs_reasoning": true, "confidence": 0.8}'
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

    def test_needs_reasoning_string_true_converted(self):
        text = '{"needs_reasoning": "true"}'
        result = _parse_meta_json(text)
        assert result["needs_reasoning"] is True

    def test_needs_reasoning_string_yes_converted(self):
        text = '{"needs_reasoning": "yes"}'
        result = _parse_meta_json(text)
        assert result["needs_reasoning"] is True

    def test_needs_reasoning_string_false_converted(self):
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
        text = '{}'
        result = _parse_meta_json(text)
        assert result["expected_output_tokens"] == 512
        assert result["complexity"] == "STANDARD"
        assert result["needs_reasoning"] is False
        assert result["confidence"] == 0.5


# ---------------------------------------------------------------------------
# Missing line coverage
# ---------------------------------------------------------------------------


class TestPreflightMissingLines:
    # _load_cache ------------------------------------------------------------

    def test_load_cache_no_file(self, monkeypatch):
        import mullet.core.router.preflight as pf_module

        mock_path = MagicMock()
        mock_path.exists.return_value = False
        monkeypatch.setattr(pf_module, "_CACHE_PATH", mock_path)
        result = _load_cache()
        assert result == {}

    def test_load_cache_blank_line_skipped(self, monkeypatch, tmp_path):
        import mullet.core.router.preflight as pf_module

        fake_path = tmp_path / "cache.jsonl"
        fake_path.parent.mkdir(parents=True, exist_ok=True)
        valid = {
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
        fake_path.write_text("\n\n" + json.dumps(valid) + "\n")
        monkeypatch.setattr(pf_module, "_CACHE_PATH", fake_path)
        result = _load_cache()
        assert "k1" in result
        assert len(result) == 1

    def test_load_cache_stale_entry_skipped(self, monkeypatch, tmp_path):
        import mullet.core.router.preflight as pf_module

        fake_path = tmp_path / "cache.jsonl"
        fake_path.parent.mkdir(parents=True, exist_ok=True)
        stale = {
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
        fake_path.write_text(json.dumps(stale) + "\n")
        monkeypatch.setattr(pf_module, "_CACHE_PATH", fake_path)
        result = _load_cache()
        assert "k1" not in result

    def test_load_cache_bad_json_line_skipped(self, monkeypatch, tmp_path):
        import mullet.core.router.preflight as pf_module

        fake_path = tmp_path / "cache.jsonl"
        fake_path.parent.mkdir(parents=True, exist_ok=True)
        valid = {
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
        fake_path.write_text("{ bad json }\n" + json.dumps(valid) + "\n")
        monkeypatch.setattr(pf_module, "_CACHE_PATH", fake_path)
        result = _load_cache()
        assert "k1" in result
        assert len(result) == 1

    def test_load_cache_file_read_exception_logs_debug(self, monkeypatch, caplog):
        import mullet.core.router.preflight as pf_module

        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.open.side_effect = OSError("boom")
        monkeypatch.setattr(pf_module, "_CACHE_PATH", mock_path)
        with caplog.at_level("DEBUG", logger="mullet.core.router.preflight"):
            result = _load_cache()
        assert result == {}
        assert "Failed to load preflight cache" in caplog.text

    def test_load_cache_valid_entry_loaded(self, monkeypatch, tmp_path):
        import mullet.core.router.preflight as pf_module

        fake_path = tmp_path / "cache.jsonl"
        fake_path.parent.mkdir(parents=True, exist_ok=True)
        valid = {
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
        fake_path.write_text(json.dumps(valid) + "\n")
        monkeypatch.setattr(pf_module, "_CACHE_PATH", fake_path)
        result = _load_cache()
        assert "k1" in result
        assert result["k1"].cached is True

    # _save_to_cache ---------------------------------------------------------

    def test_save_to_cache_os_replace_raises_logs_debug(self, monkeypatch, tmp_path, caplog):
        import mullet.core.router.preflight as pf_module

        fake_path = tmp_path / "cache.jsonl"
        monkeypatch.setattr(pf_module, "_CACHE_PATH", fake_path)
        monkeypatch.setattr(pf_module, "_CACHE", {})
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
            "mullet.core.router.preflight.os.replace", side_effect=OSError("replace failed")
        ):
            with caplog.at_level("DEBUG", logger="mullet.core.router.preflight"):
                _save_to_cache("key1", result)
        assert "Failed to append to preflight cache" in caplog.text

    def test_save_to_cache_rewrite_exception_logs_debug(self, monkeypatch, tmp_path, caplog):
        import mullet.core.router.preflight as pf_module

        fake_path = tmp_path / "cache.jsonl"
        monkeypatch.setattr(pf_module, "_CACHE_PATH", fake_path)
        stale_mock = MagicMock()
        stale_mock.input_tokens = 1
        stale_mock.expected_output_tokens = 512
        stale_mock.complexity_tier.value = "LOW"
        stale_mock.needs_reasoning = False
        stale_mock.confidence = 0.9
        stale_mock.analyzer = "fast_path"
        stale_mock.latency_ms = 0
        stale_mock._cached_written_at = 0
        monkeypatch.setattr(pf_module, "_CACHE", {"old_key": stale_mock})
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
        ) as mock_replace:
            with caplog.at_level("DEBUG", logger="mullet.core.router.preflight"):
                _save_to_cache("key1", result)
        assert mock_replace.call_count == 2
        assert "Failed to rewrite preflight cache" in caplog.text

    # analyze_prompt ---------------------------------------------------------

    def test_analyze_prompt_local_none_non_economy_tries_gemini(self):
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
            patch("mullet.core.router.preflight.is_simple_message", return_value=False),
            patch("mullet.core.router.preflight._try_local_llm", return_value=None),
            patch(
                "mullet.core.router.preflight._try_gemini_flash", return_value=gemini_result
            ),
        ):
            result = analyze_prompt("design a system", speed_mode="balanced")
        assert result.analyzer == "gemini_flash"

    def test_analyze_prompt_cache_hit_pre_populated(self, monkeypatch):
        import mullet.core.router.preflight as pf_module

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
        monkeypatch.setattr(pf_module, "_CACHE", {"abc": cached})
        with patch("mullet.core.router.preflight._cache_key", return_value="abc"):
            result = analyze_prompt("any task")
        assert result.cached is True
        assert result.analyzer == "local_llm"

    # _try_local_llm ---------------------------------------------------------

    def test_try_local_llm_provider_none_returns_none(self):
        with patch(
            "mullet.core.providers.local_detection.detect_local_provider",
            return_value=None,
        ):
            result = _try_local_llm("some task")
        assert result is None

    def test_try_local_llm_good_json_returns_preflight_result(self):
        mock_response = MagicMock()
        mock_response.content = (
            '{"expected_output_tokens": 500, "complexity": "LOW",'
            ' "needs_reasoning": false, "confidence": 0.9}'
        )
        mock_provider = MagicMock()
        mock_provider.create.return_value = mock_response
        with patch(
            "mullet.core.providers.local_detection.detect_local_provider",
            return_value=mock_provider,
        ):
            result = _try_local_llm("some task")
        assert result is not None
        assert result.analyzer == "local_llm"
        assert result.complexity_tier == Tier.LOW
        assert result.expected_output_tokens == 500

    def test_try_local_llm_exception_returns_none(self, caplog):
        mock_provider = MagicMock()
        mock_provider.create.side_effect = RuntimeError("boom")
        with (
            patch(
                "mullet.core.providers.local_detection.detect_local_provider",
                return_value=mock_provider,
            ),
            caplog.at_level("DEBUG", logger="mullet.core.router.preflight"),
        ):
            result = _try_local_llm("some task")
        assert result is None
        assert "Local LLM preflight failed" in caplog.text

    def test_try_local_llm_parse_meta_json_none_returns_none(self):
        mock_response = MagicMock()
        mock_response.content = "no json here"
        mock_provider = MagicMock()
        mock_provider.create.return_value = mock_response
        with patch(
            "mullet.core.providers.local_detection.detect_local_provider",
            return_value=mock_provider,
        ):
            result = _try_local_llm("some task")
        assert result is None

    def test_try_local_llm_malformed_json_returns_none(self):
        mock_response = MagicMock()
        mock_response.content = "{ bad json }"
        mock_provider = MagicMock()
        mock_provider.create.return_value = mock_response
        with patch(
            "mullet.core.providers.local_detection.detect_local_provider",
            return_value=mock_provider,
        ):
            result = _try_local_llm("some task")
        assert result is None

    # _try_gemini_flash ------------------------------------------------------

    def test_try_gemini_flash_no_api_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        result = _try_gemini_flash("some task")
        assert result is None

    def test_try_gemini_flash_good_json_returns_preflight_result(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake_key")
        mock_response = MagicMock()
        mock_response.content = (
            '{"expected_output_tokens": 500, "complexity": "LOW",'
            ' "needs_reasoning": false, "confidence": 0.9}'
        )
        mock_provider = MagicMock()
        mock_provider.create.return_value = mock_response
        with patch(
            "mullet.core.providers.gemini.GeminiProvider",
            return_value=mock_provider,
        ):
            result = _try_gemini_flash("some task")
        assert result is not None
        assert result.analyzer == "gemini_flash"
        assert result.complexity_tier == Tier.LOW
        assert result.expected_output_tokens == 500

    def test_try_gemini_flash_exception_returns_none(self, monkeypatch, caplog):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake_key")
        with (
            patch(
                "mullet.core.providers.gemini.GeminiProvider",
                side_effect=RuntimeError("boom"),
            ),
            caplog.at_level("DEBUG", logger="mullet.core.router.preflight"),
        ):
            result = _try_gemini_flash("some task")
        assert result is None
        assert "Gemini flash preflight failed" in caplog.text

    def test_try_gemini_flash_parse_meta_json_none_returns_none(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake_key")
        mock_response = MagicMock()
        mock_response.content = "no json here"
        mock_provider = MagicMock()
        mock_provider.create.return_value = mock_response
        with patch(
            "mullet.core.providers.gemini.GeminiProvider",
            return_value=mock_provider,
        ):
            result = _try_gemini_flash("some task")
        assert result is None

    def test_try_gemini_flash_gemini_api_key_works(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        mock_response = MagicMock()
        mock_response.content = (
            '{"expected_output_tokens": 500, "complexity": "LOW",'
            ' "needs_reasoning": false, "confidence": 0.9}'
        )
        mock_provider = MagicMock()
        mock_provider.create.return_value = mock_response
        with patch(
            "mullet.core.providers.gemini.GeminiProvider",
            return_value=mock_provider,
        ):
            result = _try_gemini_flash("some task")
        assert result is not None
        assert result.analyzer == "gemini_flash"
        assert result.complexity_tier == Tier.LOW
        assert result.expected_output_tokens == 500


def test_parse_meta_json_invalid_confidence_type():
    """Lines 180-181: TypeError/ValueError on confidence → default 0.5."""
    from mullet.core.router.preflight import _parse_meta_json
    data = _parse_meta_json('{"expected_output_tokens": 200, "complexity": "LOW", "needs_reasoning": false, "confidence": "not-a-number"}')
    assert data is not None
    assert data["confidence"] == 0.5
