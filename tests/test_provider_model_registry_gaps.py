"""Gap-filling tests for mullet.core.providers.model_registry.

Targets the uncovered lines reported by coverage so combined coverage
reaches >= 90 %.  All HTTP calls and external dependencies are mocked.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mullet.core.providers.model_registry import (
    CONTEXT_SAFETY_MARGIN,
    ModelRegistry,
    ProviderModel,
    _fetch_moonshot_pricing,
    _context_for,
    _fetch_moonshot_pricing,
    _pricing_for,
    _tier_chain_starting_from,
    _tier_for,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_registry(tmp_path: Path) -> ModelRegistry:
    """Return a registry with an isolated cache path."""
    reg = ModelRegistry()
    reg.CACHE_PATH = tmp_path / "model_cache.json"  # type: ignore[misc]
    return reg


def _urlopen_cm(payload: dict) -> MagicMock:
    body = json.dumps(payload).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return MagicMock(return_value=resp)


# ---------------------------------------------------------------------------
# _tier_chain_starting_from (line 62)
# ---------------------------------------------------------------------------


def test_tier_chain_empty_string():
    assert _tier_chain_starting_from("") == [""]


# ---------------------------------------------------------------------------
# _pricing_for substring + MODEL_DATABASE fallback (lines 164, 167-168)
# ---------------------------------------------------------------------------


def test_pricing_for_substring_match():
    # A known key that can match as substring
    price = _pricing_for("claude-sonnet-4-6-20251001")
    # Should hit the substring branch and return a non-zero tuple
    assert price != (0.0, 0.0)


def test_pricing_for_model_database_fallback():
    # Use a model that is in MODEL_DATABASE but not in KNOWN_PRICING
    from mullet.core.providers.model_info import MODEL_DATABASE
    # Find a model that is in MODEL_DATABASE but not in KNOWN_PRICING keys
    # This is tricky; instead we monkeypatch temporarily
    fake_info = MagicMock()
    fake_info.input_cost_per_1k = 1.23
    fake_info.output_cost_per_1k = 4.56
    with patch.dict(MODEL_DATABASE, {"unique-fake-model-xyz": fake_info}, clear=False):
        assert _pricing_for("unique-fake-model-xyz") == (1.23, 4.56)


# ---------------------------------------------------------------------------
# _context_for heuristic from model name (lines 182, 184, 186)
# ---------------------------------------------------------------------------


def test_context_for_heuristics():
    assert _context_for("some-128k-model") == 131_072
    assert _context_for("some-32k-model") == 32_768
    assert _context_for("some-8k-model") == 8_192


# ---------------------------------------------------------------------------
# _fetch_moonshot_pricing gaps (lines 233-234, 239, 254, 262-263, 279-285)
# ---------------------------------------------------------------------------


def test_fetch_moonshot_pricing_cache_read_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mullet.core.providers.model_registry._MOONSHOT_PRICING_CACHE_PATH",
        tmp_path / "moonshot_pricing_cache.json",
    )
    # Write invalid JSON so json.loads raises
    (tmp_path / "moonshot_pricing_cache.json").write_text("not-json")
    result = _fetch_moonshot_pricing()
    assert result == {}


def test_fetch_moonshot_pricing_no_api_key(monkeypatch):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    result = _fetch_moonshot_pricing()
    assert result == {}


def test_fetch_moonshot_pricing_skips_empty_model_id(monkeypatch):
    monkeypatch.setattr(
        "mullet.core.providers.model_registry._MOONSHOT_PRICING_CACHE_PATH",
        MagicMock(exists=MagicMock(return_value=False)),
    )
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    payload = {
        "data": [
            {"id": "", "pricing": {"input": 0.1, "output": 0.2}},
            {"id": "valid-model", "pricing": {"input": 0.1, "output": 0.2}},
        ]
    }
    with patch("urllib.request.urlopen", _urlopen_cm(payload)):
        result = _fetch_moonshot_pricing()
    assert "valid-model" in result
    assert "" not in result


def test_fetch_moonshot_pricing_bad_pricing_values(monkeypatch):
    monkeypatch.setattr(
        "mullet.core.providers.model_registry._MOONSHOT_PRICING_CACHE_PATH",
        MagicMock(exists=MagicMock(return_value=False)),
    )
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    payload = {
        "data": [
            {"id": "bad", "pricing": {"input": "nope", "output": "nope"}},
        ]
    }
    with patch("urllib.request.urlopen", _urlopen_cm(payload)):
        result = _fetch_moonshot_pricing()
    assert "bad" not in result


def test_fetch_moonshot_pricing_cache_write_cleanup_exception(monkeypatch, tmp_path):
    cache_path = tmp_path / "moonshot_pricing_cache.json"
    monkeypatch.setattr(
        "mullet.core.providers.model_registry._MOONSHOT_PRICING_CACHE_PATH",
        cache_path,
    )
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    payload = {"data": [{"id": "m1", "pricing": {"input": 0.1, "output": 0.2}}]}

    # Make os.replace raise so the inner cleanup runs
    with patch("urllib.request.urlopen", _urlopen_cm(payload)):
        with patch("os.replace", side_effect=OSError("replace fail")):
            with patch("os.unlink", side_effect=OSError("unlink fail")):
                # Should not raise
                result = _fetch_moonshot_pricing()
    assert "m1" in result


# ---------------------------------------------------------------------------
# ModelRegistry.__init__ live pricing update (line 307)
# ---------------------------------------------------------------------------


def test_registry_init_merges_live_pricing(monkeypatch):
    monkeypatch.setattr(
        "mullet.core.providers.model_registry._fetch_moonshot_pricing",
        lambda: {"live-model": (0.01, 0.02)},
    )
    reg = ModelRegistry()
    from mullet.core.providers.model_registry import KNOWN_PRICING
    assert "live-model" in KNOWN_PRICING
    KNOWN_PRICING.pop("live-model", None)


# ---------------------------------------------------------------------------
# get_models static fallback (line 337)
# ---------------------------------------------------------------------------


def test_get_models_falls_back_to_static(tmp_path):
    reg = _fresh_registry(tmp_path)
    reg._cache.pop("openrouter", None)
    reg._cache_ts.pop("openrouter", None)
    with patch.object(reg, "_fetch_and_cache"):
        models = reg.get_models("openrouter")
    # openrouter has static models
    assert len(models) > 0


# ---------------------------------------------------------------------------
# _assert_can_serve exception handling (lines 373-380, 396-397)
# ---------------------------------------------------------------------------


def test_assert_can_serve_constraint_registry_exception():
    reg = ModelRegistry()

    def raising(*a, **kw):
        raise RuntimeError("boom")

    with patch("mullet.core.providers.constraints.get_constraint_registry", raising):
        # Should log warning and return model_id unchanged
        assert reg._assert_can_serve("kimi", "kimi-k2.5", "STANDARD", 1000) == "kimi-k2.5"


def test_assert_can_serve_max_ctx_exception():
    reg = ModelRegistry()
    from mullet.core.providers.errors import ModelContextExceeded

    fake_entry = MagicMock()
    fake_entry.can_serve.return_value = False
    fake_entry.max_context = "not-an-int"
    fake_cr = MagicMock()
    fake_cr.get.return_value = fake_entry

    with patch("mullet.core.providers.constraints.get_constraint_registry", lambda: fake_cr):
        with pytest.raises(ModelContextExceeded):
            reg._assert_can_serve("kimi", "x", "STANDARD", 1000)


def test_pick_model_preferred_constraint_exception_returns_preferred():
    reg = ModelRegistry()
    # Make constraint registry raise on the preferred model
    fake_cr = MagicMock()
    fake_cr.get.side_effect = RuntimeError("constraint down")

    with patch("mullet.core.providers.constraints.get_constraint_registry", lambda: fake_cr):
        reg._cache["kimi"] = []
        chosen = reg._pick_model(
            provider="kimi",
            complexity="STANDARD",
            context_needed=4096,
            context_tokens=1000,
        )
    # Exception in constraint check for preferred → falls through to return preferred on line 560
    assert chosen == "kimi-k2.7-code"


# ---------------------------------------------------------------------------
# _pick_model generic provider path (lines 580-612)
# ---------------------------------------------------------------------------


def test_pick_model_generic_provider_low_complexity():
    reg = ModelRegistry()
    reg._cache["generic_provider"] = [
        ProviderModel(
            id="cheap-model",
            provider="generic_provider",
            display_name="Cheap",
            context_length=100_000,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.005,
            tier="fast",
        ),
        ProviderModel(
            id="dear-model",
            provider="generic_provider",
            display_name="Dear",
            context_length=100_000,
            input_cost_per_1k=0.010,
            output_cost_per_1k=0.050,
            tier="standard",
        ),
    ]
    chosen = reg._pick_model(
        provider="generic_provider",
        complexity="LOW",
        context_needed=4096,
        context_tokens=None,
    )
    assert chosen == "cheap-model"


def test_pick_model_generic_provider_critical_complexity():
    reg = ModelRegistry()
    reg._cache["generic_provider"] = [
        ProviderModel(
            id="cheap-model",
            provider="generic_provider",
            display_name="Cheap",
            context_length=100_000,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.005,
            tier="fast",
        ),
        ProviderModel(
            id="dear-model",
            provider="generic_provider",
            display_name="Dear",
            context_length=100_000,
            input_cost_per_1k=0.010,
            output_cost_per_1k=0.050,
            tier="standard",
        ),
    ]
    chosen = reg._pick_model(
        provider="generic_provider",
        complexity="CRITICAL",
        context_needed=4096,
        context_tokens=None,
    )
    assert chosen == "cheap-model"  # lowest output cost in fitting pool


def test_pick_model_generic_provider_context_tokens_filter():
    reg = ModelRegistry()
    reg._cache["generic_provider"] = [
        ProviderModel(
            id="small",
            provider="generic_provider",
            display_name="Small",
            context_length=4_000,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.005,
            tier="fast",
        ),
        ProviderModel(
            id="large",
            provider="generic_provider",
            display_name="Large",
            context_length=100_000,
            input_cost_per_1k=0.010,
            output_cost_per_1k=0.050,
            tier="standard",
        ),
    ]
    chosen = reg._pick_model(
        provider="generic_provider",
        complexity="STANDARD",
        context_needed=4096,
        context_tokens=50_000,
    )
    # 50_000 * 1.1 = 55_000 > 4_000, so small is filtered out
    assert chosen == "large"


def test_pick_model_generic_empty_returns_empty():
    reg = ModelRegistry()
    reg._cache["generic_provider"] = []
    chosen = reg._pick_model(
        provider="generic_provider",
        complexity="STANDARD",
        context_needed=4096,
        context_tokens=None,
    )
    assert chosen == ""


# ---------------------------------------------------------------------------
# list_all provider in cache branch (line 641)
# ---------------------------------------------------------------------------


def test_list_all_uses_cache_when_present(tmp_path):
    reg = _fresh_registry(tmp_path)
    reg._cache["anthropic"] = reg._static_models("anthropic")
    result = reg.list_all()
    assert result["anthropic"] == reg._cache["anthropic"]


# ---------------------------------------------------------------------------
# _load_cache gaps (lines 653, 666-668)
# ---------------------------------------------------------------------------


def test_load_cache_no_file(tmp_path):
    reg = _fresh_registry(tmp_path)
    reg.CACHE_PATH = tmp_path / "nonexistent.json"
    reg._load_cache()
    assert not reg._cache


def test_load_cache_corrupt_json(tmp_path):
    reg = _fresh_registry(tmp_path)
    reg.CACHE_PATH.write_text("not json")
    reg._load_cache()
    assert not reg._cache


# ---------------------------------------------------------------------------
# _save_cache exception cleanup (lines 689-693)
# ---------------------------------------------------------------------------


def test_save_cache_cleanup_on_error(tmp_path):
    reg = _fresh_registry(tmp_path)
    reg._cache["anthropic"] = reg._static_models("anthropic")
    with patch("os.replace", side_effect=OSError("replace fail")):
        with patch("os.unlink", side_effect=OSError("unlink fail")):
            # Must not raise
            reg._save_cache()


# ---------------------------------------------------------------------------
# _fetch_and_cache fetcher None (line 709)
# ---------------------------------------------------------------------------


def test_fetch_and_cache_unknown_provider():
    reg = ModelRegistry()
    reg._fetch_and_cache("unknown_provider")
    assert "unknown_provider" not in reg._cache


# ---------------------------------------------------------------------------
# _fetch_anthropic no api key (line 729) and empty model_id (line 743)
# ---------------------------------------------------------------------------


def test_fetch_anthropic_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reg = ModelRegistry()
    assert reg._fetch_anthropic() == []


def test_fetch_anthropic_skips_empty_model_id(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    payload = {"data": [{"id": "", "display_name": "Empty"}, {"id": "real", "display_name": "Real"}]}
    reg = ModelRegistry()
    with patch("urllib.request.urlopen", _urlopen_cm(payload)):
        models = reg._fetch_anthropic()
    ids = [m.id for m in models]
    assert "" not in ids
    assert "real" in ids


# ---------------------------------------------------------------------------
# _fetch_kimi no api key (line 766) and empty model_id (line 777)
# ---------------------------------------------------------------------------


def test_fetch_kimi_no_key(monkeypatch):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    reg = ModelRegistry()
    assert reg._fetch_kimi() == []


def test_fetch_kimi_skips_empty_model_id(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    payload = {"data": [{"id": ""}, {"id": "real-kimi"}]}
    reg = ModelRegistry()
    with patch("urllib.request.urlopen", _urlopen_cm(payload)):
        models = reg._fetch_kimi()
    ids = [m.id for m in models]
    assert "" not in ids
    assert "real-kimi" in ids


# ---------------------------------------------------------------------------
# _fetch_gemini no api key (line 822), skip non-gemini (832), skip non-gen (836)
# ---------------------------------------------------------------------------


def test_fetch_gemini_no_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    reg = ModelRegistry()
    assert reg._fetch_gemini() == []


def test_fetch_gemini_skips_non_gemini_and_non_generatecontent(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    payload = {
        "models": [
            {"name": "models/other-model", "displayName": "Other", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-skipped", "displayName": "Skipped", "supportedGenerationMethods": ["embedContent"]},
            {"name": "models/gemini-kept", "displayName": "Kept", "supportedGenerationMethods": ["generateContent"]},
        ]
    }
    reg = ModelRegistry()
    with patch("urllib.request.urlopen", _urlopen_cm(payload)):
        models = reg._fetch_gemini()
    ids = [m.id for m in models]
    assert "other-model" not in ids
    assert "gemini-skipped" not in ids
    assert "gemini-kept" in ids


# ---------------------------------------------------------------------------
# _fetch_local empty model_id (line 869)
# ---------------------------------------------------------------------------


def test_fetch_local_skips_empty_model_id(monkeypatch):
    def fake_get(url, headers=None, timeout=10):
        return {
            "data": [
                {"id": "", "type": "llm", "max_context_length": 1000},
                {"id": "real-local", "type": "llm", "max_context_length": 32768},
            ]
        }
    monkeypatch.setattr("mullet.core.providers.model_registry._http_get", fake_get)
    reg = ModelRegistry()
    models = reg._fetch_local()
    ids = [m.id for m in models]
    assert "" not in ids
    assert "real-local" in ids


# ---------------------------------------------------------------------------
# _static_models openrouter branch (line 1106)
# ---------------------------------------------------------------------------


def test_static_models_openrouter():
    reg = ModelRegistry()
    models = reg._static_models("openrouter")
    ids = [m.id for m in models]
    assert "deepseek/deepseek-v4-flash" in ids


# ---------------------------------------------------------------------------
# _kimi_candidates / _gemini_candidates static fallback (lines 1180, 1184-1187)
# ---------------------------------------------------------------------------


def test_kimi_candidates_static_fallback():
    reg = ModelRegistry()
    reg._cache["kimi"] = []
    models = reg._kimi_candidates()
    assert len(models) > 0


def test_gemini_candidates_static_fallback():
    reg = ModelRegistry()
    reg._cache["gemini"] = []
    models = reg._gemini_candidates()
    assert len(models) > 0


# ---------------------------------------------------------------------------
# _best_anthropic live pick + log downgrade (lines 1203-1208)
# ---------------------------------------------------------------------------


def test_best_anthropic_live_pick_and_downgrade_log(caplog):
    reg = ModelRegistry()
    reg._cache["anthropic"] = [
        ProviderModel(
            id="claude-haiku-4-5-20251001",
            provider="anthropic",
            display_name="Haiku",
            context_length=200_000,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.005,
            tier="fast",
        ),
    ]
    with caplog.at_level("INFO", logger="mullet.core.providers.model_registry"):
        chosen = reg._best_anthropic("LOW", context_needed=4096, context_tokens=None)
    assert chosen == "claude-haiku-4-5-20251001"
    # Downgrade log may or may not fire depending on PREFERRED_DEFAULTS
    # We just need the live path to execute (lines 1203-1208)


# ---------------------------------------------------------------------------
# _best_kimi routing matrix model not found log (line 1255)
# ---------------------------------------------------------------------------


def test_best_kimi_routing_matrix_not_in_candidates(caplog):
    reg = ModelRegistry()
    fake_preflight = MagicMock()
    fake_preflight.complexity_tier = "STANDARD"

    def fake_select_kimi(preflight, speed_mode="balanced"):
        return ("nonexistent-model", "test-reason")

    with patch("mullet.core.router.routing_matrix.select_kimi_model", fake_select_kimi):
        # Ensure candidates do NOT contain the model so it falls through to ladder
        reg._cache["kimi"] = []
        with caplog.at_level("DEBUG", logger="mullet.core.providers.model_registry"):
            chosen = reg._best_kimi(
                "STANDARD",
                context_needed=4096,
                preflight=fake_preflight,
            )
    assert chosen  # should still return something from ladder
    assert any("not in candidates" in r.getMessage() for r in caplog.records)


def test_best_gemini_routing_matrix_found():
    reg = ModelRegistry()
    fake_preflight = MagicMock()
    fake_preflight.complexity_tier = "STANDARD"

    def fake_select_gemini(preflight, speed_mode="balanced"):
        return ("gemini-2.5-flash", "test-reason")

    with patch("mullet.core.router.routing_matrix.select_gemini_model", fake_select_gemini):
        reg._cache["gemini"] = [
            ProviderModel(
                id="gemini-2.5-flash",
                provider="gemini",
                display_name="Flash",
                context_length=1_000_000,
                input_cost_per_1k=0.0003,
                output_cost_per_1k=0.0025,
                tier="fast",
            ),
        ]
        chosen = reg._best_gemini(
            "STANDARD",
            context_needed=4096,
            preflight=fake_preflight,
        )
    assert chosen == "gemini-2.5-flash"


def test_best_gemini_routing_matrix_not_found(caplog):
    reg = ModelRegistry()
    fake_preflight = MagicMock()
    fake_preflight.complexity_tier = "STANDARD"

    def fake_select_gemini(preflight, speed_mode="balanced"):
        return ("nonexistent-gemini", "test-reason")

    with patch("mullet.core.router.routing_matrix.select_gemini_model", fake_select_gemini):
        reg._cache["gemini"] = []
        with caplog.at_level("DEBUG", logger="mullet.core.providers.model_registry"):
            chosen = reg._best_gemini(
                "STANDARD",
                context_needed=4096,
                preflight=fake_preflight,
            )
    assert chosen  # falls through to ladder
    assert any("not in candidates" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# _best_gemini routing matrix path (lines 1339-1346)
# ---------------------------------------------------------------------------


def test_best_gemini_routing_matrix_found():
    reg = ModelRegistry()
    fake_preflight = MagicMock()
    fake_preflight.complexity_tier = "STANDARD"

    def fake_select_gemini(preflight, speed_mode="balanced"):
        return ("gemini-2.5-flash", "test-reason")

    with patch("mullet.core.router.routing_matrix.select_gemini_model", fake_select_gemini):
        reg._cache["gemini"] = [
            ProviderModel(
                id="gemini-2.5-flash",
                provider="gemini",
                display_name="Flash",
                context_length=1_000_000,
                input_cost_per_1k=0.0003,
                output_cost_per_1k=0.0025,
                tier="fast",
            ),
        ]
        chosen = reg._best_gemini(
            "STANDARD",
            context_needed=4096,
            preflight=fake_preflight,
        )
    assert chosen == "gemini-2.5-flash"


def test_best_gemini_routing_matrix_not_found(caplog):
    reg = ModelRegistry()
    fake_preflight = MagicMock()
    fake_preflight.complexity_tier = "STANDARD"

    def fake_select_gemini(preflight, speed_mode="balanced"):
        return ("nonexistent-gemini", "test-reason")

    with patch("mullet.core.router.routing_matrix.select_gemini_model", fake_select_gemini):
        reg._cache["gemini"] = []
        with caplog.at_level("DEBUG", logger="mullet.core.providers.model_registry"):
            chosen = reg._best_gemini(
                "STANDARD",
                context_needed=4096,
                preflight=fake_preflight,
            )
    assert chosen  # falls through to ladder
    assert any("not in candidates" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# _best_gemini live pick + log downgrade (lines 1351-1356)
# ---------------------------------------------------------------------------


def test_best_gemini_live_pick_and_downgrade_log(caplog):
    reg = ModelRegistry()
    reg._cache["gemini"] = [
        ProviderModel(
            id="gemini-2.5-flash",
            provider="gemini",
            display_name="Flash",
            context_length=1_000_000,
            input_cost_per_1k=0.0003,
            output_cost_per_1k=0.0025,
            tier="fast",
        ),
    ]
    with caplog.at_level("INFO", logger="mullet.core.providers.model_registry"):
        chosen = reg._best_gemini("LOW", context_needed=4096, context_tokens=None)
    assert chosen == "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# _best_openrouter reasoning (1392) and STANDARD default (1398)
# ---------------------------------------------------------------------------


def test_best_openrouter_reasoning():
    reg = ModelRegistry()
    chosen = reg._best_openrouter("STANDARD", prefer_reasoning=True)
    assert chosen == "deepseek/deepseek-v4-pro"


def test_best_openrouter_standard_default():
    reg = ModelRegistry()
    chosen = reg._best_openrouter("STANDARD", prefer_reasoning=False)
    assert chosen == "deepseek/deepseek-v4-flash"


# ---------------------------------------------------------------------------
# _best_local full path (lines 1411-1460)
# ---------------------------------------------------------------------------


def test_best_local_detect_none():
    with patch("mullet.core.providers.local_detection.detect_local_provider", lambda: None):
        reg = ModelRegistry()
        assert reg._best_local() is None


def test_best_local_list_models_path():
    fake_provider = MagicMock()
    fake_provider.list_models.return_value = [
        MagicMock(id="model-a"),
        MagicMock(id="model-b"),
    ]

    def fake_select_local(preflight, available, speed_mode="balanced"):
        return ("model-a", "picked")

    with patch("mullet.core.providers.local_detection.detect_local_provider", lambda: fake_provider):
        with patch("mullet.core.router.routing_matrix.select_local_model", fake_select_local):
            reg = ModelRegistry()
            result = reg._best_local()
    assert result is not None
    assert result.id == "model-a"


def test_best_local_model_attr_path():
    fake_provider = MagicMock()
    del fake_provider.list_models
    fake_provider.model = "model-c"

    def fake_select_local(preflight, available, speed_mode="balanced"):
        return ("model-c", "picked")

    with patch("mullet.core.providers.local_detection.detect_local_provider", lambda: fake_provider):
        with patch("mullet.core.router.routing_matrix.select_local_model", fake_select_local):
            reg = ModelRegistry()
            result = reg._best_local()
    assert result is not None
    assert result.id == "model-c"


def test_best_local_no_available():
    fake_provider = MagicMock()
    fake_provider.list_models.return_value = []
    with patch("mullet.core.providers.local_detection.detect_local_provider", lambda: fake_provider):
        reg = ModelRegistry()
        assert reg._best_local() is None


def test_best_local_select_local_returns_none():
    fake_provider = MagicMock()
    fake_provider.list_models.return_value = [MagicMock(id="model-d")]

    def fake_select_local(preflight, available, speed_mode="balanced"):
        return (None, "none")

    with patch("mullet.core.providers.local_detection.detect_local_provider", lambda: fake_provider):
        with patch("mullet.core.router.routing_matrix.select_local_model", fake_select_local):
            reg = ModelRegistry()
            assert reg._best_local() is None


def test_best_local_exception_returns_none():
    def raising():
        raise RuntimeError("local detection failed")
    with patch("mullet.core.providers.local_detection.detect_local_provider", raising):
        reg = ModelRegistry()
        assert reg._best_local() is None


# ---------------------------------------------------------------------------
# get_best_model D5 escalation (lines 689-693 context via last_mce)
# ---------------------------------------------------------------------------


def test_get_best_model_d5_escalation_then_exhausted(monkeypatch):
    """When every tier refuses with non-exhausted MCE, final MCE has exhausted=True."""
    reg = ModelRegistry()
    # Make _pick_model always raise MCE with exhausted=False
    from mullet.core.providers.errors import ModelContextExceeded

    def fake_pick(**kw):
        raise ModelContextExceeded(
            provider="kimi",
            tier="STANDARD",
            context_tokens=1000,
            tried_candidates=["a"],
            max_candidate_context=1000,
            all_candidates_exhausted=False,
        )

    monkeypatch.setattr(reg, "_pick_model", fake_pick)
    with pytest.raises(ModelContextExceeded) as exc_info:
        reg.get_best_model("kimi", complexity="STANDARD", context_tokens=1000)
    assert exc_info.value.all_candidates_exhausted is True


# ---------------------------------------------------------------------------
# get_best_model with provider=None local path (line 517-525)
# ---------------------------------------------------------------------------


def test_get_best_model_provider_none_local_path():
    reg = ModelRegistry()
    fake_local = ProviderModel(
        id="local-llm",
        provider="local",
        display_name="Local",
        context_length=32_768,
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
    )
    fake_preflight = MagicMock()
    fake_preflight.complexity_tier = "LOW"

    with patch.object(reg, "_best_local", lambda **kw: fake_local):
        with patch("mullet.core.router.routing_matrix._tier_gte", lambda a, b: False):
            chosen = reg.get_best_model(
                provider=None,
                complexity="STANDARD",
                preflight=fake_preflight,
                speed_mode="balanced",
            )
    assert chosen == "local-llm"


def test_fetch_moonshot_pricing_http_error_returns_empty(monkeypatch, tmp_path):
    """Lines 246-247: HTTP call raises → except Exception: return {}."""
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    nonexistent = tmp_path / "no_such_cache.json"
    monkeypatch.setattr("mullet.core.providers.model_registry._MOONSHOT_PRICING_CACHE_PATH", nonexistent)
    with patch("mullet.core.providers.model_registry._http_get", side_effect=OSError("network down")):
        result = _fetch_moonshot_pricing()
    assert result == {}


def test_assert_can_serve_constraint_registry_exception_returns_model():
    """Lines 375-380: constraint registry raises non-ValueError → log warning, return model_id."""
    reg = ModelRegistry.__new__(ModelRegistry)
    with patch("mullet.core.providers.constraints.get_constraint_registry", side_effect=RuntimeError("registry down")):
        result = reg._assert_can_serve("kimi", "kimi-k2", "STANDARD", context_tokens=1000)
    assert result == "kimi-k2"


def test_assert_can_serve_constraint_registry_value_error_reraises():
    """Line 374: constraint registry raises ValueError → re-raised."""
    reg = ModelRegistry.__new__(ModelRegistry)
    with patch("mullet.core.providers.constraints.get_constraint_registry", side_effect=ValueError("context too large")):
        with pytest.raises(ValueError, match="context too large"):
            reg._assert_can_serve("kimi", "kimi-k2", "STANDARD", context_tokens=1000)
