"""Tests for mullet.core.providers.model_registry.

10 tests using pytest + unittest.mock.  All HTTP calls are mocked —
no real API calls are made.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mullet.core.providers.model_registry import (
    ModelRegistry,
    ProviderModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry(tmp_path: Path | None = None) -> ModelRegistry:
    """Return a fresh ModelRegistry with an isolated cache path."""
    reg = ModelRegistry()
    if tmp_path is not None:
        reg.CACHE_PATH = tmp_path / "model_cache.json"  # type: ignore[misc]
    return reg


def _urlopen_response(payload: dict) -> MagicMock:
    """Return a context-manager mock that yields a urllib response for *payload*."""
    body = json.dumps(payload).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen = MagicMock(return_value=mock_resp)
    return mock_urlopen


# ---------------------------------------------------------------------------
# 1. Cache hit — get_models should NOT call the fetcher
# ---------------------------------------------------------------------------

def test_get_models_uses_cache(tmp_path: Path) -> None:
    """When cache is fresh, get_models returns cached data without fetching."""
    reg = _make_registry(tmp_path)
    static = reg._static_models("anthropic")
    # Seed in-memory cache as fresh
    reg._cache["anthropic"] = static
    reg._cache_ts["anthropic"] = time.time()

    with patch.object(reg, "_fetch_anthropic") as mock_fetch:
        models = reg.get_models("anthropic")

    mock_fetch.assert_not_called()
    assert len(models) == len(static)


# ---------------------------------------------------------------------------
# 2. Stale cache — get_models should trigger a fetch
# ---------------------------------------------------------------------------

def test_get_models_refreshes_stale_cache(tmp_path: Path) -> None:
    """When cache is stale, get_models triggers a provider fetch."""
    reg = _make_registry(tmp_path)
    static = reg._static_models("anthropic")
    # Make cache look old (>24h)
    reg._cache["anthropic"] = static
    reg._cache_ts["anthropic"] = time.time() - (reg.CACHE_TTL_HOURS * 3600 + 1)

    fetched_models = [
        ProviderModel(
            id="claude-sonnet-4-6",
            provider="anthropic",
            display_name="Claude Sonnet 4.6",
            context_length=1_000_000,
            input_cost_per_1k=0.003,
            output_cost_per_1k=0.015,
            supports_vision=True,
            tier="standard",
        )
    ]

    with (
        patch.object(reg, "_fetch_anthropic", return_value=fetched_models) as mock_fetch,
        patch.object(reg, "_save_cache"),
    ):
        models = reg.get_models("anthropic")

    mock_fetch.assert_called_once()
    assert any(m.id == "claude-sonnet-4-6" for m in models)


# ---------------------------------------------------------------------------
# 3. Force refresh — refresh=True fetches even if cache is fresh
# ---------------------------------------------------------------------------

def test_get_models_force_refresh(tmp_path: Path) -> None:
    """refresh=True bypasses a fresh cache and triggers a new fetch."""
    reg = _make_registry(tmp_path)
    static = reg._static_models("anthropic")
    # Cache is deliberately fresh
    reg._cache["anthropic"] = static
    reg._cache_ts["anthropic"] = time.time()

    new_model = ProviderModel(
        id="claude-opus-4-7",
        provider="anthropic",
        display_name="Claude Opus 4.7",
        context_length=1_000_000,
        input_cost_per_1k=0.005,
        output_cost_per_1k=0.025,
        supports_vision=True,
        tier="powerful",
    )

    with (
        patch.object(reg, "_fetch_anthropic", return_value=[new_model]) as mock_fetch,
        patch.object(reg, "_save_cache"),
    ):
        models = reg.get_models("anthropic", refresh=True)

    mock_fetch.assert_called_once()
    assert len(models) == 1
    assert models[0].id == "claude-opus-4-7"


# ---------------------------------------------------------------------------
# 4. _fetch_anthropic parses a mocked API response
# ---------------------------------------------------------------------------

def test_fetch_anthropic_parses_response(tmp_path: Path) -> None:
    """_fetch_anthropic correctly maps API response fields to ProviderModel."""
    reg = _make_registry(tmp_path)

    payload = {
        "data": [
            {"id": "claude-sonnet-4-6", "display_name": "Claude Sonnet 4.6"},
            {"id": "claude-haiku-4-5-20251001", "display_name": "Claude Haiku 4.5"},
        ]
    }

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        patch("urllib.request.urlopen", _urlopen_response(payload)),
    ):
        models = reg._fetch_anthropic()

    ids = {m.id for m in models}
    assert "claude-sonnet-4-6" in ids
    assert "claude-haiku-4-5-20251001" in ids
    for m in models:
        assert m.provider == "anthropic"
        assert m.context_length > 0  # pulled from MODEL_DATABASE


# ---------------------------------------------------------------------------
# 5. _fetch_kimi parses a mocked API response
# ---------------------------------------------------------------------------

def test_fetch_kimi_parses_response(tmp_path: Path) -> None:
    """_fetch_kimi correctly maps OpenAI-compat response fields to ProviderModel."""
    reg = _make_registry(tmp_path)

    payload = {
        "data": [
            {"id": "kimi-k2-0711-preview"},
            {"id": "kimi-k2-0905-preview"},
        ]
    }

    with (
        patch.dict("os.environ", {"MOONSHOT_API_KEY": "test-key"}),
        patch("urllib.request.urlopen", _urlopen_response(payload)),
    ):
        models = reg._fetch_kimi()

    ids = {m.id for m in models}
    assert "kimi-k2-0711-preview" in ids
    assert "kimi-k2-0905-preview" in ids
    for m in models:
        assert m.provider == "kimi"


# ---------------------------------------------------------------------------
# 6. _fetch_gemini parses a mocked API response
# ---------------------------------------------------------------------------

def test_fetch_gemini_parses_response(tmp_path: Path) -> None:
    """_fetch_gemini correctly maps Google API response fields to ProviderModel."""
    reg = _make_registry(tmp_path)

    payload = {
        "models": [
            {
                "name": "models/gemini-2.5-flash",
                "displayName": "Gemini 2.5 Flash",
                "inputTokenLimit": 1_000_000,
                "supportedGenerationMethods": ["generateContent"],
            },
            {
                "name": "models/gemini-2.5-pro",
                "displayName": "Gemini 2.5 Pro",
                "inputTokenLimit": 1_000_000,
                "supportedGenerationMethods": ["generateContent"],
            },
        ]
    }

    with (
        patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
        patch("urllib.request.urlopen", _urlopen_response(payload)),
    ):
        models = reg._fetch_gemini()

    ids = {m.id for m in models}
    assert "gemini-2.5-flash" in ids
    assert "gemini-2.5-pro" in ids
    for m in models:
        assert m.provider == "gemini"
        assert m.context_length == 1_000_000


# ---------------------------------------------------------------------------
# 7. HTTP error → static fallback
# ---------------------------------------------------------------------------

def test_fetch_falls_back_to_static_on_error(tmp_path: Path) -> None:
    """When urlopen raises, get_models still returns (static) models."""
    reg = _make_registry(tmp_path)
    # Clear cache so a fetch is attempted
    reg._cache.pop("anthropic", None)
    reg._cache_ts.pop("anthropic", None)

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        patch("urllib.request.urlopen", side_effect=OSError("network down")),
        patch.object(reg, "_save_cache"),
    ):
        models = reg.get_models("anthropic")

    assert len(models) > 0
    assert all("claude" in m.id for m in models)


# ---------------------------------------------------------------------------
# 8. get_best_model — LOW complexity returns cheap/fast model
# ---------------------------------------------------------------------------

def test_get_best_model_low_complexity() -> None:
    """LOW complexity should return a haiku/flash-class model."""
    reg = ModelRegistry()
    model_id = reg.get_best_model("anthropic", complexity="LOW")
    assert "haiku" in model_id.lower(), f"Expected haiku for LOW, got {model_id!r}"

    model_id_gemini = reg.get_best_model("gemini", complexity="LOW")
    assert "flash" in model_id_gemini.lower(), (
        f"Expected flash for LOW, got {model_id_gemini!r}"
    )


# ---------------------------------------------------------------------------
# 9. get_best_model — CRITICAL complexity returns powerful model
# ---------------------------------------------------------------------------

def test_get_best_model_critical_complexity() -> None:
    """CRITICAL complexity should return a model clearing the CRITICAL quality floor.

    Post routing-v3 (2026-07-10): the value picker prefers claude-sonnet-5 over
    any Opus variant here, since sonnet-5's coding score (71) clears the
    CRITICAL floor (65, anchored to claude-opus-4-7) at ~60% lower cost. This
    is intentional cost optimization, not a regression -- see
    docs/design/routing-v3-value-picker.md.
    """
    reg = ModelRegistry()
    model_id = reg.get_best_model("anthropic", complexity="CRITICAL")
    assert model_id == "claude-sonnet-5", f"Expected claude-sonnet-5 for CRITICAL, got {model_id!r}"

    model_id_gemini = reg.get_best_model("gemini", complexity="CRITICAL")
    assert "pro" in model_id_gemini.lower(), (
        f"Expected pro for CRITICAL (gemini), got {model_id_gemini!r}"
    )


# ---------------------------------------------------------------------------
# 10. refresh_all returns provider model counts
# ---------------------------------------------------------------------------

def test_refresh_all_returns_counts(tmp_path: Path) -> None:
    """refresh_all should return a dict mapping provider names to model counts."""
    reg = _make_registry(tmp_path)

    anthropic_models = reg._static_models("anthropic")
    kimi_models = reg._static_models("kimi")
    gemini_models = reg._static_models("gemini")

    with (
        patch.object(reg, "_fetch_anthropic", return_value=anthropic_models),
        patch.object(reg, "_fetch_kimi", return_value=kimi_models),
        patch.object(reg, "_fetch_gemini", return_value=gemini_models),
        patch.object(reg, "_save_cache"),
    ):
        counts = reg.refresh_all()

    assert "anthropic" in counts
    assert "kimi" in counts
    assert "gemini" in counts
    assert counts["anthropic"] == len(anthropic_models)
    assert counts["kimi"] == len(kimi_models)
    assert counts["gemini"] == len(gemini_models)


# ---------------------------------------------------------------------------
# 11. STANDARD kimi → prefers kimi-k2.6 when present in cache
# ---------------------------------------------------------------------------

def test_get_best_model_prefers_kimi_k26_for_standard() -> None:
    """STANDARD kimi task must return kimi-k2.6, not a deprecated model.

    This is the regression test for the HUD/wire desync bug: the live cache
    contained a Kimi K2 model and a cheaper deprecated model; the old heuristic
    picked the cheaper one while the HUD displayed Kimi K2.
    K2.6 is the preferred default as of 2026-05.
    """
    reg = ModelRegistry()
    # Populate live cache with kimi-k2.5 AND the preferred k2.6 model
    reg._cache["kimi"] = [
        ProviderModel(
            id="kimi-k2.5",
            provider="kimi",
            display_name="Kimi K2.5",
            context_length=262_144,
            input_cost_per_1k=0.00060,
            output_cost_per_1k=0.00300,
            tier="standard",
        ),
        ProviderModel(
            id="kimi-k2.6",
            provider="kimi",
            display_name="Kimi K2.6",
            context_length=262_144,
            input_cost_per_1k=0.00095,
            output_cost_per_1k=0.00400,
            tier="powerful",
        ),
    ]
    reg._cache_ts["kimi"] = time.time()
    model_id = reg.get_best_model("kimi", complexity="STANDARD", context_needed=4096)
    assert model_id == "kimi-k2.7-code", (
        f"Expected kimi-k2.7-code for STANDARD, got {model_id!r} — "
        "this is the HUD/wire desync bug"
    )


# ---------------------------------------------------------------------------
# 12. STANDARD kimi → injects kimi-k2.6 when missing from live fetch
# ---------------------------------------------------------------------------

def test_get_best_model_injects_k26_when_missing_from_live() -> None:
    """When live cache has only deprecated models, kimi-k2.6 is still returned.

    Fix #1 (PREFERRED_DEFAULTS short-circuit) returns kimi-k2.6 via static
    MODEL_DATABASE even when the live fetch didn't include it.
    """
    reg = ModelRegistry()
    # Live cache has ONLY older preview models — kimi-k2.6 missing
    reg._cache["kimi"] = [
        ProviderModel(
            id="kimi-k2-0905-preview",
            provider="kimi",
            display_name="Kimi K2 (0905-preview)",
            context_length=262_144,
            input_cost_per_1k=0.00060,
            output_cost_per_1k=0.00250,
            tier="standard",
        ),
    ]
    reg._cache_ts["kimi"] = time.time()
    model_id = reg.get_best_model("kimi", complexity="STANDARD", context_needed=4096)
    assert model_id == "kimi-k2.7-code", (
        f"Expected kimi-k2.7-code via static fallback, got {model_id!r}"
    )


# ---------------------------------------------------------------------------
# 13. LOW complexity — PREFERRED_DEFAULTS does NOT override
# ---------------------------------------------------------------------------

def test_get_best_model_low_complexity_still_picks_cheap() -> None:
    """LOW complexity must NOT be forced to kimi-k2.6 — pick cheapest available.

    Fix #1 only applies to STANDARD complexity. LOW tasks should still route
    to the cheapest eligible model (now kimi-k2.5 after deprecated previews
    were removed from the auto-pick ladder).
    """
    reg = ModelRegistry()
    reg._cache["kimi"] = [
        ProviderModel(
            id="kimi-k2.5",
            provider="kimi",
            display_name="Kimi K2.5",
            context_length=262_144,
            input_cost_per_1k=0.00060,
            output_cost_per_1k=0.00300,
            tier="standard",
        ),
        ProviderModel(
            id="kimi-k2.6",
            provider="kimi",
            display_name="Kimi K2.6",
            context_length=262_144,
            input_cost_per_1k=0.00095,
            output_cost_per_1k=0.00400,
            tier="powerful",
        ),
    ]
    reg._cache_ts["kimi"] = time.time()
    model_id = reg.get_best_model("kimi", complexity="LOW", context_needed=4096)
    # Should NOT be kimi-k2.6 — LOW tasks pick cheaper output cost
    assert model_id != "kimi-k2.6", "LOW should pick a cheaper model than k2.6"
    # The key invariant: PREFERRED_DEFAULTS is NOT applied for LOW
    assert model_id == "kimi-k2.5"


# ---------------------------------------------------------------------------
# 14. Non-kimi provider — PREFERRED_DEFAULTS not in dict → no change
# ---------------------------------------------------------------------------

def test_get_best_model_explicit_preferred_not_set(monkeypatch) -> None:
    """Providers not in PREFERRED_DEFAULTS are unaffected by Fix #1."""
    from mullet.core.providers.model_registry import PREFERRED_DEFAULTS, ProviderModel
    # Confirm "local" is not in PREFERRED_DEFAULTS
    assert "local" not in PREFERRED_DEFAULTS
    # get_best_model for "local" with empty cache returns empty string (no local models known)
    reg = ModelRegistry()
    reg._cache["local"] = []
    reg._cache_ts["local"] = time.time()
    # Pin _best_local so the test doesn't depend on a real LM Studio instance
    monkeypatch.setattr(
        reg,
        "_best_local",
        lambda **kw: ProviderModel(
            id="qwen2.5-7b-instruct",
            provider="local",
            display_name="Qwen2.5 7B Instruct",
            context_length=32_768,
            input_cost_per_1k=0.0,
            output_cost_per_1k=0.0,
        ),
    )
    model_id = reg.get_best_model("local", complexity="STANDARD", context_needed=4096)
    # Should not crash and should not return a kimi/anthropic/gemini model
    assert "kimi" not in model_id
    assert "claude" not in model_id
    assert "gemini" not in model_id


# --- internals (untested before 2026-04-21) -----------------------------

def test_pricing_for_known_and_unknown():
    from mullet.core.providers.model_registry import _pricing_for
    assert _pricing_for("kimi-k2.5") == (0.00060, 0.00300)
    assert _pricing_for("nonsense-xyz") == (0.0, 0.0)


def test_tier_for_classifies_models():
    from mullet.core.providers.model_registry import _tier_for
    assert _tier_for("claude-haiku-4-5") == "fast"
    assert _tier_for("claude-opus-4-7") == "powerful"
    assert _tier_for("gemini-2.5-flash") == "fast"
    assert _tier_for("random-standard-model") == "standard"


def test_context_for_known_and_unknown():
    from mullet.core.providers.model_registry import _context_for
    assert _context_for("kimi-k2.5") >= 200_000
    assert _context_for("unknown-model-xyz") == 32_000


def test_fetch_local_empty_on_http_error(monkeypatch):
    from mullet.core.providers import model_registry as mr
    def raising(*a, **kw):
        raise ConnectionError("lm studio offline")
    monkeypatch.setattr(mr, "_http_get", raising)
    reg = mr.ModelRegistry()
    assert reg._fetch_local() == []


def test_fetch_local_parses_models(monkeypatch):
    from mullet.core.providers import model_registry as mr
    def fake_get(*a, **kw):
        return {"data": [{
            "id": "qwen2.5-7b",
            "type": "llm",
            "max_context_length": 32768,
            "state": "loaded",
            "capabilities": ["tool_use"],
        }]}
    monkeypatch.setattr(mr, "_http_get", fake_get)
    reg = mr.ModelRegistry()
    out = reg._fetch_local()
    assert len(out) == 1
    assert out[0].id == "qwen2.5-7b"
    assert out[0].supports_tools is True


def test_list_all_contains_known_providers():
    from mullet.core.providers.model_registry import ModelRegistry
    reg = ModelRegistry()
    keys = set(reg.list_all().keys())
    assert {"anthropic", "kimi", "gemini", "local"}.issubset(keys)


def test_get_registry_is_singleton():
    from mullet.core.providers.model_registry import get_registry
    a = get_registry()
    b = get_registry()
    assert a is b


def test_fetch_and_cache_never_raises(monkeypatch):
    from mullet.core.providers import model_registry as mr
    reg = mr.ModelRegistry()
    def raising():
        raise RuntimeError("simulated API down")
    monkeypatch.setattr(reg, "_fetch_kimi", raising)
    reg._fetch_and_cache("kimi")
    assert "kimi" in reg._cache
    assert len(reg._cache["kimi"]) > 0


# ---------------------------------------------------------------------------
# 15-16. PREFERRED_DEFAULTS for anthropic + gemini
# ---------------------------------------------------------------------------


def test_get_best_model_prefers_sonnet_for_anthropic_standard() -> None:
    """Anthropic STANDARD should route to claude-sonnet-5 per PREFERRED_DEFAULTS."""
    from mullet.core.providers.model_registry import (
        ModelRegistry, PREFERRED_DEFAULTS, ProviderModel,
    )
    assert PREFERRED_DEFAULTS["anthropic"] == "claude-sonnet-5"

    reg = ModelRegistry()
    reg._cache["anthropic"] = [
        ProviderModel(
            id="claude-haiku-4-5-20251001",
            provider="anthropic",
            display_name="Claude Haiku 4.5",
            context_length=200_000,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.005,
            supports_vision=True,
            tier="fast",
        ),
        ProviderModel(
            id="claude-sonnet-5",
            provider="anthropic",
            display_name="Claude Sonnet 5",
            context_length=1_000_000,
            input_cost_per_1k=0.002,
            output_cost_per_1k=0.01,
            supports_vision=True,
            tier="standard",
        ),
        ProviderModel(
            id="claude-sonnet-4-6",
            provider="anthropic",
            display_name="Claude Sonnet 4.6",
            context_length=1_000_000,
            input_cost_per_1k=0.003,
            output_cost_per_1k=0.015,
            supports_vision=True,
            tier="standard",
        ),
    ]
    reg._cache_ts["anthropic"] = time.time()

    model_id = reg.get_best_model("anthropic", complexity="STANDARD", context_needed=4096)
    assert model_id == "claude-sonnet-5", f"expected sonnet via PREFERRED_DEFAULTS; got {model_id!r}"


def test_get_best_model_prefers_gemini_flash_for_gemini_standard() -> None:
    """Gemini STANDARD should route to gemini-2.5-flash per PREFERRED_DEFAULTS."""
    from mullet.core.providers.model_registry import (
        ModelRegistry, PREFERRED_DEFAULTS, ProviderModel,
    )
    assert PREFERRED_DEFAULTS["gemini"] == "gemini-2.5-flash"

    reg = ModelRegistry()
    reg._cache["gemini"] = [
        ProviderModel(
            id="gemini-3.1-pro",
            provider="gemini",
            display_name="Gemini 3.1 Pro",
            context_length=2_000_000,
            input_cost_per_1k=0.002,
            output_cost_per_1k=0.012,
            supports_vision=True,
            tier="powerful",
        ),
        ProviderModel(
            id="gemini-3.5-flash",
            provider="gemini",
            display_name="Gemini 3.5 Flash",
            context_length=1_048_576,
            input_cost_per_1k=0.0015,
            output_cost_per_1k=0.009,
            supports_vision=True,
            tier="fast",
        ),
        ProviderModel(
            id="gemini-2.5-pro",
            provider="gemini",
            display_name="Gemini 2.5 Pro",
            context_length=1_000_000,
            input_cost_per_1k=0.00125,
            output_cost_per_1k=0.010,
            supports_vision=True,
            tier="powerful",
        ),
        ProviderModel(
            id="gemini-2.5-flash",
            provider="gemini",
            display_name="Gemini 2.5 Flash",
            context_length=1_000_000,
            input_cost_per_1k=0.00015,
            output_cost_per_1k=0.00035,
            supports_vision=True,
            tier="fast",
        ),
    ]
    reg._cache_ts["gemini"] = time.time()

    model_id = reg.get_best_model("gemini", complexity="STANDARD", context_needed=4096)
    assert model_id == "gemini-2.5-flash", f"expected flash via PREFERRED_DEFAULTS; got {model_id!r}"


def test_get_best_model_preferred_skipped_when_context_too_small() -> None:
    """If preferred model can't fit context_needed, fall back to alternatives."""
    from mullet.core.providers.model_registry import (
        ModelRegistry, ProviderModel,
    )

    reg = ModelRegistry()
    # Populate only a short-context model for kimi
    reg._cache["kimi"] = [
        ProviderModel(
            id="kimi-k2-0711-preview",
            provider="kimi",
            display_name="Kimi K2 (0711-preview)",
            context_length=131_072,
            input_cost_per_1k=0.00060,
            output_cost_per_1k=0.00250,
            tier="standard",
        ),
    ]
    reg._cache_ts["kimi"] = time.time()

    # Request STANDARD with context > 131k. kimi-k2.5 is NOT in the cache → preferred
    # cannot be served from cache. Fix #2 (static fallback inject) picks it up anyway.
    model_id = reg.get_best_model("kimi", complexity="STANDARD", context_needed=150_000)
    # Should either return kimi-k2.5 (via static fallback inject) or any model with
    # context >= 150_000. Assert NOT k2-0711 since that has only 131k context.
    assert model_id != "kimi-k2-0711-preview", (
        f"kimi-k2-0711-preview can't fit 150k context; router should skip it; got {model_id!r}"
    )


# ---------------------------------------------------------------------------
# Allowlist enforcement — G11
# ---------------------------------------------------------------------------


def test_get_best_model_raises_when_provider_not_in_allowlist() -> None:
    """Providers outside the allowlist are rejected with ValueError."""
    import pytest
    reg = ModelRegistry()
    with pytest.raises(ValueError, match="not in allowed_providers"):
        reg.get_best_model("gemini", complexity="STANDARD", allowed_providers={"kimi"})


def test_get_best_model_allowlist_hit_returns_model() -> None:
    """Providers inside the allowlist behave normally."""
    reg = ModelRegistry()
    model_id = reg.get_best_model(
        "kimi", complexity="STANDARD", allowed_providers={"kimi"}
    )
    assert model_id


def test_get_best_model_allowlist_none_preserves_behaviour() -> None:
    """No allowlist is a no-op."""
    reg = ModelRegistry()
    model_id = reg.get_best_model("kimi", complexity="STANDARD")
    assert model_id


# ---------------------------------------------------------------------------
# Constraint-registry consultation on PREFERRED_DEFAULTS — G17
# ---------------------------------------------------------------------------


def test_preferred_default_falls_through_when_constraint_rejects(monkeypatch) -> None:
    """When context_tokens is given and constraint registry rejects preferred,
    get_best_model must fall through to heuristic selection (never return the preferred).

    Uses a context_tokens value that FITS the kimi-k2 family (262K) but NOT
    kimi-k2-0711-preview (max 131K). Reject the preferred kimi-k2.5 only; let the
    real registry decide for other candidates so the D2 post-pick assertion
    doesn't spuriously raise.
    """
    from mullet.core.providers import constraints as _c

    reg = ModelRegistry()

    real_get = _c.ConstraintRegistry.get

    def selective_get(self, model_id):
        if model_id == "kimi-k2.5":
            fake = MagicMock()
            fake.can_serve.return_value = False
            fake.max_context = 262_144
            return fake
        return real_get(self, model_id)

    monkeypatch.setattr(_c.ConstraintRegistry, "get", selective_get)

    # 200_000 tokens fits all kimi-k2 models (262K) but not k2-0711 (131K).
    model_id = reg.get_best_model(
        "kimi",
        complexity="STANDARD",
        context_needed=4096,
        context_tokens=200_000,
    )
    assert isinstance(model_id, str) and model_id
    # Key invariant: must not be kimi-k2-0711-preview (it can't hold 200k)
    assert model_id != "kimi-k2-0711-preview", (
        f"heuristic must respect context_tokens; got {model_id!r}"
    )
    # Key invariant: must not be kimi-k2.5 either — it was forced to fall-through
    assert model_id != "kimi-k2.5", (
        f"preferred kimi-k2.5 was rejected; heuristic must pick a sibling, got {model_id!r}"
    )


def test_preferred_default_returns_when_no_context_tokens(monkeypatch) -> None:
    """Without context_tokens, preferred short-circuits and constraint registry is not consulted."""
    from mullet.core.providers import constraints as _c

    reg = ModelRegistry()
    # Make constraint registry raise if accidentally consulted.
    fake_constraint = MagicMock()
    fake_constraint.can_serve.side_effect = AssertionError("should not be called")
    monkeypatch.setattr(
        _c.ConstraintRegistry, "get",
        lambda self, model_id: fake_constraint,
    )

    # No context_tokens → preferred returned without consulting constraints.
    model_id = reg.get_best_model(
        "kimi",
        complexity="STANDARD",
        context_needed=4096,
    )
    assert model_id == "kimi-k2.7-code"


# ---------------------------------------------------------------------------
# P1-3 — Context-aware model picker + log downgrades at INFO
# ---------------------------------------------------------------------------


def test_best_kimi_for_large_context_picks_k2_5_not_moonshot_v1(monkeypatch):
    """With context_tokens=150_000 (>128K), must not downgrade to a tiny-context model.

    Before P1-3 the heuristic ignored prompt size and could silently pick a
    small-context model as a "STANDARD cheap" choice, which the constraint
    registry then rejects downstream. D2 adds the 10% safety margin so
    only models with sufficient context are eligible.
    """
    from mullet.core.providers import constraints as _c
    from mullet.core.providers.model_registry import ModelRegistry

    # Reject only the preferred kimi-k2.5 so we exercise the heuristic.
    # Letting the real registry decide for other candidates lets the D2
    # post-pick assertion validate the final choice.
    real_get = _c.ConstraintRegistry.get

    def selective_get(self, model_id):
        if model_id == "kimi-k2.5":
            fake = MagicMock()
            fake.can_serve.return_value = False
            fake.max_context = 262_144
            return fake
        return real_get(self, model_id)

    monkeypatch.setattr(_c.ConstraintRegistry, "get", selective_get)

    reg = ModelRegistry()
    # Empty live cache → static fallback in _best_kimi runs.
    reg._cache["kimi"] = []
    reg._cache_ts["kimi"] = time.time()

    chosen = reg.get_best_model(
        "kimi",
        complexity="STANDARD",
        context_needed=4096,
        context_tokens=150_000,
    )
    assert chosen in ("kimi-k2.7-code", "kimi-k2.6", "kimi-k2.5"), (
        f"STANDARD + 150K context must pick a suitable K2 model; got {chosen!r}"
    )


def test_best_kimi_raises_clear_error_when_all_candidates_too_small(monkeypatch):
    """context_tokens > every candidate's max_context → ValueError with candidate list."""
    from mullet.core.providers import constraints as _c
    from mullet.core.providers.model_registry import ModelRegistry

    # Force heuristic path — preferred constraint rejects.
    fake_constraint = MagicMock()
    fake_constraint.can_serve.return_value = False
    monkeypatch.setattr(
        _c.ConstraintRegistry, "get",
        lambda self, model_id: fake_constraint,
    )

    reg = ModelRegistry()
    reg._cache["kimi"] = []
    reg._cache_ts["kimi"] = time.time()

    with pytest.raises(ValueError) as exc_info:
        reg.get_best_model(
            "kimi",
            complexity="STANDARD",
            context_needed=4096,
            context_tokens=1_000_000,
        )
    msg = str(exc_info.value)
    assert "kimi" in msg.lower()
    assert "candidates tried" in msg.lower()
    assert "max context" in msg.lower()


def test_downgrade_is_logged_at_info(monkeypatch, caplog):
    """Any time we move off the provider's PREFERRED_DEFAULT, emit INFO log."""
    from mullet.core.providers import constraints as _c
    from mullet.core.providers.model_registry import ModelRegistry

    # Reject the preferred kimi-k2.6 so the heuristic falls back to k2.5 and
    # logs the downgrade. Let real registry handle other candidates so the D2
    # post-pick assertion accepts them.
    real_get = _c.ConstraintRegistry.get

    def selective_get(self, model_id):
        if model_id == "kimi-k2.6":
            fake = MagicMock()
            fake.can_serve.return_value = False
            fake.max_context = 262_144
            return fake
        return real_get(self, model_id)

    monkeypatch.setattr(_c.ConstraintRegistry, "get", selective_get)

    reg = ModelRegistry()
    reg._cache["kimi"] = []
    reg._cache_ts["kimi"] = time.time()

    # context_tokens=4096 at LOW tier → heuristic picks cheapest eligible model.
    # Downgrade from the preferred default is logged.
    with caplog.at_level("INFO", logger="mullet.core.providers.model_registry"):
        chosen = reg.get_best_model(
            "kimi",
            complexity="LOW",  # LOW + small context → cheapest eligible model (downgrade)
            context_needed=4096,
            context_tokens=4096,
        )
    # A downgrade from kimi-k2.6 to k2.5 at LOW tier should log.
    if chosen != "kimi-k2.6":
        assert any("downgrade" in rec.getMessage() for rec in caplog.records), (
            f"expected a 'downgrade' log line in INFO records; got {[r.getMessage() for r in caplog.records]}"
        )


def test_low_tier_short_prompt_picks_cheapest_k2():
    """Prompt <=7500 tokens, LOW tier, kimi-only → cheapest K2 model (cost win)."""
    reg = ModelRegistry()
    reg._cache["kimi"] = []  # force static fallback
    reg._cache_ts["kimi"] = time.time()
    chosen = reg.get_best_model("kimi", complexity="LOW", context_tokens=4_000)
    assert chosen == "kimi-k2.5", f"Short LOW prompt should pick cheapest K2 model, got {chosen!r}"


def test_low_tier_medium_prompt_picks_cheapest_k2_not_turbo():
    """Prompt 8K-240K tokens, LOW tier → cheapest K2 model, NOT k2-turbo-preview."""
    reg = ModelRegistry()
    reg._cache["kimi"] = []
    reg._cache_ts["kimi"] = time.time()
    chosen = reg.get_best_model("kimi", complexity="LOW", context_tokens=50_000)
    assert chosen == "kimi-k2.5", f"Medium LOW prompt should pick cheapest K2 model, got {chosen!r}"
    assert chosen != "kimi-k2-turbo-preview", "kimi-k2-turbo-preview is expensive and must not auto-pick for LOW"


def test_kimi_capabilities_include_low():
    from mullet.core.router.complexity import PROVIDER_TIER_CAPABILITIES
    assert "LOW" in PROVIDER_TIER_CAPABILITIES["kimi"], "kimi must advertise LOW tier"


def test_reasoning_prompt_routes_to_thinking():
    reg = ModelRegistry()
    reg._cache["kimi"] = []
    reg._cache_ts["kimi"] = time.time()
    chosen = reg.get_best_model("kimi", complexity="STANDARD", prefer_reasoning=True)
    assert chosen == "kimi-k2.7-code", f"Reasoning flag should pick kimi-k2.7-code (kimi-k2-thinking EOL 2026-05-25), got {chosen!r}"


# ---------------------------------------------------------------------------
# OpenRouter routing tests
# ---------------------------------------------------------------------------

def test_openrouter_standard_returns_v4_flash():
    """STANDARD complexity for openrouter must short-circuit to deepseek-v4-flash."""
    reg = ModelRegistry()
    reg._cache["openrouter"] = []
    reg._cache_ts["openrouter"] = time.time()
    chosen = reg.get_best_model("openrouter", complexity="STANDARD", context_needed=4096)
    assert chosen == "deepseek/deepseek-v4-flash", (
        f"openrouter STANDARD should pick v4-flash via PREFERRED_DEFAULTS, got {chosen!r}"
    )


def test_openrouter_advanced_returns_v4_flash():
    """ADVANCED complexity for openrouter picks the cheapest floor-clearing candidate.

    Post routing-v3 (2026-07-10): the value picker prefers deepseek-v4-flash
    over v4-pro here, since v4-flash's coding score (59) already clears the
    ADVANCED floor (54) at lower cost. Intentional cost optimization, not a
    regression -- see docs/design/routing-v3-value-picker.md.
    """
    reg = ModelRegistry()
    reg._cache["openrouter"] = []
    reg._cache_ts["openrouter"] = time.time()
    chosen = reg.get_best_model("openrouter", complexity="ADVANCED", context_needed=4096)
    assert chosen == "deepseek/deepseek-v4-flash", (
        f"openrouter ADVANCED should pick v4-flash via _best_openrouter, got {chosen!r}"
    )


def test_openrouter_low_returns_v4_flash():
    """LOW complexity for openrouter picks deepseek-v4-flash.

    The free-tier list (OPENROUTER_FREE_TIER_CANDIDATES in
    value_picker_constants.py) is empty since OpenRouter withdrew
    qwen/qwen3-coder:free and tencent/hy3:free, so LOW uses the paid ladder.
    """
    reg = ModelRegistry()
    reg._cache["openrouter"] = []
    reg._cache_ts["openrouter"] = time.time()
    chosen = reg.get_best_model("openrouter", complexity="LOW", context_needed=4096)
    assert chosen == "deepseek/deepseek-v4-flash", (
        f"openrouter LOW should pick v4-flash, got {chosen!r}"
    )


def test_fetch_moonshot_pricing_no_api_key_returns_empty(monkeypatch):
    """Line 239: no MOONSHOT_API_KEY or KIMI_API_KEY → early return {}."""
    from mullet.core.providers.model_registry import _fetch_moonshot_pricing
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    with patch("mullet.core.providers.model_registry._MOONSHOT_PRICING_CACHE_PATH",
               Path("/nonexistent/path.json")):
        result = _fetch_moonshot_pricing()
    assert result == {}
