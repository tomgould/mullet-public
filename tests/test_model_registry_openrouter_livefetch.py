"""Tests for OpenRouter live-fetch in ModelRegistry.

All HTTP calls are mocked — no real API calls are made.
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
# 1. _fetch_openrouter parses a mocked HTTP response correctly
# ---------------------------------------------------------------------------

def test_fetch_openrouter_parses_response(tmp_path: Path) -> None:
    """_fetch_openrouter correctly maps API response fields to ProviderModel, including ×1000 pricing."""
    reg = _make_registry(tmp_path)

    payload = {
        "data": [
            {
                "id": "deepseek/deepseek-v4-flash",
                "name": "DeepSeek V4 Flash",
                "context_length": 1_000_000,
                "pricing": {"prompt": "0.000000098", "completion": "0.000000196"},
            },
            {
                "id": "deepseek/deepseek-v4-pro",
                "name": "DeepSeek V4 Pro",
                "context_length": 1_000_000,
                "pricing": {"prompt": "0.000000435", "completion": "0.00000087"},
            },
        ]
    }

    with patch("urllib.request.urlopen", _urlopen_response(payload)):
        models = reg._fetch_openrouter()

    assert len(models) == 2
    by_id = {m.id: m for m in models}

    flash = by_id["deepseek/deepseek-v4-flash"]
    assert flash.provider == "openrouter"
    assert flash.display_name == "DeepSeek V4 Flash"
    assert flash.context_length == 1_000_000
    # pricing is per-token in API → per-1k in ProviderModel
    assert flash.input_cost_per_1k == pytest.approx(0.000098)
    assert flash.output_cost_per_1k == pytest.approx(0.000196)
    assert flash.supports_tools is True
    assert flash.supports_vision is False
    assert flash.is_reasoning is False

    v4_pro = by_id["deepseek/deepseek-v4-pro"]
    assert v4_pro.input_cost_per_1k == pytest.approx(0.000435)
    assert v4_pro.output_cost_per_1k == pytest.approx(0.00087)
    assert v4_pro.is_reasoning is False


# ---------------------------------------------------------------------------
# 2. _fetch_openrouter skips malformed entries gracefully
# ---------------------------------------------------------------------------

def test_fetch_openrouter_skips_malformed_entries(tmp_path: Path) -> None:
    """Missing id or bad pricing dict should not crash; bad entries are skipped."""
    reg = _make_registry(tmp_path)

    payload = {
        "data": [
            {"id": "good/model", "context_length": 128_000, "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
            {"name": "No ID", "context_length": 128_000, "pricing": {"prompt": "0.000001", "completion": "0.000002"}},  # missing id
            {"id": "bad-pricing", "context_length": 128_000, "pricing": {"prompt": None, "completion": "oops"}},  # bad pricing
            {"id": "missing-pricing", "context_length": 128_000},  # no pricing at all
        ]
    }

    with patch("urllib.request.urlopen", _urlopen_response(payload)):
        models = reg._fetch_openrouter()

    ids = [m.id for m in models]
    assert "good/model" in ids
    assert "bad-pricing" in ids  # falls back to 0.0, 0.0 pricing
    assert "missing-pricing" in ids  # falls back to 0.0, 0.0 pricing
    assert "No ID" not in ids
    assert "name" not in ids


# ---------------------------------------------------------------------------
# 3. _fetch_and_cache("openrouter") populates the cache
# ---------------------------------------------------------------------------

def test_fetch_and_cache_populates_openrouter(tmp_path: Path) -> None:
    """Calling _fetch_and_cache for openrouter should populate self._cache."""
    reg = _make_registry(tmp_path)

    fetched = [
        ProviderModel(
            id="deepseek/deepseek-v4-flash",
            provider="openrouter",
            display_name="DeepSeek V4 Flash",
            context_length=1_000_000,
            input_cost_per_1k=0.000098,
            output_cost_per_1k=0.000196,
            tier="standard",
        )
    ]

    with (
        patch.object(reg, "_fetch_openrouter", return_value=fetched),
        patch.object(reg, "_save_cache"),
    ):
        reg._fetch_and_cache("openrouter")

    assert "openrouter" in reg._cache
    assert len(reg._cache["openrouter"]) == 1
    assert reg._cache["openrouter"][0].id == "deepseek/deepseek-v4-flash"


# ---------------------------------------------------------------------------
# 4. refresh_all() and list_all() include openrouter
# ---------------------------------------------------------------------------

def test_refresh_all_includes_openrouter(tmp_path: Path) -> None:
    """refresh_all must include 'openrouter' in returned counts."""
    reg = _make_registry(tmp_path)

    static = reg._static_models("openrouter")

    with (
        patch.object(reg, "_fetch_anthropic", return_value=reg._static_models("anthropic")),
        patch.object(reg, "_fetch_kimi", return_value=reg._static_models("kimi")),
        patch.object(reg, "_fetch_gemini", return_value=reg._static_models("gemini")),
        patch.object(reg, "_fetch_openrouter", return_value=static),
        patch.object(reg, "_fetch_local", return_value=[]),
        patch.object(reg, "_save_cache"),
    ):
        counts = reg.refresh_all()

    assert "openrouter" in counts
    assert counts["openrouter"] == len(static)


def test_list_all_includes_openrouter() -> None:
    """list_all must include 'openrouter' in its returned keys."""
    reg = ModelRegistry()
    keys = set(reg.list_all().keys())
    assert "openrouter" in keys


# ---------------------------------------------------------------------------
# 5. _best_openrouter backward-compat: empty live cache returns same ids as before
# ---------------------------------------------------------------------------

def test_best_openrouter_empty_cache_backward_compat() -> None:
    """With an empty live cache, _best_openrouter must return the exact same hardcoded ids as before."""
    reg = ModelRegistry()
    reg._cache["openrouter"] = []
    reg._cache_ts["openrouter"] = time.time()

    # No free-tier candidates are left (OpenRouter withdrew them), so LOW
    # takes the cheapest paid model.
    assert reg._best_openrouter("LOW") == "deepseek/deepseek-v4-flash"
    assert reg._best_openrouter("STANDARD") == "deepseek/deepseek-v4-flash"
    # routing-v3 Phase 3a (OpenRouter): with an empty live cache, ADVANCED now
    # goes through the value picker instead of the old static "elif ADVANCED"
    # branch (that branch is only reached as a fallback on exception/None).
    # deepseek-v4-flash (coding=59) clears the recalibrated ADVANCED floor (54)
    # and costs ~4.8x less than deepseek-v4-pro -- an approved, intentional
    # divergence from the old static default, not a bug. See
    # test_openrouter_value_picker_integration.py for the full rationale.
    assert reg._best_openrouter("ADVANCED") == "deepseek/deepseek-v4-flash"
    assert reg._best_openrouter("CRITICAL") == "deepseek/deepseek-v4-pro"
    assert reg._best_openrouter("STANDARD", prefer_reasoning=True) == "deepseek/deepseek-v4-pro"


# ---------------------------------------------------------------------------
# 6. _best_openrouter self-heal with live cache
# ---------------------------------------------------------------------------

def test_best_openrouter_self_heal_present_in_live() -> None:
    """When preferred model is present in live cache, return it unchanged."""
    reg = ModelRegistry()
    reg._cache["openrouter"] = [
        ProviderModel(
            id="deepseek/deepseek-v4-pro",
            provider="openrouter",
            display_name="DeepSeek V4 Pro",
            context_length=1_000_000,
            input_cost_per_1k=0.000435,
            output_cost_per_1k=0.00087,
            tier="powerful",
        ),
    ]
    reg._cache_ts["openrouter"] = time.time()

    assert reg._best_openrouter("ADVANCED") == "deepseek/deepseek-v4-pro"


def test_best_openrouter_self_heal_missing_in_live(monkeypatch) -> None:
    """When preferred model is absent from live cache, fall back via _pick_from_live."""
    reg = ModelRegistry()
    reg._cache["openrouter"] = [
        ProviderModel(
            id="deepseek/deepseek-v4-flash",
            provider="openrouter",
            display_name="DeepSeek V4 Flash",
            context_length=1_000_000,
            input_cost_per_1k=0.000098,
            output_cost_per_1k=0.000196,
            tier="standard",
        ),
    ]
    reg._cache_ts["openrouter"] = time.time()

    # ADVANCED preferred is deepseek-v4-pro, which is NOT in the live cache.
    chosen = reg._best_openrouter("ADVANCED")
    assert chosen != "deepseek/deepseek-v4-pro"
    # It should fall back to the only live model available
    assert chosen == "deepseek/deepseek-v4-flash"
