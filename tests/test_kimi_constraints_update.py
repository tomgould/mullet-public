"""Regression tests for Kimi/Moonshot constraint + registry updates.

Audit against https://platform.kimi.ai/docs/models via the `kimi_docs` Trawl
pool (2026-04-24). Locks the corrections we made after comparing Mullet's
internal data to Moonshot's authoritative spec:

- `kimi-k2-0711-preview` context is 128K, NOT 256K.
- `kimi-k2.6` / `kimi-k2.5` are native multimodal — `supports_vision=True`.
- `kimi-k2-turbo-preview`, `kimi-k2-thinking`, `kimi-k2-thinking-turbo`
  exist in the constraint + pricing tables.
"""

from __future__ import annotations

import pytest

from mullet.core.providers.constraints import get_constraint_registry
from mullet.core.providers.model_info import MODEL_DATABASE, _ALIASES
from mullet.core.providers.model_registry import KNOWN_PRICING


def test_k2_0711_preview_context_is_128k() -> None:
    reg = get_constraint_registry()
    c = reg.get("kimi-k2-0711-preview")
    assert c is not None
    assert c.max_context == 131_072, "0711-preview is 128K per Moonshot docs, not 256K"
    assert MODEL_DATABASE["kimi-k2-0711-preview"].context_length == 131_072


def test_k2_6_is_multimodal() -> None:
    c = get_constraint_registry().get("kimi-k2.6")
    assert c is not None
    assert c.supports_vision is True


def test_k2_5_is_multimodal() -> None:
    c = get_constraint_registry().get("kimi-k2.5")
    assert c is not None
    assert c.supports_vision is True


def test_k2_preview_series_is_not_multimodal() -> None:
    """K2 preview series ("Does not support vision functionality" per docs)."""
    for model_id in (
        "kimi-k2-0711-preview",
        "kimi-k2-0905-preview",
        "kimi-k2-turbo-preview",
        "kimi-k2-thinking",
        "kimi-k2-thinking-turbo",
    ):
        c = get_constraint_registry().get(model_id)
        assert c is not None, model_id
        assert c.supports_vision is False, f"{model_id} shouldn't claim vision"


@pytest.mark.parametrize(
    "model_id",
    [
        "kimi-k2-turbo-preview",
        "kimi-k2-thinking",
        "kimi-k2-thinking-turbo",
    ],
)
def test_new_k2_preview_models_registered(model_id: str) -> None:
    c = get_constraint_registry().get(model_id)
    assert c is not None, f"{model_id} missing from constraints"
    assert c.max_context == 262_144
    assert c.supports_tools is True
    assert c.supports_json_mode is True
    assert model_id in KNOWN_PRICING
    assert model_id in MODEL_DATABASE


def test_max_output_tokens_populated_for_k2_family() -> None:
    """K2 series output ceiling should equal max_context per Moonshot FAQ."""
    for model_id in (
        "kimi-k2.6",
        "kimi-k2.5",
        "kimi-k2-0905-preview",
        "kimi-k2-turbo-preview",
        "kimi-k2-thinking",
        "kimi-k2-thinking-turbo",
    ):
        c = get_constraint_registry().get(model_id)
        assert c is not None, model_id
        assert c.max_output_tokens == 262_144, (
            f"{model_id}: max_output should mirror 256K context ceiling"
        )


def test_resolve_model_aliases_for_kimi() -> None:
    assert _ALIASES["kimi-k2-turbo"] == "kimi-k2-turbo-preview"
    assert _ALIASES["kimi-thinking"] == "kimi-k2-thinking"
    assert _ALIASES["kimi-thinking-turbo"] == "kimi-k2-thinking-turbo"


# ---------------------------------------------------------------------------
# thinking_required — shape_request() strips enable_thinking=False
# ---------------------------------------------------------------------------

def test_k2_7_code_has_thinking_required() -> None:
    c = get_constraint_registry().get("kimi-k2.7-code")
    assert c is not None
    assert c.thinking_required is True


def test_k2_7_code_highspeed_has_thinking_required() -> None:
    c = get_constraint_registry().get("kimi-k2.7-code-highspeed")
    assert c is not None
    assert c.thinking_required is True


def test_k2_6_does_not_require_thinking() -> None:
    c = get_constraint_registry().get("kimi-k2.6")
    assert c is not None
    assert c.thinking_required is False


def test_shape_request_strips_enable_thinking_false_for_k2_7() -> None:
    from mullet.core.providers.constraints import shape_request
    raw = {"messages": [], "enable_thinking": False, "max_tokens": 1000}
    shaped = shape_request("kimi-k2.7-code", raw)
    assert "enable_thinking" not in shaped, "enable_thinking=False must be stripped for thinking_required models"


def test_shape_request_keeps_enable_thinking_true_for_k2_7() -> None:
    from mullet.core.providers.constraints import shape_request
    raw = {"messages": [], "enable_thinking": True, "max_tokens": 1000}
    shaped = shape_request("kimi-k2.7-code", raw)
    assert shaped.get("enable_thinking") is True


def test_shape_request_does_not_strip_enable_thinking_for_k2_6() -> None:
    from mullet.core.providers.constraints import shape_request
    raw = {"messages": [], "enable_thinking": False, "max_tokens": 1000}
    shaped = shape_request("kimi-k2.6", raw)
    assert shaped.get("enable_thinking") is False, "k2.6 does not require thinking — must keep the flag"
