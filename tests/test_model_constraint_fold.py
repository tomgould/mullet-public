"""Cycle 25 — regression guards for ModelInfo.supports_* → ModelConstraints fold.

These tests pin the behaviour where ModelConstraints is the single source of
truth for capability flags and ModelInfo delegates via back-compat properties.
"""

from __future__ import annotations


from mullet.core.providers.constraints import get_constraint_registry
from mullet.core.providers.model_info import ModelInfo, get_model_info


def test_constraints_supports_tools_reflects_static_data():
    """ConstraintRegistry.get must return correct supports_tools per static table.

    Expected failure mode if regressed: moonshot-v1-8k.supports_tools flips to True
    (defaulting instead of consulting STATIC_CONSTRAINTS) or kimi-k2.5 flips to False.
    """
    reg = get_constraint_registry()
    assert reg.get("moonshot-v1-8k").supports_tools is False
    assert reg.get("kimi-k2.5").supports_tools is True


def test_constraints_supports_streaming_matches_static_data():
    """o1-mini must be supports_streaming=False, kimi-k2.5 must be True.

    Expected failure mode if regressed: dataclass default leaks through for o1-mini.
    """
    reg = get_constraint_registry()
    assert reg.get("o1-mini").supports_streaming is False
    assert reg.get("kimi-k2.5").supports_streaming is True


def test_constraints_supports_vision_matches_static_data():
    """claude-sonnet-4-6 must have vision, moonshot-v1-8k must not.

    Expected failure mode if regressed: vision default (False) overrides static True.
    """
    reg = get_constraint_registry()
    assert reg.get("claude-sonnet-4-6").supports_vision is True
    assert reg.get("moonshot-v1-8k").supports_vision is False


def test_modelinfo_back_compat_supports_tools_routes_to_constraints():
    """ModelInfo.supports_tools must delegate to ConstraintRegistry.

    Pre-fold bug: ModelInfo.supports_tools defaults True, so moonshot-v1-8k
    reported tool support even though ConstraintRegistry correctly said False.
    Post-fold: both must agree — moonshot-v1-8k reports False.

    Expected failure mode if regressed: the field is re-added to ModelInfo and
    diverges from ConstraintRegistry.
    """
    info = get_model_info("moonshot-v1-8k")
    assert info.supports_tools is False, (
        "ModelInfo.supports_tools must route through ConstraintRegistry; "
        "moonshot-v1-8k does not support tools"
    )
    info2 = get_model_info("kimi-k2.5")
    assert info2.supports_tools is True


def test_modelinfo_back_compat_supports_vision_routes_to_constraints():
    """ModelInfo.supports_vision must delegate to ConstraintRegistry.

    Expected failure mode if regressed: reverts to reading a local field and
    picks up the default False even when constraints say True.
    """
    info = get_model_info("claude-sonnet-4-6")
    assert info.supports_vision is True
    info2 = get_model_info("moonshot-v1-8k")
    assert info2.supports_vision is False


def test_modelinfo_back_compat_supports_streaming_routes_to_constraints():
    """ModelInfo.supports_streaming must delegate to ConstraintRegistry.

    Expected failure mode if regressed: o1-mini reports True from dataclass default.
    """
    info = get_model_info("o1-mini")
    assert info.supports_streaming is False
    info2 = get_model_info("kimi-k2.5")
    assert info2.supports_streaming is True


def test_can_serve_checks_supports_flags():
    """ModelConstraints.can_serve must enforce requires_tools / requires_streaming.

    Expected failure mode if regressed: can_serve returns True for a tool-requiring
    task on moonshot-v1-8k (a known tool-less model).
    """
    reg = get_constraint_registry()
    moon = reg.get("moonshot-v1-8k")
    kimi = reg.get("kimi-k2.5")
    assert moon.can_serve({"requires_tools": True}) is False
    assert kimi.can_serve({"requires_tools": True}) is True
    o1 = reg.get("o1-mini")
    assert o1.can_serve({"requires_streaming": True}) is False
    assert kimi.can_serve({"requires_streaming": True}) is True


def test_modelinfo_does_not_carry_its_own_supports_fields():
    """ModelInfo dataclass must NOT declare supports_* as fields (single source of truth).

    Expected failure mode if regressed: someone re-adds a dataclass field and we
    re-enter the two-sources-disagree state.
    """
    field_names = set(ModelInfo.__dataclass_fields__.keys())
    assert "supports_tools" not in field_names, (
        "supports_tools must not be a ModelInfo field — "
        "it belongs to ModelConstraints. Use the back-compat property."
    )
    assert "supports_vision" not in field_names
    assert "supports_streaming" not in field_names


def test_modelinfo_back_compat_unknown_model_returns_permissive():
    """An unknown model should fall through to the permissive ConstraintRegistry default.

    Expected failure mode if regressed: property raises KeyError or returns None
    for a model the constraint registry has never heard of.
    """
    info = get_model_info("totally-unknown-xyz")
    # Permissive default → supports_tools True, supports_streaming True, supports_vision False
    assert info.supports_tools is True
    assert info.supports_streaming is True
    assert info.supports_vision is False
