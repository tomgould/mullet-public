"""G15 - tiers_for_providers helper."""
from mullet.core.router.complexity import tiers_for_providers


def test_local_only():
    # Phase E: local is a free fallback at every tier.
    assert tiers_for_providers({"local"}) == ["LOW", "STANDARD", "ADVANCED", "CRITICAL"]


def test_sonnet_only():
    assert tiers_for_providers({"sonnet"}) == ["ADVANCED", "CRITICAL"]


def test_union_covers_full_range():
    assert tiers_for_providers({"local", "sonnet"}) == ["LOW", "STANDARD", "ADVANCED", "CRITICAL"]


def test_unknown_provider_empty():
    assert tiers_for_providers({"unknown-xyz"}) == []


def test_empty_allowed_returns_empty():
    assert tiers_for_providers(set()) == []
