"""Tests for canonical local-provider auto-detection (Cycle 7 C1).

`detect_local_provider` was extracted from `mullet.cli.main` into
`mullet.core.providers.local_detection`. These tests pin:

- the no-server-running path returns None,
- the 30s TTL cache avoids a second LMStudioManager probe,
- the cache invalidates when a cached provider's health_check fails,
- the back-compat re-export from `mullet.cli.main` is the same callable.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mullet.core.providers import local_detection as ld


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset the module-level cache before and after each test."""
    ld.clear_local_provider_cache()
    yield
    ld.clear_local_provider_cache()


def test_detect_returns_none_when_no_local_server():
    """Expected failure when broken: returns a stub provider instead of None.

    All probes fail (LMStudioManager raises, port-scan urlopen raises).
    """
    with patch(
        "mullet.core.providers.lmstudio_manager.LMStudioManager"
    ) as MockMgr, patch(
        "urllib.request.urlopen", side_effect=OSError("connection refused")
    ):
        MockMgr.side_effect = RuntimeError("no lms cli")
        result = ld.detect_local_provider()
    assert result is None, f"expected None when no server reachable, got {result!r}"


def test_detect_caches_within_ttl():
    """Expected failure when broken: LMStudioManager.get_or_load_provider called twice.

    Two consecutive calls within the TTL must hit the cache; the manager
    is probed at most once, and the same provider object is returned.
    """
    with patch(
        "mullet.core.providers.lmstudio_manager.LMStudioManager"
    ) as MockMgr:
        instance = MockMgr.return_value
        instance.get_or_load_provider.return_value = ("test-model-id", True)

        # OpenAICompatibleProvider returned by detect — give it a healthy
        # health_check so the second call hits the cached branch cleanly.
        with patch(
            "mullet.core.providers.local_detection.OpenAICompatibleProvider"
        ) as MockProv:
            stub = MagicMock()
            stub.health_check.return_value = True
            MockProv.return_value = stub

            first = ld.detect_local_provider()
            second = ld.detect_local_provider()

        assert first is second, "cache should return the same provider object"
        assert instance.get_or_load_provider.call_count == 1, (
            f"expected 1 LMStudioManager probe, got "
            f"{instance.get_or_load_provider.call_count}"
        )


def test_detect_cache_invalidates_on_unhealthy():
    """Expected failure when broken: stale cached provider returned despite failed health_check.

    First call seeds the cache. health_check then flips False → next call
    must drop the cache and re-probe LMStudioManager.
    """
    with patch(
        "mullet.core.providers.lmstudio_manager.LMStudioManager"
    ) as MockMgr:
        instance = MockMgr.return_value
        instance.get_or_load_provider.return_value = ("model-id", False)

        # Two distinct provider stubs, second healthier than the first.
        sick = MagicMock(name="sick")
        sick.health_check.return_value = True  # healthy on initial probe
        fresh = MagicMock(name="fresh")
        fresh.health_check.return_value = True

        with patch(
            "mullet.core.providers.local_detection.OpenAICompatibleProvider",
            side_effect=[sick, fresh],
        ):
            first = ld.detect_local_provider()
            assert first is sick

            # Flip the cached provider's health to False so the next call
            # treats it as stale and re-probes.
            sick.health_check.return_value = False

            second = ld.detect_local_provider()

        assert second is fresh, (
            f"expected re-probed provider after unhealthy cache, got {second!r}"
        )
        assert instance.get_or_load_provider.call_count == 2, (
            f"expected 2 LMStudioManager probes after cache invalidation, "
            f"got {instance.get_or_load_provider.call_count}"
        )


def test_back_compat_import_from_cli_main_still_works():
    """Expected failure when broken: ImportError or two distinct callables.

    `from mullet.cli.main import detect_local_provider` must resolve to
    the same function object as the canonical import path.
    """
    from mullet.cli.main import detect_local_provider as cli_export
    from mullet.core.providers.local_detection import (
        detect_local_provider as canonical,
    )
    assert cli_export is canonical, (
        "back-compat re-export must be the SAME function object as the canonical one; "
        f"got cli_export={cli_export!r} canonical={canonical!r}"
    )
