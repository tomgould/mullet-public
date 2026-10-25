"""Tests for local_detection missing coverage paths."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from mullet.core.providers import local_detection as ld


@pytest.fixture(autouse=True)
def _clear_cache():
    ld.clear_local_provider_cache()
    yield
    ld.clear_local_provider_cache()


# ───────────────────────────────────────────────────────────────
# LMStudio path
# ───────────────────────────────────────────────────────────────

def test_detect_logs_info_on_fresh_load():
    """Line 80-81: logger.info when model was not already loaded."""
    with patch(
        "mullet.core.providers.lmstudio_manager.LMStudioManager"
    ) as MockMgr, patch(
        "mullet.core.providers.local_detection.OpenAICompatibleProvider"
    ) as MockProv:
        instance = MockMgr.return_value
        instance.get_or_load_provider.return_value = ("model-x", False)  # was_loaded=False
        stub = MagicMock()
        stub.health_check.return_value = True
        MockProv.return_value = stub
        with patch.object(ld.logger, "info") as mock_log:
            result = ld.detect_local_provider()
        assert result is stub
        mock_log.assert_called_once()
        assert "Loaded local model" in mock_log.call_args[0][0]


def test_detect_no_log_when_already_loaded():
    """Line 80-81: no logger.info when was_loaded=True."""
    with patch(
        "mullet.core.providers.lmstudio_manager.LMStudioManager"
    ) as MockMgr, patch(
        "mullet.core.providers.local_detection.OpenAICompatibleProvider"
    ) as MockProv:
        instance = MockMgr.return_value
        instance.get_or_load_provider.return_value = ("model-x", True)
        stub = MagicMock()
        stub.health_check.return_value = True
        MockProv.return_value = stub
        with patch.object(ld.logger, "info") as mock_log:
            result = ld.detect_local_provider()
        assert result is stub
        mock_log.assert_not_called()


def test_detect_lmstudio_exception_falls_back_to_port_scan(monkeypatch):
    """Lines 83-84: LMStudioManager raises, fallback port scan succeeds."""
    # The default URLs are what's under test; every request here is mocked.
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
    with patch(
        "mullet.core.providers.lmstudio_manager.LMStudioManager",
        side_effect=RuntimeError("no lms"),
    ), patch("urllib.request.urlopen") as mock_urlopen, patch(
        "mullet.core.providers.local_detection.OpenAICompatibleProvider"
    ) as MockProv:
        response = MagicMock()
        response.status = 200
        response.read.return_value = json.dumps({"data": [{"id": "llama2"}]}).encode()
        mock_urlopen.return_value.__enter__.return_value = response
        stub = MagicMock()
        stub.health_check.return_value = True
        MockProv.return_value = stub
        result = ld.detect_local_provider()
        assert result is stub
        MockProv.assert_called_once_with(
            base_url="http://localhost:11434/v1", model="llama2"
        )


# ───────────────────────────────────────────────────────────────
# Port-scan fallback
# ───────────────────────────────────────────────────────────────

def test_detect_port_scan_skips_embedding_models():
    """Lines 99-103: embedding markers are skipped."""
    with patch(
        "mullet.core.providers.lmstudio_manager.LMStudioManager",
        side_effect=Exception("fail"),
    ), patch("urllib.request.urlopen") as mock_urlopen, patch(
        "mullet.core.providers.local_detection.OpenAICompatibleProvider"
    ) as MockProv:
        response = MagicMock()
        response.status = 200
        response.read.return_value = json.dumps(
            {"data": [{"id": "nomic-embed-text"}, {"id": "llama3"}]}
        ).encode()
        mock_urlopen.return_value.__enter__.return_value = response
        stub = MagicMock()
        stub.health_check.return_value = True
        MockProv.return_value = stub
        result = ld.detect_local_provider()
        assert result is not None
        assert MockProv.call_args.kwargs["model"] == "llama3"


def test_detect_port_scan_no_models_returns_none():
    """Lines 97-98: empty models list continues to next port."""
    with patch(
        "mullet.core.providers.lmstudio_manager.LMStudioManager",
        side_effect=Exception("fail"),
    ), patch("urllib.request.urlopen") as mock_urlopen, patch(
        "mullet.core.providers.local_detection.OpenAICompatibleProvider"
    ):
        response = MagicMock()
        response.status = 200
        response.read.return_value = json.dumps({"data": []}).encode()
        mock_urlopen.return_value.__enter__.return_value = response
        result = ld.detect_local_provider()
        assert result is None


def test_detect_port_scan_non_200_status_continues():
    """Lines 93-94: non-200 status continues to next port."""
    with patch(
        "mullet.core.providers.lmstudio_manager.LMStudioManager",
        side_effect=Exception("fail"),
    ), patch("urllib.request.urlopen") as mock_urlopen:
        response = MagicMock()
        response.status = 404
        mock_urlopen.return_value.__enter__.return_value = response
        result = ld.detect_local_provider()
        assert result is None


def test_detect_port_scan_default_model_when_all_embedding():
    """Line 103: default model when all are embedding models."""
    with patch(
        "mullet.core.providers.lmstudio_manager.LMStudioManager",
        side_effect=Exception("fail"),
    ), patch("urllib.request.urlopen") as mock_urlopen, patch(
        "mullet.core.providers.local_detection.OpenAICompatibleProvider"
    ) as MockProv:
        response = MagicMock()
        response.status = 200
        response.read.return_value = json.dumps(
            {"data": [{"id": "clip-vit"}]}
        ).encode()
        mock_urlopen.return_value.__enter__.return_value = response
        stub = MagicMock()
        stub.health_check.return_value = True
        MockProv.return_value = stub
        result = ld.detect_local_provider()
        assert MockProv.call_args.kwargs["model"] == "default"


def test_detect_sets_none_and_timestamp_on_total_failure():
    """Lines 119-121: after all probes fail, cache is set to None."""
    with patch(
        "mullet.core.providers.lmstudio_manager.LMStudioManager",
        side_effect=Exception("fail"),
    ), patch("urllib.request.urlopen", side_effect=OSError("refused")):
        result = ld.detect_local_provider()
    assert result is None
    assert ld._cached_local_provider is None
    assert ld._cache_timestamp > 0


# ───────────────────────────────────────────────────────────────
# _cache_is_fresh
# ───────────────────────────────────────────────────────────────

def test_cache_is_fresh_true():
    """Lines 124-131: _cache_is_fresh returns True when cache is valid."""
    with patch(
        "mullet.core.providers.lmstudio_manager.LMStudioManager"
    ) as MockMgr, patch(
        "mullet.core.providers.local_detection.OpenAICompatibleProvider"
    ) as MockProv:
        instance = MockMgr.return_value
        instance.get_or_load_provider.return_value = ("model", True)
        stub = MagicMock()
        stub.health_check.return_value = True
        MockProv.return_value = stub
        ld.detect_local_provider()
        assert ld._cache_is_fresh() is True


def test_cache_is_fresh_no_cache():
    assert ld._cache_is_fresh() is False


# ───────────────────────────────────────────────────────────────
# get_local_provider_nonblocking
# ───────────────────────────────────────────────────────────────

def test_get_local_provider_nonblocking_fresh_cache():
    """Lines 153-154: fresh cache returns immediately."""
    with patch(
        "mullet.core.providers.lmstudio_manager.LMStudioManager"
    ) as MockMgr, patch(
        "mullet.core.providers.local_detection.OpenAICompatibleProvider"
    ) as MockProv:
        instance = MockMgr.return_value
        instance.get_or_load_provider.return_value = ("model", True)
        stub = MagicMock()
        stub.health_check.return_value = True
        MockProv.return_value = stub
        ld.detect_local_provider()
        result = asyncio.run(ld.get_local_provider_nonblocking())
        assert result is stub


def test_get_local_provider_nonblocking_stale_kicks_background():
    """Lines 156-169: stale cache kicks background refresh."""
    ld._cached_local_provider = MagicMock()
    ld._cache_timestamp = 0.0  # definitely stale
    with patch("asyncio.get_running_loop") as mock_loop:
        task = MagicMock()
        task.done.return_value = True
        mock_loop.return_value.create_task.return_value = task
        result = asyncio.run(ld.get_local_provider_nonblocking())
        assert result is ld._cached_local_provider
        mock_loop.return_value.create_task.assert_called_once()


def test_get_local_provider_nonblocking_cold_cache():
    """Lines 142-144: cold cache returns None and kicks background."""
    ld.clear_local_provider_cache()
    with patch("asyncio.get_running_loop") as mock_loop:
        task = MagicMock()
        task.done.return_value = True
        mock_loop.return_value.create_task.return_value = task
        result = asyncio.run(ld.get_local_provider_nonblocking())
        assert result is None
        mock_loop.return_value.create_task.assert_called_once()


def test_get_local_provider_nonblocking_no_loop_fallback():
    """Lines 162-166: no running loop falls back to sync detect."""
    ld.clear_local_provider_cache()
    with patch(
        "mullet.core.providers.local_detection.detect_local_provider",
        return_value=None,
    ) as mock_detect, patch(
        "mullet.core.providers.local_detection.asyncio.get_running_loop",
        side_effect=RuntimeError("no loop"),
    ):
        result = asyncio.run(ld.get_local_provider_nonblocking())
        assert result is None
        mock_detect.assert_called_once()


# ───────────────────────────────────────────────────────────────
# _refresh_cache_in_background
# ───────────────────────────────────────────────────────────────

def test_refresh_cache_in_background_success():
    """Lines 172-185: background refresh logs on success."""
    with patch(
        "mullet.core.providers.local_detection.detect_local_provider"
    ) as mock_detect, patch.object(ld.logger, "debug") as mock_log:
        provider = MagicMock()
        provider.base_url = "http://localhost:1234/v1"
        provider.model = "m"
        mock_detect.return_value = provider
        asyncio.run(ld._refresh_cache_in_background())
        mock_log.assert_called_once()
        assert "background refresh: local provider available" in mock_log.call_args[0][0]


def test_refresh_cache_in_background_none():
    """Lines 186-187: background refresh logs when no provider found."""
    with patch(
        "mullet.core.providers.local_detection.detect_local_provider",
        return_value=None,
    ), patch.object(ld.logger, "debug") as mock_log:
        asyncio.run(ld._refresh_cache_in_background())
        mock_log.assert_called_once()
        assert "no local provider available" in mock_log.call_args[0][0]


def test_detect_cache_hit_healthy_returns_immediately():
    """Line 65: healthy cached provider returned immediately on second call."""
    with patch(
        "mullet.core.providers.lmstudio_manager.LMStudioManager"
    ) as MockMgr, patch(
        "mullet.core.providers.local_detection.OpenAICompatibleProvider"
    ) as MockProv:
        instance = MockMgr.return_value
        instance.get_or_load_provider.return_value = ("model", True)
        stub = MagicMock()
        stub.health_check.return_value = True
        MockProv.return_value = stub
        first = ld.detect_local_provider()
        second = ld.detect_local_provider()
        assert first is second is stub
        assert instance.get_or_load_provider.call_count == 1


def test_detect_cache_invalidates_on_unhealthy_health_check():
    """Lines 64-66: failed health_check drops cache and re-probes."""
    with patch(
        "mullet.core.providers.lmstudio_manager.LMStudioManager"
    ) as MockMgr, patch(
        "mullet.core.providers.local_detection.OpenAICompatibleProvider"
    ) as MockProv:
        instance = MockMgr.return_value
        instance.get_or_load_provider.return_value = ("model", True)
        sick = MagicMock()
        sick.health_check.return_value = False
        fresh = MagicMock()
        fresh.health_check.return_value = True
        MockProv.side_effect = [sick, fresh]
        first = ld.detect_local_provider()
        assert first is sick
        second = ld.detect_local_provider()
        assert second is fresh
        assert instance.get_or_load_provider.call_count == 2
