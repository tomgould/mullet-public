"""Coverage-gap tests for mullet.core.providers.lmstudio_manager."""

import json
import logging
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from mullet.core.providers.lmstudio_manager import LMStudioManager


class TestMakeRequestGaps:
    """Lines 56-57, 59-60, 62-63: error branches in _make_request."""

    def test_make_request_http_non_200(self, caplog):
        caplog.set_level(logging.DEBUG, logger="mullet.core.providers.lmstudio_manager")
        mgr = LMStudioManager()
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.status = 404
            mock_urlopen.return_value.__enter__.return_value = mock_response
            result = mgr._make_request("/test")
        assert result is None
        assert "HTTP error 404" in caplog.text

    def test_make_request_urlerror(self, caplog):
        caplog.set_level(logging.DEBUG, logger="mullet.core.providers.lmstudio_manager")
        mgr = LMStudioManager()
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
            result = mgr._make_request("/test")
        assert result is None
        assert "Network error" in caplog.text

    def test_make_request_json_decode_error(self, caplog):
        caplog.set_level(logging.DEBUG, logger="mullet.core.providers.lmstudio_manager")
        mgr = LMStudioManager()
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.read.return_value = b"not-json"
            mock_urlopen.return_value.__enter__.return_value = mock_response
            result = mgr._make_request("/test")
        assert result is None
        assert "JSON decode error" in caplog.text


class TestPickBestModelOverflow:
    """Lines 126-137: overflow fallback when no model fits context_needed."""

    def test_pick_best_model_overflow_fallback(self, caplog):
        caplog.set_level(logging.WARNING, logger="mullet.core.providers.lmstudio_manager")
        mgr = LMStudioManager()
        models = [
            {"id": "small", "type": "llm", "state": "loaded", "loaded_context_length": 1024},
            {"id": "medium", "type": "llm", "state": "not-loaded", "loaded_context_length": 2048},
        ]
        with patch.object(mgr, "get_models") as mock_get:
            mock_get.return_value = models
            result = mgr.pick_best_model(context_needed=4096)
        assert result is not None
        # largest loaded first, then largest available
        assert result["id"] == "small"
        assert "no local model has context >= 4096 tokens" in caplog.text

    def test_pick_best_model_overflow_no_candidates(self):
        mgr = LMStudioManager()
        with patch.object(mgr, "get_models") as mock_get:
            mock_get.return_value = []
            result = mgr.pick_best_model(context_needed=4096)
        assert result is None


class TestRunLmsCommandGaps:
    """Lines 153-155: generic Exception in _run_lms_command."""

    def test_run_lms_command_generic_exception(self, caplog):
        caplog.set_level(logging.ERROR, logger="mullet.core.providers.lmstudio_manager")
        mgr = LMStudioManager()
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = OSError("something went wrong")
            result = mgr._run_lms_command(["test"])
        assert result == (False, "something went wrong")
        assert "Error running LMS command" in caplog.text


class TestEnsureLoadedGaps:
    """Lines 187-191: timeout waiting for model load."""

    def test_ensure_loaded_timeout(self, caplog):
        caplog.set_level(logging.ERROR, logger="mullet.core.providers.lmstudio_manager")
        mgr = LMStudioManager()
        with patch.object(mgr, "get_loaded_models") as mock_loaded, \
             patch.object(mgr, "_run_lms_command") as mock_run, \
             patch("time.time") as mock_time, \
             patch("time.sleep"):
            mock_loaded.return_value = []  # never loads
            mock_run.return_value = (True, "initiated")
            # Simulate time advancing past 30s timeout; use a generator that
            # yields large values once the sequence is exhausted so logging's
            # internal time.time() calls don't blow up with StopIteration.
            time_seq = iter([0, 5, 10, 20, 35])
            mock_time.side_effect = lambda: next(time_seq, 999)
            result = mgr.ensure_loaded("slow-model")
        assert result is False
        assert "Timeout waiting for model slow-model to load" in caplog.text


class TestReloadCurrentModelGaps:
    """Lines 211-212, 215: no model loaded / empty model_id."""

    def test_reload_current_model_no_loaded(self, caplog):
        caplog.set_level(logging.WARNING, logger="mullet.core.providers.lmstudio_manager")
        mgr = LMStudioManager()
        with patch.object(mgr, "get_loaded_models") as mock_loaded:
            mock_loaded.return_value = []
            result = mgr.reload_current_model()
        assert result is False
        assert "no model loaded, nothing to reload" in caplog.text

    def test_reload_current_model_empty_id(self):
        mgr = LMStudioManager()
        with patch.object(mgr, "get_loaded_models") as mock_loaded:
            mock_loaded.return_value = [{"id": "", "modelKey": ""}]
            result = mgr.reload_current_model()
        assert result is False


class TestGetOrLoadProviderGaps:
    """Lines 238-245: get_or_load_provider load-failure branch."""

    def test_get_or_load_provider_load_fails(self, caplog):
        caplog.set_level(logging.ERROR, logger="mullet.core.providers.lmstudio_manager")
        mgr = LMStudioManager()
        with patch.object(mgr, "get_loaded_models") as mock_loaded, \
             patch.object(mgr, "pick_best_model") as mock_pick, \
             patch.object(mgr, "ensure_loaded") as mock_ensure:
            mock_loaded.return_value = []
            mock_pick.return_value = {"id": "best-model", "type": "llm", "state": "not-loaded"}
            mock_ensure.return_value = False
            result = mgr.get_or_load_provider()
        assert result == (None, False)
        assert "Failed to ensure model best-model is loaded" in caplog.text

    def test_get_or_load_provider_loads_new_model(self, caplog):
        caplog.set_level(logging.INFO, logger="mullet.core.providers.lmstudio_manager")
        mgr = LMStudioManager()
        with patch.object(mgr, "get_loaded_models") as mock_loaded, \
             patch.object(mgr, "pick_best_model") as mock_pick, \
             patch.object(mgr, "ensure_loaded") as mock_ensure:
            mock_loaded.return_value = []
            mock_pick.return_value = {"id": "new-model", "type": "llm", "state": "not-loaded"}
            mock_ensure.return_value = True
            result = mgr.get_or_load_provider()
        assert result == ("new-model", False)
        assert "Picked model new-model for loading" in caplog.text
