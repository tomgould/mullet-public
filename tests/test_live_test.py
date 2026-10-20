"""Tests for mullet.cli.live_test — unit coverage for missing lines."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mullet.cli.live_test import (
    _load_mcp_env_fallback,
    _discover_targets,
    _fire_one,
    _fire_kimi_swarm_holds,
    run_live_test,
    run_cli,
    Target,
    _MCP_ENV_KEYS,
)


# ---------------------------------------------------------------------------
# Lines 120-121: Gemini provider unavailable — exception is silently swallowed
# ---------------------------------------------------------------------------

class TestDiscoverTargetsGeminiFailure:
    """Line 120-121: Exception inside gemini try-block is caught and logged."""

    def test_gemini_import_error_is_caught(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        # Remove all other keys so only gemini branch triggers
        for k in ("KIMI_API_KEY", "MOONSHOT_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(k, raising=False)

        import sys

        # Remove the gemini module from sys.modules so the 'from ... import' fails
        old_mod = sys.modules.pop("mullet.core.providers.gemini", None)
        try:
            with patch.dict("sys.modules", {"mullet.core.providers.gemini": None}), \
                 patch("mullet.core.providers.local_detection.detect_local_provider", return_value=None):
                # Should not raise even though GeminiProvider import fails
                targets = _discover_targets()
        finally:
            if old_mod is not None:
                sys.modules["mullet.core.providers.gemini"] = old_mod

        # No gemini targets since the provider could not be built
        assert all(t.provider_label != "gemini" for t in targets)


# ---------------------------------------------------------------------------
# Lines 144-145: ClaudeMax probe failed — exception is caught
# ---------------------------------------------------------------------------

class TestDiscoverTargetsClaudeMaxFailure:
    """Lines 144-145: Exception from ClaudeMaxProvider probe is caught."""

    def test_claude_max_probe_exception_is_caught(self, monkeypatch):
        # No ANTHROPIC_API_KEY → falls into the ClaudeMax branch
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        for k in ("KIMI_API_KEY", "MOONSHOT_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
            monkeypatch.delenv(k, raising=False)

        # Make _read_oauth_token raise an exception so lines 144-145 (except block) run.
        # Use patch.object on the already-imported module to avoid string-path lookup issues.
        import mullet.core.providers.claude_max as cm_mod
        with patch.object(cm_mod, "_read_oauth_token", side_effect=RuntimeError("oauth token read failed")), \
             patch("mullet.core.providers.local_detection.detect_local_provider", return_value=None):
            targets = _discover_targets()
        assert all(t.provider_label != "claude_max" for t in targets)


# ---------------------------------------------------------------------------
# Lines 155-156: local detection failed — exception is caught
# ---------------------------------------------------------------------------

class TestDiscoverTargetsLocalFailure:
    """Lines 155-156: Exception from detect_local_provider is caught."""

    def test_local_detection_exception_is_caught(self, monkeypatch):
        for k in ("KIMI_API_KEY", "MOONSHOT_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(k, raising=False)

        with patch(
            "mullet.core.providers.local_detection.detect_local_provider",
            side_effect=RuntimeError("lm studio not running"),
        ):
            targets = _discover_targets()
        assert all(t.provider_label != "local" for t in targets)


# ---------------------------------------------------------------------------
# Line 197: remaining > 0 → time.sleep is called in _fire_one
# ---------------------------------------------------------------------------

class TestFireOneHold:
    """Line 196-197: remaining hold time triggers time.sleep."""

    def test_sleep_called_when_time_remaining(self):
        mock_provider = MagicMock()
        mock_provider.create.return_value = MagicMock()

        target = Target(
            provider_label="test",
            model_id="test-model",
            build=lambda: mock_provider,
        )
        results: dict = {}
        sleep_calls: list[float] = []

        with patch("mullet.cli.live_test.add_agent"), \
             patch("mullet.cli.live_test.remove_agent"), \
             patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)), \
             patch("time.monotonic", side_effect=[0.0, 0.1, 0.1]):
            # hold_seconds=30; elapsed~0.1 → remaining~29.9 > 0
            _fire_one(target, hold_seconds=30, results=results)

        assert len(sleep_calls) > 0
        assert sleep_calls[0] > 0

    def test_no_sleep_when_already_elapsed(self):
        """remaining <= 0 → no sleep call."""
        mock_provider = MagicMock()
        mock_provider.create.return_value = MagicMock()

        target = Target(
            provider_label="test",
            model_id="test-model",
            build=lambda: mock_provider,
        )
        results: dict = {}
        sleep_calls: list[float] = []

        with patch("mullet.cli.live_test.add_agent"), \
             patch("mullet.cli.live_test.remove_agent"), \
             patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)), \
             patch("time.monotonic", side_effect=[0.0, 5.0, 5.0]):
            # hold_seconds=1; elapsed=5.0 → remaining=-4.0 ≤ 0
            _fire_one(target, hold_seconds=1, results=results)

        assert len(sleep_calls) == 0


# ---------------------------------------------------------------------------
# Lines 228-229: remove_agent exception in swarm hold is silently swallowed
# ---------------------------------------------------------------------------

class TestFireKimiSwarmHoldsRemoveAgentException:
    """Lines 228-229: exception from remove_agent in swarm hold is caught."""

    def test_remove_agent_exception_caught(self):
        with patch("mullet.cli.live_test.add_agent"), \
             patch("mullet.cli.live_test.remove_agent", side_effect=RuntimeError("gone")), \
             patch("time.sleep"):
            # Should not raise
            _fire_kimi_swarm_holds(seconds=0, size=1)


# ---------------------------------------------------------------------------
# Line 305: error_detail printed when present in failures summary
# ---------------------------------------------------------------------------

class TestRunLiveTestFailurePrinting:
    """Line 304-305: info['error'] non-empty → second print line."""

    def test_error_detail_printed(self, capsys):
        def _fake_fire_one(target, hold_seconds, results):
            results[f"{target.provider_label}/{target.model_id}"] = {
                "ok": False,
                "error_class": "TimeoutError",
                "error": "Connection timed out after 30s",
                "duration": 31.0,
            }

        target = Target("kimi", "kimi-k2.7", build=lambda: MagicMock())

        with patch("mullet.cli.live_test._discover_targets", return_value=[target]), \
             patch("mullet.cli.live_test._fire_one", side_effect=_fake_fire_one):
            rc = run_live_test(seconds=1, include_swarm=False, swarm_size=0)

        assert rc == 1
        out = capsys.readouterr().out
        assert "Connection timed out" in out


# ---------------------------------------------------------------------------
# Line 358: run_cli seconds < 1 → return 2
# ---------------------------------------------------------------------------

class TestRunCli:
    """Line 347: seconds < 1 exits with code 2."""

    def test_seconds_less_than_one_returns_2(self, capsys):
        rc = run_cli(["0"])
        assert rc == 2
        err = capsys.readouterr().err
        assert "seconds must be >= 1" in err

    def test_valid_seconds_delegates_to_run_live_test(self):
        with patch("mullet.cli.live_test.run_live_test", return_value=0) as mock_run:
            rc = run_cli(["5", "--no-swarm", "--no-local"])
        assert rc == 0
        mock_run.assert_called_once_with(
            seconds=5,
            include_swarm=False,
            swarm_size=3,
            include_local=False,
        )
