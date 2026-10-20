"""Tests for mullet.cli.live_test."""

import json
import os
import sys
from unittest.mock import MagicMock

import pytest

from mullet.cli.live_test import (
    Target,
    _discover_targets,
    _fire_kimi_swarm_holds,
    _fire_one,
    _load_mcp_env_fallback,
    _parse_args,
    run_cli,
    run_live_test,
)


# --------------------------------------------------------------------------- #
# _load_mcp_env_fallback
# --------------------------------------------------------------------------- #
class TestLoadMcpEnvFallback:
    def test_no_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(os, "environ", {})
        monkeypatch.setattr("mullet.cli.live_test.Path.home", lambda: tmp_path)
        _load_mcp_env_fallback()
        assert os.environ == {}

    def test_malformed_json(self, monkeypatch, tmp_path):
        monkeypatch.setattr(os, "environ", {})
        monkeypatch.setattr("mullet.cli.live_test.Path.home", lambda: tmp_path)
        (tmp_path / ".mcp.json").write_text("not json")
        _load_mcp_env_fallback()
        assert os.environ == {}

    def test_missing_mcp_servers(self, monkeypatch, tmp_path):
        monkeypatch.setattr(os, "environ", {})
        monkeypatch.setattr("mullet.cli.live_test.Path.home", lambda: tmp_path)
        (tmp_path / ".mcp.json").write_text(json.dumps({}))
        _load_mcp_env_fallback()
        assert os.environ == {}

    def test_mullet_server_no_env(self, monkeypatch, tmp_path):
        monkeypatch.setattr(os, "environ", {})
        monkeypatch.setattr("mullet.cli.live_test.Path.home", lambda: tmp_path)
        payload = {"mcpServers": {"mullet": {}}}
        (tmp_path / ".mcp.json").write_text(json.dumps(payload))
        _load_mcp_env_fallback()
        assert os.environ == {}

    def test_does_not_overwrite_existing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(os, "environ", {"KIMI_API_KEY": "existing"})
        monkeypatch.setattr("mullet.cli.live_test.Path.home", lambda: tmp_path)
        payload = {
            "mcpServers": {"mullet": {"env": {"KIMI_API_KEY": "from-file"}}}
        }
        (tmp_path / ".mcp.json").write_text(json.dumps(payload))
        _load_mcp_env_fallback()
        assert os.environ["KIMI_API_KEY"] == "existing"

    def test_sets_missing_key(self, monkeypatch, tmp_path):
        monkeypatch.setattr(os, "environ", {})
        monkeypatch.setattr("mullet.cli.live_test.Path.home", lambda: tmp_path)
        payload = {
            "mcpServers": {"mullet": {"env": {"KIMI_API_KEY": "from-file"}}}
        }
        (tmp_path / ".mcp.json").write_text(json.dumps(payload))
        _load_mcp_env_fallback()
        assert os.environ["KIMI_API_KEY"] == "from-file"


# --------------------------------------------------------------------------- #
# _discover_targets
# --------------------------------------------------------------------------- #
class TestDiscoverTargets:
    def test_no_providers(self, monkeypatch):
        monkeypatch.setattr(os, "environ", {})
        monkeypatch.setattr("mullet.cli.live_test._load_mcp_env_fallback", lambda: None)
        monkeypatch.setattr(
            "mullet.core.providers.local_detection.detect_local_provider", lambda: None
        )
        monkeypatch.setattr(
            "mullet.core.providers.claude_max._read_oauth_token", lambda: None
        )
        targets = _discover_targets()
        assert targets == []

    def test_kimi_key(self, monkeypatch):
        monkeypatch.setattr(os, "environ", {"KIMI_API_KEY": "sk-test"})
        monkeypatch.setattr("mullet.cli.live_test._load_mcp_env_fallback", lambda: None)
        monkeypatch.setattr(
            "mullet.core.providers.local_detection.detect_local_provider", lambda: None
        )
        monkeypatch.setattr(
            "mullet.core.providers.claude_max._read_oauth_token", lambda: None
        )
        monkeypatch.setattr(
            "mullet.core.providers.kimi.KimiProvider", lambda **kw: MagicMock()
        )
        targets = _discover_targets()
        assert any(t.provider_label == "kimi" for t in targets)

    def test_google_key(self, monkeypatch):
        monkeypatch.setattr(os, "environ", {"GOOGLE_API_KEY": "gk-test"})
        monkeypatch.setattr("mullet.cli.live_test._load_mcp_env_fallback", lambda: None)
        monkeypatch.setattr(
            "mullet.core.providers.local_detection.detect_local_provider", lambda: None
        )
        monkeypatch.setattr(
            "mullet.core.providers.claude_max._read_oauth_token", lambda: None
        )
        monkeypatch.setattr(
            "mullet.core.providers.gemini.GeminiProvider", lambda **kw: MagicMock()
        )
        targets = _discover_targets()
        assert any(t.provider_label == "gemini" for t in targets)

    def test_anthropic_key(self, monkeypatch):
        monkeypatch.setattr(os, "environ", {"ANTHROPIC_API_KEY": "ak-test"})
        monkeypatch.setattr("mullet.cli.live_test._load_mcp_env_fallback", lambda: None)
        monkeypatch.setattr(
            "mullet.core.providers.local_detection.detect_local_provider", lambda: None
        )
        monkeypatch.setattr(
            "mullet.core.providers.anthropic.AnthropicProvider", lambda **kw: MagicMock()
        )
        targets = _discover_targets()
        assert any(t.provider_label == "anthropic" for t in targets)

    def test_local_provider(self, monkeypatch):
        monkeypatch.setattr(os, "environ", {})
        monkeypatch.setattr("mullet.cli.live_test._load_mcp_env_fallback", lambda: None)
        monkeypatch.setattr(
            "mullet.core.providers.claude_max._read_oauth_token", lambda: None
        )
        mock_local = MagicMock()
        mock_local.model = "test-model"
        monkeypatch.setattr(
            "mullet.core.providers.local_detection.detect_local_provider",
            lambda: mock_local,
        )
        targets = _discover_targets()
        assert any(
            t.provider_label == "local" and t.model_id == "test-model" for t in targets
        )


# --------------------------------------------------------------------------- #
# _fire_one
# --------------------------------------------------------------------------- #
class TestFireOne:
    def test_success(self, monkeypatch):
        monkeypatch.setattr("mullet.cli.live_test.add_agent", lambda *a, **kw: None)
        monkeypatch.setattr("mullet.cli.live_test.remove_agent", lambda *a: None)
        monkeypatch.setattr(
            "mullet.cli.live_test.time.monotonic",
            MagicMock(side_effect=[0.0, 0.1, 0.2]),
        )
        monkeypatch.setattr("mullet.cli.live_test.time.sleep", lambda s: None)

        mock_provider = MagicMock()
        target = Target("test", "m1", lambda: mock_provider)
        results = {}
        _fire_one(target, 0, results)
        assert results["test/m1"]["ok"] is True
        assert results["test/m1"]["error_class"] == ""
        mock_provider.create.assert_called_once()

    def test_failure(self, monkeypatch):
        monkeypatch.setattr("mullet.cli.live_test.add_agent", lambda *a, **kw: None)
        monkeypatch.setattr("mullet.cli.live_test.remove_agent", lambda *a: None)
        monkeypatch.setattr(
            "mullet.cli.live_test.time.monotonic",
            MagicMock(side_effect=[0.0, 0.1, 0.2]),
        )
        monkeypatch.setattr("mullet.cli.live_test.time.sleep", lambda s: None)

        mock_provider = MagicMock()
        mock_provider.create.side_effect = RuntimeError("oops")
        target = Target("test", "m1", lambda: mock_provider)
        results = {}
        _fire_one(target, 0, results)
        assert results["test/m1"]["ok"] is False
        assert results["test/m1"]["error_class"] == "RuntimeError"

    def test_remove_agent_exception(self, monkeypatch):
        monkeypatch.setattr("mullet.cli.live_test.add_agent", lambda *a, **kw: None)
        monkeypatch.setattr(
            "mullet.cli.live_test.remove_agent",
            MagicMock(side_effect=RuntimeError("remove failed")),
        )
        monkeypatch.setattr(
            "mullet.cli.live_test.time.monotonic",
            MagicMock(side_effect=[0.0, 0.1, 0.2]),
        )
        monkeypatch.setattr("mullet.cli.live_test.time.sleep", lambda s: None)

        mock_provider = MagicMock()
        target = Target("test", "m1", lambda: mock_provider)
        results = {}
        _fire_one(target, 0, results)
        assert results["test/m1"]["ok"] is True

    def test_multiline_error(self, monkeypatch):
        monkeypatch.setattr("mullet.cli.live_test.add_agent", lambda *a, **kw: None)
        monkeypatch.setattr("mullet.cli.live_test.remove_agent", lambda *a: None)
        monkeypatch.setattr(
            "mullet.cli.live_test.time.monotonic",
            MagicMock(side_effect=[0.0, 0.1, 0.2]),
        )
        monkeypatch.setattr("mullet.cli.live_test.time.sleep", lambda s: None)

        mock_provider = MagicMock()
        mock_provider.create.side_effect = RuntimeError("first line\nsecond line")
        target = Target("test", "m1", lambda: mock_provider)
        results = {}
        _fire_one(target, 0, results)
        assert results["test/m1"]["error"] == "first line"


# --------------------------------------------------------------------------- #
# _fire_kimi_swarm_holds
# --------------------------------------------------------------------------- #
class TestFireKimiSwarmHolds:
    def test_success(self, monkeypatch):
        add_calls = []
        monkeypatch.setattr(
            "mullet.cli.live_test.add_agent", lambda *a, **kw: add_calls.append((a, kw))
        )
        monkeypatch.setattr("mullet.cli.live_test.remove_agent", lambda *a: None)
        monkeypatch.setattr("mullet.cli.live_test.time.sleep", lambda s: None)
        _fire_kimi_swarm_holds(0, 2)
        assert len(add_calls) == 2

    def test_size_zero(self, monkeypatch):
        add_calls = []
        monkeypatch.setattr(
            "mullet.cli.live_test.add_agent", lambda *a, **kw: add_calls.append((a, kw))
        )
        monkeypatch.setattr("mullet.cli.live_test.remove_agent", lambda *a: None)
        monkeypatch.setattr("mullet.cli.live_test.time.sleep", lambda s: None)
        _fire_kimi_swarm_holds(0, 0)
        assert len(add_calls) == 0


# --------------------------------------------------------------------------- #
# run_live_test
# --------------------------------------------------------------------------- #
class TestRunLiveTest:
    def test_no_targets(self, monkeypatch, capsys):
        monkeypatch.setattr("mullet.cli.live_test._discover_targets", lambda: [])
        ret = run_live_test(1)
        captured = capsys.readouterr()
        assert ret == 1
        assert "no providers" in captured.out

    def test_all_succeed(self, monkeypatch, capsys):
        targets = [
            Target("kimi", "k1", lambda: MagicMock()),
            Target("gemini", "g1", lambda: MagicMock()),
        ]

        def mock_fire_one(target, hold_seconds, results):
            results[f"{target.provider_label}/{target.model_id}"] = {
                "ok": True,
                "error_class": "",
                "error": "",
                "duration": 0.1,
            }

        monkeypatch.setattr("mullet.cli.live_test._discover_targets", lambda: targets)
        monkeypatch.setattr("mullet.cli.live_test._fire_one", mock_fire_one)
        monkeypatch.setattr("mullet.cli.live_test._fire_kimi_swarm_holds", lambda s, n: None)
        ret = run_live_test(1, include_swarm=False)
        captured = capsys.readouterr()
        assert ret == 0
        assert "✓  health: all 2/2" in captured.out

    def test_one_fails(self, monkeypatch, capsys):
        targets = [
            Target("kimi", "k1", lambda: MagicMock()),
            Target("gemini", "g1", lambda: MagicMock()),
        ]

        def mock_fire_one(target, hold_seconds, results):
            ok = target.provider_label != "gemini"
            results[f"{target.provider_label}/{target.model_id}"] = {
                "ok": ok,
                "error_class": "" if ok else "RuntimeError",
                "error": "",
                "duration": 0.1,
            }

        monkeypatch.setattr("mullet.cli.live_test._discover_targets", lambda: targets)
        monkeypatch.setattr("mullet.cli.live_test._fire_one", mock_fire_one)
        monkeypatch.setattr("mullet.cli.live_test._fire_kimi_swarm_holds", lambda s, n: None)
        ret = run_live_test(1, include_swarm=False)
        captured = capsys.readouterr()
        assert ret == 1
        assert "⚠  health: 1/2" in captured.out

    def test_exclude_local(self, monkeypatch):
        targets = [
            Target("local", "lm", lambda: MagicMock()),
            Target("kimi", "k1", lambda: MagicMock()),
        ]
        fired = []

        def mock_fire_one(target, hold_seconds, results):
            fired.append(target.provider_label)
            results[f"{target.provider_label}/{target.model_id}"] = {
                "ok": True,
                "error_class": "",
                "error": "",
                "duration": 0.1,
            }

        monkeypatch.setattr("mullet.cli.live_test._discover_targets", lambda: targets)
        monkeypatch.setattr("mullet.cli.live_test._fire_one", mock_fire_one)
        monkeypatch.setattr("mullet.cli.live_test._fire_kimi_swarm_holds", lambda s, n: None)
        ret = run_live_test(1, include_swarm=False, include_local=False)
        assert "local" not in fired
        assert "kimi" in fired
        assert ret == 0

    def test_include_swarm(self, monkeypatch):
        targets = [Target("kimi", "k1", lambda: MagicMock())]
        swarm_called = []

        def mock_fire_one(target, hold_seconds, results):
            results[f"{target.provider_label}/{target.model_id}"] = {
                "ok": True,
                "error_class": "",
                "error": "",
                "duration": 0.1,
            }

        def mock_swarm(seconds, size):
            swarm_called.append((seconds, size))

        monkeypatch.setattr("mullet.cli.live_test._discover_targets", lambda: targets)
        monkeypatch.setattr("mullet.cli.live_test._fire_one", mock_fire_one)
        monkeypatch.setattr("mullet.cli.live_test._fire_kimi_swarm_holds", mock_swarm)
        ret = run_live_test(1, include_swarm=True, swarm_size=3)
        assert swarm_called == [(1, 3)]
        assert ret == 0


# --------------------------------------------------------------------------- #
# _parse_args
# --------------------------------------------------------------------------- #
class TestParseArgs:
    def test_defaults(self):
        ns = _parse_args([])
        assert ns.seconds == 60
        assert ns.no_swarm is False
        assert ns.swarm_size == 3
        assert ns.no_local is False

    def test_no_swarm(self):
        ns = _parse_args(["30", "--no-swarm"])
        assert ns.seconds == 30
        assert ns.no_swarm is True
        assert ns.swarm_size == 3
        assert ns.no_local is False

    def test_swarm_size_and_no_local(self):
        ns = _parse_args(["60", "--swarm-size", "5", "--no-local"])
        assert ns.seconds == 60
        assert ns.swarm_size == 5
        assert ns.no_local is True
        assert ns.no_swarm is False


# --------------------------------------------------------------------------- #
# run_cli
# --------------------------------------------------------------------------- #
class TestRunCli:
    def test_zero_seconds(self, capsys):
        ret = run_cli(["0"])
        captured = capsys.readouterr()
        assert ret == 2
        assert "seconds must be >= 1" in captured.err

    def test_one_second(self, monkeypatch):
        called = []

        def mock_run(**kw):
            called.append(kw)
            return 0

        monkeypatch.setattr("mullet.cli.live_test.run_live_test", mock_run)
        ret = run_cli(["1"])
        assert ret == 0
        assert len(called) == 1
        assert called[0] == {
            "seconds": 1,
            "include_swarm": True,
            "swarm_size": 3,
            "include_local": True,
        }

    def test_options(self, monkeypatch):
        called = []

        def mock_run(**kw):
            called.append(kw)
            return 0

        monkeypatch.setattr("mullet.cli.live_test.run_live_test", mock_run)
        ret = run_cli(["30", "--no-swarm", "--no-local"])
        assert ret == 0
        assert len(called) == 1
        assert called[0] == {
            "seconds": 30,
            "include_swarm": False,
            "swarm_size": 3,
            "include_local": False,
        }
