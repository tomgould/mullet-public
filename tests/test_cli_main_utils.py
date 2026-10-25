"""Tests for pure/mockable utility functions in mullet.cli.main."""

import argparse
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mullet.core.agent.types import AgentResult, TokenUsage

# Import the module under test
import mullet.cli.main as main_module


# ---------------------------------------------------------------------------
# get_local_loaded_context
# ---------------------------------------------------------------------------

class TestGetLocalLoadedContext:
    def test_loaded_model_returns_context_length(self):
        payload = json.dumps({
            "data": [
                {"state": "loaded", "type": "llm", "loaded_context_length": 128000},
            ]
        }).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert main_module.get_local_loaded_context() == 128000

    def test_multiple_loaded_returns_max(self):
        payload = json.dumps({
            "data": [
                {"state": "loaded", "type": "llm", "loaded_context_length": 64000},
                {"state": "loaded", "type": "llm", "loaded_context_length": 128000},
            ]
        }).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert main_module.get_local_loaded_context() == 128000

    def test_empty_data_returns_zero(self):
        payload = json.dumps({"data": []}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert main_module.get_local_loaded_context() == 0

    def test_server_unreachable_returns_zero(self):
        with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
            assert main_module.get_local_loaded_context() == 0

    def test_custom_base_url(self):
        payload = json.dumps({
            "data": [
                {"state": "loaded", "type": "llm", "loaded_context_length": 32000},
            ]
        }).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            assert main_module.get_local_loaded_context("http://192.168.1.5:8080") == 32000
            mock_urlopen.assert_called_once_with(
                "http://192.168.1.5:8080/api/v0/models", timeout=3
            )


# ---------------------------------------------------------------------------
# _parse_providers_allowlist
# ---------------------------------------------------------------------------

class TestParseProvidersAllowlist:
    def test_none_returns_none(self):
        args = SimpleNamespace(providers=None)
        assert main_module._parse_providers_allowlist(args) is None

    def test_empty_string_returns_none(self):
        args = SimpleNamespace(providers="")
        assert main_module._parse_providers_allowlist(args) is None

    def test_single_provider(self):
        args = SimpleNamespace(providers="kimi")
        assert main_module._parse_providers_allowlist(args) == frozenset({"kimi"})

    def test_normalises_aliases(self):
        args = SimpleNamespace(providers="anthropic,google")
        assert main_module._parse_providers_allowlist(args) == frozenset({"sonnet", "gemini"})

    def test_whitespace_handling(self):
        args = SimpleNamespace(providers="openrouter, kimi")
        assert main_module._parse_providers_allowlist(args) == frozenset({"openrouter", "kimi"})

    def test_all_whitespace_returns_none(self):
        args = SimpleNamespace(providers="  ,  ,  ")
        assert main_module._parse_providers_allowlist(args) is None


# ---------------------------------------------------------------------------
# _pick_speed_provider
# ---------------------------------------------------------------------------

class TestPickSpeedProvider:
    def test_fast_no_allowed(self):
        assert main_module._pick_speed_provider("fast", None) == "openrouter"

    def test_max_no_allowed(self):
        assert main_module._pick_speed_provider("max", None) == "sonnet"

    def test_economy_no_allowed(self):
        assert main_module._pick_speed_provider("economy", None) == "local"

    def test_balanced_first_match_in_allowed(self):
        allowed = frozenset({"kimi"})
        assert main_module._pick_speed_provider("balanced", allowed) == "kimi"

    def test_fast_with_allowed_match(self):
        allowed = frozenset({"kimi"})
        assert main_module._pick_speed_provider("fast", allowed) == "kimi"

    def test_fast_no_match_raises(self):
        allowed = frozenset({"gemini"})
        with pytest.raises(ValueError):
            main_module._pick_speed_provider("fast", allowed)

    def test_unknown_mode_returns_none(self):
        assert main_module._pick_speed_provider("unknown", None) is None

    def test_premium_no_allowed(self):
        assert main_module._pick_speed_provider("premium", None) == "opus"

    def test_balanced_no_allowed(self):
        assert main_module._pick_speed_provider("balanced", None) == "openrouter"


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------

class TestBuildParser:
    def test_returns_argument_parser(self):
        parser = main_module.build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_parse_provider(self):
        parser = main_module.build_parser()
        args = parser.parse_args(["--provider", "kimi", "hello"])
        assert args.provider == "kimi"
        assert args.remaining == ["hello"]

    def test_parse_json(self):
        parser = main_module.build_parser()
        args = parser.parse_args(["--json"])
        assert args.json_output is True

    def test_parse_verbose(self):
        parser = main_module.build_parser()
        args = parser.parse_args(["--verbose"])
        assert args.verbose is True

    def test_parse_speed(self):
        parser = main_module.build_parser()
        args = parser.parse_args(["--speed", "fast"])
        assert args.speed == "fast"

    def test_parse_providers(self):
        parser = main_module.build_parser()
        args = parser.parse_args(["--providers", "kimi,sonnet"])
        assert args.providers == "kimi,sonnet"


# ---------------------------------------------------------------------------
# _print_result
# ---------------------------------------------------------------------------

class TestPrintResult:
    def test_plain_output(self, capsys):
        result = AgentResult(
            completed=True,
            final_response="Hello world",
            tokens_used=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            provider_used="kimi",
            tier_used="STANDARD",
        )
        args = SimpleNamespace(json_output=False, verbose=False)
        main_module._print_result(result, args, duration=1.23)
        captured = capsys.readouterr()
        assert captured.out.strip() == "Hello world"

    def test_json_output(self, capsys):
        result = AgentResult(
            completed=True,
            final_response="Hello world",
            tool_calls_made=3,
            iterations_used=2,
            tokens_used=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            provider_used="kimi",
            tier_used="STANDARD",
        )
        args = SimpleNamespace(json_output=True, verbose=False)

        with patch("mullet.core.agent.cost.estimate_cost", return_value=0.001234), \
             patch("mullet.core.agent.cost.estimate_savings", return_value=0.000500):
            main_module._print_result(result, args, duration=1.23)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["completed"] is True
        assert data["response"] == "Hello world"
        assert data["tool_calls"] == 3
        assert data["iterations"] == 2
        assert data["tokens"]["input"] == 10
        assert data["tokens"]["output"] == 5
        assert data["tokens"]["total"] == 15
        assert data["provider"] == "kimi"
        assert data["tier"] == "STANDARD"
        assert data["cost"] == 0.001234
        assert data["savings"] == 0.0005
        assert data["duration"] == 1.23


# ---------------------------------------------------------------------------
# _print_verbose
# ---------------------------------------------------------------------------

class TestPrintVerbose:
    def test_verbose_false_no_output(self, capsys):
        result = AgentResult(
            completed=True,
            final_response="Hi",
            tokens_used=TokenUsage(total_tokens=7),
            provider_used="local",
        )
        args = SimpleNamespace(verbose=False)
        main_module._print_verbose(result, args)
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_verbose_true_prints_stats(self, capsys):
        result = AgentResult(
            completed=True,
            final_response="Hi",
            tool_calls_made=1,
            iterations_used=2,
            tokens_used=TokenUsage(input_tokens=3, output_tokens=4, total_tokens=7),
            provider_used="sonnet",
            tier_used="ADVANCED",
        )
        args = SimpleNamespace(verbose=True)

        with patch("mullet.core.agent.cost.format_cost_summary", return_value="Cost: $0.01"):
            main_module._print_verbose(result, args)

        captured = capsys.readouterr()
        assert "Tools called: 1" in captured.err
        assert "Iterations: 2" in captured.err
        assert "Tokens: 7" in captured.err
        assert "Provider: sonnet (ADVANCED)" in captured.err
        assert "Cost: $0.01" in captured.err


# ---------------------------------------------------------------------------
# _run_models_command
# ---------------------------------------------------------------------------

class TestRunModelsCommand:
    def test_list_all(self, capsys):
        mock_model = MagicMock()
        mock_model.id = "gpt-4"
        mock_model.tier = "STANDARD"
        mock_model.context_length = 128000
        mock_model.input_cost_per_1k = 0.03
        mock_model.output_cost_per_1k = 0.06

        mock_registry = MagicMock()
        mock_registry.list_all.return_value = {"openai": [mock_model]}

        with patch("mullet.core.providers.model_registry.get_registry", return_value=mock_registry):
            main_module._run_models_command([])

        captured = capsys.readouterr()
        assert "OPENAI" in captured.out
        assert "gpt-4" in captured.out

    def test_refresh(self, capsys):
        mock_registry = MagicMock()
        mock_registry.refresh_all.return_value = {"openai": 42}

        with patch("mullet.core.providers.model_registry.get_registry", return_value=mock_registry):
            main_module._run_models_command(["refresh"])

        captured = capsys.readouterr()
        assert "Refreshing model registry" in captured.out
        assert "openai: 42 models" in captured.out
        assert "Done." in captured.out

    def test_provider_filter(self, capsys):
        mock_model = MagicMock()
        mock_model.id = "gpt-4"
        mock_model.tier = "STANDARD"
        mock_model.context_length = 128000
        mock_model.input_cost_per_1k = 0.03
        mock_model.output_cost_per_1k = 0.06

        mock_registry = MagicMock()
        mock_registry.list_all.return_value = {
            "openai": [mock_model],
            "anthropic": [],
        }

        with patch("mullet.core.providers.model_registry.get_registry", return_value=mock_registry):
            main_module._run_models_command(["--provider", "openai"])

        captured = capsys.readouterr()
        assert "OPENAI" in captured.out
        # Anthropic should be filtered out
        assert "gpt-4" in captured.out

    def test_list_sorts_by_output_cost(self, capsys):
        cheap = MagicMock()
        cheap.id = "cheap"
        cheap.tier = "LOW"
        cheap.context_length = 8000
        cheap.input_cost_per_1k = 0.01
        cheap.output_cost_per_1k = 0.01

        expensive = MagicMock()
        expensive.id = "expensive"
        expensive.tier = "STANDARD"
        expensive.context_length = 8000
        expensive.input_cost_per_1k = 0.01
        expensive.output_cost_per_1k = 0.10

        mock_registry = MagicMock()
        mock_registry.list_all.return_value = {"test": [expensive, cheap]}

        with patch("mullet.core.providers.model_registry.get_registry", return_value=mock_registry):
            main_module._run_models_command([])

        captured = capsys.readouterr()
        # cheap should appear before expensive because sorted by output_cost
        cheap_idx = captured.out.index("cheap")
        expensive_idx = captured.out.index("expensive")
        assert cheap_idx < expensive_idx
