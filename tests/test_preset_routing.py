"""Tests for preset-based routing in MCP handlers."""

import json
from unittest.mock import MagicMock, patch

import pytest

from mullet.core.presets import Preset


def _make_local_only_preset() -> Preset:
    return Preset(
        name="local-only",
        label="Local",
        allowed_providers=["local"],
        speed_mode="economy",
        opencode_model="lmstudio/qwen2.5-coder-14b-instruct",
    )


def _make_claude_only_preset() -> Preset:
    return Preset(
        name="claude-only",
        label="Claude Code only",
        allowed_providers=["sonnet", "haiku", "opus-4-8"],
        speed_mode="balanced",
        opencode_model=None,
        native_only=True,
    )


class TestPresetRoutingRun:
    def test_local_only_preset_sets_allowlist_and_speed_mode(self):
        """No caller providers + local-only preset → allowlist {'local'}, speed_mode 'economy'."""
        preset = _make_local_only_preset()
        with patch("mullet.server.mcp_handlers._active_preset", return_value=preset), \
             patch("mullet.server.mcp_handlers._resolve_provider") as mock_resolve, \
             patch("mullet.server.mcp_handlers._build_fallback_chain", return_value=[]), \
             patch("mullet.cli.main.build_tool_registry", return_value={}), \
             patch("mullet.core.agent.runner.AgentRunner") as MockRunner:
            mock_resolve.return_value = MagicMock(name="local", model="local-model")
            mock_instance = MagicMock()
            mock_instance.run.return_value = MagicMock(
                completed=True,
                final_response="done",
                tool_calls_made=0,
                tokens_used=MagicMock(total_tokens=0),
            )
            MockRunner.return_value = mock_instance

            from mullet.server.mcp_handlers import _handle_mullet_run_inner
            result = _handle_mullet_run_inner({"task": "do something"})

        mock_resolve.assert_called_once()
        _, kwargs = mock_resolve.call_args
        assert kwargs.get("allowed") == {"local"}
        assert kwargs.get("speed_mode") == "economy"

    def test_explicit_providers_override_preset_run(self):
        """Caller supplies providers → preset allowlist is ignored."""
        preset = _make_local_only_preset()
        with patch("mullet.server.mcp_handlers._active_preset", return_value=preset), \
             patch("mullet.server.mcp_handlers._resolve_provider") as mock_resolve, \
             patch("mullet.server.mcp_handlers._build_fallback_chain", return_value=[]), \
             patch("mullet.cli.main.build_tool_registry", return_value={}), \
             patch("mullet.core.agent.runner.AgentRunner") as MockRunner:
            mock_resolve.return_value = MagicMock(name="kimi", model="kimi-model")
            mock_instance = MagicMock()
            mock_instance.run.return_value = MagicMock(
                completed=True,
                final_response="done",
                tool_calls_made=0,
                tokens_used=MagicMock(total_tokens=0),
            )
            MockRunner.return_value = mock_instance

            from mullet.server.mcp_handlers import _handle_mullet_run_inner
            result = _handle_mullet_run_inner({"task": "do something", "providers": "kimi,gemini"})

        mock_resolve.assert_called_once()
        _, kwargs = mock_resolve.call_args
        assert kwargs.get("allowed") == {"kimi", "gemini"}
        # Caller wins: preset speed_mode should NOT be applied
        assert kwargs.get("speed_mode") == "balanced"


class TestPresetRoutingDelegate:
    def test_local_only_preset_sets_allowlist_and_speed_mode(self):
        preset = _make_local_only_preset()
        with patch("mullet.server.mcp_handlers._active_preset", return_value=preset), \
             patch("mullet.server.mcp_handlers._resolve_provider") as mock_resolve, \
             patch("mullet.cli.main.build_tool_registry", return_value={}), \
             patch("mullet.core.agent.runner.AgentRunner") as MockRunner, \
             patch("mullet.core.agent.delegation.delegate_task", return_value=MagicMock(completed=True, summary="ok", error=None)):
            mock_resolve.return_value = MagicMock(name="local", model="local-model")
            MockRunner.return_value = MagicMock()

            from mullet.server.mcp_handlers import handle_mullet_delegate
            result = json.loads(handle_mullet_delegate({"goal": "do something"}))

        mock_resolve.assert_called_once()
        _, kwargs = mock_resolve.call_args
        assert kwargs.get("allowed") == {"local"}
        assert kwargs.get("speed_mode") == "economy"

    def test_explicit_providers_override_preset_delegate(self):
        preset = _make_local_only_preset()
        with patch("mullet.server.mcp_handlers._active_preset", return_value=preset), \
             patch("mullet.server.mcp_handlers._resolve_provider") as mock_resolve, \
             patch("mullet.cli.main.build_tool_registry", return_value={}), \
             patch("mullet.core.agent.runner.AgentRunner") as MockRunner, \
             patch("mullet.core.agent.delegation.delegate_task", return_value=MagicMock(completed=True, summary="ok", error=None)):
            mock_resolve.return_value = MagicMock(name="kimi", model="kimi-model")
            MockRunner.return_value = MagicMock()

            from mullet.server.mcp_handlers import handle_mullet_delegate
            result = json.loads(handle_mullet_delegate({"goal": "do something", "providers": "kimi,gemini"}))

        mock_resolve.assert_called_once()
        _, kwargs = mock_resolve.call_args
        assert kwargs.get("allowed") == {"kimi", "gemini"}
        assert kwargs.get("speed_mode") == "balanced"


class TestPresetRoutingOpencode:
    def test_native_only_preset_blocks_opencode(self):
        preset = _make_claude_only_preset()
        with patch("mullet.server.mcp_handlers._active_preset", return_value=preset), \
             patch("mullet.tools.opencode_tool.opencode_run") as mock_run:
            from mullet.server.mcp_handlers import handle_mullet_opencode
            result = handle_mullet_opencode({"task": "edit foo.py"})

        assert "error" in result
        assert "native" in result["error"].lower() or "disabled" in result["error"].lower()
        assert "claude-only" in result["error"] or "Preset" in result["error"]
        mock_run.assert_not_called()

    def test_local_only_preset_uses_lightweight_agent_when_reachable(self):
        # lmstudio/-prefixed models now route through the lightweight AgentRunner
        # path (mullet_opencode's CLI overhead doesn't fit local's context budget)
        # instead of opencode_run directly — see _run_via_lightweight_agent.
        preset = _make_local_only_preset()
        lightweight_result = {"completed": True, "response": "ok", "files_changed": []}
        with patch("mullet.server.mcp_handlers._active_preset", return_value=preset), \
             patch("mullet.server.mcp_handlers._run_via_lightweight_agent", return_value=lightweight_result) as mock_lightweight, \
             patch("mullet.tools.opencode_tool.opencode_run") as mock_run, \
             patch("mullet.hud.agent_state.add_agent"), \
             patch("mullet.hud.agent_state.remove_agent"):
            from mullet.server.mcp_handlers import handle_mullet_opencode
            result = handle_mullet_opencode({"task": "edit foo.py"})

        mock_lightweight.assert_called_once()
        args, _ = mock_lightweight.call_args
        assert args[0] == "edit foo.py"
        mock_run.assert_not_called()
        assert result["completed"] is True

    def test_local_only_preset_falls_back_to_cloud_when_local_unreachable(self):
        # When the lightweight agent can't run locally (detect_local_provider
        # returns None, or the run fails/incompletes), handle_mullet_opencode
        # must fall through to opencode_run with a real cloud model, not retry
        # the same unreachable local model.
        preset = _make_local_only_preset()
        with patch("mullet.server.mcp_handlers._active_preset", return_value=preset), \
             patch("mullet.server.mcp_handlers._run_via_lightweight_agent", return_value=None), \
             patch("mullet.tools.opencode_tool.opencode_run") as mock_run, \
             patch("mullet.hud.agent_state.add_agent"), \
             patch("mullet.hud.agent_state.remove_agent"):
            from mullet.server.mcp_handlers import handle_mullet_opencode
            handle_mullet_opencode({"task": "edit foo.py"})

        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs.get("model") == "openrouter/deepseek/deepseek-v4-flash"

    def test_explicit_model_arg_overrides_preset(self):
        preset = _make_local_only_preset()
        with patch("mullet.server.mcp_handlers._active_preset", return_value=preset), \
             patch("mullet.tools.opencode_tool.opencode_run") as mock_run, \
             patch("mullet.hud.agent_state.add_agent"), \
             patch("mullet.hud.agent_state.remove_agent"):
            from mullet.server.mcp_handlers import handle_mullet_opencode
            result = handle_mullet_opencode({"task": "edit foo.py", "model": "custom/model"})

        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs.get("model") == "custom/model"


class TestPresetRoutingFailOpen:
    def test_no_preset_behavior_unchanged_run(self):
        with patch("mullet.server.mcp_handlers._active_preset", return_value=None), \
             patch("mullet.server.mcp_handlers._resolve_provider") as mock_resolve, \
             patch("mullet.server.mcp_handlers._build_fallback_chain", return_value=[]), \
             patch("mullet.cli.main.build_tool_registry", return_value={}), \
             patch("mullet.core.agent.runner.AgentRunner") as MockRunner:
            mock_resolve.return_value = MagicMock(name="kimi", model="kimi-model")
            mock_instance = MagicMock()
            mock_instance.run.return_value = MagicMock(
                completed=True,
                final_response="done",
                tool_calls_made=0,
                tokens_used=MagicMock(total_tokens=0),
            )
            MockRunner.return_value = mock_instance

            from mullet.server.mcp_handlers import _handle_mullet_run_inner
            result = _handle_mullet_run_inner({"task": "do something"})

        mock_resolve.assert_called_once()
        _, kwargs = mock_resolve.call_args
        assert kwargs.get("allowed") is None
        assert kwargs.get("speed_mode") == "balanced"

    def test_no_preset_behavior_unchanged_delegate(self):
        with patch("mullet.server.mcp_handlers._active_preset", return_value=None), \
             patch("mullet.server.mcp_handlers._resolve_provider") as mock_resolve, \
             patch("mullet.cli.main.build_tool_registry", return_value={}), \
             patch("mullet.core.agent.runner.AgentRunner") as MockRunner, \
             patch("mullet.core.agent.delegation.delegate_task", return_value=MagicMock(completed=True, summary="ok", error=None)):
            mock_resolve.return_value = MagicMock(name="kimi", model="kimi-model")
            MockRunner.return_value = MagicMock()

            from mullet.server.mcp_handlers import handle_mullet_delegate
            result = json.loads(handle_mullet_delegate({"goal": "do something"}))

        mock_resolve.assert_called_once()
        _, kwargs = mock_resolve.call_args
        assert kwargs.get("allowed") is None
        assert kwargs.get("speed_mode") == "balanced"

    def test_no_preset_behavior_unchanged_opencode(self):
        with patch("mullet.server.mcp_handlers._active_preset", return_value=None), \
             patch("mullet.tools.opencode_tool.opencode_run") as mock_run, \
             patch("mullet.tools.opencode_tool._resolve_configured_model", return_value="opencode/big-pickle"), \
             patch("mullet.core.providers.local_detection.detect_local_provider", return_value=None), \
             patch("mullet.hud.agent_state.add_agent"), \
             patch("mullet.hud.agent_state.remove_agent"), \
             patch("mullet.core.router.complexity.is_simple_message", return_value=True):
            from mullet.server.mcp_handlers import handle_mullet_opencode
            result = handle_mullet_opencode({"task": "hello world"})

        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs.get("model") == "opencode/big-pickle"


class TestPresetRoutingNativeOnly:
    def test_native_only_preset_refuses_run_without_caller_providers(self):
        """No caller providers + native_only preset → refusal, _resolve_provider not called."""
        preset = _make_claude_only_preset()

        def _raising_resolve(*args, **kwargs):
            raise AssertionError("_resolve_provider should not be called")

        with patch("mullet.server.mcp_handlers._active_preset", return_value=preset), \
             patch("mullet.server.mcp_handlers._resolve_provider", side_effect=_raising_resolve):
            from mullet.server.mcp_handlers import _handle_mullet_run_inner
            result = _handle_mullet_run_inner({"task": "do something"})

        assert "error" in result
        assert "native-only" in result["error"].lower()
        assert "claude-only" in result["error"]

    def test_native_only_preset_refuses_delegate_without_caller_providers(self):
        """No caller providers + native_only preset → refusal, _resolve_provider not called."""
        preset = _make_claude_only_preset()

        def _raising_resolve(*args, **kwargs):
            raise AssertionError("_resolve_provider should not be called")

        with patch("mullet.server.mcp_handlers._active_preset", return_value=preset), \
             patch("mullet.server.mcp_handlers._resolve_provider", side_effect=_raising_resolve):
            from mullet.server.mcp_handlers import handle_mullet_delegate
            result = json.loads(handle_mullet_delegate({"goal": "do something"}))

        assert "error" in result
        assert "native-only" in result["error"].lower()
        assert "claude-only" in result["error"]

    def test_explicit_providers_override_native_only_preset_run(self):
        """Caller supplies providers → native_only preset does not refuse."""
        preset = _make_claude_only_preset()
        with patch("mullet.server.mcp_handlers._active_preset", return_value=preset), \
             patch("mullet.server.mcp_handlers._resolve_provider") as mock_resolve, \
             patch("mullet.server.mcp_handlers._build_fallback_chain", return_value=[]), \
             patch("mullet.cli.main.build_tool_registry", return_value={}), \
             patch("mullet.core.agent.runner.AgentRunner") as MockRunner:
            mock_resolve.return_value = MagicMock(name="local", model="local-model")
            mock_instance = MagicMock()
            mock_instance.run.return_value = MagicMock(
                completed=True,
                final_response="done",
                tool_calls_made=0,
                tokens_used=MagicMock(total_tokens=0),
            )
            MockRunner.return_value = mock_instance

            from mullet.server.mcp_handlers import _handle_mullet_run_inner
            result = _handle_mullet_run_inner({"task": "do something", "providers": "local"})

        mock_resolve.assert_called_once()
        _, kwargs = mock_resolve.call_args
        assert kwargs.get("allowed") == {"local"}
        assert kwargs.get("speed_mode") == "balanced"

    def test_explicit_providers_override_native_only_preset_delegate(self):
        """Caller supplies providers → native_only preset does not refuse."""
        preset = _make_claude_only_preset()
        with patch("mullet.server.mcp_handlers._active_preset", return_value=preset), \
             patch("mullet.server.mcp_handlers._resolve_provider") as mock_resolve, \
             patch("mullet.cli.main.build_tool_registry", return_value={}), \
             patch("mullet.core.agent.runner.AgentRunner") as MockRunner, \
             patch("mullet.core.agent.delegation.delegate_task", return_value=MagicMock(completed=True, summary="ok", error=None)):
            mock_resolve.return_value = MagicMock(name="local", model="local-model")
            MockRunner.return_value = MagicMock()

            from mullet.server.mcp_handlers import handle_mullet_delegate
            result = json.loads(handle_mullet_delegate({"goal": "do something", "providers": "local"}))

        mock_resolve.assert_called_once()
        _, kwargs = mock_resolve.call_args
        assert kwargs.get("allowed") == {"local"}
        assert kwargs.get("speed_mode") == "balanced"
