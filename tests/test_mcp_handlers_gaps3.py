"""Tests covering missing branches in mcp_handlers.py (lines 77, 265-417, 610-682, 691-940, 964-969)."""
import json
import logging
from unittest.mock import MagicMock, patch

import pytest

import mullet.server.mcp_handlers as handlers
from mullet.server.mcp_handlers import (
    _rlog,
    _resolve_provider,
    _provider_for_tier,
    _handle_mullet_run_inner,
    handle_mullet_run,
    handle_mullet_delegate,
)


# ---------------------------------------------------------------------------
# _rlog (line 77)
# ---------------------------------------------------------------------------

class TestRlog:
    def test_rlog_writes_to_stderr(self, capsys):
        _rlog("hello world")
        captured = capsys.readouterr()
        assert "hello world" in captured.err
        assert "[mullet]" in captured.err


# ---------------------------------------------------------------------------
# _resolve_provider (lines 265-308)
# ---------------------------------------------------------------------------

class TestResolveProvider:
    def test_empty_allowlist_raises(self):
        with pytest.raises(ValueError, match="empty set"):
            _resolve_provider(allowed=set())

    def test_force_tier_calls_provider_for_tier(self):
        mock_prov = MagicMock()
        mock_prov.name = "haiku"
        with patch.object(handlers, "_provider_for_tier", return_value=mock_prov) as mock_ptf:
            result = _resolve_provider(force_tier="STANDARD")
        mock_ptf.assert_called_once()
        assert result is mock_prov

    def test_assert_raises_when_provider_not_in_allowlist(self):
        mock_prov = MagicMock()
        mock_prov.name = "kimi"
        with patch.object(handlers, "_provider_for_tier", return_value=mock_prov):
            with pytest.raises(RuntimeError, match="not in allowlist"):
                _resolve_provider(force_tier="STANDARD", allowed={"haiku"})

    def test_no_task_routes_to_low(self):
        mock_prov = MagicMock()
        mock_prov.name = "local"
        with patch.object(handlers, "_provider_for_tier", return_value=mock_prov) as mock_ptf:
            result = _resolve_provider()  # no task, no force_tier
        call_args = mock_ptf.call_args
        assert call_args[0][0] == "LOW"

    def test_simple_task_routes_to_low(self):
        mock_prov = MagicMock()
        mock_prov.name = "local"
        with patch.object(handlers, "_provider_for_tier", return_value=mock_prov), \
             patch("mullet.core.router.complexity.is_simple_message", return_value=True):
            result = _resolve_provider(task="hi")
        assert result is mock_prov


# ---------------------------------------------------------------------------
# _provider_for_tier (lines 332-417)
# ---------------------------------------------------------------------------

class TestProviderForTier:
    def _env_no_keys(self, monkeypatch):
        monkeypatch.delenv("KIMI_API_KEY", raising=False)
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def test_gemini_no_key_returns_none(self, monkeypatch):
        """_mk_gemini: no GEMINI_API_KEY → factory returns None."""
        self._env_no_keys(monkeypatch)
        result = _provider_for_tier("STANDARD", allowed={"gemini"})
        assert result is None

    def test_gemini_provider_raises_inner_caught(self, monkeypatch, caplog):
        """_mk_gemini: has key but GeminiProvider raises → caught internally, returns None."""
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        self._env_no_keys(monkeypatch)
        # Re-set GOOGLE_API_KEY because _env_no_keys cleared it
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        with patch("mullet.core.providers.gemini.GeminiProvider", side_effect=RuntimeError("boom")), \
             caplog.at_level(logging.WARNING):
            result = _provider_for_tier("STANDARD", allowed={"gemini"})
        assert result is None
        assert "Gemini unavailable" in caplog.text

    def test_local_blocked_by_resource_guard_skipped(self, monkeypatch):
        """_mk_local: resource_guard.can_dispatch_local()=(False, reason) → None."""
        self._env_no_keys(monkeypatch)
        mock_local = MagicMock()
        mock_guard = MagicMock()
        mock_guard.can_dispatch_local.return_value = (False, "GPU busy")
        with patch("mullet.core.providers.local_detection.detect_local_provider", return_value=mock_local), \
             patch("mullet.core.router.resource_guard.get_resource_guard", return_value=mock_guard):
            result = _provider_for_tier("STANDARD", allowed={"local"})
        assert result is None

    def test_factory_exception_continues_to_next(self, monkeypatch):
        """Candidate factory raises Exception in outer loop → logged, continues to next candidate."""
        self._env_no_keys(monkeypatch)
        with patch("mullet.core.providers.local_detection.detect_local_provider", side_effect=RuntimeError("factory crash")):
            result = _provider_for_tier("LOW", allowed={"local"})
        assert result is None

    def test_ok_false_skips_candidate(self, monkeypatch):
        """_ok(name) returns False → candidate skipped before factory is called."""
        self._env_no_keys(monkeypatch)
        result = _provider_for_tier("LOW", allowed=set())
        assert result is None


# ---------------------------------------------------------------------------
# _handle_mullet_run_inner (lines 691+)
# ---------------------------------------------------------------------------

class TestHandleMulletRunInner:
    def _mock_base(self):
        """Return a mock provider + agent runner."""
        mock_prov = MagicMock()
        mock_prov.name = "kimi"
        mock_prov.model = "k2"

        mock_result = MagicMock()
        mock_result.completed = True
        mock_result.final_response = "task done"
        mock_result.tool_calls_made = 2
        mock_result.tokens_used.total_tokens = 500
        mock_result.provider_used = "kimi"

        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result
        mock_runner.provider = mock_prov
        mock_runner._tried_providers = {"kimi"}
        return mock_prov, mock_runner

    def test_empty_task_returns_error(self):
        result = _handle_mullet_run_inner({"task": ""})
        assert result == {"error": "task is required"}

    def test_apply_blocks_returns_error(self):
        result = _handle_mullet_run_inner({"task": "do something", "apply": True})
        assert "apply=true is not supported" in result.get("error", "")

    def test_resolve_provider_raises_returns_error(self):
        with patch.object(handlers, "_resolve_provider", side_effect=ValueError("no provider")), \
             patch("mullet.server.mcp_handlers._write_agent_state_fn"), \
             patch("mullet.server.mcp_handlers._clear_agent_state"):
            result = _handle_mullet_run_inner({"task": "do something"})
        assert "provider resolution failed" in result.get("error", "")

    def test_no_provider_returns_error(self):
        with patch.object(handlers, "_resolve_provider", return_value=None), \
             patch("mullet.server.mcp_handlers._write_agent_state_fn"), \
             patch("mullet.server.mcp_handlers._clear_agent_state"):
            result = _handle_mullet_run_inner({"task": "do something"})
        assert result == {"error": "No LLM provider available"}

    def test_knowledge_load_failure_logged_and_continues(self, caplog):
        mock_prov, mock_runner = self._mock_base()
        with patch.object(handlers, "_resolve_provider", return_value=mock_prov), \
             patch("mullet.cli.main.build_tool_registry", return_value={}), \
             patch("mullet.server.mcp_handlers._write_agent_state_fn"), \
             patch("mullet.server.mcp_handlers._clear_agent_state"), \
             patch("mullet.core.agent.runner.AgentRunner", return_value=mock_runner), \
             patch("mullet.core.agent.prompts.build_base_prompt", return_value="sys"), \
             patch("mullet.core.providers.tokenize.estimate_tokens", return_value=100), \
             patch("mullet.server.mcp_handlers._build_fallback_chain", return_value=[]), \
             patch("mullet.core.knowledge.manager.build_knowledge_environment", side_effect=RuntimeError("db down")), \
             caplog.at_level(logging.WARNING):
            result = _handle_mullet_run_inner({"task": "do something", "client": "mypool"})
        assert "Knowledge load failed" in caplog.text
        assert "completed" in result

    def test_estimate_tokens_exception_uses_fallback(self, caplog):
        mock_prov, mock_runner = self._mock_base()
        with patch.object(handlers, "_resolve_provider", return_value=mock_prov), \
             patch("mullet.cli.main.build_tool_registry", return_value={}), \
             patch("mullet.server.mcp_handlers._write_agent_state_fn"), \
             patch("mullet.server.mcp_handlers._clear_agent_state"), \
             patch("mullet.core.agent.runner.AgentRunner", return_value=mock_runner), \
             patch("mullet.core.agent.prompts.build_base_prompt", return_value="sys"), \
             patch("mullet.server.mcp_handlers._build_fallback_chain", return_value=[]), \
             patch("mullet.core.providers.tokenize.estimate_tokens", side_effect=RuntimeError("tiktoken err")), \
             caplog.at_level(logging.WARNING):
            result = _handle_mullet_run_inner({"task": "do something"})
        assert "completed" in result
        assert "estimate_tokens failed" in caplog.text

    def test_removed_mode_returns_error(self):
        """ultrawork and ralph were removed; asking for them is an error, not a silent standard run."""
        with patch.object(handlers, "_resolve_provider") as resolve:
            for mode in ("ultrawork", "ralph"):
                result = _handle_mullet_run_inner({"task": "do complex work", "mode": mode})
                assert "was removed" in result["error"]
                assert "mullet job" in result["error"]
        resolve.assert_not_called()

    def test_agent_run_exception_returns_error(self):
        mock_prov, mock_runner = self._mock_base()
        mock_runner.run.side_effect = RuntimeError("agent crashed")
        with patch.object(handlers, "_resolve_provider", return_value=mock_prov), \
             patch("mullet.cli.main.build_tool_registry", return_value={}), \
             patch("mullet.server.mcp_handlers._write_agent_state_fn"), \
             patch("mullet.server.mcp_handlers._clear_agent_state"), \
             patch("mullet.core.agent.runner.AgentRunner", return_value=mock_runner), \
             patch("mullet.core.agent.prompts.build_base_prompt", return_value="sys"), \
             patch("mullet.server.mcp_handlers._build_fallback_chain", return_value=[]), \
             patch("mullet.core.providers.tokenize.estimate_tokens", return_value=100):
            result = _handle_mullet_run_inner({"task": "do something"})
        assert "Agent execution failed" in result.get("error", "")


# ---------------------------------------------------------------------------
# handle_mullet_run routing priors (lines 610, 615-616, 637-649, 681-682)
# ---------------------------------------------------------------------------

class TestHandleMulletRunRoutingPriors:
    def test_priors_mode_on_enables_priors(self, monkeypatch):
        """MULLET_ROUTING_PRIORS=on → priors_enabled=True (line 610)."""
        monkeypatch.setenv("MULLET_ROUTING_PRIORS", "on")
        result_value = {"completed": True, "response": "done", "actual_provider": "kimi", "actual_model": "k2"}
        with patch.object(handlers, "_handle_mullet_run_inner", return_value=result_value), \
             patch("mullet.core.routing_history.make_outcome", return_value=MagicMock()), \
             patch("mullet.core.routing_history.record"):
            result = handle_mullet_run({"task": "do something", "providers": ""})
        assert result.get("completed") is True

    def test_priors_auto_iter_recent_exception_disables_priors(self, monkeypatch):
        """MULLET_ROUTING_PRIORS=auto, iter_recent raises → priors disabled (lines 615-616)."""
        monkeypatch.setenv("MULLET_ROUTING_PRIORS", "auto")
        result_value = {"completed": True, "response": "done", "actual_provider": "kimi", "actual_model": "k2"}
        with patch.object(handlers, "_handle_mullet_run_inner", return_value=result_value), \
             patch("mullet.core.routing_history.iter_recent", side_effect=RuntimeError("db error")), \
             patch("mullet.core.routing_history.make_outcome", return_value=MagicMock()), \
             patch("mullet.core.routing_history.record"):
            result = handle_mullet_run({"task": "do something"})
        assert result.get("completed") is True

    def test_priors_rerank_fires_when_priors_enabled(self, monkeypatch):
        """When priors enabled and providers given, re-rank may modify arguments (lines 637-649)."""
        monkeypatch.setenv("MULLET_ROUTING_PRIORS", "on")
        result_value = {"completed": True, "actual_provider": "kimi", "actual_model": "k2"}

        captured_args = {}

        def capture_inner(args):
            captured_args.update(dict(args))
            return result_value

        with patch.object(handlers, "_handle_mullet_run_inner", side_effect=capture_inner), \
             patch("mullet.core.routing_history.make_outcome", return_value=MagicMock()), \
             patch("mullet.core.routing_history.record"), \
             patch("mullet.core.routing_priors.pick_better_provider_semantic", return_value=("gemini", "better quality")), \
             patch("mullet.core.routing_priors.pick_better_provider", return_value=("kimi", None)), \
             patch("mullet.core.routing_history.make_signature", return_value="sig123"):
            handle_mullet_run({"task": "do something", "providers": "kimi,gemini"})
        # The re-rank should have fired (semantic returned gemini), modifying providers
        assert captured_args.get("providers", "").startswith("gemini")

    def test_routing_history_record_exception_non_fatal(self, monkeypatch):
        """make_outcome raises → warning logged, result still returned (lines 706-707)."""
        monkeypatch.setenv("MULLET_ROUTING_PRIORS", "off")
        result_value = {"completed": True, "actual_provider": "kimi", "actual_model": "k2"}
        with patch.object(handlers, "_handle_mullet_run_inner", return_value=result_value), \
             patch("mullet.core.routing_history.make_outcome", side_effect=RuntimeError("history down")), \
             patch("mullet.core.routing_history.record"):
            result = handle_mullet_run({"task": "do something"})
        assert result.get("completed") is True


# ---------------------------------------------------------------------------
# handle_mullet_delegate (lines 964-969)
# ---------------------------------------------------------------------------

class TestHandleMulletDelegate:
    def test_missing_goal_returns_error(self):
        result = json.loads(handle_mullet_delegate({}))
        assert result == {"error": "goal is required"}

    def test_resolve_provider_raises_returns_error(self):
        with patch.object(handlers, "_resolve_provider", side_effect=ValueError("no prov")), \
             patch("mullet.server.mcp_handlers._write_agent_state_fn"), \
             patch("mullet.server.mcp_handlers._clear_agent_state"):
            result = json.loads(handle_mullet_delegate({"goal": "do task"}))
        assert "provider resolution failed" in result.get("error", "")

    def test_no_provider_returns_error(self):
        with patch.object(handlers, "_resolve_provider", return_value=None), \
             patch("mullet.server.mcp_handlers._write_agent_state_fn"), \
             patch("mullet.server.mcp_handlers._clear_agent_state"):
            result = json.loads(handle_mullet_delegate({"goal": "do task"}))
        assert result.get("error") == "No LLM provider available"

    def test_resource_flush_exception_non_fatal(self):
        """Lines 991-995: resource_guard.flush_resources raises → debug log, no propagation."""
        mock_prov = MagicMock()
        mock_prov.name = "kimi"
        mock_prov.model = "k2"
        mock_parent = MagicMock()
        mock_delegate_result = MagicMock()
        mock_delegate_result.completed = True
        mock_delegate_result.summary = "done"
        mock_delegate_result.error = None
        mock_guard = MagicMock()
        mock_guard.flush_resources.side_effect = RuntimeError("guard down")
        with patch.object(handlers, "_resolve_provider", return_value=mock_prov), \
             patch("mullet.server.mcp_handlers._write_agent_state_fn"), \
             patch("mullet.server.mcp_handlers._clear_agent_state"), \
             patch("mullet.core.agent.runner.AgentRunner", return_value=mock_parent), \
             patch("mullet.cli.main.build_tool_registry", return_value={}), \
             patch("mullet.core.agent.delegation.delegate_task", return_value=mock_delegate_result), \
             patch("mullet.core.router.resource_guard.get_resource_guard", return_value=mock_guard):
            result = json.loads(handle_mullet_delegate({"goal": "do task"}))
        assert result.get("completed") is True


# ---------------------------------------------------------------------------
# Regression tests for Bug 1 (provider-name vs allowlist-alias mismatch)
# ---------------------------------------------------------------------------

class TestAliasAllowlistMismatch:
    def test_assert_allows_provider_when_alias_tag_matches_allowlist(self):
        """_assert must accept a provider whose .name differs from allowlist
        when _mcp_resolved_alias was set by _provider_for_tier."""
        mock_prov = MagicMock()
        mock_prov.name = "anthropic"
        mock_prov._mcp_resolved_alias = "sonnet"
        with patch.object(handlers, "_provider_for_tier", return_value=mock_prov):
            result = _resolve_provider(force_tier="STANDARD", allowed={"sonnet"})
        assert result is mock_prov

    def test_assert_still_raises_when_neither_name_nor_alias_matches(self):
        """_assert must still fire BUG when provider is truly outside allowlist."""
        mock_prov = MagicMock()
        mock_prov.name = "anthropic"
        # no _mcp_resolved_alias set
        with patch.object(handlers, "_provider_for_tier", return_value=mock_prov):
            with pytest.raises(RuntimeError, match="not in allowlist"):
                _resolve_provider(force_tier="STANDARD", allowed={"sonnet"})

    def test_provider_for_tier_none_without_anthropic_key(self, monkeypatch):
        """Anthropic-only routing with no ANTHROPIC_API_KEY must resolve to
        None (matching how Kimi/Gemini behave when unconfigured)."""
        from mullet.server.mcp_handlers import _provider_for_tier
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("KIMI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with patch("mullet.core.providers.local_detection.detect_local_provider", return_value=None):
            result = _provider_for_tier("STANDARD", allowed={"sonnet"})
        assert result is None


class TestAnthropicOnlyNativeHandoff:
    """Anthropic-only routing with no ANTHROPIC_API_KEY must hand the task
    back to the caller (Claude Code) with a clear message instead of
    erroring opaquely."""

    def test_helper_true_for_anthropic_only_no_key(self, monkeypatch):
        from mullet.server.mcp_handlers import _anthropic_only_needs_native_handoff
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert _anthropic_only_needs_native_handoff({"sonnet"}) is True
        assert _anthropic_only_needs_native_handoff({"haiku", "opus-4-8"}) is True

    def test_helper_false_when_key_present(self, monkeypatch):
        from mullet.server.mcp_handlers import _anthropic_only_needs_native_handoff
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        assert _anthropic_only_needs_native_handoff({"sonnet"}) is False

    def test_helper_false_when_not_anthropic_only(self, monkeypatch):
        from mullet.server.mcp_handlers import _anthropic_only_needs_native_handoff
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert _anthropic_only_needs_native_handoff({"sonnet", "kimi"}) is False
        assert _anthropic_only_needs_native_handoff(None) is False
        assert _anthropic_only_needs_native_handoff(set()) is False

    def test_handle_mullet_run_returns_native_handoff(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = _handle_mullet_run_inner({"task": "hi", "providers": "sonnet"})
        assert "error" in result
        assert "in-context" in result["error"]
        assert "ANTHROPIC_API_KEY" in result["error"]

    def test_handle_mullet_delegate_returns_native_handoff(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = json.loads(handle_mullet_delegate({"goal": "hi", "providers": "haiku"}))
        assert "error" in result
        assert "in-context" in result["error"]
        assert "ANTHROPIC_API_KEY" in result["error"]


# ---------------------------------------------------------------------------
# Regression tests for Bug 3 (auto-swarm zero-task fallback)
# ---------------------------------------------------------------------------

class TestAutoSwarmZeroTaskFallback:
    def test_auto_swarm_zero_tasks_falls_back_to_single_dispatch(self, monkeypatch):
        """When run_swarm returns an empty SwarmResult, _handle_mullet_run_inner
        must fall through to single-provider dispatch instead of returning
        a misleading empty success."""
        from mullet.core.orchestration.swarm_types import SwarmResult

        fake_result = SwarmResult(
            goal="test",
            tasks=[],
            synthesis="",
        )

        async def fake_run_swarm(goal, ctx, **kw):
            return fake_result

        monkeypatch.setattr("mullet.core.orchestration.swarm.run_swarm", fake_run_swarm)
        monkeypatch.setattr("mullet.core.router.auto_swarm._should_swarm", lambda *a, **kw: (True, "many files"))

        mock_prov = MagicMock()
        mock_prov.name = "kimi"
        mock_prov.model = "k2"
        mock_result = MagicMock()
        mock_result.completed = True
        mock_result.final_response = "single provider done"
        mock_result.tool_calls_made = 0
        mock_result.tokens_used.total_tokens = 100
        mock_result.provider_used = "kimi"
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result
        mock_runner.provider = mock_prov
        mock_runner._tried_providers = {"kimi"}

        with patch.object(handlers, "_resolve_provider", return_value=mock_prov), \
             patch("mullet.cli.main.build_tool_registry", return_value={}), \
             patch("mullet.server.mcp_handlers._write_agent_state_fn"), \
             patch("mullet.server.mcp_handlers._clear_agent_state"), \
             patch("mullet.core.agent.runner.AgentRunner", return_value=mock_runner), \
             patch("mullet.core.agent.prompts.build_base_prompt", return_value="sys"), \
             patch("mullet.core.providers.tokenize.estimate_tokens", return_value=100), \
             patch("mullet.server.mcp_handlers._build_fallback_chain", return_value=[]):
            result = _handle_mullet_run_inner({"task": "review files", "files": ["a.py"] * 5})

        # Should NOT be a swarm result
        assert result.get("auto_swarm") is not True
        assert "single provider done" in result.get("response", "")
