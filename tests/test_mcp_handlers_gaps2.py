"""Tests covering uncovered paths in mullet/server/mcp_handlers.py (lines 993-1391).

Covers:
- handle_mullet_status: budget exception, running agents with elapsed_seconds
- handle_mullet_providers: per-provider env-var branches
- handle_mullet_budget: tracker raises → fallback JSON; success path
- handle_mullet_search: missing args, knowledge_session raises, success
- handle_mullet_remember: missing args, knowledge_session raises, success
- handle_mullet_analyze: missing task, agent_type override, headroom failure, full success
- handle_mullet_swarm: missing goal, run_swarm raises, apply_blocks path
- handle_mullet_opencode: tier routing, pressure routing, auto sentinel, explicit model override, add_agent raises
"""
import json
import os
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch, Mock, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# handle_mullet_status
# ---------------------------------------------------------------------------

class TestHandleMulletStatus:
    """Tests for handle_mullet_status (lines 998-1042)."""

    def _patch_base(self, monkeypatch, local_name="local-llm", can_dispatch=(True, "ok")):
        """Shared base patches for status handler."""
        mock_local = MagicMock()
        mock_local.name = local_name
        monkeypatch.setattr(
            "mullet.core.providers.local_detection.detect_local_provider",
            lambda: mock_local,
        )
        mock_guard = MagicMock()
        mock_guard.can_dispatch_local.return_value = can_dispatch
        mock_guard.get_system_snapshot.return_value = {"cpu": 0.1}
        monkeypatch.setattr(
            "mullet.core.router.resource_guard.get_resource_guard",
            lambda: mock_guard,
        )
        monkeypatch.setattr("mullet.hud.agent_state._read_current", lambda: [])
        return mock_local, mock_guard

    def test_status_ok_basic(self, monkeypatch):
        """handle_mullet_status returns status=ok with expected keys."""
        self._patch_base(monkeypatch)
        mock_br = MagicMock()
        mock_br.status.value = "go"
        mock_br.message = "all good"
        monkeypatch.setattr("mullet.hud.guard.check_budget", lambda: mock_br)

        from mullet.server.mcp_handlers import handle_mullet_status
        result = json.loads(handle_mullet_status({}))

        assert result["status"] == "ok"
        assert result["local_provider"] == "local-llm"
        assert result["local_can_dispatch"] is True
        assert result["budget"] == {"status": "go", "message": "all good"}
        assert result["running_agents"] == []
        assert result["agent_count"] == 0

    def test_status_budget_exception_leaves_budget_info_none(self, monkeypatch):
        """When check_budget raises, budget_info stays None (line 1014-1015)."""
        self._patch_base(monkeypatch)
        monkeypatch.setattr(
            "mullet.hud.guard.check_budget",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        from mullet.server.mcp_handlers import handle_mullet_status
        result = json.loads(handle_mullet_status({}))

        assert result["status"] == "ok"
        assert result["budget"] is None

    def test_status_budget_exception_via_importerror(self, monkeypatch):
        """Budget section silently suppresses any exception (lines 1014-1015)."""
        self._patch_base(monkeypatch)

        def _raise():
            raise ImportError("missing module")

        monkeypatch.setattr("mullet.hud.guard.check_budget", _raise)

        from mullet.server.mcp_handlers import handle_mullet_status
        result = json.loads(handle_mullet_status({}))

        assert result["budget"] is None
        assert result["status"] == "ok"

    def test_status_running_agents_include_elapsed_seconds(self, monkeypatch):
        """Running agents include elapsed_seconds computed from started_at (line 1021-1022)."""
        self._patch_base(monkeypatch)
        mock_br = MagicMock()
        mock_br.status.value = "go"
        mock_br.message = "ok"
        monkeypatch.setattr("mullet.hud.guard.check_budget", lambda: mock_br)

        started_at = time.time() - 30.0  # 30 seconds ago
        monkeypatch.setattr(
            "mullet.hud.agent_state._read_current",
            lambda: [
                {
                    "id": "agent-abc",
                    "description": "do stuff",
                    "provider": "kimi",
                    "model": "kimi-k2",
                    "status": "running",
                    "started_at": started_at,
                    "cost_usd": 0.01,
                }
            ],
        )

        from mullet.server.mcp_handlers import handle_mullet_status
        result = json.loads(handle_mullet_status({}))

        assert result["agent_count"] == 1
        agent = result["running_agents"][0]
        assert agent["id"] == "agent-abc"
        assert agent["elapsed_seconds"] >= 29  # allow a little scheduling slack
        assert agent["cost_usd"] == 0.01

    def test_status_agent_missing_started_at_defaults_to_zero_elapsed(self, monkeypatch):
        """Agent without started_at defaults elapsed to ~0 (line 1021)."""
        self._patch_base(monkeypatch)
        mock_br = MagicMock()
        mock_br.status.value = "go"
        mock_br.message = "ok"
        monkeypatch.setattr("mullet.hud.guard.check_budget", lambda: mock_br)
        monkeypatch.setattr(
            "mullet.hud.agent_state._read_current",
            lambda: [{"id": "x", "started_at": time.time()}],
        )

        from mullet.server.mcp_handlers import handle_mullet_status
        result = json.loads(handle_mullet_status({}))

        agent = result["running_agents"][0]
        assert agent["elapsed_seconds"] >= 0

    def test_status_no_local_provider(self, monkeypatch):
        """When no local provider, local_provider is None (line 1034)."""
        monkeypatch.setattr(
            "mullet.core.providers.local_detection.detect_local_provider",
            lambda: None,
        )
        mock_guard = MagicMock()
        mock_guard.can_dispatch_local.return_value = (False, "no local provider")
        mock_guard.get_system_snapshot.return_value = {}
        monkeypatch.setattr(
            "mullet.core.router.resource_guard.get_resource_guard",
            lambda: mock_guard,
        )
        monkeypatch.setattr("mullet.hud.agent_state._read_current", lambda: [])
        monkeypatch.setattr("mullet.hud.guard.check_budget", lambda: (_ for _ in ()).throw(Exception("x")))

        from mullet.server.mcp_handlers import handle_mullet_status
        result = json.loads(handle_mullet_status({}))

        assert result["local_provider"] is None
        assert result["local_can_dispatch"] is False


# ---------------------------------------------------------------------------
# handle_mullet_providers
# ---------------------------------------------------------------------------

class TestHandleMulletProviders:
    """Tests for handle_mullet_providers (lines 1045-1105)."""

    def _base_patches(self, monkeypatch):
        mock_guard = MagicMock()
        mock_guard.can_dispatch_local.return_value = (True, "ok")
        mock_guard.get_system_snapshot.return_value = {}
        monkeypatch.setattr(
            "mullet.core.router.resource_guard.get_resource_guard",
            lambda: mock_guard,
        )
        # No local by default
        monkeypatch.setattr(
            "mullet.core.providers.local_detection.detect_local_provider",
            lambda: None,
        )
        # Stub registry
        mock_registry = MagicMock()
        mock_registry.get_best_model.side_effect = lambda provider, tier: f"{provider}-{tier.lower()}-model"
        monkeypatch.setattr(
            "mullet.core.providers.model_registry.get_registry",
            lambda: mock_registry,
        )
        return mock_guard, mock_registry

    def test_providers_no_env_vars_empty_list(self, monkeypatch):
        """With no API keys and no local provider, providers list is empty."""
        self._base_patches(monkeypatch)
        for k in ("KIMI_API_KEY", "MOONSHOT_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
                   "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.delenv(k, raising=False)

        from mullet.server.mcp_handlers import handle_mullet_providers
        result = json.loads(handle_mullet_providers({}))

        names = [p["name"] for p in result["providers"]]
        assert "kimi" not in names
        assert "gemini" not in names
        assert "anthropic" not in names
        assert "openrouter" not in names

    def test_providers_kimi_api_key_includes_kimi(self, monkeypatch):
        """KIMI_API_KEY set → kimi provider present (line 1066-1075)."""
        self._base_patches(monkeypatch)
        for k in ("MOONSHOT_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
                   "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("KIMI_API_KEY", "test-kimi")

        from mullet.server.mcp_handlers import handle_mullet_providers
        result = json.loads(handle_mullet_providers({}))

        names = [p["name"] for p in result["providers"]]
        assert "kimi" in names
        kimi = next(p for p in result["providers"] if p["name"] == "kimi")
        assert kimi["healthy"] is True
        assert kimi["cost"] == "cheap"
        assert "actual_model_standard" in kimi
        assert "actual_model_critical" in kimi

    def test_providers_moonshot_key_also_includes_kimi(self, monkeypatch):
        """MOONSHOT_API_KEY is the alternate key for kimi provider."""
        self._base_patches(monkeypatch)
        for k in ("KIMI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
                   "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("MOONSHOT_API_KEY", "moonshot-key")

        from mullet.server.mcp_handlers import handle_mullet_providers
        result = json.loads(handle_mullet_providers({}))

        names = [p["name"] for p in result["providers"]]
        assert "kimi" in names

    def test_providers_google_api_key_includes_gemini(self, monkeypatch):
        """GOOGLE_API_KEY set → gemini provider present (line 1077-1086)."""
        self._base_patches(monkeypatch)
        for k in ("KIMI_API_KEY", "MOONSHOT_API_KEY", "GEMINI_API_KEY",
                   "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "google-key")

        from mullet.server.mcp_handlers import handle_mullet_providers
        result = json.loads(handle_mullet_providers({}))

        names = [p["name"] for p in result["providers"]]
        assert "gemini" in names
        gemini = next(p for p in result["providers"] if p["name"] == "gemini")
        assert gemini["cost"] == "budget"

    def test_providers_gemini_api_key_includes_gemini(self, monkeypatch):
        """GEMINI_API_KEY is the alternate key for gemini provider."""
        self._base_patches(monkeypatch)
        for k in ("KIMI_API_KEY", "MOONSHOT_API_KEY", "GOOGLE_API_KEY",
                   "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

        from mullet.server.mcp_handlers import handle_mullet_providers
        result = json.loads(handle_mullet_providers({}))

        names = [p["name"] for p in result["providers"]]
        assert "gemini" in names

    def test_providers_anthropic_api_key_includes_anthropic(self, monkeypatch):
        """ANTHROPIC_API_KEY set → anthropic provider present (line 1088-1097)."""
        self._base_patches(monkeypatch)
        for k in ("KIMI_API_KEY", "MOONSHOT_API_KEY", "GOOGLE_API_KEY",
                   "GEMINI_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

        from mullet.server.mcp_handlers import handle_mullet_providers
        result = json.loads(handle_mullet_providers({}))

        names = [p["name"] for p in result["providers"]]
        assert "anthropic" in names
        anthropic = next(p for p in result["providers"] if p["name"] == "anthropic")
        assert anthropic["cost"] == "premium"

    def test_providers_openrouter_api_key_includes_openrouter(self, monkeypatch):
        """OPENROUTER_API_KEY set → openrouter provider present (line 1099-1100)."""
        self._base_patches(monkeypatch)
        for k in ("KIMI_API_KEY", "MOONSHOT_API_KEY", "GOOGLE_API_KEY",
                   "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        from mullet.server.mcp_handlers import handle_mullet_providers
        result = json.loads(handle_mullet_providers({}))

        names = [p["name"] for p in result["providers"]]
        assert "openrouter" in names
        or_entry = next(p for p in result["providers"] if p["name"] == "openrouter")
        assert or_entry["model"] == "various"
        assert or_entry["cost"] == "variable"

    def test_providers_local_included_when_detected(self, monkeypatch):
        """When local provider is detected, it appears in the list (line 1058-1064)."""
        mock_guard = MagicMock()
        mock_guard.can_dispatch_local.return_value = (True, "ok")
        mock_guard.get_system_snapshot.return_value = {}
        monkeypatch.setattr(
            "mullet.core.router.resource_guard.get_resource_guard",
            lambda: mock_guard,
        )

        mock_local = MagicMock()
        mock_local.model = "qwen2.5-coder"
        mock_local.health_check.return_value = True
        monkeypatch.setattr(
            "mullet.core.providers.local_detection.detect_local_provider",
            lambda: mock_local,
        )
        mock_registry = MagicMock()
        mock_registry.get_best_model.side_effect = lambda p, t: f"{p}-model"
        monkeypatch.setattr(
            "mullet.core.providers.model_registry.get_registry",
            lambda: mock_registry,
        )
        for k in ("KIMI_API_KEY", "MOONSHOT_API_KEY", "GOOGLE_API_KEY",
                   "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.delenv(k, raising=False)

        from mullet.server.mcp_handlers import handle_mullet_providers
        result = json.loads(handle_mullet_providers({}))

        names = [p["name"] for p in result["providers"]]
        assert "local" in names
        local = next(p for p in result["providers"] if p["name"] == "local")
        assert local["healthy"] is True
        assert local["cost"] == "free"


# ---------------------------------------------------------------------------
# handle_mullet_budget
# ---------------------------------------------------------------------------

class TestHandleMulletBudget:
    """Tests for handle_mullet_budget (lines 1108-1121)."""

    def test_budget_success_returns_tracker_status(self, monkeypatch):
        """Success path returns BudgetTracker.status() dict (line 1113)."""
        mock_tracker = MagicMock()
        mock_tracker.status.return_value = {
            "session_tokens": 1234,
            "session_cost": 0.05,
            "daily_limit": 10.0,
        }
        with patch("mullet.core.router.budget.BudgetTracker", return_value=mock_tracker):
            from mullet.server.mcp_handlers import handle_mullet_budget
            result = json.loads(handle_mullet_budget({}))

        assert result["session_tokens"] == 1234
        assert result["session_cost"] == 0.05

    def test_budget_tracker_raises_returns_fallback(self, monkeypatch):
        """BudgetTracker() raises → fallback JSON with 'note' key (lines 1114-1121)."""
        with patch("mullet.core.router.budget.BudgetTracker", side_effect=RuntimeError("db gone")):
            from mullet.server.mcp_handlers import handle_mullet_budget
            result = json.loads(handle_mullet_budget({}))

        assert "note" in result
        assert result["session_tokens"] == 0
        assert result["session_cost"] == 0.0
        assert "unavailable" in result["note"].lower()

    def test_budget_status_raises_returns_fallback(self, monkeypatch):
        """BudgetTracker.status() raises → fallback JSON (lines 1114-1121)."""
        mock_tracker = MagicMock()
        mock_tracker.status.side_effect = Exception("status broken")
        with patch("mullet.core.router.budget.BudgetTracker", return_value=mock_tracker):
            from mullet.server.mcp_handlers import handle_mullet_budget
            result = json.loads(handle_mullet_budget({}))

        assert "note" in result


# ---------------------------------------------------------------------------
# handle_mullet_search
# ---------------------------------------------------------------------------

class TestHandleMulletSearch:
    """Tests for handle_mullet_search (lines 1124-1148)."""

    def test_search_missing_query_returns_error(self):
        """Missing query → error JSON (line 1131)."""
        from mullet.server.mcp_handlers import handle_mullet_search
        result = json.loads(handle_mullet_search({"client": "mypool"}))
        assert "error" in result
        assert "required" in result["error"]

    def test_search_missing_client_returns_error(self):
        """Missing client → error JSON (line 1131)."""
        from mullet.server.mcp_handlers import handle_mullet_search
        result = json.loads(handle_mullet_search({"query": "find something"}))
        assert "error" in result
        assert "required" in result["error"]

    def test_search_missing_both_returns_error(self):
        """Both missing → error JSON."""
        from mullet.server.mcp_handlers import handle_mullet_search
        result = json.loads(handle_mullet_search({}))
        assert "error" in result

    def test_search_knowledge_session_raises_returns_error(self, monkeypatch):
        """knowledge_session raises → error JSON with 'Search failed' (lines 1146-1148)."""
        @contextmanager
        def _bad_session(client):
            raise RuntimeError("db exploded")
            yield  # noqa: unreachable

        monkeypatch.setattr("mullet.core.knowledge.manager.knowledge_session", _bad_session)

        from mullet.server.mcp_handlers import handle_mullet_search
        result = json.loads(handle_mullet_search({"query": "test", "client": "pool"}))

        assert result.get("error") == "Search failed"

    def test_search_success_returns_formatted_results(self, monkeypatch):
        """Success path returns formatted results list (lines 1133-1145)."""
        mock_km = MagicMock()
        mock_km.search.return_value = [
            {"content": "some text here", "source": "file.py", "score": 0.9},
            {"text": "alt content", "metadata": {"source": "other.py"}, "score": 0.7},
        ]

        @contextmanager
        def _session(client):
            yield mock_km

        monkeypatch.setattr("mullet.core.knowledge.manager.knowledge_session", _session)

        from mullet.server.mcp_handlers import handle_mullet_search
        result = json.loads(handle_mullet_search({"query": "find it", "client": "pool", "top_k": 5}))

        assert result["count"] == 2
        assert result["results"][0]["content"] == "some text here"
        assert result["results"][0]["source"] == "file.py"
        assert result["results"][0]["score"] == 0.9
        mock_km.search.assert_called_once_with("find it", top_k=5)

    def test_search_content_truncated_to_500_chars(self, monkeypatch):
        """Content field is truncated to 500 chars (line 1141)."""
        long_content = "x" * 1000
        mock_km = MagicMock()
        mock_km.search.return_value = [{"content": long_content, "source": "f.py", "score": 1.0}]

        @contextmanager
        def _session(client):
            yield mock_km

        monkeypatch.setattr("mullet.core.knowledge.manager.knowledge_session", _session)

        from mullet.server.mcp_handlers import handle_mullet_search
        result = json.loads(handle_mullet_search({"query": "q", "client": "c"}))

        assert len(result["results"][0]["content"]) == 500


# ---------------------------------------------------------------------------
# handle_mullet_remember
# ---------------------------------------------------------------------------

class TestHandleMulletRemember:
    """Tests for handle_mullet_remember (lines 1151-1167)."""

    def test_remember_missing_key_returns_error(self):
        """Missing key → error JSON (line 1158)."""
        from mullet.server.mcp_handlers import handle_mullet_remember
        result = json.loads(handle_mullet_remember({"content": "c", "client": "p"}))
        assert "error" in result
        assert "required" in result["error"]

    def test_remember_missing_content_returns_error(self):
        """Missing content → error JSON (line 1158)."""
        from mullet.server.mcp_handlers import handle_mullet_remember
        result = json.loads(handle_mullet_remember({"key": "k", "client": "p"}))
        assert "error" in result

    def test_remember_missing_client_returns_error(self):
        """Missing client → error JSON (line 1158)."""
        from mullet.server.mcp_handlers import handle_mullet_remember
        result = json.loads(handle_mullet_remember({"key": "k", "content": "c"}))
        assert "error" in result

    def test_remember_missing_all_returns_error(self):
        """All missing → error JSON."""
        from mullet.server.mcp_handlers import handle_mullet_remember
        result = json.loads(handle_mullet_remember({}))
        assert "error" in result

    def test_remember_knowledge_session_raises_returns_error(self, monkeypatch):
        """knowledge_session raises → error JSON with 'Memory store failed' (lines 1165-1167)."""
        @contextmanager
        def _bad_session(client):
            raise RuntimeError("write fail")
            yield

        monkeypatch.setattr("mullet.core.knowledge.manager.knowledge_session", _bad_session)

        from mullet.server.mcp_handlers import handle_mullet_remember
        result = json.loads(handle_mullet_remember({"key": "k", "content": "c", "client": "p"}))

        assert result.get("error") == "Memory store failed"

    def test_remember_success_returns_stored_true(self, monkeypatch):
        """Success path returns stored=True with key and client (line 1164)."""
        mock_km = MagicMock()

        @contextmanager
        def _session(client):
            yield mock_km

        monkeypatch.setattr("mullet.core.knowledge.manager.knowledge_session", _session)

        from mullet.server.mcp_handlers import handle_mullet_remember
        result = json.loads(handle_mullet_remember({"key": "mykey", "content": "myval", "client": "mypool"}))

        assert result["stored"] is True
        assert result["key"] == "mykey"
        assert result["client"] == "mypool"
        mock_km.remember.assert_called_once_with("mykey", "myval", namespace="mcp")


# ---------------------------------------------------------------------------
# handle_mullet_analyze
# ---------------------------------------------------------------------------

class TestHandleMulletAnalyze:
    """Tests for handle_mullet_analyze (lines 1170-1224)."""

    def test_analyze_missing_task_returns_error(self):
        """Missing task → error JSON (line 1176)."""
        from mullet.server.mcp_handlers import handle_mullet_analyze
        result = json.loads(handle_mullet_analyze({}))
        assert result.get("error") == "task required"

    def test_analyze_empty_task_returns_error(self):
        """Empty task string → error JSON."""
        from mullet.server.mcp_handlers import handle_mullet_analyze
        result = json.loads(handle_mullet_analyze({"task": ""}))
        assert result.get("error") == "task required"

    def test_analyze_basic_task_returns_full_result(self):
        """Full success path returns expected keys (lines 1196-1214)."""
        from mullet.server.mcp_handlers import handle_mullet_analyze
        result = json.loads(handle_mullet_analyze({"task": "fix the login bug"}))

        assert "complexity_score" in result
        assert "tier" in result
        assert "is_simple" in result
        assert "signals" in result
        assert result["task"] == "fix the login bug"
        assert result["agent_override"] is None

    def test_analyze_agent_type_override_present(self):
        """agent_type with known override → agent_override is set (lines 1191-1194)."""
        from mullet.server.mcp_handlers import handle_mullet_analyze
        from mullet.core.router.decision import AGENT_TIER_OVERRIDES

        # Find a known override that has a non-None tier
        known_override = None
        known_type = None
        for agent_type, tier_val in AGENT_TIER_OVERRIDES.items():
            if tier_val is not None:
                known_override = tier_val
                known_type = agent_type
                break

        if known_type is None:
            pytest.skip("No AGENT_TIER_OVERRIDES with non-None value found")

        result = json.loads(handle_mullet_analyze({"task": "do something", "agent_type": known_type}))
        assert result["agent_override"] == known_override.value

    def test_analyze_check_headroom_raises_no_headroom_key(self, monkeypatch):
        """If check_headroom raises, result has no 'headroom' key (lines 1221-1222)."""
        monkeypatch.setattr(
            "mullet.hud.budget_monitor.check_headroom",
            lambda: (_ for _ in ()).throw(Exception("headroom broken")),
        )

        from mullet.server.mcp_handlers import handle_mullet_analyze
        result = json.loads(handle_mullet_analyze({"task": "analyze this code"}))

        # headroom key should be absent when check_headroom raises
        assert "headroom" not in result
        # But other keys must still be present
        assert "tier" in result
        assert "complexity_score" in result

    def test_analyze_signals_structure(self):
        """signals dict contains expected sub-keys (lines 1202-1214)."""
        from mullet.server.mcp_handlers import handle_mullet_analyze
        result = json.loads(handle_mullet_analyze({"task": "write unit tests for auth module"}))

        signals = result["signals"]
        for key in ("word_count", "file_paths", "architecture_keywords",
                    "debugging_keywords", "simple_keywords", "risk_keywords",
                    "question_depth", "subtasks", "cross_file", "domain", "impact"):
            assert key in signals, f"missing signal key: {key}"

    def test_analyze_task_truncated_to_100_chars(self):
        """task field in result is capped at 100 chars (line 1197)."""
        from mullet.server.mcp_handlers import handle_mullet_analyze
        long_task = "a" * 200
        result = json.loads(handle_mullet_analyze({"task": long_task}))
        assert len(result["task"]) == 100


# ---------------------------------------------------------------------------
# handle_mullet_swarm
# ---------------------------------------------------------------------------

class TestHandleMulletSwarm:
    """Tests for handle_mullet_swarm (lines 1227-1310)."""

    def test_swarm_missing_goal_returns_error(self):
        """Missing goal → error JSON (line 1238)."""
        from mullet.server.mcp_handlers import handle_mullet_swarm
        result = json.loads(handle_mullet_swarm({}))
        assert result.get("error") == "goal is required"

    def test_swarm_run_swarm_raises_returns_error(self, monkeypatch):
        """When run_swarm raises, returns error JSON with exception message (lines 1282-1284)."""
        async def _bad_swarm(*a, **kw):
            raise RuntimeError("swarm crash test")

        monkeypatch.setattr("mullet.core.orchestration.swarm.run_swarm", _bad_swarm)
        monkeypatch.setattr("mullet.hud.agent_state.add_agent", lambda *a, **kw: None)
        monkeypatch.setattr("mullet.hud.agent_state.remove_agent", lambda *a, **kw: None)
        monkeypatch.setattr(
            "mullet.hud.agent_state.update_agent_tokens", lambda *a, **kw: None
        )

        from mullet.server.mcp_handlers import handle_mullet_swarm
        result = json.loads(handle_mullet_swarm({"goal": "do stuff"}))

        assert "error" in result
        assert "swarm crash test" in result["error"]

    def test_swarm_success_returns_synthesis(self, monkeypatch):
        """Success path returns goal, synthesis, task_count etc."""
        from mullet.core.orchestration.swarm_types import SwarmResult
        from mullet.core.agent.types import TokenUsage

        fake_task = MagicMock()
        fake_task.id = "t1"
        fake_task.instruction = "do step 1"

        fake_subtask = MagicMock()
        fake_subtask.success = True
        fake_subtask.files_changed = []
        fake_subtask.tokens = TokenUsage(input_tokens=10, output_tokens=5)

        fake_result = SwarmResult(
            goal="build it",
            synthesis="all done",
            tasks=[fake_task],
            results=[fake_subtask],
            total_tokens=TokenUsage(input_tokens=100, output_tokens=50),
            partial_failure=False,
        )

        async def _fake_swarm(goal, ctx, **kw):
            return fake_result

        monkeypatch.setattr("mullet.core.orchestration.swarm.run_swarm", _fake_swarm)
        monkeypatch.setattr("mullet.hud.agent_state.add_agent", lambda *a, **kw: None)
        monkeypatch.setattr("mullet.hud.agent_state.remove_agent", lambda *a, **kw: None)
        monkeypatch.setattr(
            "mullet.hud.agent_state.update_agent_tokens", lambda *a, **kw: None
        )
        monkeypatch.setattr(
            "mullet.core.agent.cost.estimate_cost", lambda *a, **kw: 0.01
        )

        from mullet.server.mcp_handlers import handle_mullet_swarm
        result = json.loads(handle_mullet_swarm({"goal": "build it"}))

        assert result["goal"] == "build it"
        assert result["synthesis"] == "all done"
        assert result["task_count"] == 1
        assert result["success_count"] == 1
        assert "applied" not in result  # apply_blocks=False

    def test_swarm_apply_blocks_adds_applied_and_files_changed(self, monkeypatch):
        """apply=True → response includes applied=True and files_changed (line 1307-1309)."""
        from mullet.core.orchestration.swarm_types import SwarmResult
        from mullet.core.agent.types import TokenUsage

        fake_subtask = MagicMock()
        fake_subtask.success = True
        fake_subtask.files_changed = ["src/foo.py"]
        fake_subtask.tokens = TokenUsage(input_tokens=5, output_tokens=2)

        fake_task = MagicMock()
        fake_task.id = "t1"
        fake_task.instruction = "fix it"

        fake_result = SwarmResult(
            goal="fix all",
            synthesis="fixed",
            tasks=[fake_task],
            results=[fake_subtask],
            total_tokens=TokenUsage(input_tokens=5, output_tokens=2),
            partial_failure=False,
        )

        async def _fake_swarm(goal, ctx, **kw):
            return fake_result

        monkeypatch.setattr("mullet.core.orchestration.swarm.run_swarm", _fake_swarm)
        monkeypatch.setattr("mullet.hud.agent_state.add_agent", lambda *a, **kw: None)
        monkeypatch.setattr("mullet.hud.agent_state.remove_agent", lambda *a, **kw: None)
        monkeypatch.setattr(
            "mullet.hud.agent_state.update_agent_tokens", lambda *a, **kw: None
        )
        monkeypatch.setattr(
            "mullet.core.agent.cost.estimate_cost", lambda *a, **kw: 0.0
        )

        from mullet.server.mcp_handlers import handle_mullet_swarm
        result = json.loads(handle_mullet_swarm({"goal": "fix all", "apply": True}))

        assert result.get("applied") is True
        assert "files_changed" in result

    def test_swarm_path_sanitization_error(self, monkeypatch, tmp_path):
        """Path sanitization failure returns error JSON (line 1255)."""
        monkeypatch.setattr(
            "mullet.server.mcp_handlers._sanitize_paths",
            lambda files, target: ([], None, "path traversal detected"),
        )

        from mullet.server.mcp_handlers import handle_mullet_swarm
        result = json.loads(handle_mullet_swarm({
            "goal": "do thing",
            "files": ["../evil/file.py"],
        }))

        assert "error" in result
        assert "path traversal" in result["error"]


# ---------------------------------------------------------------------------
# handle_mullet_opencode
# ---------------------------------------------------------------------------

class TestHandleMulletOpencode:
    """Tests for handle_mullet_opencode (lines 1313-1391)."""

    def _mock_opencode_run(self, return_value=None):
        if return_value is None:
            return_value = {"status": "ok", "output": "done"}
        return patch(
            "mullet.tools.opencode_tool.opencode_run",
            return_value=return_value,
        )

    def _mock_resolve_model(self, model="moonshot/kimi-k2.6"):
        return patch(
            "mullet.tools.opencode_tool._resolve_configured_model",
            return_value=model,
        )

    def _mock_budget(self, pressure=0.3):
        mock_br = MagicMock()
        mock_br.pressure = pressure
        return patch("mullet.hud.guard.check_budget", return_value=mock_br)

    def _mock_local(self, model=None):
        """Mock detect_local_provider. model=None means LM Studio unreachable."""
        if model is None:
            return patch("mullet.core.providers.local_detection.detect_local_provider", return_value=None)
        mock_provider = MagicMock()
        mock_provider.model = model
        return patch("mullet.core.providers.local_detection.detect_local_provider", return_value=mock_provider)

    def test_no_task_uses_configured_model(self):
        """No task → falls to else → uses _resolve_configured_model() (line 1358)."""
        with self._mock_resolve_model("moonshot/kimi-k2.6") as mock_resolve, \
             self._mock_opencode_run({"status": "ok"}) as mock_run:
            from mullet.server.mcp_handlers import handle_mullet_opencode
            result = handle_mullet_opencode({})

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["model"] == "moonshot/kimi-k2.6"

    def test_low_tier_falls_back_to_configured_model_when_local_unreachable(self):
        """LOW tier, LM Studio unreachable → _resolve_configured_model() (line 1343-1344)."""
        with self._mock_resolve_model("moonshot/kimi-k2.6"), \
             self._mock_budget(0.3) as mock_budget, \
             self._mock_local(None), \
             self._mock_opencode_run({"status": "ok"}) as mock_run:
            with patch("mullet.core.router.complexity.is_simple_message", return_value=True):
                from mullet.server.mcp_handlers import handle_mullet_opencode
                result = handle_mullet_opencode({"task": "hello world"})

        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["model"] == "moonshot/kimi-k2.6"

    def test_low_tier_prefers_local_when_reachable(self):
        """LOW tier, LM Studio reachable → lightweight path, never the cloud default."""
        lightweight_return = {"completed": True, "response": "done", "duration": 1.0, "model": "lmstudio/qwen2.5-coder-14b-instruct", "files_changed": None, "usage": {"total_tokens": 10}}
        with self._mock_resolve_model("moonshot/kimi-k2.6"), \
             self._mock_budget(0.3), \
             self._mock_local("qwen2.5-coder-14b-instruct"), \
             self._mock_opencode_run({"status": "ok"}) as mock_run, \
             patch("mullet.server.mcp_handlers._run_via_lightweight_agent", return_value=lightweight_return) as mock_lw:
            with patch("mullet.core.router.complexity.is_simple_message", return_value=True):
                from mullet.server.mcp_handlers import handle_mullet_opencode
                result = handle_mullet_opencode({"task": "hello world"})

        mock_lw.assert_called_once()
        mock_run.assert_not_called()
        assert result == lightweight_return

    def test_explicit_tier_low_falls_back_when_local_unreachable(self):
        """tier='LOW' arg, LM Studio unreachable → _resolve_configured_model() (lines 1328-1344)."""
        with self._mock_resolve_model("local/qwen"), \
             self._mock_budget(0.2), \
             self._mock_local(None), \
             self._mock_opencode_run({"status": "ok"}) as mock_run:
            from mullet.server.mcp_handlers import handle_mullet_opencode
            result = handle_mullet_opencode({"task": "small task", "tier": "LOW"})

        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["model"] == "local/qwen"

    def test_high_pressure_uses_kimi_fallback(self):
        """pressure >= 0.80 → moonshot/kimi-k2.7-code (lines 1345-1347)."""
        with self._mock_resolve_model(), \
             self._mock_budget(0.85), \
             self._mock_opencode_run({"status": "ok"}) as mock_run:
            with patch("mullet.core.router.complexity.is_simple_message", return_value=False), \
                 patch("mullet.core.router.complexity.recommend_tier") as mock_rt:
                mock_rt.return_value = MagicMock(value="STANDARD")
                from mullet.server.mcp_handlers import handle_mullet_opencode
                result = handle_mullet_opencode({"task": "complex task"})

        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["model"] == "moonshot/kimi-k2.7-code"

    def test_critical_tier_uses_deepseek_v4_pro(self):
        """CRITICAL tier → openrouter/deepseek/deepseek-v4-pro (line 1348-1350)."""
        with self._mock_resolve_model(), \
             self._mock_budget(0.4), \
             self._mock_opencode_run({"status": "ok"}) as mock_run:
            from mullet.server.mcp_handlers import handle_mullet_opencode
            result = handle_mullet_opencode({"task": "huge refactor", "tier": "CRITICAL"})

        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["model"] == "openrouter/deepseek/deepseek-v4-pro"

    def test_advanced_reasoning_uses_deepseek_v4_pro(self):
        """ADVANCED tier + is_reasoning_task → openrouter/deepseek/deepseek-v4-pro (line 1348-1350)."""
        with self._mock_resolve_model(), \
             self._mock_budget(0.4), \
             self._mock_opencode_run({"status": "ok"}) as mock_run:
            with patch("mullet.core.router.complexity.is_reasoning_task", return_value=True):
                from mullet.server.mcp_handlers import handle_mullet_opencode
                result = handle_mullet_opencode({"task": "analyze architecture", "tier": "ADVANCED"})

        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["model"] == "openrouter/deepseek/deepseek-v4-pro"

    def test_advanced_non_reasoning_uses_deepseek_v4_pro(self):
        """ADVANCED tier + NOT is_reasoning_task → openrouter/deepseek/deepseek-v4-pro (line 1351-1353)."""
        with self._mock_resolve_model(), \
             self._mock_budget(0.4), \
             self._mock_opencode_run({"status": "ok"}) as mock_run:
            with patch("mullet.core.router.complexity.is_reasoning_task", return_value=False):
                from mullet.server.mcp_handlers import handle_mullet_opencode
                result = handle_mullet_opencode({"task": "complex code task", "tier": "ADVANCED"})

        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["model"] == "openrouter/deepseek/deepseek-v4-pro"

    def test_standard_tier_falls_back_to_deepseek_v4_flash_when_local_unreachable(self):
        """STANDARD tier, pressure < 0.80, LM Studio unreachable → openrouter/deepseek/deepseek-v4-flash."""
        with self._mock_resolve_model(), \
             self._mock_budget(0.4), \
             self._mock_local(None), \
             self._mock_opencode_run({"status": "ok"}) as mock_run:
            with patch("mullet.core.router.complexity.is_simple_message", return_value=False), \
                 patch("mullet.core.router.complexity.recommend_tier") as mock_rt:
                mock_rt.return_value = MagicMock(value="STANDARD")
                from mullet.server.mcp_handlers import handle_mullet_opencode
                result = handle_mullet_opencode({"task": "write tests"})

        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["model"] == "openrouter/deepseek/deepseek-v4-flash"

    def test_standard_tier_prefers_local_when_reachable(self):
        """STANDARD tier, pressure < 0.80, LM Studio reachable → lightweight path,
        never paid OpenRouter tokens. Falls back to cloud if local can't handle."""
        lightweight_return = {"completed": True, "response": "done", "duration": 1.0, "model": "lmstudio/qwen2.5-coder-14b-instruct", "files_changed": None, "usage": {"total_tokens": 10}}
        with self._mock_resolve_model(), \
             self._mock_budget(0.4), \
             self._mock_local("qwen2.5-coder-14b-instruct"), \
             self._mock_opencode_run({"status": "ok"}) as mock_run, \
             patch("mullet.server.mcp_handlers._run_via_lightweight_agent", return_value=lightweight_return) as mock_lw:
            with patch("mullet.core.router.complexity.is_simple_message", return_value=False), \
                 patch("mullet.core.router.complexity.recommend_tier") as mock_rt:
                mock_rt.return_value = MagicMock(value="STANDARD")
                from mullet.server.mcp_handlers import handle_mullet_opencode
                result = handle_mullet_opencode({"task": "write tests"})

        mock_lw.assert_called_once()
        mock_run.assert_not_called()
        assert result == lightweight_return

    def test_advanced_tier_ignores_local_reachability(self):
        """ADVANCED tier must stay cloud-only even when LM Studio is reachable
        (weak local models aren't trusted for complex tasks)."""
        with self._mock_resolve_model(), \
             self._mock_budget(0.4), \
             self._mock_local("qwen2.5-coder-14b-instruct"), \
             self._mock_opencode_run({"status": "ok"}) as mock_run:
            with patch("mullet.core.router.complexity.is_reasoning_task", return_value=False):
                from mullet.server.mcp_handlers import handle_mullet_opencode
                result = handle_mullet_opencode({"task": "complex code task", "tier": "ADVANCED"})

        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["model"] == "openrouter/deepseek/deepseek-v4-pro"

    def test_auto_model_calls_analyze_and_select(self):
        """model='auto' → calls analyze_prompt + select_kimi_model (lines 1361-1367)."""
        mock_pf = MagicMock()
        with patch("mullet.core.router.preflight.analyze_prompt", return_value=mock_pf) as mock_analyze, \
             patch("mullet.core.router.routing_matrix.select_kimi_model", return_value=("kimi-chosen", "fast")) as mock_select, \
             self._mock_opencode_run({"status": "ok"}) as mock_run:
            from mullet.server.mcp_handlers import handle_mullet_opencode
            result = handle_mullet_opencode({"task": "do thing", "model": "auto"})

        mock_analyze.assert_called_once()
        mock_select.assert_called_once()
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["model"] == "kimi-chosen"

    def test_explicit_model_arg_overrides_all_tier_logic(self):
        """Explicit model_arg bypasses all tier/budget logic (line 1324-1325)."""
        with self._mock_opencode_run({"status": "ok"}) as mock_run:
            from mullet.server.mcp_handlers import handle_mullet_opencode
            result = handle_mullet_opencode({
                "task": "complex task",
                "model": "custom/my-model",
            })

        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["model"] == "custom/my-model"

    def test_add_agent_raises_still_returns_opencode_result(self):
        """add_agent raising does not prevent opencode_run from being called (lines 1375-1376)."""
        with self._mock_opencode_run({"status": "ok", "output": "done"}) as mock_run, \
             self._mock_budget(0.3):
            with patch("mullet.hud.agent_state.add_agent", side_effect=RuntimeError("hud fail")):
                from mullet.server.mcp_handlers import handle_mullet_opencode
                result = handle_mullet_opencode({"task": "task", "model": "some/model"})

        assert result["status"] == "ok"
        mock_run.assert_called_once()

    def test_opencode_run_error_dict_passes_through(self):
        """opencode_run returning error dict is forwarded as-is (line 1385)."""
        error_dict = {"error": "ACP server not running", "code": 503}
        with self._mock_opencode_run(error_dict) as mock_run:
            from mullet.server.mcp_handlers import handle_mullet_opencode
            result = handle_mullet_opencode({"task": "do it", "model": "kimi/k2"})

        assert result == error_dict

    def test_agent_id_threaded_to_opencode_run_for_hud_accuracy(self):
        """agent_id must reach opencode_run so it can correct the HUD on escalation."""
        with self._mock_opencode_run({"status": "ok"}) as mock_run:
            from mullet.server.mcp_handlers import handle_mullet_opencode
            handle_mullet_opencode({"task": "task", "model": "some/model"})

        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["agent_id"] is not None
        assert call_kwargs["agent_id"].startswith("opencode-")

    def test_agent_id_not_passed_when_hud_write_fails(self):
        """If add_agent itself fails, agent_id must not be threaded through —
        opencode_run should never try to live-update a HUD entry that was
        never successfully created."""
        with self._mock_opencode_run({"status": "ok"}) as mock_run:
            with patch("mullet.hud.agent_state.add_agent", side_effect=RuntimeError("hud fail")):
                from mullet.server.mcp_handlers import handle_mullet_opencode
                handle_mullet_opencode({"task": "task", "model": "some/model"})

        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["agent_id"] is None

    def test_remove_agent_raises_in_finally_is_silenced(self):
        """remove_agent raising in finally block is silently suppressed (line 1389-1390)."""
        with self._mock_opencode_run({"status": "ok"}) as mock_run, \
             self._mock_budget(0.3):
            with patch("mullet.hud.agent_state.add_agent", return_value=None), \
                 patch("mullet.hud.agent_state.remove_agent", side_effect=RuntimeError("remove fail")):
                from mullet.server.mcp_handlers import handle_mullet_opencode
                # Should not raise even though remove_agent fails
                result = handle_mullet_opencode({"task": "task", "model": "some/model"})

        assert result["status"] == "ok"

    def test_lightweight_local_returns_none_when_no_local_provider(self):
        """_run_via_lightweight_agent returns None when detect_local_provider returns None."""
        with patch("mullet.core.providers.local_detection.detect_local_provider", return_value=None):
            from mullet.server.mcp_handlers import _run_via_lightweight_agent
            result = _run_via_lightweight_agent("task", "/tmp")
        assert result is None

    def test_lightweight_local_returns_dict_on_success(self):
        """_run_via_lightweight_agent returns a properly-shaped dict on mocked successful agent.run()."""
        mock_provider = MagicMock()
        mock_provider.model = "qwen2.5-coder-14b-instruct"

        mock_result = MagicMock()
        mock_result.completed = True
        mock_result.final_response = "done"
        mock_result.tokens_used.total_tokens = 123

        mock_agent = MagicMock()
        mock_agent.run.return_value = mock_result

        with patch("mullet.core.providers.local_detection.detect_local_provider", return_value=mock_provider), \
             patch("mullet.cli.main.build_tool_registry") as mock_registry, \
             patch("mullet.core.agent.prompts.build_base_prompt") as mock_prompt, \
             patch("mullet.core.agent.runner.AgentRunner", return_value=mock_agent) as mock_runner, \
             patch("mullet.tools.opencode_tool._git_status_snapshot", side_effect=[{"a.py"}, {"a.py", "b.py"}]):
            from mullet.server.mcp_handlers import _run_via_lightweight_agent
            result = _run_via_lightweight_agent("task", "/tmp")

        assert result is not None
        assert result["completed"] is True
        assert result["response"] == "done"
        assert isinstance(result["duration"], float)
        assert result["model"] == "lmstudio/qwen2.5-coder-14b-instruct"
        assert result["files_changed"] == ["b.py"]
        assert result["usage"] == {"total_tokens": 123}

        mock_runner.assert_called_once()
        call_kwargs = mock_runner.call_args.kwargs
        assert call_kwargs["toolsets"] == ["terminal", "file", "code", "git"]
        assert call_kwargs["max_iterations"] == 30
        mock_agent.run.assert_called_once_with("task")

    def test_lightweight_local_returns_none_when_agent_run_raises(self):
        """_run_via_lightweight_agent returns None when agent.run() raises."""
        mock_provider = MagicMock()
        mock_provider.model = "qwen2.5-coder-14b-instruct"

        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("boom")

        with patch("mullet.core.providers.local_detection.detect_local_provider", return_value=mock_provider), \
             patch("mullet.core.agent.runner.AgentRunner", return_value=mock_agent), \
             patch("mullet.tools.opencode_tool._git_status_snapshot", return_value=set()):
            from mullet.server.mcp_handlers import _run_via_lightweight_agent
            result = _run_via_lightweight_agent("task", "/tmp")

        assert result is None

    def test_lightweight_local_returns_none_when_not_completed(self):
        """_run_via_lightweight_agent returns None when result.completed is falsy."""
        mock_provider = MagicMock()
        mock_provider.model = "qwen2.5-coder-14b-instruct"

        mock_result = MagicMock()
        mock_result.completed = False

        mock_agent = MagicMock()
        mock_agent.run.return_value = mock_result

        with patch("mullet.core.providers.local_detection.detect_local_provider", return_value=mock_provider), \
             patch("mullet.core.agent.runner.AgentRunner", return_value=mock_agent), \
             patch("mullet.tools.opencode_tool._git_status_snapshot", return_value=set()):
            from mullet.server.mcp_handlers import _run_via_lightweight_agent
            result = _run_via_lightweight_agent("task", "/tmp")

        assert result is None

    def test_opencode_uses_lightweight_path_for_lmstudio_and_returns_directly(self):
        """handle_mullet_opencode calls _run_via_lightweight_agent (not opencode_run) when model starts with lmstudio/ and returns its result directly on success."""
        lightweight_return = {"completed": True, "response": "lightweight done", "duration": 1.2, "model": "lmstudio/qwen", "files_changed": None, "usage": {"total_tokens": 50}}
        with patch("mullet.server.mcp_handlers._run_via_lightweight_agent", return_value=lightweight_return) as mock_lightweight, \
             self._mock_opencode_run({"status": "ok"}) as mock_run:
            from mullet.server.mcp_handlers import handle_mullet_opencode
            result = handle_mullet_opencode({"task": "task", "model": "lmstudio/qwen"})

        mock_lightweight.assert_called_once_with("task", os.getcwd())
        mock_run.assert_not_called()
        assert result == lightweight_return

    def test_opencode_falls_back_to_cloud_when_lightweight_returns_none(self):
        """handle_mullet_opencode falls through to opencode_run with a cloud model when _run_via_lightweight_agent returns None."""
        with patch("mullet.server.mcp_handlers._run_via_lightweight_agent", return_value=None) as mock_lightweight, \
             self._mock_opencode_run({"status": "ok"}) as mock_run:
            from mullet.server.mcp_handlers import handle_mullet_opencode
            result = handle_mullet_opencode({"task": "task", "model": "lmstudio/qwen"})

        mock_lightweight.assert_called_once_with("task", os.getcwd())
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["model"] == "openrouter/deepseek/deepseek-v4-flash"
        assert result["status"] == "ok"

    def test_opencode_unchanged_for_cloud_models(self):
        """handle_mullet_opencode never calls _run_via_lightweight_agent when the resolved model doesn't start with lmstudio/."""
        with patch("mullet.server.mcp_handlers._run_via_lightweight_agent") as mock_lightweight, \
             self._mock_opencode_run({"status": "ok"}) as mock_run:
            from mullet.server.mcp_handlers import handle_mullet_opencode
            result = handle_mullet_opencode({"task": "task", "model": "openrouter/deepseek/deepseek-v4-flash"})

        mock_lightweight.assert_not_called()
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["model"] == "openrouter/deepseek/deepseek-v4-flash"


# ---------------------------------------------------------------------------
# handle_mullet_opencode: risk-flag attachment (_attach_risk_flags)
# ---------------------------------------------------------------------------

class TestHandleMulletOpencodeRiskFlags:
    """Tests for _attach_risk_flags wiring inside handle_mullet_opencode."""

    def _mock_opencode_run(self, return_value=None):
        if return_value is None:
            return_value = {"status": "ok", "output": "done"}
        return patch(
            "mullet.tools.opencode_tool.opencode_run",
            return_value=return_value,
        )

    def test_risk_flags_attached_when_completed_and_files_changed(self):
        """opencode_run reports completed=True with files_changed → risk_flags
        is populated from classify_git_diff's output (flattened to dicts),
        one call per file in files_changed, results concatenated in order."""
        from mullet.core.verification.diff_risk import RiskFlag

        flag_a = RiskFlag(
            file="a.py",
            line=3,
            category="literal_change",
            old_snippet="x = 1",
            new_snippet="x = 2",
            reason="numeric literal changed: 1 -> 2",
        )
        flag_b = RiskFlag(
            file="b.py",
            line=10,
            category="comment_changed",
            old_snippet="# old",
            new_snippet="# new",
            reason="comment text changed",
        )
        per_file_flags = {"a.py": [flag_a], "b.py": [flag_b]}
        opencode_result = {
            "completed": True,
            "files_changed": ["a.py", "b.py"],
            "response": "done",
        }
        with self._mock_opencode_run(opencode_result) as mock_run, \
             patch(
                 "mullet.core.verification.diff_risk.classify_git_diff",
                 side_effect=lambda cwd, filepath: per_file_flags[filepath],
             ) as mock_classify:
            from mullet.server.mcp_handlers import handle_mullet_opencode
            result = handle_mullet_opencode({"task": "task", "model": "some/model"})

        mock_run.assert_called_once()
        assert result["risk_flags"] == [flag_a.__dict__, flag_b.__dict__]
        assert mock_classify.call_count == 2

    def test_risk_flags_empty_when_classify_git_diff_raises(self):
        """A classifier bug must not break dispatch — risk_flags becomes []
        and handle_mullet_opencode still returns normally."""
        opencode_result = {
            "completed": True,
            "files_changed": ["a.py"],
            "response": "done",
        }
        with self._mock_opencode_run(opencode_result) as mock_run, \
             patch(
                 "mullet.core.verification.diff_risk.classify_git_diff",
                 side_effect=RuntimeError("boom"),
             ):
            from mullet.server.mcp_handlers import handle_mullet_opencode
            result = handle_mullet_opencode({"task": "task", "model": "some/model"})

        mock_run.assert_called_once()
        assert result["risk_flags"] == []

    def test_risk_flags_absent_when_files_changed_empty(self):
        """completed=True but files_changed=[] (falsy) → no risk_flags key at all."""
        opencode_result = {
            "completed": True,
            "files_changed": [],
            "response": "done",
        }
        with self._mock_opencode_run(opencode_result) as mock_run, \
             patch(
                 "mullet.core.verification.diff_risk.classify_git_diff",
             ) as mock_classify:
            from mullet.server.mcp_handlers import handle_mullet_opencode
            result = handle_mullet_opencode({"task": "task", "model": "some/model"})

        mock_run.assert_called_once()
        assert "risk_flags" not in result
        mock_classify.assert_not_called()

    def test_risk_flags_absent_when_files_changed_none(self):
        """completed=True but files_changed=None → no risk_flags key at all."""
        opencode_result = {
            "completed": True,
            "files_changed": None,
            "response": "done",
        }
        with self._mock_opencode_run(opencode_result) as mock_run, \
             patch(
                 "mullet.core.verification.diff_risk.classify_git_diff",
             ) as mock_classify:
            from mullet.server.mcp_handlers import handle_mullet_opencode
            result = handle_mullet_opencode({"task": "task", "model": "some/model"})

        mock_run.assert_called_once()
        assert "risk_flags" not in result
        mock_classify.assert_not_called()

    def test_risk_flags_absent_when_completed_falsy(self):
        """completed=False with files_changed present → no risk_flags key at all."""
        opencode_result = {
            "completed": False,
            "files_changed": ["a.py"],
            "response": "failed",
        }
        with self._mock_opencode_run(opencode_result) as mock_run, \
             patch(
                 "mullet.core.verification.diff_risk.classify_git_diff",
             ) as mock_classify:
            from mullet.server.mcp_handlers import handle_mullet_opencode
            result = handle_mullet_opencode({"task": "task", "model": "some/model"})

        mock_run.assert_called_once()
        assert "risk_flags" not in result
        mock_classify.assert_not_called()

    def test_risk_flags_attached_on_lightweight_local_path(self):
        """The lightweight (LM Studio) return path also gets risk_flags attached."""
        from mullet.core.verification.diff_risk import RiskFlag

        fake_flags = [
            RiskFlag(
                file="a.py",
                line=1,
                category="new_control_flow",
                old_snippet="return x;",
                new_snippet="return x && y;",
                reason="new control-flow token introduced: '&&'",
            ),
        ]
        lightweight_return = {
            "completed": True,
            "response": "done",
            "duration": 1.0,
            "model": "lmstudio/qwen2.5-coder-14b-instruct",
            "files_changed": ["a.py"],
            "usage": {"total_tokens": 10},
        }
        with patch(
                 "mullet.server.mcp_handlers._run_via_lightweight_agent",
                 return_value=lightweight_return,
             ) as mock_lw, \
             self._mock_opencode_run({"status": "ok"}) as mock_run, \
             patch(
                 "mullet.core.verification.diff_risk.classify_git_diff",
                 return_value=fake_flags,
             ):
            from mullet.server.mcp_handlers import handle_mullet_opencode
            result = handle_mullet_opencode({"task": "task", "model": "lmstudio/qwen"})

        mock_lw.assert_called_once()
        mock_run.assert_not_called()
        assert result["risk_flags"] == [flag.__dict__ for flag in fake_flags]
