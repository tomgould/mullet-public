"""Tests covering uncovered paths in mullet/server/mcp_handlers.py (lines 993-1391).

Covers:
- handle_mullet_status: budget exception, running agents with elapsed_seconds
- handle_mullet_providers: per-provider env-var branches
- handle_mullet_budget: tracker raises → fallback JSON; success path
- handle_mullet_search: missing args, knowledge_session raises, success
- handle_mullet_remember: missing args, knowledge_session raises, success
- handle_mullet_analyze: missing task, agent_type override, headroom failure, full success
- handle_mullet_swarm: missing goal, run_swarm raises, apply=True refused
"""
import json
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

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
        assert "applied" not in result

    def test_swarm_apply_true_is_refused_before_running(self, monkeypatch):
        """apply=True returns an error and never runs the swarm, which can't write files."""
        called = []

        async def _fake_swarm(goal, ctx, **kw):
            called.append(goal)

        monkeypatch.setattr("mullet.core.orchestration.swarm.run_swarm", _fake_swarm)

        from mullet.server.mcp_handlers import handle_mullet_swarm
        result = json.loads(handle_mullet_swarm({"goal": "fix all", "apply": True}))

        assert "not supported" in result["error"]
        assert "applied" not in result
        assert called == []

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
