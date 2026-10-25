"""Tests for MCP handler dispatch and utility functions."""
import json
from unittest.mock import patch, MagicMock
from mullet.server.mcp_handlers import (
    _write_agent_state, _clear_agent_state, _build_fallback_chain,
    handle_mullet_status, handle_mullet_budget, handle_mullet_providers,
    MCP_TOOLS, MCP_HANDLERS,
)


def test_mcp_tools_list():
    assert isinstance(MCP_TOOLS, list)
    names = [t["name"] for t in MCP_TOOLS]
    assert "mullet_run" in names
    assert "mullet_delegate" in names
    assert "mullet_status" in names


def test_mcp_handlers_dict():
    assert "mullet_run" in MCP_HANDLERS
    assert "mullet_status" in MCP_HANDLERS
    assert callable(MCP_HANDLERS["mullet_run"])


@patch("mullet.core.providers.local_detection.detect_local_provider", return_value=None)
@patch("mullet.core.router.resource_guard.get_resource_guard")
def test_handle_status(mock_guard, mock_detect):
    mock_guard.return_value.can_dispatch_local.return_value = (False, "no local")
    mock_guard.return_value.get_system_snapshot.return_value = {"load_avg": 1.0}
    result = json.loads(handle_mullet_status({}))
    assert result["status"] == "ok"


def test_handle_budget():
    result = json.loads(handle_mullet_budget({}))
    assert "session_tokens" in result or "note" in result


@patch("mullet.core.providers.local_detection.detect_local_provider", return_value=None)
@patch("mullet.core.router.resource_guard.get_resource_guard")
def test_handle_providers(mock_guard, mock_detect):
    mock_guard.return_value.can_dispatch_local.return_value = (False, "none")
    mock_guard.return_value.get_system_snapshot.return_value = {}
    result = json.loads(handle_mullet_providers({}))
    assert "providers" in result


@patch.dict("os.environ", {"KIMI_API_KEY": "test-kimi-key"}, clear=False)
@patch("mullet.core.providers.local_detection.detect_local_provider", return_value=None)
@patch("mullet.core.router.resource_guard.get_resource_guard")
def test_handle_providers_kimi_model(mock_guard, mock_detect):
    """Kimi default is now k2.7-code (promoted over k2.6)."""
    mock_guard.return_value.get_system_snapshot.return_value = {}
    result = json.loads(handle_mullet_providers({}))
    kimi_entry = next((p for p in result["providers"] if p["name"] == "kimi"), None)
    assert kimi_entry is not None, "kimi provider should be present"
    assert kimi_entry["model"] == "kimi-k2.7-code", f"expected 'kimi-k2.7-code', got {kimi_entry['model']!r}"
    assert kimi_entry["actual_model_standard"] == "kimi-k2.7-code"


def test_write_clear_agent_state_is_now_a_noop(tmp_path):
    """_write_agent_state and _clear_agent_state are intentional no-ops now.

    The provider hook in OpenAICompatibleProvider._hud_agent_track is the single
    source of truth for HUD agent state: it writes on create() entry with a
    call-scoped id and removes on exit, so concurrent calls coexist cleanly
    without duplicated rows. Any legacy writes from the handler would double
    the row count for every task.
    """
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        _write_agent_state("test task", "local", "model-x")
        state_file = tmp_path / ".mullet" / "state" / "agents.json"
        # Shim is a no-op — no file should be written here.
        assert not state_file.exists()

        _clear_agent_state()
        assert not state_file.exists()
    finally:
        os.chdir(old_cwd)


@patch.dict("os.environ", {"GOOGLE_API_KEY": ""}, clear=False)
def test_build_fallback_chain_no_keys():
    chain = _build_fallback_chain()
    # With no API keys, chain may be empty or have CLI providers
    assert isinstance(chain, list)


def test_all_handlers_callable():
    for name, handler in MCP_HANDLERS.items():
        assert callable(handler), f"{name} handler is not callable"


def test_provider_for_tier_low_no_local_falls_to_cloud(monkeypatch):
    """When local LLM is unavailable, LOW tier falls through to cloud."""
    from mullet.server.mcp_handlers import _provider_for_tier
    monkeypatch.setenv("KIMI_API_KEY", "test-kimi-key")
    with patch("mullet.core.providers.local_detection.detect_local_provider", return_value=None):
        with patch("mullet.core.providers.kimi.KimiProvider") as mock_kimi:
            mock_kimi.return_value = MagicMock(name="kimi")
            result = _provider_for_tier("LOW")
            # Should fall through to cloud (Kimi)
            assert result is not None


def test_provider_for_tier_standard_prefers_kimi(monkeypatch):
    """STANDARD tier prefers Kimi over Gemini when both available."""
    from mullet.server.mcp_handlers import _provider_for_tier
    monkeypatch.setenv("KIMI_API_KEY", "kimi-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "gemini-test")
    with patch("mullet.core.providers.kimi.KimiProvider") as mock_kimi:
        mock_kimi.return_value = MagicMock(name="kimi")
        result = _provider_for_tier("STANDARD")
        mock_kimi.assert_called_once()


def test_provider_for_tier_no_keys_and_no_local_returns_none(monkeypatch):
    """When no API keys, no local LLM, returns None."""
    from mullet.server.mcp_handlers import _provider_for_tier
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with patch("mullet.core.providers.local_detection.detect_local_provider", return_value=None):
        result = _provider_for_tier("LOW")
        assert result is None


# --- Input validation tests ---

def test_handle_mullet_run_requires_task():
    """handle_mullet_run returns error when task is missing."""
    from mullet.server.mcp_handlers import handle_mullet_run
    result = handle_mullet_run({})
    assert isinstance(result, dict)  # D7: dict-not-string contract
    assert result.get("error") == "task is required"


def test_handle_mullet_run_empty_task():
    """handle_mullet_run returns error when task is empty string."""
    from mullet.server.mcp_handlers import handle_mullet_run
    result = handle_mullet_run({"task": ""})
    assert isinstance(result, dict)  # D7: dict-not-string contract
    assert result.get("error") == "task is required"


def test_handle_mullet_delegate_requires_goal():
    """handle_mullet_delegate returns error when goal is missing."""
    from mullet.server.mcp_handlers import handle_mullet_delegate
    result = json.loads(handle_mullet_delegate({}))
    assert result.get("error") == "goal is required"


def test_handle_mullet_swarm_requires_goal():
    """handle_mullet_swarm returns error when goal is missing."""
    from mullet.server.mcp_handlers import handle_mullet_swarm
    result = json.loads(handle_mullet_swarm({}))
    assert result.get("error") == "goal is required"


def test_handle_mullet_search_requires_query_and_client():
    """handle_mullet_search returns error when query or client missing."""
    from mullet.server.mcp_handlers import handle_mullet_search
    result = json.loads(handle_mullet_search({"query": "test"}))
    assert "required" in result.get("error", "")
    result2 = json.loads(handle_mullet_search({"client": "mypool"}))
    assert "required" in result2.get("error", "")


def test_handle_mullet_analyze_requires_task():
    """handle_mullet_analyze returns error when task is missing."""
    from mullet.server.mcp_handlers import handle_mullet_analyze
    result = json.loads(handle_mullet_analyze({}))
    assert result.get("error") == "task required"


# --- Auto-swarm routing tests ---

def test_auto_swarm_fires_on_many_files(monkeypatch):
    """handle_mullet_run routes to swarm when file count ≥ 5."""
    from mullet.core.orchestration.swarm_types import SwarmResult, SwarmTask
    from mullet.core.agent.types import TokenUsage

    fake_result = SwarmResult(
        goal="test",
        tasks=[SwarmTask(id="t1", instruction="do thing")],
        synthesis="swarm output",
        total_tokens=TokenUsage(input_tokens=10, output_tokens=5),
        partial_failure=False,
    )

    monkeypatch.setattr(
        "mullet.server.mcp_handlers._resolve_provider", lambda **kw: None
    )

    async def fake_run_swarm(goal, ctx, **kw):
        return fake_result

    monkeypatch.setattr("mullet.core.orchestration.swarm.run_swarm", fake_run_swarm)

    from mullet.server.mcp_handlers import handle_mullet_run
    files = [f"src/module{i}.py" for i in range(5)]
    result = handle_mullet_run({"task": "review all modules", "files": files})

    assert result.get("auto_swarm") is True
    assert "swarm_reason" in result
    assert result["response"] == "swarm output"


def test_auto_swarm_bypassed_with_force_single(monkeypatch):
    """force_single=True prevents auto-swarm even when heuristic would fire."""
    monkeypatch.setattr(
        "mullet.server.mcp_handlers._resolve_provider", lambda **kw: None
    )

    from mullet.server.mcp_handlers import handle_mullet_run
    files = [f"src/module{i}.py" for i in range(5)]
    result = handle_mullet_run({
        "task": "review all modules",
        "files": files,
        "force_single": True,
    })

    # Should NOT have taken the swarm path — will hit "no provider" instead
    assert result.get("auto_swarm") is not True


def test_auto_swarm_not_triggered_for_small_task(monkeypatch):
    """Small tasks without swarm signals do not route to swarm."""
    monkeypatch.setattr(
        "mullet.server.mcp_handlers._resolve_provider", lambda **kw: None
    )

    from mullet.server.mcp_handlers import handle_mullet_run
    result = handle_mullet_run({"task": "tiny task", "files": ["a.py"]})

    assert result.get("auto_swarm") is not True


def test_auto_swarm_fires_on_critical_tier(monkeypatch):
    """CRITICAL tier triggers auto-swarm routing."""
    from mullet.core.orchestration.swarm_types import SwarmResult, SwarmTask
    from mullet.core.agent.types import TokenUsage

    fake_result = SwarmResult(
        goal="critical task",
        tasks=[SwarmTask(id="t1", instruction="do thing")],
        synthesis="critical output",
        total_tokens=TokenUsage(input_tokens=20, output_tokens=10),
        partial_failure=False,
    )

    async def fake_run_swarm(goal, ctx, **kw):
        return fake_result

    monkeypatch.setattr("mullet.core.orchestration.swarm.run_swarm", fake_run_swarm)
    monkeypatch.setattr(
        "mullet.server.mcp_handlers._resolve_provider", lambda **kw: None
    )

    from mullet.server.mcp_handlers import handle_mullet_run
    result = handle_mullet_run({
        "task": "critical complex refactor",
        "files": ["a.py"],
        "tier": "CRITICAL",
    })

    assert result.get("auto_swarm") is True
    assert "CRITICAL" in result.get("swarm_reason", "")


def test_auto_swarm_fallback_on_swarm_error(monkeypatch):
    """If swarm raises, handle_mullet_run falls back to single-provider path."""
    async def bad_swarm(*a, **kw):
        raise RuntimeError("swarm exploded")

    monkeypatch.setattr("mullet.core.orchestration.swarm.run_swarm", bad_swarm)
    monkeypatch.setattr(
        "mullet.server.mcp_handlers._resolve_provider", lambda **kw: None
    )

    from mullet.server.mcp_handlers import handle_mullet_run
    files = [f"f{i}.py" for i in range(5)]
    result = handle_mullet_run({"task": "review", "files": files})

    # Fell back to single-provider path which has no provider → error
    assert result.get("auto_swarm") is not True


# ---------------------------------------------------------------------------
# Allowlist obedience — G2, G3, G7
# ---------------------------------------------------------------------------


def test_build_fallback_chain_empty_allowlist_raises():
    """Empty allowlist is a programming error — must raise loudly."""
    import pytest
    with pytest.raises(ValueError, match="empty set"):
        _build_fallback_chain(allowed=set())


def test_resolve_provider_empty_allowlist_raises():
    """Empty allowlist in _resolve_provider raises immediately."""
    import pytest
    from mullet.server.mcp_handlers import _resolve_provider
    with pytest.raises(ValueError, match="empty set"):
        _resolve_provider(task="hi", allowed=set())


def test_build_fallback_chain_allowlist_constraint_conflict_raises(monkeypatch):
    """When allowlist + constraints eliminates every candidate, raise ValueError."""
    import pytest
    monkeypatch.setenv("KIMI_API_KEY", "test-key")
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # Force constraint registry to reject every model.
    from mullet.core.providers import constraints as _c
    fake_constraint = MagicMock()
    fake_constraint.can_serve.return_value = False
    monkeypatch.setattr(
        _c.ConstraintRegistry, "get",
        lambda self, model_id: fake_constraint,
    )

    with pytest.raises(ValueError, match="filtered out by constraint registry"):
        _build_fallback_chain(
            allowed={"kimi"},
            requested_tier="STANDARD",
            prompt_tokens=1_000_000_000,
        )


def test_build_fallback_chain_no_allowlist_ignores_constraint_conflict(monkeypatch):
    """Without allowlist, constraint filter returning nothing just returns [] — no raise."""
    monkeypatch.setenv("KIMI_API_KEY", "test-key")
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    from mullet.core.providers import constraints as _c
    fake_constraint = MagicMock()
    fake_constraint.can_serve.return_value = False
    monkeypatch.setattr(
        _c.ConstraintRegistry, "get",
        lambda self, model_id: fake_constraint,
    )

    # No allowlist -> no raise, just empty list
    result = _build_fallback_chain(
        allowed=None,
        requested_tier="STANDARD",
        prompt_tokens=1_000_000_000,
    )
    assert result == []


def test_bridge_mode_skips_out_of_allowlist_providers(monkeypatch, tmp_path):
    """Defence-in-depth: bridge-mode inner loop skips providers not in allowlist."""
    # Create a throwaway file so bridge-mode engages
    f = tmp_path / "hello.py"
    f.write_text("print('hi')\n")

    class _FakeProv:
        def __init__(self, name):
            self.name = name
            self.model = f"{name}-model"

        def create(self, *a, **kw):  # pragma: no cover - should never be called for banned
            class _R:
                content = ""
                usage = None
            return _R()

    # preferred is a banned provider; fallbacks contains an allowed one that returns content.
    banned = _FakeProv("gemini")
    allowed_provider = _FakeProv("kimi")

    class _Resp:
        content = "done"
        usage = None
        tool_calls = []
        reasoning_content = None

    allowed_provider.create = lambda *a, **kw: _Resp()  # type: ignore

    monkeypatch.setattr(
        "mullet.server.mcp_handlers._resolve_provider",
        lambda **kw: banned,
    )
    monkeypatch.setattr(
        "mullet.server.mcp_handlers._build_fallback_chain",
        lambda **kw: [allowed_provider],
    )

    from mullet.server.mcp_handlers import handle_mullet_run
    result = handle_mullet_run({
        "task": "rewrite",
        "files": [str(f)],
        "target_dir": str(tmp_path),
        "providers": "kimi",
    })

    # The banned gemini provider should have been skipped.
    # The allowed kimi provider should have been the actual provider used.
    assert result.get("actual_provider") == "kimi"


# --- Cluster 2: allowed_providers threading through swarm paths ---

def test_handle_mullet_run_auto_swarm_passes_allowed_providers():
    """handle_mullet_run auto-swarm path passes allowed_providers to run_swarm."""
    from mullet.server import mcp_handlers
    from mullet.core.agent.types import TokenUsage
    from mullet.core.orchestration.swarm_types import SwarmResult

    from mullet.core.orchestration.swarm_types import SwarmTask
    fake_result = SwarmResult(
        goal="g",
        tasks=[SwarmTask(id="t1", instruction="do thing")],
        results=[],
        synthesis="s",
        partial_failure=False,
        total_tokens=TokenUsage(),
    )

    captured = {}

    async def _fake_run_swarm(task, ctx, *, allowed_providers=None, **kwargs):
        captured["allowed_providers"] = allowed_providers
        return fake_result

    with patch("mullet.core.router.auto_swarm._should_swarm", return_value=(True, "test")), \
         patch("mullet.core.orchestration.swarm.run_swarm", _fake_run_swarm):
        result = mcp_handlers.handle_mullet_run({
            "task": "do a thing",
            "providers": "kimi",
        })

    assert captured["allowed_providers"] == frozenset({"kimi"})
    assert result.get("auto_swarm") is True


def test_handle_mullet_swarm_parses_providers():
    """handle_mullet_swarm parses providers arg into frozenset and forwards to run_swarm."""
    from mullet.server import mcp_handlers
    from mullet.core.agent.types import TokenUsage
    from mullet.core.orchestration.swarm_types import SwarmResult

    fake_result = SwarmResult(
        goal="g",
        tasks=[],
        results=[],
        synthesis="s",
        partial_failure=False,
        total_tokens=TokenUsage(),
    )

    captured = {}

    async def _fake_run_swarm(goal, ctx, *, dry_run=False, allowed_providers=None, **kwargs):
        captured["allowed_providers"] = allowed_providers
        return fake_result

    with patch("mullet.core.orchestration.swarm.run_swarm", _fake_run_swarm):
        mcp_handlers.handle_mullet_swarm({
            "goal": "build it",
            "providers": "kimi,haiku",
        })

    assert captured["allowed_providers"] == frozenset({"kimi", "haiku"})


def test_handle_mullet_swarm_no_providers_is_none():
    """handle_mullet_swarm without providers arg passes allowed_providers=None."""
    from mullet.server import mcp_handlers
    from mullet.core.agent.types import TokenUsage
    from mullet.core.orchestration.swarm_types import SwarmResult

    fake_result = SwarmResult(
        goal="g",
        tasks=[],
        results=[],
        synthesis="s",
        partial_failure=False,
        total_tokens=TokenUsage(),
    )

    captured = {}

    async def _fake_run_swarm(goal, ctx, *, dry_run=False, allowed_providers=None, **kwargs):
        captured["allowed_providers"] = allowed_providers
        return fake_result

    with patch("mullet.core.orchestration.swarm.run_swarm", _fake_run_swarm):
        mcp_handlers.handle_mullet_swarm({"goal": "build it"})

    assert captured["allowed_providers"] is None


@patch("mullet.core.agent.delegation.delegate_task")
@patch("mullet.core.agent.runner.AgentRunner")
@patch("mullet.cli.main.build_tool_registry")
@patch("mullet.server.mcp_handlers._resolve_provider")
def test_handle_mullet_delegate_forwards_providers(
    mock_resolve, mock_registry, mock_runner, mock_delegate_task
):
    from mullet.server.mcp_handlers import handle_mullet_delegate

    mock_provider = MagicMock()
    mock_provider.name = "kimi"
    mock_provider.model = "kimi-k2"
    mock_resolve.return_value = mock_provider
    mock_registry.return_value = MagicMock()
    mock_runner.return_value = MagicMock()

    result_mock = MagicMock()
    result_mock.completed = True
    result_mock.summary = "done"
    result_mock.error = None
    mock_delegate_task.return_value = result_mock

    handle_mullet_delegate({"goal": "do x", "providers": "kimi,gemini"})

    # _resolve_provider called with allowed={"kimi","gemini"}
    assert mock_resolve.called
    call_allowed = mock_resolve.call_args.kwargs.get("allowed")
    assert call_allowed == {"kimi", "gemini"}

    # AgentRunner constructed with allowed_providers=frozenset({"kimi","gemini"})
    assert mock_runner.called
    ar_allowed = mock_runner.call_args.kwargs.get("allowed_providers")
    assert ar_allowed == frozenset({"kimi", "gemini"})


@patch("mullet.core.agent.delegation.delegate_task")
@patch("mullet.core.agent.runner.AgentRunner")
@patch("mullet.cli.main.build_tool_registry")
@patch("mullet.server.mcp_handlers._resolve_provider")
def test_handle_mullet_delegate_no_providers_defaults_to_none(
    mock_resolve, mock_registry, mock_runner, mock_delegate_task
):
    from mullet.server.mcp_handlers import handle_mullet_delegate

    mock_provider = MagicMock()
    mock_provider.name = "kimi"
    mock_provider.model = "kimi-k2"
    mock_resolve.return_value = mock_provider
    mock_registry.return_value = MagicMock()
    mock_runner.return_value = MagicMock()

    result_mock = MagicMock()
    result_mock.completed = True
    result_mock.summary = "done"
    result_mock.error = None
    mock_delegate_task.return_value = result_mock

    handle_mullet_delegate({"goal": "do x"})

    # _resolve_provider called with allowed=None
    assert mock_resolve.called
    call_allowed = mock_resolve.call_args.kwargs.get("allowed")
    assert call_allowed is None

    # AgentRunner constructed with allowed_providers=None
    assert mock_runner.called
    ar_allowed = mock_runner.call_args.kwargs.get("allowed_providers")
    assert ar_allowed is None


def test_sanitize_paths_path_traversal_detected(tmp_path):
    """Lines 61-64: file resolving outside target_dir → traversal error returned."""
    from mullet.server.mcp_handlers import _sanitize_paths
    files, target_str, err = _sanitize_paths(["../../etc/passwd"], str(tmp_path))
    assert files == []
    assert "traversal" in (err or "").lower()


def test_sanitize_paths_resolution_failure():
    """Lines 67-72: exception in path resolution → error string returned."""
    from mullet.server.mcp_handlers import _sanitize_paths
    # None target_dir causes TypeError in Path(None)
    files, target_str, err = _sanitize_paths(["somefile"], None)
    assert files == []
    assert err is not None
