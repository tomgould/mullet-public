"""Tests for UltraworkMode — parallel agent execution."""
from unittest.mock import MagicMock, patch

from mullet.core.agent.types import DelegationResult, TokenUsage
from mullet.core.orchestration.ultrawork import UltraworkMode, UltraworkResult


def test_ultrawork_result_defaults():
    r = UltraworkResult()
    assert r.completed is False
    assert r.subtask_results == []
    assert r.total_subtasks == 0
    assert r.summary == ""


def test_ultrawork_result_with_data():
    r = UltraworkResult(
        completed=True,
        total_subtasks=3,
        completed_subtasks=2,
        failed_subtasks=1,
        summary="Done",
    )
    assert r.completed is True
    assert r.failed_subtasks == 1


def test_ultrawork_mode_init():
    uw = UltraworkMode(max_workers=2)
    assert uw.max_workers == 2
    assert uw.reinforcement_count == 0
    assert uw.router is None


def test_ultrawork_mode_reinforcement_tracking():
    uw = UltraworkMode()
    assert uw.reinforcement_count == 0
    uw.reinforcement_count += 1
    assert uw.reinforcement_count == 1


def test_ultrawork_mode_init_allowed_providers_default_none():
    uw = UltraworkMode()
    assert uw.allowed_providers is None


def test_ultrawork_mode_init_accepts_allowed_providers():
    allow = frozenset({"kimi", "gemini"})
    uw = UltraworkMode(allowed_providers=allow)
    assert uw.allowed_providers == allow


def test_ultrawork_execute_forwards_allowed_providers():
    """UltraworkMode.execute must forward allowed_providers to route_delegation + delegate_batch."""
    allow = frozenset({"kimi", "gemini"})

    router = MagicMock()
    decision = MagicMock()
    decision.tier = MagicMock(value="STANDARD")
    decision.provider = "kimi"
    router.route_delegation.return_value = decision

    uw = UltraworkMode(router=router, max_workers=2, allowed_providers=allow)

    parent_agent = MagicMock()
    parent_agent.tier_used = "STANDARD"
    parent_agent.provider = MagicMock()
    parent_agent.provider.name = "kimi"

    subtasks = ["alpha", "beta"]

    with patch(
        "mullet.core.orchestration.ultrawork.delegate_batch",
        return_value=[],
    ) as mock_batch:
        uw.execute("parent task", parent_agent, subtasks=subtasks)

    # route_delegation called for each subtask with allowed_providers
    assert router.route_delegation.call_count == len(subtasks)
    for call in router.route_delegation.call_args_list:
        assert call.kwargs.get("allowed_providers") == allow

    # delegate_batch called once with allowed_providers
    assert mock_batch.call_count == 1
    assert mock_batch.call_args.kwargs.get("allowed_providers") == allow


class TestUltraworkMissingLines:
    """Cover specific uncovered lines in ultrawork.py."""

    def test_execute_no_subtasks_calls_decompose(self):
        uw = UltraworkMode()
        parent_agent = MagicMock()
        parent_agent.provider.name = "local"
        with patch.object(uw, "_decompose", return_value=["sub1"]) as mock_decomp:
            with patch("mullet.core.orchestration.ultrawork.delegate_batch", return_value=[]):
                uw.execute("task", parent_agent)
        mock_decomp.assert_called_once_with("task", parent_agent)

    def test_execute_empty_decompose_runs_single_task(self):
        uw = UltraworkMode()
        parent_agent = MagicMock()
        parent_agent.provider.name = "local"
        result_mock = MagicMock()
        result_mock.completed = True
        result_mock.final_response = "single result"
        result_mock.tokens_used = TokenUsage()
        parent_agent.run.return_value = result_mock
        with patch.object(uw, "_decompose", return_value=[]):
            res = uw.execute("task", parent_agent)
        assert res.completed is True
        assert res.summary == "single result"
        assert res.total_tokens == TokenUsage()
        parent_agent.run.assert_called_once_with("task")

    def test_router_provider_resolver_returns_provider(self):
        router = MagicMock()
        decision = MagicMock()
        decision.tier = MagicMock(value="STANDARD")
        decision.provider = "kimi"
        router.route_delegation.return_value = decision
        uw = UltraworkMode(router=router)
        parent_agent = MagicMock()
        parent_agent.tier_used = "STANDARD"
        parent_agent.provider.name = "local"
        with patch("mullet.core.router.resolver.ProviderResolver") as MockResolver:
            instance = MockResolver.return_value
            instance.resolve.return_value = "kimi-provider"
            with patch("mullet.core.orchestration.ultrawork.delegate_batch", return_value=[]) as mock_batch:
                uw.execute("task", parent_agent, subtasks=["st1"])
        call_kwargs = mock_batch.call_args.kwargs
        # Per-task overrides live in task dicts; batch-level override is None
        assert call_kwargs.get("provider_override") is None
        tasks = call_kwargs.get("tasks") or mock_batch.call_args.args[0]
        assert tasks[0].get("_provider_override") == "kimi-provider"

    def test_router_provider_resolver_raises_exception(self):
        router = MagicMock()
        decision = MagicMock()
        decision.tier = MagicMock(value="STANDARD")
        decision.provider = "kimi"
        router.route_delegation.return_value = decision
        uw = UltraworkMode(router=router)
        parent_agent = MagicMock()
        parent_agent.tier_used = "STANDARD"
        parent_agent.provider.name = "local"
        with patch("mullet.core.router.resolver.ProviderResolver") as MockResolver:
            instance = MockResolver.return_value
            instance.resolve.side_effect = RuntimeError("fail")
            with patch("mullet.core.orchestration.ultrawork.delegate_batch", return_value=[]) as mock_batch:
                res = uw.execute("task", parent_agent, subtasks=["st1"])
        call_kwargs = mock_batch.call_args.kwargs
        assert call_kwargs.get("provider_override") is None
        assert isinstance(res, UltraworkResult)

    def test_provider_override_stays_per_task(self):
        router = MagicMock()
        decision = MagicMock()
        decision.tier = MagicMock(value="STANDARD")
        decision.provider = "kimi"
        router.route_delegation.return_value = decision
        uw = UltraworkMode(router=router)
        parent_agent = MagicMock()
        parent_agent.tier_used = "STANDARD"
        parent_agent.provider.name = "local"
        with patch("mullet.core.router.resolver.ProviderResolver") as MockResolver:
            instance = MockResolver.return_value
            instance.resolve.return_value = "override-provider"
            with patch("mullet.core.orchestration.ultrawork.delegate_batch", return_value=[]) as mock_batch:
                uw.execute("task", parent_agent, subtasks=["st1", "st2", "st3"])
        call_kwargs = mock_batch.call_args.kwargs
        # Batch-level override is None; per-task overrides remain in task dicts
        assert call_kwargs.get("provider_override") is None
        tasks = call_kwargs.get("tasks") or mock_batch.call_args.args[0]
        for t in tasks:
            assert t.get("_provider_override") == "override-provider"

    def test_decompose_numbered_list(self):
        uw = UltraworkMode(max_workers=5)
        agent = MagicMock()
        result_mock = MagicMock()
        result_mock.completed = True
        result_mock.final_response = "1. First task\n2. Second task\n3. Third task"
        agent.run.return_value = result_mock
        subtasks = uw._decompose("task", agent)
        assert subtasks == ["First task", "Second task", "Third task"]

    def test_decompose_dash_list(self):
        uw = UltraworkMode(max_workers=5)
        agent = MagicMock()
        result_mock = MagicMock()
        result_mock.completed = True
        result_mock.final_response = "- First task\n- Second task"
        agent.run.return_value = result_mock
        subtasks = uw._decompose("task", agent)
        assert subtasks == ["First task", "Second task"]

    def test_decompose_not_completed_returns_empty(self):
        uw = UltraworkMode()
        agent = MagicMock()
        result_mock = MagicMock()
        result_mock.completed = False
        agent.run.return_value = result_mock
        subtasks = uw._decompose("task", agent)
        assert subtasks == []

    def test_decompose_lines_not_matching_pattern_ignored(self):
        uw = UltraworkMode(max_workers=5)
        agent = MagicMock()
        result_mock = MagicMock()
        result_mock.completed = True
        result_mock.final_response = "intro\n1. Valid\nRandom\n- Dash\noutro"
        agent.run.return_value = result_mock
        subtasks = uw._decompose("task", agent)
        assert subtasks == ["Valid", "Dash"]

    def test_decompose_capped_at_max_workers(self):
        uw = UltraworkMode(max_workers=2)
        agent = MagicMock()
        result_mock = MagicMock()
        result_mock.completed = True
        result_mock.final_response = "1. a\n2. b\n3. c\n4. d\n5. e"
        agent.run.return_value = result_mock
        subtasks = uw._decompose("task", agent)
        assert len(subtasks) == 4  # max_workers * 2

    def test_merge_results_completed_with_agent_result(self):
        uw = UltraworkMode()
        original = ["task A"]
        deduped, subtask_map = uw._deduplicate_subtasks(original)
        agent_res = MagicMock()
        agent_res.tokens_used = TokenUsage(input_tokens=5, output_tokens=10)
        agent_res.tier_used = "PREMIUM"
        result = DelegationResult(
            goal="task A", completed=True, summary="done", agent_result=agent_res
        )
        res = uw._merge_results([result], original, subtask_map)
        assert res.completed is True
        assert res.completed_subtasks == 1
        assert res.total_tokens.input_tokens == 5
        assert res.tiers_used == {"PREMIUM": 1}

    def test_merge_results_failed_with_agent_result(self):
        uw = UltraworkMode()
        original = ["task A"]
        deduped, subtask_map = uw._deduplicate_subtasks(original)
        agent_res = MagicMock()
        agent_res.tokens_used = TokenUsage(input_tokens=3, output_tokens=6)
        agent_res.tier_used = "STANDARD"
        result = DelegationResult(
            goal="task A", completed=False, summary="fail", agent_result=agent_res
        )
        res = uw._merge_results([result], original, subtask_map)
        assert res.completed is False
        assert res.failed_subtasks == 1
        assert res.total_tokens.input_tokens == 3
        assert res.tiers_used == {"STANDARD": 1}

    def test_merge_results_duplicate_not_in_deduped_map(self):
        uw = UltraworkMode()
        original = ["task A", "task B"]
        deduped, subtask_map = uw._deduplicate_subtasks(original)
        result = DelegationResult(
            goal="task A", completed=True, summary="done", agent_result=None
        )
        res = uw._merge_results([result], original, subtask_map)
        assert "DEDUPLICATED" in res.summary
        assert res.completed_subtasks == 1

    def test_merge_results_failed_subtasks_in_summary(self):
        uw = UltraworkMode()
        original = ["task A", "task B"]
        deduped, subtask_map = uw._deduplicate_subtasks(original)
        result_a = DelegationResult(
            goal="task A", completed=True, summary="done", agent_result=None
        )
        result_b = DelegationResult(
            goal="task B", completed=False, summary="fail", agent_result=None
        )
        res = uw._merge_results([result_a, result_b], original, subtask_map)
        assert res.failed_subtasks == 1
        assert "1 failed" in res.summary
        assert res.completed is False

    def test_router_same_provider_no_override(self):
        router = MagicMock()
        decision = MagicMock()
        decision.tier = MagicMock(value="STANDARD")
        decision.provider = "local"
        router.route_delegation.return_value = decision
        uw = UltraworkMode(router=router)
        parent_agent = MagicMock()
        parent_agent.tier_used = "STANDARD"
        parent_agent.provider.name = "local"
        with patch("mullet.core.router.resolver.ProviderResolver") as MockResolver:
            with patch("mullet.core.orchestration.ultrawork.delegate_batch", return_value=[]) as mock_batch:
                uw.execute("task", parent_agent, subtasks=["st1"])
        MockResolver.assert_not_called()
        call_kwargs = mock_batch.call_args.kwargs
        assert call_kwargs.get("provider_override") is None

    def test_merge_results_all_completed_overall_status(self):
        uw = UltraworkMode()
        original = ["task A", "task B"]
        deduped, subtask_map = uw._deduplicate_subtasks(original)
        result_a = DelegationResult(
            goal="task A", completed=True, summary="done", agent_result=None
        )
        result_b = DelegationResult(
            goal="task B", completed=True, summary="done", agent_result=None
        )
        res = uw._merge_results([result_a, result_b], original, subtask_map)
        assert "All subtasks completed successfully" in res.summary
        assert res.completed is True
        assert res.failed_subtasks == 0

    def test_execute_with_subtasks_skips_decompose(self):
        uw = UltraworkMode()
        parent_agent = MagicMock()
        parent_agent.provider.name = "local"
        with patch.object(uw, "_decompose") as mock_decomp:
            with patch("mullet.core.orchestration.ultrawork.delegate_batch", return_value=[]):
                uw.execute("task", parent_agent, subtasks=["st1"])
        mock_decomp.assert_not_called()

    def test_merge_results_partial_completed_status(self):
        uw = UltraworkMode()
        original = ["task A", "task B"]
        deduped, subtask_map = uw._deduplicate_subtasks(original)
        result_a = DelegationResult(
            goal="task A", completed=True, summary="done", agent_result=None
        )
        res = uw._merge_results([result_a], original, subtask_map)
        assert "1 out of 2 subtasks completed" in res.summary
        assert res.completed is True  # failed_subtasks == 0

    def test_decompose_empty_lines_ignored(self):
        uw = UltraworkMode(max_workers=5)
        agent = MagicMock()
        result_mock = MagicMock()
        result_mock.completed = True
        result_mock.final_response = "\n\n1. Task one\n\n2. Task two\n\n"
        agent.run.return_value = result_mock
        subtasks = uw._decompose("task", agent)
        assert subtasks == ["Task one", "Task two"]

    def test_merge_results_no_results_all_pending(self):
        uw = UltraworkMode()
        original = ["task A", "task B"]
        deduped, subtask_map = uw._deduplicate_subtasks(original)
        res = uw._merge_results([], original, subtask_map)
        assert res.completed_subtasks == 0
        assert res.failed_subtasks == 0
        assert "0 out of 2 subtasks completed" in res.summary
        assert res.completed is True

