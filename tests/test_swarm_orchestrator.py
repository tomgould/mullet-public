"""Tests for mullet.core.orchestration.swarm module."""

import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from mullet.core.orchestration.swarm import (
    _estimate_cost,
    _retry_failed_tasks,
    _sum_tokens,
    run_swarm,
)
from mullet.core.orchestration.swarm_types import (
    SwarmContext,
    SwarmResult,
    SwarmTask,
    TaskResult,
)
from mullet.core.agent.types import TokenUsage


class TestEmptyTasksPath:
    """Test 1: Empty tasks path - mock decomposer returns [] tasks."""

    def test_empty_tasks_returns_swarm_result(self):
        """Test that empty tasks list returns valid SwarmResult."""
        # Arrange
        mock_decomposer = MagicMock()
        mock_decomposer.decompose.return_value = []
        
        ctx = SwarmContext()
        
        # Act
        result = asyncio.run(run_swarm("test goal", ctx, decomposer=mock_decomposer))
        
        # Assert
        assert isinstance(result, SwarmResult)
        assert result.tasks == []
        assert result.goal == "test goal"


class TestDryRunReturnsEstimatedTokens:
    """Test 2: dry_run returns estimated tokens and cost."""

    def test_dry_run_returns_estimates(self):
        """Test that dry_run=True returns estimated tokens and cost."""
        # Arrange
        mock_decomposer = MagicMock()
        task1 = SwarmTask(id="1", instruction="short instruction")
        task2 = SwarmTask(id="2", instruction="another instruction that is longer")
        mock_decomposer.decompose.return_value = [task1, task2]
        
        ctx = SwarmContext()
        
        # Act
        result = asyncio.run(run_swarm("test goal", ctx, decomposer=mock_decomposer, dry_run=True))
        
        # Assert
        assert isinstance(result, SwarmResult)
        assert result.estimated_input_tokens > 0
        assert result.estimated_cost_usd >= 0.0
        assert len(result.tasks) == 2


class TestEstimateCost:
    """Test 3 & 4: _estimate_cost function tests."""

    def test_estimate_cost_with_tasks(self):
        """Test _estimate_cost with known instruction lengths."""
        # Arrange
        task1 = SwarmTask(id="1", instruction="a" * 100)  # 100 chars
        task2 = SwarmTask(id="2", instruction="b" * 200)  # 200 chars
        tasks = [task1, task2]
        
        # Act
        tokens, cost = _estimate_cost(tasks)
        
        # Assert
        assert tokens > 0
        assert cost >= 0.0
        # Should have tokens for 2 tasks plus overhead
        assert tokens >= 400  # (300 chars / 4 * 1.3) + 400 overhead

    def test_estimate_cost_empty_list(self):
        """Test _estimate_cost with empty list returns zero."""
        # Act
        tokens, cost = _estimate_cost([])
        
        # Assert
        assert tokens == 0
        assert cost == 0.0


class TestRetryFailedTasks:
    """Test 5: retry failed tasks - no retryable failures."""

    @pytest.mark.asyncio
    async def test_no_retryable_failures_returns_same_results(self):
        """Test that _retry_failed_tasks returns same results when no failures."""
        # Arrange
        task1 = SwarmTask(id="1", instruction="test1")
        task2 = SwarmTask(id="2", instruction="test2")
        tasks = [task1, task2]
        
        result1 = TaskResult(
            task_id="1",
            success=True,
            response="success",
            error=None,
            tokens=TokenUsage(input_tokens=10, output_tokens=5)
        )
        result2 = TaskResult(
            task_id="2",
            success=True,
            response="success",
            error=None,
            tokens=TokenUsage(input_tokens=15, output_tokens=7)
        )
        results = [result1, result2]
        
        # Act
        retry_results = await _retry_failed_tasks(tasks, results, concurrency=5)
        
        # Assert
        assert retry_results == results
        assert len(retry_results) == 2


class TestSumTokens:
    """Test 6: _sum_tokens sums correctly."""

    def test_sum_tokens_with_known_values(self):
        """Test _sum_tokens calculates correct totals."""
        # Arrange
        result1 = TaskResult(
            task_id="1",
            success=True,
            response="test1",
            error=None,
            tokens=TokenUsage(input_tokens=10, output_tokens=5)
        )
        result2 = TaskResult(
            task_id="2",
            success=True,
            response="test2",
            error=None,
            tokens=TokenUsage(input_tokens=20, output_tokens=15)
        )
        result3 = TaskResult(
            task_id="3",
            success=True,
            response="test3",
            error=None,
            tokens=TokenUsage(input_tokens=30, output_tokens=25)
        )
        results = [result1, result2, result3]
        
        # Act
        total = _sum_tokens(results)
        
        # Assert
        assert total.input_tokens == 60  # 10 + 20 + 30
        assert total.output_tokens == 45  # 5 + 15 + 25

    def test_sum_tokens_empty_list(self):
        """Test _sum_tokens with empty list returns zero tokens."""
        # Act
        total = _sum_tokens([])
        
        # Assert
        assert total.input_tokens == 0
        assert total.output_tokens == 0
        assert total.total_tokens == 0


class TestRunSwarmWithoutAPIKey:
    """Test 7: run_swarm uses StaticDecomposer when no KIMI_API_KEY."""

    @patch.dict('os.environ', {'KIMI_API_KEY': '', 'MOONSHOT_API_KEY': ''})
    @pytest.mark.asyncio
    async def test_uses_static_decomposer_without_api_key(self):
        """Test that StaticDecomposer is used when no API keys are available."""
        # Arrange
        ctx = SwarmContext()
        
        # Mock the dependencies
        with patch('mullet.core.orchestration.swarm.SwarmDispatcher') as mock_dispatcher, \
             patch('mullet.core.orchestration.synthesizer.Synthesizer') as mock_synthesizer, \
             patch('mullet.core.orchestration.static_decomposer.StaticDecomposer') as mock_static_decomposer:
            
            # Configure mocks
            mock_dispatcher_instance = AsyncMock()
            mock_dispatcher_instance.dispatch.return_value = []
            mock_dispatcher.return_value = mock_dispatcher_instance
            
            mock_synthesizer_instance = AsyncMock()
            mock_synthesizer_instance.synthesize.return_value = ("", TokenUsage())
            mock_synthesizer.return_value = mock_synthesizer_instance
            
            mock_decomposer = MagicMock()
            mock_decomposer.decompose.return_value = []
            mock_static_decomposer.return_value = mock_decomposer
            
            # Act
            result = await run_swarm("test goal", ctx, concurrency=1)
            
            # Assert
            assert isinstance(result, SwarmResult)
            assert result.goal == "test goal"
            mock_static_decomposer.assert_called_once()

    def test_static_decomposer_used_with_none_decomposer_param(self):
        """Test that StaticDecomposer is used when decomposer=None and no API key."""
        # Arrange
        ctx = SwarmContext()
        
        with patch.dict('os.environ', {'KIMI_API_KEY': '', 'MOONSHOT_API_KEY': ''}):
            with patch('mullet.core.orchestration.static_decomposer.StaticDecomposer') as mock_static, \
                 patch('mullet.core.orchestration.swarm.SwarmDispatcher') as mock_dispatcher:
                
                mock_decomposer = MagicMock()
                mock_decomposer.decompose.return_value = []
                mock_static.return_value = mock_decomposer
                
                mock_dispatcher_instance = AsyncMock()
                mock_dispatcher_instance.dispatch.return_value = []
                mock_dispatcher.return_value = mock_dispatcher_instance

                # Act & Assert - should not raise an exception
                result = asyncio.run(run_swarm("goal", ctx, decomposer=None))
                assert isinstance(result, SwarmResult)


class TestAllowedProviders:
    """Cluster 2: allowed_providers threading through run_swarm."""

    def test_run_swarm_with_kimi_allowlist(self):
        """run_swarm accepts allowed_providers when 'kimi' is included (dry_run)."""
        stub = MagicMock()
        stub.decompose.return_value = [SwarmTask(id="t1", instruction="do one thing")]
        ctx = SwarmContext(files=[], output_shape="markdown", concurrency=1)
        result = asyncio.run(run_swarm(
            "goal",
            ctx,
            decomposer=stub,
            dry_run=True,
            allowed_providers=frozenset({"kimi"}),
        ))
        assert isinstance(result, SwarmResult)
        assert result.goal == "goal"

    def test_run_swarm_with_haiku_only_allowlist_accepted(self):
        """Phase C2: haiku is now swarm-capable — run_swarm with haiku-only no longer raises.
        (Reconciled from test_run_swarm_with_haiku_only_allowlist_rejects: the old guard
        required kimi; after C2 any SWARM_CAPABLE_PROVIDERS member is accepted.)
        Tasks will fail at the LLM call (no real key), but SwarmDispatcher itself constructs OK.
        """
        stub = MagicMock()
        stub.decompose.return_value = [
            SwarmTask(id="t1", instruction="do thing one"),
            SwarmTask(id="t2", instruction="do thing two"),
        ]
        ctx = SwarmContext(files=[], output_shape="markdown", concurrency=2)
        # Should not raise ValueError — haiku is now swarm-capable
        result = asyncio.run(run_swarm(
            "goal",
            ctx,
            decomposer=stub,
            allowed_providers=frozenset({"haiku"}),
        ))
        # Tasks may fail (no real API key in test env), but swarm completes
        assert result is not None

    def test_select_decomposer_falls_back_to_static_when_llm_providers_disallowed(self):
        """_select_decomposer returns StaticDecomposer when no supported LLM provider is allowed."""
        from mullet.core.orchestration.swarm import _select_decomposer
        from mullet.core.orchestration.static_decomposer import StaticDecomposer

        d = _select_decomposer(frozenset({"gemini"}))
        assert isinstance(d, StaticDecomposer)


class TestSelectSynthesizerProvider:
    """G19: _select_synthesizer_provider respects the allowlist."""

    def _clear_keys(self, monkeypatch):
        for k in ("KIMI_API_KEY", "MOONSHOT_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(k, raising=False)

    def test_none_allowlist_with_kimi_key_returns_kimi(self, monkeypatch):
        from mullet.core.orchestration.swarm import _select_synthesizer_provider
        self._clear_keys(monkeypatch)
        monkeypatch.setenv("KIMI_API_KEY", "test-key")
        p = _select_synthesizer_provider(None)
        assert p is not None
        assert p._sync.__class__.__name__ == "KimiProvider"

    def test_none_allowlist_no_keys_returns_none(self, monkeypatch):
        """allowed_providers=None with no Kimi API key → None (caller falls back).

        Legacy behaviour constructed KimiProvider anyway; G19 swallows the
        ValueError and returns None so run_swarm falls back to raw
        concatenation instead of crashing. Integration tests that patch
        KimiProvider still see the wrapper because the try block succeeds.
        """
        from mullet.core.orchestration.swarm import _select_synthesizer_provider
        self._clear_keys(monkeypatch)
        assert _select_synthesizer_provider(None) is None

    def test_empty_frozenset_returns_none(self, monkeypatch):
        from mullet.core.orchestration.swarm import _select_synthesizer_provider
        self._clear_keys(monkeypatch)
        monkeypatch.setenv("KIMI_API_KEY", "test-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        assert _select_synthesizer_provider(frozenset()) is None

    def test_local_only_returns_none(self, monkeypatch):
        from mullet.core.orchestration.swarm import _select_synthesizer_provider
        self._clear_keys(monkeypatch)
        monkeypatch.setenv("KIMI_API_KEY", "test-key")
        assert _select_synthesizer_provider(frozenset({"local"})) is None

    def test_haiku_allowlist_with_anthropic_key(self, monkeypatch):
        from mullet.core.orchestration.swarm import _select_synthesizer_provider
        self._clear_keys(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        p = _select_synthesizer_provider(frozenset({"haiku"}))
        assert p is not None
        assert p._sync.__class__.__name__ == "AnthropicProvider"
        assert p._sync.model == "claude-haiku-4-5"

    def test_sonnet_allowlist_with_anthropic_key(self, monkeypatch):
        from mullet.core.orchestration.swarm import _select_synthesizer_provider
        self._clear_keys(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        p = _select_synthesizer_provider(frozenset({"sonnet"}))
        assert p is not None
        assert p._sync.__class__.__name__ == "AnthropicProvider"
        assert p._sync.model == "claude-sonnet-4-6"

    def test_kimi_preferred_over_haiku_when_both_allowed(self, monkeypatch):
        from mullet.core.orchestration.swarm import _select_synthesizer_provider
        self._clear_keys(monkeypatch)
        monkeypatch.setenv("KIMI_API_KEY", "test-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        p = _select_synthesizer_provider(frozenset({"kimi", "haiku"}))
        assert p is not None
        assert p._sync.__class__.__name__ == "KimiProvider"

    def test_haiku_excluded_falls_to_sonnet(self, monkeypatch):
        """kimi excluded, haiku excluded, only sonnet — returns sonnet."""
        from mullet.core.orchestration.swarm import _select_synthesizer_provider
        self._clear_keys(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        p = _select_synthesizer_provider(frozenset({"sonnet"}))
        assert p is not None
        assert p._sync.model == "claude-sonnet-4-6"