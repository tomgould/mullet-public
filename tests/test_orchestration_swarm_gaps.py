"""Coverage-gap tests for mullet.core.orchestration.swarm."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mullet.core.agent.types import TokenUsage
from mullet.core.orchestration.swarm import (
    _estimate_cost,
    _select_decomposer,
    _select_synthesizer_provider,
    _write_report,
    run_swarm,
)
from mullet.core.orchestration.swarm_types import (
    SwarmContext,
    SwarmResult,
    SwarmTask,
    TaskResult,
)


def _clear_keys(monkeypatch):
    for k in ("KIMI_API_KEY", "MOONSHOT_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)


class TestEstimateCostRegistrySuccess:
    """Cover _estimate_cost lines 33-34 (get_registry/get_model_costs success)."""

    def test_estimate_cost_uses_registry_when_available(self, monkeypatch):
        mock_registry = MagicMock()
        mock_registry.get_best_model.return_value = "kimi-k2.5"
        monkeypatch.setattr(
            "mullet.core.providers.model_info.get_model_costs",
            lambda model_id: (0.0006, 0.003),
            raising=False,
        )
        monkeypatch.setattr(
            "mullet.core.providers.model_registry.get_registry",
            lambda: mock_registry,
        )
        task = SwarmTask(id="t1", instruction="hello world")
        tokens, cost = _estimate_cost([task])
        assert tokens > 0
        assert cost >= 0.0
        mock_registry.get_best_model.assert_called_once_with("kimi", "STANDARD")


class TestSelectSynthesizerProviderGaps:
    """Cover lines 146, 158."""

    def test_kimi_require_key_no_key_returns_none(self, monkeypatch):
        _clear_keys(monkeypatch)
        p = _select_synthesizer_provider(frozenset({"kimi"}))
        assert p is None

    def test_haiku_no_anthropic_key_returns_none(self, monkeypatch):
        _clear_keys(monkeypatch)
        p = _select_synthesizer_provider(frozenset({"haiku"}))
        assert p is None


class TestSelectDecomposerGaps:
    """Cover lines 198-200, 203-208, 216-223."""

    def test_kimi_decomposer_success(self, monkeypatch):
        _clear_keys(monkeypatch)
        monkeypatch.setenv("KIMI_API_KEY", "test-key")
        d = _select_decomposer(frozenset({"kimi"}))
        from mullet.core.orchestration.llm_decomposer import LLMDecomposer

        assert isinstance(d, LLMDecomposer)

    def test_haiku_decomposer_success(self, monkeypatch):
        _clear_keys(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        d = _select_decomposer(frozenset({"haiku"}))
        from mullet.core.orchestration.llm_decomposer import LLMDecomposer

        assert isinstance(d, LLMDecomposer)

    def test_sonnet_decomposer_success(self, monkeypatch):
        _clear_keys(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        d = _select_decomposer(frozenset({"sonnet"}))
        from mullet.core.orchestration.llm_decomposer import LLMDecomposer

        assert isinstance(d, LLMDecomposer)

    def test_kimi_fails_falls_through_to_haiku(self, monkeypatch):
        _clear_keys(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        d = _select_decomposer(frozenset({"kimi", "haiku"}))
        from mullet.core.orchestration.llm_decomposer import LLMDecomposer

        assert isinstance(d, LLMDecomposer)


class TestRetryFailedTasksGaps:
    """Cover lines 97-120 (_retry_failed_tasks actual retry path)."""

    @pytest.mark.asyncio
    async def test_retry_failed_tasks_with_retryable_errors(self):
        from mullet.core.orchestration.swarm import _retry_failed_tasks

        task1 = SwarmTask(id="1", instruction="test1")
        task2 = SwarmTask(id="2", instruction="test2")
        tasks = [task1, task2]

        result1 = TaskResult(
            task_id="1", success=True, response="ok", error=None, tokens=TokenUsage()
        )
        result2 = TaskResult(
            task_id="2", success=False, response="", error="timeout", tokens=TokenUsage()
        )
        results = [result1, result2]

        with patch("mullet.core.orchestration.swarm.SwarmDispatcher") as mock_dispatcher:
            mock_dispatcher_instance = AsyncMock()
            mock_dispatcher_instance.dispatch.return_value = [
                TaskResult(
                    task_id="2",
                    success=True,
                    response="retried ok",
                    error=None,
                    tokens=TokenUsage(),
                )
            ]
            mock_dispatcher.return_value = mock_dispatcher_instance

            retry_results = await _retry_failed_tasks(tasks, results, concurrency=5)

            assert len(retry_results) == 2
            assert retry_results[0] == result1
            assert retry_results[1].success is True
            assert retry_results[1].response == "retried ok"
            mock_dispatcher_instance.dispatch.assert_called_once()
            called_tasks = mock_dispatcher_instance.dispatch.call_args[0][0]
            assert called_tasks[0].tier == "CRITICAL"
            assert called_tasks[0].id == "2"


class TestSelectDecomposerSonnetNoKey:
    """Cover line 205 (_select_decomposer sonnet no-key path)."""

    def test_sonnet_decomposer_no_key_falls_to_static(self, monkeypatch):
        _clear_keys(monkeypatch)
        d = _select_decomposer(frozenset({"sonnet"}))
        from mullet.core.orchestration.static_decomposer import StaticDecomposer

        assert isinstance(d, StaticDecomposer)


class TestAutoStaticDecomposer:
    """Cover lines 290-291 (auto StaticDecomposer when files/items present)."""

    def test_auto_static_decomposer_when_files_present(self):
        ctx = SwarmContext(files=["file.txt"], output_shape="markdown", concurrency=1)

        with patch(
            "mullet.core.orchestration.static_decomposer.StaticDecomposer"
        ) as mock_static, patch(
            "mullet.core.orchestration.swarm.SwarmDispatcher"
        ) as mock_dispatcher:
            mock_decomposer = MagicMock()
            mock_decomposer.decompose.return_value = []
            mock_static.return_value = mock_decomposer

            mock_dispatcher_instance = AsyncMock()
            mock_dispatcher_instance.dispatch.return_value = []
            mock_dispatcher.return_value = mock_dispatcher_instance

            result = asyncio.run(run_swarm("goal", ctx, decomposer=None))
            mock_static.assert_called_once()
            assert isinstance(result, SwarmResult)

    def test_auto_static_decomposer_when_items_present(self):
        ctx = SwarmContext(items=["item1"], output_shape="markdown", concurrency=1)

        with patch(
            "mullet.core.orchestration.static_decomposer.StaticDecomposer"
        ) as mock_static, patch(
            "mullet.core.orchestration.swarm.SwarmDispatcher"
        ) as mock_dispatcher:
            mock_decomposer = MagicMock()
            mock_decomposer.decompose.return_value = []
            mock_static.return_value = mock_decomposer

            mock_dispatcher_instance = AsyncMock()
            mock_dispatcher_instance.dispatch.return_value = []
            mock_dispatcher.return_value = mock_dispatcher_instance

            result = asyncio.run(run_swarm("goal", ctx, decomposer=None))
            mock_static.assert_called_once()
            assert isinstance(result, SwarmResult)


class TestRunSwarmInnerGaps:
    """Cover lines 354-361, 373-377, 395, 398-400."""

    def test_no_synthesizer_fallback(self):
        stub = MagicMock()
        stub.decompose.return_value = [SwarmTask(id="t1", instruction="do one thing")]
        ctx = SwarmContext(files=[], output_shape="markdown", concurrency=1)

        with patch("mullet.core.orchestration.swarm.SwarmDispatcher") as mock_dispatcher, \
             patch("mullet.core.orchestration.swarm._select_synthesizer_provider", return_value=None):
            mock_dispatcher_instance = AsyncMock()
            mock_dispatcher_instance.dispatch.return_value = [
                TaskResult(task_id="t1", success=True, response="ok", error=None, tokens=TokenUsage())
            ]
            mock_dispatcher.return_value = mock_dispatcher_instance

            result = asyncio.run(run_swarm("goal", ctx, decomposer=stub))
            assert result.synthesis_fallback is True
            assert "ok" in result.synthesis

    def test_synthesizer_exception_fallback(self):
        stub = MagicMock()
        stub.decompose.return_value = [SwarmTask(id="t1", instruction="do one thing")]
        ctx = SwarmContext(files=[], output_shape="markdown", concurrency=1)

        with patch("mullet.core.orchestration.swarm.SwarmDispatcher") as mock_dispatcher, \
             patch("mullet.core.orchestration.swarm.Synthesizer") as mock_synth, \
             patch("mullet.core.orchestration.swarm._select_synthesizer_provider") as mock_select:
            mock_select.return_value = MagicMock()
            mock_dispatcher_instance = AsyncMock()
            mock_dispatcher_instance.dispatch.return_value = [
                TaskResult(task_id="t1", success=True, response="ok", error=None, tokens=TokenUsage())
            ]
            mock_dispatcher.return_value = mock_dispatcher_instance

            mock_synth_instance = AsyncMock()
            mock_synth_instance.synthesize.side_effect = ValueError("synth boom")
            mock_synth.return_value = mock_synth_instance

            result = asyncio.run(run_swarm("goal", ctx, decomposer=stub))
            assert result.synthesis_fallback is True
            assert "ok" in result.synthesis

    def test_dispatcher_exception_reraises(self):
        stub = MagicMock()
        stub.decompose.return_value = [SwarmTask(id="t1", instruction="do one thing")]
        ctx = SwarmContext(files=[], output_shape="markdown", concurrency=1)

        with patch("mullet.core.orchestration.swarm.SwarmDispatcher") as mock_dispatcher:
            mock_dispatcher_instance = AsyncMock()
            mock_dispatcher_instance.dispatch.side_effect = RuntimeError("dispatch boom")
            mock_dispatcher.return_value = mock_dispatcher_instance

            with pytest.raises(RuntimeError, match="dispatch boom"):
                asyncio.run(run_swarm("goal", ctx, decomposer=stub))

    def test_report_path_writes_file(self, tmp_path):
        stub = MagicMock()
        stub.decompose.return_value = [SwarmTask(id="t1", instruction="do one thing")]
        ctx = SwarmContext(files=[], output_shape="markdown", concurrency=1)
        report_path = tmp_path / "report.json"

        with patch("mullet.core.orchestration.swarm.SwarmDispatcher") as mock_dispatcher, \
             patch("mullet.core.orchestration.swarm.Synthesizer") as mock_synth, \
             patch("mullet.core.orchestration.swarm._select_synthesizer_provider") as mock_select:
            mock_select.return_value = MagicMock()
            mock_dispatcher_instance = AsyncMock()
            mock_dispatcher_instance.dispatch.return_value = [
                TaskResult(task_id="t1", success=True, response="ok", error=None, tokens=TokenUsage())
            ]
            mock_dispatcher.return_value = mock_dispatcher_instance

            mock_synth_instance = AsyncMock()
            mock_synth_instance.synthesize.return_value = ("synthesized", TokenUsage())
            mock_synth.return_value = mock_synth_instance

            result = asyncio.run(run_swarm("goal", ctx, decomposer=stub, report_path=report_path))
            assert report_path.exists()
            payload = json.loads(report_path.read_text())
            assert payload["goal"] == "goal"
            assert payload["task_count"] == 1
            assert payload["success_count"] == 1
            assert payload["synthesis"] == "synthesized"


class TestWriteReport:
    """Cover lines 404-438."""

    def test_write_report_creates_file(self, tmp_path):
        report_path = tmp_path / "nested" / "report.json"
        result = SwarmResult(
            goal="test goal",
            tasks=[SwarmTask(id="t1", instruction="hello")],
            results=[
                TaskResult(
                    task_id="t1",
                    success=True,
                    response="ok",
                    error=None,
                    tokens=TokenUsage(input_tokens=5, output_tokens=3),
                )
            ],
            synthesis="synth text",
            partial_failure=False,
            synthesis_fallback=False,
            total_tokens=TokenUsage(input_tokens=5, output_tokens=3),
            wall_clock_seconds=1.23,
            timestamp="2024-01-01T00:00:00+00:00",
        )
        _write_report(result, report_path)
        assert report_path.exists()
        payload = json.loads(report_path.read_text())
        assert payload["goal"] == "test goal"
        assert payload["wall_clock_seconds"] == 1.23
        assert payload["partial_failure"] is False
        assert payload["synthesis_fallback"] is False
        assert payload["total_tokens"]["input"] == 5
        assert payload["total_tokens"]["output"] == 3
        assert payload["task_count"] == 1
        assert payload["success_count"] == 1
        assert payload["tasks"][0]["id"] == "t1"
        assert payload["results"][0]["task_id"] == "t1"
        assert payload["results"][0]["success"] is True
