"""End-to-end swarm pipeline tests — no real LLM calls."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from mullet.core.orchestration.swarm import run_swarm
from mullet.core.orchestration.swarm_types import SwarmContext, SwarmResult, TaskResult
from mullet.core.orchestration.static_decomposer import StaticDecomposer
from mullet.core.orchestration.swarm_dispatcher import SwarmDispatcher
from mullet.core.orchestration.synthesizer import Synthesizer
from mullet.core.agent.types import TokenUsage


@pytest.mark.asyncio
async def test_static_decomposer_to_synthesizer_pipeline():
    """Test StaticDecomposer → SwarmDispatcher (mocked) → Synthesizer (mocked) → SwarmResult."""
    # Setup
    files = ["file1.py", "file2.py", "file3.py"]
    context = SwarmContext(files=files)
    decomposer = StaticDecomposer()
    
    # Decompose
    tasks = decomposer.decompose("analyze code", context)
    assert len(tasks) == 3
    
    # Mock dispatcher
    mock_results = [
        TaskResult(
            task_id="task-000",
            success=True,
            response="Analysis of file1.py",
            error=None,
            tokens=TokenUsage(input_tokens=10, output_tokens=20)
        ),
        TaskResult(
            task_id="task-001",
            success=True,
            response="Analysis of file2.py",
            error=None,
            tokens=TokenUsage(input_tokens=15, output_tokens=25)
        ),
        TaskResult(
            task_id="task-002",
            success=True,
            response="Analysis of file3.py",
            error=None,
            tokens=TokenUsage(input_tokens=12, output_tokens=22)
        )
    ]
    
    with patch.object(SwarmDispatcher, 'dispatch', return_value=mock_results):
        dispatcher = SwarmDispatcher()
        results = await dispatcher.dispatch(tasks)
    
    # Mock synthesizer
    mock_synthesis = "# Synthesized Report\n\nAll files analyzed successfully."
    mock_syn_usage = TokenUsage(input_tokens=100, output_tokens=200)
    
    with patch.object(Synthesizer, 'synthesize', return_value=(mock_synthesis, mock_syn_usage)):
        provider = AsyncMock()
        synthesizer = Synthesizer(provider)
        synthesis, usage = await synthesizer.synthesize("analyze code", results)
    
    # Verify
    assert synthesis == mock_synthesis
    assert usage == mock_syn_usage
    assert len(results) == 3
    assert all(r.success for r in results)


@pytest.mark.asyncio
async def test_run_swarm_full_pipeline():
    """Test run_swarm() full pipeline with StaticDecomposer, mocked dispatcher and synthesizer."""
    context = SwarmContext(files=["test.py", "utils.py"])
    
    # Mock dispatcher results
    mock_results = [
        TaskResult(
            task_id="task-000",
            success=True,
            response="Test file analysis",
            error=None,
            tokens=TokenUsage(input_tokens=50, output_tokens=100)
        ),
        TaskResult(
            task_id="task-001",
            success=True,
            response="Utils file analysis",
            error=None,
            tokens=TokenUsage(input_tokens=75, output_tokens=125)
        )
    ]
    
    # Mock synthesis
    mock_synthesis = "# Analysis Report\n\nBoth files processed successfully."
    mock_syn_usage = TokenUsage(input_tokens=200, output_tokens=300)
    
    mock_kimi = MagicMock()
    with patch("mullet.core.providers.kimi.KimiProvider", return_value=mock_kimi), \
         patch("mullet.core.providers.kimi_async.KimiAsyncProvider", return_value=mock_kimi), \
         patch.object(SwarmDispatcher, 'dispatch', return_value=mock_results), \
         patch.object(Synthesizer, 'synthesize', return_value=(mock_synthesis, mock_syn_usage)):
        result = await run_swarm("analyze codebase", context)

    assert isinstance(result, SwarmResult)
    assert result.goal == "analyze codebase"
    assert len(result.tasks) == 2
    assert len(result.results) == 2
    assert result.synthesis == mock_synthesis
    assert result.total_tokens.input_tokens == 325  # 50+75+200
    assert result.total_tokens.output_tokens == 525  # 100+125+300
    assert not result.partial_failure
    assert not result.synthesis_fallback


@pytest.mark.asyncio
async def test_run_swarm_dry_run():
    """Test run_swarm() dry_run=True: decompose only, no dispatch."""
    context = SwarmContext(files=["file1.py", "file2.py"])
    
    result = await run_swarm("analyze code", context, dry_run=True)
    
    assert isinstance(result, SwarmResult)
    assert result.goal == "analyze code"
    assert len(result.tasks) == 2
    assert result.tasks[0].instruction.startswith("analyze code")
    assert "file1.py" in result.tasks[0].instruction
    assert result.tasks[1].instruction.startswith("analyze code")
    assert "file2.py" in result.tasks[1].instruction
    assert result.results == []
    assert result.synthesis == ""
    assert result.total_tokens.input_tokens == 0
    assert result.total_tokens.output_tokens == 0


@pytest.mark.asyncio
async def test_swarm_dispatcher_handles_task_failure():
    """Test SwarmDispatcher handles task-level failure gracefully."""
    context = SwarmContext(files=["good.py", "bad.py", "ok.py"])
    
    # Mock results with one failure
    mock_results = [
        TaskResult(
            task_id="task-000",
            success=True,
            response="Good file processed",
            error=None,
            tokens=TokenUsage(input_tokens=10, output_tokens=20)
        ),
        TaskResult(
            task_id="task-001",
            success=False,
            response="",
            error="Syntax error in file",
            tokens=TokenUsage(input_tokens=5, output_tokens=0)
        ),
        TaskResult(
            task_id="task-002",
            success=True,
            response="OK file processed",
            error=None,
            tokens=TokenUsage(input_tokens=15, output_tokens=25)
        )
    ]
    
    # Mock synthesis
    mock_synthesis = "# Analysis Report\n\n2/3 files processed successfully."
    mock_syn_usage = TokenUsage(input_tokens=150, output_tokens=250)
    
    mock_kimi = MagicMock()
    with patch("mullet.core.providers.kimi.KimiProvider", return_value=mock_kimi), \
         patch("mullet.core.providers.kimi_async.KimiAsyncProvider", return_value=mock_kimi), \
         patch.object(SwarmDispatcher, 'dispatch', return_value=mock_results), \
         patch.object(Synthesizer, 'synthesize', return_value=(mock_synthesis, mock_syn_usage)):
        result = await run_swarm("process files", context)

    assert isinstance(result, SwarmResult)
    assert len(result.tasks) == 3
    assert len(result.results) == 3
    assert result.partial_failure is True
    assert result.results[1].success is False
    assert result.results[1].error == "Syntax error in file"
    assert result.synthesis == mock_synthesis
    assert result.total_tokens.input_tokens == 180  # 10+5+15+150
    assert result.total_tokens.output_tokens == 295  # 20+0+25+250


@pytest.mark.asyncio
async def test_swarm_retries_transient_failures():
    """Cross-task retry: timeout/rate_limit tasks are re-dispatched at CRITICAL tier."""
    context = SwarmContext(files=["a.py", "b.py"])

    first_call = True

    async def mock_dispatch(tasks, on_progress=None):
        nonlocal first_call
        if first_call:
            first_call = False
            return [
                TaskResult(task_id="task-000", success=True, response="ok", error=None,
                           tokens=TokenUsage(input_tokens=10, output_tokens=10)),
                TaskResult(task_id="task-001", success=False, response="", error="timeout",
                           tokens=TokenUsage()),
            ]
        # Retry pass — task-001 escalated to CRITICAL, now succeeds
        assert tasks[0].id == "task-001"
        assert tasks[0].tier == "CRITICAL"
        return [
            TaskResult(task_id="task-001", success=True, response="ok on retry", error=None,
                       tokens=TokenUsage(input_tokens=20, output_tokens=20)),
        ]

    mock_synthesis = "All good."
    mock_syn_usage = TokenUsage(input_tokens=50, output_tokens=50)
    mock_kimi = MagicMock()

    with patch("mullet.core.providers.kimi.KimiProvider", return_value=mock_kimi), \
         patch("mullet.core.providers.kimi_async.KimiAsyncProvider", return_value=mock_kimi), \
         patch.object(SwarmDispatcher, 'dispatch', side_effect=mock_dispatch), \
         patch.object(Synthesizer, 'synthesize', return_value=(mock_synthesis, mock_syn_usage)):
        result = await run_swarm("analyze code", context)

    assert not result.partial_failure
    assert all(r.success for r in result.results)
    assert result.results[1].response == "ok on retry"


@pytest.mark.live
@pytest.mark.asyncio
async def test_run_swarm_with_real_api():
    """Integration test with the real Kimi API. Marked live: Mullet loads .env,
    so a key is nearly always present and an unmarked run spends real money."""
    import os
    if not (os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY")):
        pytest.skip("Requires KIMI_API_KEY or MOONSHOT_API_KEY environment variable")
    
    context = SwarmContext(files=["README.md"])
    result = await run_swarm("summarize README", context)
    assert isinstance(result, SwarmResult)
    assert len(result.tasks) == 1
