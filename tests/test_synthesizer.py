"""Tests for the synthesizer module."""

import pytest
from unittest.mock import AsyncMock

from mullet.core.agent.types import TokenUsage
from mullet.core.orchestration.synthesizer import Synthesizer, _format_raw_results
from mullet.core.orchestration.swarm_types import TaskResult
from mullet.core.providers.kimi_async import KimiAsyncProvider


@pytest.mark.asyncio
async def test_synthesize_all_success():
    """Test synthesis with all successful results."""
    provider = AsyncMock(spec=KimiAsyncProvider)
    provider.generate.return_value.text = "# Synthesized Report\n\nAll tasks completed successfully."
    provider.generate.return_value.usage = TokenUsage(input_tokens=50, output_tokens=100)
    
    synthesizer = Synthesizer(provider)
    
    results = [
        TaskResult(task_id="1", success=True, response="Result 1 content", error=None),
        TaskResult(task_id="2", success=True, response="Result 2 content", error=None),
        TaskResult(task_id="3", success=True, response="Result 3 content", error=None),
    ]
    
    text, usage = await synthesizer.synthesize("Test goal", results)
    
    assert text == "# Synthesized Report\n\nAll tasks completed successfully."
    assert usage.input_tokens == 50
    assert usage.output_tokens == 100
    provider.generate.assert_called_once()


@pytest.mark.asyncio
async def test_synthesize_partial_failure():
    """Test synthesis with some failures."""
    provider = AsyncMock(spec=KimiAsyncProvider)
    provider.generate.return_value.text = "# Mixed Results\n\nSome tasks failed."
    provider.generate.return_value.usage = TokenUsage(input_tokens=75, output_tokens=150)
    
    synthesizer = Synthesizer(provider)
    
    results = [
        TaskResult(task_id="1", success=True, response="Success 1", error=None),
        TaskResult(task_id="2", success=True, response="Success 2", error=None),
        TaskResult(task_id="3", success=False, response="", error="Task 3 failed"),
    ]
    
    text, usage = await synthesizer.synthesize("Test goal", results)
    
    assert text == "# Mixed Results\n\nSome tasks failed."
    assert usage.input_tokens == 75
    
    # Verify provider was called with prompt containing failure info
    call_args = provider.generate.call_args
    prompt = call_args[0][0]  # First positional argument
    assert "## Failures" in prompt
    assert "Task 3 failed" in prompt


@pytest.mark.asyncio
async def test_synthesize_all_failed_returns_empty():
    """Test that all failed results return empty string and no provider call."""
    provider = AsyncMock(spec=KimiAsyncProvider)
    
    synthesizer = Synthesizer(provider)
    
    results = [
        TaskResult(task_id="1", success=False, response="", error="Error 1"),
        TaskResult(task_id="2", success=False, response="", error="Error 2"),
    ]
    
    text, usage = await synthesizer.synthesize("Test goal", results)
    
    assert text == ""
    assert usage == TokenUsage()
    provider.generate.assert_not_called()


@pytest.mark.asyncio
async def test_synthesize_returns_token_usage():
    """Test that token usage from provider is correctly returned."""
    provider = AsyncMock(spec=KimiAsyncProvider)
    expected_usage = TokenUsage(
        input_tokens=100,
        output_tokens=200, 
        total_tokens=300,
        cache_read_tokens=10,
        cache_write_tokens=20
    )
    provider.generate.return_value.text = "Test synthesis"
    provider.generate.return_value.usage = expected_usage
    
    synthesizer = Synthesizer(provider)
    
    results = [TaskResult(task_id="1", success=True, response="Test result", error=None)]
    
    text, usage = await synthesizer.synthesize("Test goal", results)
    
    assert usage == expected_usage
    assert text == "Test synthesis"


@pytest.mark.asyncio
async def test_synthesize_provider_failure_returns_raw_fallback():
    """Test that provider failures return raw concatenation without re-raising."""
    provider = AsyncMock(spec=KimiAsyncProvider)
    provider.generate.side_effect = Exception("Provider error")
    
    synthesizer = Synthesizer(provider)
    
    results = [
        TaskResult(task_id="1", success=True, response="Success 1", error=None),
        TaskResult(task_id="2", success=True, response="Success 2", error=None),
    ]
    
    text, usage = await synthesizer.synthesize("Test goal", results)
    
    # Should return raw concatenation, not raise exception
    assert "=== Task 1 (SUCCESS) ===" in text
    assert "Success 1" in text
    assert "=== Task 2 (SUCCESS) ===" in text
    assert "Success 2" in text
    assert usage == TokenUsage()


@pytest.mark.asyncio
async def test_synthesize_prompt_contains_goal():
    """Test that the original goal is included in the prompt."""
    provider = AsyncMock(spec=KimiAsyncProvider)
    provider.generate.return_value.text = "Report"
    provider.generate.return_value.usage = TokenUsage()
    
    synthesizer = Synthesizer(provider)
    
    results = [TaskResult(task_id="1", success=True, response="Test", error=None)]
    goal = "Find and document all API endpoints"
    
    await synthesizer.synthesize(goal, results)
    
    call_args = provider.generate.call_args
    prompt = call_args[0][0]
    assert goal in prompt


@pytest.mark.asyncio
async def test_synthesize_prompt_contains_output_shape():
    """Test that output_shape is included in the prompt."""
    provider = AsyncMock(spec=KimiAsyncProvider)
    provider.generate.return_value.text = "Report"
    provider.generate.return_value.usage = TokenUsage()
    
    synthesizer = Synthesizer(provider)
    
    results = [TaskResult(task_id="1", success=True, response="Test", error=None)]
    output_shape = "structured JSON report"
    
    await synthesizer.synthesize("Test goal", results, output_shape=output_shape)
    
    call_args = provider.generate.call_args
    prompt = call_args[0][0]
    assert output_shape in prompt


def test_format_raw_results_includes_all_responses():
    """Test that _format_raw_results includes all responses with proper headers."""
    results = [
        TaskResult(task_id="1", success=True, response="Success content", error=None),
        TaskResult(task_id="2", success=False, response="Partial content", error="Failure reason"),
    ]
    
    formatted = _format_raw_results(results)
    
    assert "=== Task 1 (SUCCESS) ===" in formatted
    assert "Success content" in formatted
    assert "=== Task 2 (FAILED) ===" in formatted
    assert "Partial content" in formatted
    assert "Error: Failure reason" in formatted