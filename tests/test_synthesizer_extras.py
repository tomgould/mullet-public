"""Additional tests for mullet.core.orchestration.synthesizer formatters + fallback."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from mullet.core.agent.types import TokenUsage
from mullet.core.orchestration.swarm_types import TaskResult
from mullet.core.orchestration.synthesizer import (
    Synthesizer,
    _format_failures,
    _format_raw_results,
    _format_results,
)


def _task(task_id: str, response: str = "", success: bool = True, error: str | None = None) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        success=success,
        response=response,
        error=error,
    )


# --- _format_results -------------------------------------------------------


def test_format_results_empty_returns_empty():
    assert _format_results([]) == ""


def test_format_results_single_task():
    results = [_task("t1", "first response")]
    out = _format_results(results)
    assert "Result 1" in out
    assert "Task t1" in out
    assert "first response" in out


def test_format_results_multiple_numbered():
    results = [_task("a", "one"), _task("b", "two"), _task("c", "three")]
    out = _format_results(results)
    assert "Result 1" in out
    assert "Result 2" in out
    assert "Result 3" in out
    # Task IDs present
    assert "Task a" in out
    assert "Task b" in out
    assert "Task c" in out


# --- _format_failures ------------------------------------------------------


def test_format_failures_empty_returns_empty_string():
    assert _format_failures([]) == ""


def test_format_failures_single():
    failures = [_task("fail1", "", success=False, error="api error")]
    out = _format_failures(failures)
    assert "Failures" in out
    assert "fail1" in out
    assert "api error" in out


def test_format_failures_multiple():
    failures = [
        _task("f1", "", success=False, error="timeout"),
        _task("f2", "", success=False, error="rate limit"),
    ]
    out = _format_failures(failures)
    assert "f1" in out
    assert "f2" in out
    assert "timeout" in out
    assert "rate limit" in out


# --- _format_raw_results ---------------------------------------------------


def test_format_raw_results_empty():
    assert _format_raw_results([]) == ""


def test_format_raw_results_mixed_success_failure():
    results = [
        _task("ok", "works"),
        _task("bad", "", success=False, error="boom"),
    ]
    out = _format_raw_results(results)
    assert "Task ok (SUCCESS)" in out
    assert "works" in out
    assert "Task bad (FAILED)" in out
    assert "boom" in out
    assert "Error: boom" in out


def test_format_raw_results_all_success_no_error_line():
    """Successful tasks should not include 'Error: ...' line."""
    results = [_task("a", "response")]
    out = _format_raw_results(results)
    assert "Error:" not in out


# --- Synthesizer.synthesize -----------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_empty_results_returns_empty():
    provider = MagicMock()
    syn = Synthesizer(provider=provider)
    text, tokens = await syn.synthesize(goal="g", results=[])
    assert text == ""
    assert tokens.total_tokens == 0
    provider.generate.assert_not_called()


@pytest.mark.asyncio
async def test_synthesize_no_successful_results_returns_empty():
    provider = MagicMock()
    syn = Synthesizer(provider=provider)
    results = [_task("f", "", success=False, error="err")]
    text, tokens = await syn.synthesize(goal="g", results=results)
    assert text == ""
    assert tokens.total_tokens == 0
    provider.generate.assert_not_called()


@pytest.mark.asyncio
async def test_synthesize_success_calls_provider_and_returns_text():
    provider = MagicMock()
    fake_response = MagicMock()
    fake_response.text = "synthesized output"
    fake_response.usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
    provider.generate = AsyncMock(return_value=fake_response)

    syn = Synthesizer(provider=provider)
    results = [_task("a", "one"), _task("b", "two")]
    text, tokens = await syn.synthesize(goal="combine", results=results)

    assert text == "synthesized output"
    assert tokens.total_tokens == 150
    provider.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_synthesize_on_provider_error_falls_back_to_raw():
    """Provider failure → fallback returns raw concatenation with empty TokenUsage."""
    provider = MagicMock()
    provider.generate = AsyncMock(side_effect=RuntimeError("LLM down"))

    syn = Synthesizer(provider=provider)
    results = [_task("a", "content-a")]
    text, tokens = await syn.synthesize(goal="g", results=results)

    assert "content-a" in text
    assert "Task a" in text
    assert tokens.total_tokens == 0  # no successful synthesis → no tokens


@pytest.mark.asyncio
async def test_synthesize_passes_output_shape_in_prompt():
    """output_shape argument should appear in the prompt to the LLM."""
    captured_prompt = {}

    async def fake_generate(prompt, max_tokens):
        captured_prompt["p"] = prompt
        resp = MagicMock()
        resp.text = "ok"
        resp.usage = TokenUsage()
        return resp

    provider = MagicMock()
    provider.generate = fake_generate

    syn = Synthesizer(provider=provider)
    await syn.synthesize(
        goal="g", results=[_task("a", "x")], output_shape="JSON",
    )
    assert "JSON" in captured_prompt["p"]
