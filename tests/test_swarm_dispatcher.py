import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from mullet.core.agent.types import TokenUsage
from mullet.core.orchestration.swarm_dispatcher import SwarmDispatcher, RateLimitError
from mullet.core.orchestration.swarm_types import SwarmTask, TaskResult
from mullet.core.providers.kimi_async import AsyncLLMResponse


@pytest.fixture
def mock_provider():
    """Fixture for a mock KimiAsyncProvider."""
    provider = MagicMock()
    provider.generate = AsyncMock()
    return provider


@pytest.fixture
def create_task():
    """Helper to create SwarmTask instances."""
    def _create_task(task_id: str, instruction: str = "instruction") -> SwarmTask:
        return SwarmTask(id=task_id, instruction=instruction)
    return _create_task


@pytest.mark.asyncio
async def test_dispatch_single_task_success(mock_provider, create_task):
    """Test dispatch with a single successful task."""
    task = create_task("task-1")
    mock_provider.generate.return_value = AsyncLLMResponse(
        text="response-1", usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15), elapsed=0.1
    )

    dispatcher = SwarmDispatcher()
    dispatcher._make_provider = lambda tier, prompt_len, provider_name="kimi", prompt=None: mock_provider
    results = await dispatcher.dispatch([task])

    assert len(results) == 1
    result = results[0]
    assert result.task_id == "task-1"
    assert result.success is True
    assert result.response == "response-1"
    assert result.error is None
    assert result.tokens.total_tokens == 15
    assert result.elapsed_seconds > 0
    mock_provider.generate.assert_awaited_once_with(task.instruction)


@pytest.mark.asyncio
async def test_dispatch_multiple_tasks_all_succeed(mock_provider, create_task):
    """Test dispatch with multiple successful tasks."""
    tasks = [create_task("task-1"), create_task("task-2"), create_task("task-3")]
    mock_provider.generate.side_effect = [
        AsyncLLMResponse(text="res-1", usage=TokenUsage(total_tokens=10), elapsed=0.1),
        AsyncLLMResponse(text="res-2", usage=TokenUsage(total_tokens=20), elapsed=0.2),
        AsyncLLMResponse(text="res-3", usage=TokenUsage(total_tokens=30), elapsed=0.3),
    ]

    dispatcher = SwarmDispatcher(max_retries=0)
    dispatcher._make_provider = lambda tier, prompt_len, provider_name="kimi", prompt=None: mock_provider
    results = await dispatcher.dispatch(tasks)

    assert len(results) == 3
    for i, result in enumerate(results):
        assert result.task_id == f"task-{i+1}"
        assert result.success is True
        assert result.response == f"res-{i+1}"
        assert result.error is None
        assert result.tokens.total_tokens == (i + 1) * 10
        assert result.elapsed_seconds > 0
    assert mock_provider.generate.await_count == 3


@pytest.mark.asyncio
async def test_dispatch_partial_failure(mock_provider, create_task):
    """Test dispatch where some tasks succeed and some fail."""
    tasks = [create_task("task-1"), create_task("task-2"), create_task("task-3")]
    mock_provider.generate.side_effect = [
        AsyncLLMResponse(text="res-1", usage=TokenUsage(total_tokens=10), elapsed=0.1),
        Exception("simulated error"),
        AsyncLLMResponse(text="res-3", usage=TokenUsage(total_tokens=30), elapsed=0.3),
    ]

    dispatcher = SwarmDispatcher(max_retries=0)
    dispatcher._make_provider = lambda tier, prompt_len, provider_name="kimi", prompt=None: mock_provider
    results = await dispatcher.dispatch(tasks)

    assert len(results) == 3
    assert results[0].success is True
    assert results[0].response == "res-1"
    assert results[1].success is False
    assert "simulated error" in results[1].error
    assert results[2].success is True
    assert results[2].response == "res-3"


@pytest.mark.asyncio
async def test_dispatch_all_fail(mock_provider, create_task):
    """Test dispatch where all tasks fail."""
    tasks = [create_task("task-1"), create_task("task-2")]
    mock_provider.generate.side_effect = [
        ValueError("error-1"),
        RuntimeError("error-2"),
    ]

    dispatcher = SwarmDispatcher(max_retries=0)
    dispatcher._make_provider = lambda tier, prompt_len, provider_name="kimi", prompt=None: mock_provider
    results = await dispatcher.dispatch(tasks)

    assert len(results) == 2
    assert results[0].success is False
    assert "error-1" in results[0].error
    assert results[1].success is False
    assert "error-2" in results[1].error


@pytest.mark.asyncio
async def test_dispatch_returns_exceptions_as_records(mock_provider, create_task, monkeypatch):
    """Test that dispatch handles exceptions from _run_task gracefully, returning TaskResult records."""
    task = create_task("task-1")

    # Patch _run_task to raise an exception directly, bypassing its internal try/except
    async def mock_run_task(self, t: SwarmTask) -> TaskResult:
        raise ValueError(f"Direct error from _run_task for {t.id}")

    monkeypatch.setattr(SwarmDispatcher, "_run_task", mock_run_task)

    dispatcher = SwarmDispatcher()
    dispatcher._make_provider = lambda tier, prompt_len, provider_name="kimi", prompt=None: mock_provider
    results = await dispatcher.dispatch([task])

    assert len(results) == 1
    result = results[0]
    assert result.task_id == "task-1"
    assert result.success is False
    assert result.response == ""
    assert f"Direct error from _run_task for {task.id}" in result.error
    assert result.tokens == TokenUsage()
    assert result.elapsed_seconds == 0.0


@pytest.mark.asyncio
async def test_concurrency_cap_enforced(mock_provider, create_task):
    """Test that the concurrency cap is enforced."""
    concurrency = 2
    num_tasks = 5
    tasks = [create_task(f"task-{i}") for i in range(num_tasks)]

    # Use a shared list to track active tasks and a counter for max concurrency
    active_tasks_list = []
    max_concurrent = 0

    async def generate_mock(*args, **kwargs):
        nonlocal max_concurrent
        # Simulate work and track active tasks
        active_tasks_list.append(1)
        max_concurrent = max(max_concurrent, len(active_tasks_list))
        await asyncio.sleep(0)  # Yield control to the event loop
        active_tasks_list.pop()
        return AsyncLLMResponse(text="response", usage=TokenUsage(), elapsed=0.1)

    mock_provider.generate.side_effect = generate_mock

    dispatcher = SwarmDispatcher(concurrency=concurrency, max_retries=0)
    dispatcher._make_provider = lambda tier, prompt_len, provider_name="kimi", prompt=None: mock_provider
    
    # Dispatch tasks and let them run
    results = await dispatcher.dispatch(tasks)

    assert len(results) == num_tasks
    assert all(r.success for r in results)
    assert max_concurrent == concurrency # Verify that max concurrency was respected
    assert len(active_tasks_list) == 0 # All tasks should have completed


@pytest.mark.asyncio
async def test_timeout_returns_failure(mock_provider, create_task):
    """Test that a task timing out returns a failure result."""
    task = create_task("task-1")

    async def slow_generate(*args, **kwargs):
        await asyncio.sleep(0.1)  # Longer than task_timeout
        return AsyncLLMResponse(text="never reached", usage=TokenUsage(), elapsed=0.1)

    mock_provider.generate.side_effect = slow_generate

    dispatcher = SwarmDispatcher(task_timeout=0.01, max_retries=0)
    dispatcher._make_provider = lambda tier, prompt_len, provider_name="kimi", prompt=None: mock_provider
    results = await dispatcher.dispatch([task])

    assert len(results) == 1
    result = results[0]
    assert result.task_id == "task-1"
    assert result.success is False
    assert result.response == ""
    assert result.error == "timeout"
    assert result.elapsed_seconds > 0


@pytest.mark.asyncio
async def test_rate_limit_final_attempt_returns_failure(mock_provider, create_task):
    """Test that a RateLimitError after all retries returns a failure."""
    task = create_task("task-1")
    mock_provider.generate.side_effect = RateLimitError("Too many requests")

    dispatcher = SwarmDispatcher(max_retries=0)  # No retries
    dispatcher._make_provider = lambda tier, prompt_len, provider_name="kimi", prompt=None: mock_provider
    results = await dispatcher.dispatch([task])

    assert len(results) == 1
    result = results[0]
    assert result.task_id == "task-1"
    assert result.success is False
    assert result.response == ""
    assert result.error == "rate_limit"
    assert result.elapsed_seconds > 0


@pytest.mark.asyncio
async def test_general_exception_returns_failure(mock_provider, create_task):
    """Test that a general exception after all retries returns a failure."""
    task = create_task("task-1")
    mock_provider.generate.side_effect = ValueError("Something unexpected happened")

    dispatcher = SwarmDispatcher(max_retries=0)
    dispatcher._make_provider = lambda tier, prompt_len, provider_name="kimi", prompt=None: mock_provider
    results = await dispatcher.dispatch([task])

    assert len(results) == 1
    result = results[0]
    assert result.task_id == "task-1"
    assert result.success is False
    assert result.response == ""
    assert "Something unexpected happened" in result.error
    assert result.elapsed_seconds > 0


@pytest.mark.asyncio
async def test_elapsed_seconds_populated(mock_provider, create_task):
    """Test that elapsed_seconds is correctly populated for successful tasks."""
    task = create_task("task-1")
    mock_provider.generate.return_value = AsyncLLMResponse(
        text="response", usage=TokenUsage(), elapsed=0.05
    )

    dispatcher = SwarmDispatcher()
    dispatcher._make_provider = lambda tier, prompt_len, provider_name="kimi", prompt=None: mock_provider
    results = await dispatcher.dispatch([task])

    assert len(results) == 1
    result = results[0]
    assert result.success is True
    assert result.elapsed_seconds > 0


def test_dispatcher_rejects_unknown_only_allowlist():
    """SwarmDispatcher raises ValueError when allowed_providers contains no swarm-capable provider.
    Phase C2: kimi is no longer required — any provider in SWARM_CAPABLE_PROVIDERS is accepted.
    haiku is now swarm-capable (via KimiAsyncProvider duck-typed wrapper), so the old test
    that expected haiku to raise has been reconciled: only truly unknown providers raise.
    """
    with pytest.raises(ValueError, match="swarm-capable"):
        SwarmDispatcher(allowed_providers=frozenset({"completely_unknown_provider_xyz"}))


def test_dispatcher_accepts_kimi_in_allowlist():
    """SwarmDispatcher constructs OK when 'kimi' is in the allowlist."""
    d = SwarmDispatcher(allowed_providers=frozenset({"kimi"}))
    assert d is not None


def test_dispatcher_accepts_none_allowlist():
    """SwarmDispatcher constructs OK when allowed_providers is None (default)."""
    d = SwarmDispatcher(allowed_providers=None)
    assert d is not None


# ---------------------------------------------------------------------------
# SwarmContext.__post_init__ validation (lines 30-39)
# ---------------------------------------------------------------------------
def test_swarm_context_files_not_list_raises():
    """Line 30-31: SwarmContext.files must be a list."""
    from mullet.core.orchestration.swarm_types import SwarmContext
    with pytest.raises(TypeError, match="files must be list"):
        SwarmContext(files="not-a-list", items=[])


def test_swarm_context_files_element_not_str_raises():
    """Lines 32-34: SwarmContext.files elements must be str."""
    from mullet.core.orchestration.swarm_types import SwarmContext
    with pytest.raises(TypeError, match="files\\[0\\] must be str"):
        SwarmContext(files=[123], items=[])


def test_swarm_context_items_not_list_raises():
    """Lines 35-36: SwarmContext.items must be a list."""
    from mullet.core.orchestration.swarm_types import SwarmContext
    with pytest.raises(TypeError, match="items must be list"):
        SwarmContext(files=[], items="not-a-list")


def test_swarm_context_items_element_not_str_raises():
    """Lines 37-39: SwarmContext.items elements must be str."""
    from mullet.core.orchestration.swarm_types import SwarmContext
    with pytest.raises(TypeError, match="items\\[0\\] must be str"):
        SwarmContext(files=[], items=[42])


# ---------------------------------------------------------------------------
# on_progress callback exception handling (lines 140-143)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_dispatch_on_progress_exception_is_swallowed(mock_provider, create_task):
    """Lines 140-143: on_progress callback that raises is silenced."""
    task = create_task("prog-1")
    mock_provider.generate.return_value = AsyncLLMResponse(
        text="done", usage=TokenUsage(total_tokens=1), elapsed=0.01
    )
    dispatcher = SwarmDispatcher()
    dispatcher._make_provider = lambda tier, prompt_len, provider_name="kimi", prompt=None: mock_provider

    progress_calls = []
    def bad_progress(result, completed, total):
        progress_calls.append((completed, total))
        raise RuntimeError("progress boom")

    results = await dispatcher.dispatch([task], on_progress=bad_progress)
    assert len(results) == 1
    assert results[0].success is True
    assert len(progress_calls) == 1  # was called but exception was swallowed


# ---------------------------------------------------------------------------
# Backoff sleep path (lines 220-222) — RateLimitError with retry
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_dispatch_rate_limit_with_retry_sleeps(mock_provider, create_task, monkeypatch):
    """Lines 220-222: RateLimitError on first attempt triggers backoff sleep before retry."""
    task = create_task("rl-retry-1")
    call_count = 0

    async def _generate_side_effect(prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RateLimitError("429")
        return AsyncLLMResponse(text="retry-ok", usage=TokenUsage(total_tokens=2), elapsed=0.01)

    mock_provider.generate.side_effect = _generate_side_effect

    sleep_durations = []
    async def fast_sleep(duration):
        sleep_durations.append(duration)

    monkeypatch.setattr("asyncio.sleep", fast_sleep)

    dispatcher = SwarmDispatcher(max_retries=1)
    dispatcher._make_provider = lambda tier, prompt_len, provider_name="kimi", prompt=None: mock_provider

    results = await dispatcher.dispatch([task])
    assert results[0].success is True
    assert len(sleep_durations) >= 1  # backoff sleep was called
    assert sleep_durations[0] > 0


# ---------------------------------------------------------------------------
# _make_provider exception fallback (lines 113-119)
# ---------------------------------------------------------------------------
def test_make_provider_exception_fallback(monkeypatch):
    """Lines 113-119: _make_provider falls back to hardcoded model on import error."""
    from unittest.mock import MagicMock, patch

    dispatcher = SwarmDispatcher()

    mock_kimi_provider = MagicMock()
    mock_kimi_async = MagicMock()

    with patch.dict("sys.modules", {
        "mullet.core.providers.model_registry": None,  # forces import error
    }), patch("mullet.core.providers.kimi.KimiProvider", mock_kimi_provider), \
       patch("mullet.core.providers.kimi_async.KimiAsyncProvider", mock_kimi_async):
        result = dispatcher._make_provider("STANDARD", 100)
    # Should not raise — fallback always returns something
    assert result is not None


# ---------------------------------------------------------------------------
# _make_provider success path (lines 79-112)
# ---------------------------------------------------------------------------
def test_make_provider_success_path():
    """Lines 79-112: _make_provider succeeds — kimi model selected from registry."""
    from unittest.mock import patch, MagicMock

    dispatcher = SwarmDispatcher()

    mock_registry = MagicMock()
    mock_registry.get_best_model.return_value = "kimi-k2.5"
    mock_model_info = MagicMock()
    mock_model_info.context_length = 131072  # large enough for any test prompt
    mock_kp = MagicMock()
    mock_kap = MagicMock()

    # KimiProvider/KimiAsyncProvider are imported locally inside _make_provider;
    # patch them on their own modules so the local `from ... import` picks up mocks.
    with patch("mullet.core.providers.kimi.KimiProvider", mock_kp), \
         patch("mullet.core.providers.kimi_async.KimiAsyncProvider", mock_kap), \
         patch("mullet.core.providers.model_registry.get_registry", return_value=mock_registry), \
         patch("mullet.core.providers.model_info.get_model_info", return_value=mock_model_info):
        result = dispatcher._make_provider("STANDARD", 100, prompt="test prompt")

    assert result is not None
    mock_registry.get_best_model.assert_called()


def test_make_provider_all_tiers_context_too_small():
    """Lines 109-112: all tiers exhausted → falls back to CRITICAL model."""
    from unittest.mock import patch, MagicMock

    dispatcher = SwarmDispatcher()

    mock_registry = MagicMock()
    mock_registry.get_best_model.return_value = "kimi-k2.5"
    # context_length always 0 — every tier check fails
    mock_model_info = MagicMock()
    mock_model_info.context_length = 0
    mock_kp = MagicMock()
    mock_kap = MagicMock()

    with patch("mullet.core.providers.kimi.KimiProvider", mock_kp), \
         patch("mullet.core.providers.kimi_async.KimiAsyncProvider", mock_kap), \
         patch("mullet.core.providers.model_registry.get_registry", return_value=mock_registry), \
         patch("mullet.core.providers.model_info.get_model_info", return_value=mock_model_info):
        result = dispatcher._make_provider("LOW", 100)

    assert result is not None
    # Last call should be get_best_model("kimi", "CRITICAL", ...)
    last_args = mock_registry.get_best_model.call_args_list[-1]
    assert last_args[0][1] == "CRITICAL"


# ---------------------------------------------------------------------------
# estimate_tokens exception fallback (lines 91-93)
# ---------------------------------------------------------------------------
def test_make_provider_estimate_tokens_raises_uses_char_fallback():
    """Lines 91-93: estimate_tokens raising falls back to char-based estimate."""
    from unittest.mock import patch, MagicMock

    dispatcher = SwarmDispatcher()

    mock_registry = MagicMock()
    mock_registry.get_best_model.return_value = "kimi-k2.5"
    mock_model_info = MagicMock()
    mock_model_info.context_length = 131072
    mock_kp = MagicMock()
    mock_kap = MagicMock()

    with patch("mullet.core.providers.kimi.KimiProvider", mock_kp), \
         patch("mullet.core.providers.kimi_async.KimiAsyncProvider", mock_kap), \
         patch("mullet.core.providers.model_registry.get_registry", return_value=mock_registry), \
         patch("mullet.core.providers.model_info.get_model_info", return_value=mock_model_info), \
         patch("mullet.core.providers.tokenize.estimate_tokens", side_effect=RuntimeError("tokenize boom")):
        result = dispatcher._make_provider("STANDARD", 600, prompt="hello world")

    assert result is not None
    mock_registry.get_best_model.assert_called()


# ---------------------------------------------------------------------------
# CancelledError propagation (line 210)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cancelled_error_propagates_as_failed_result(mock_provider, create_task):
    """Line 210: CancelledError from generate propagates (re-raised) → captured by gather."""
    task = create_task("cancel-1")
    mock_provider.generate.side_effect = asyncio.CancelledError()

    dispatcher = SwarmDispatcher(max_retries=0)
    dispatcher._make_provider = lambda tier, prompt_len, provider_name="kimi", prompt=None: mock_provider

    # CancelledError propagates to asyncio.gather which has return_exceptions=True
    # It gets captured and turned into a TaskResult(success=False) in the outer loop
    results = await dispatcher.dispatch([task])
    assert len(results) == 1
    assert results[0].success is False
