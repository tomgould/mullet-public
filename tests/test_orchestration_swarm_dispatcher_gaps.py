"""Coverage-gap tests for mullet.core.orchestration.swarm_dispatcher."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mullet.core.agent.types import TokenUsage
from mullet.core.orchestration.swarm_dispatcher import (
    RateLimitError,
    SwarmDispatcher,
)
from mullet.core.orchestration.swarm_types import SwarmTask
from mullet.core.providers.kimi_async import AsyncLLMResponse


class TestMakeProviderGaps:
    """Cover lines 91-93, 109-119."""

    def test_estimate_tokens_exception_fallback(self):
        with patch.dict("os.environ", {"KIMI_API_KEY": "test"}), \
             patch("mullet.core.providers.tokenize.estimate_tokens", side_effect=ValueError("boom")):
            dispatcher = SwarmDispatcher()
            provider = dispatcher._make_provider("STANDARD", 100, prompt="hello")
            assert provider is not None

    def test_prompt_none_uses_char_fallback(self):
        """Line 90: prompt=None falls back to char-based token estimate."""
        with patch.dict("os.environ", {"KIMI_API_KEY": "test"}):
            dispatcher = SwarmDispatcher()
            provider = dispatcher._make_provider("STANDARD", 600, prompt=None)
            assert provider is not None

    def test_all_tiers_exhausted_uses_critical(self):
        with patch.dict("os.environ", {"KIMI_API_KEY": "test"}), \
             patch("mullet.core.providers.model_registry.get_registry") as mock_registry, \
             patch("mullet.core.providers.model_info.get_model_info") as mock_info:
            mock_registry.return_value.get_best_model.return_value = "kimi-k2.5"
            mock_info.return_value = MagicMock()
            mock_info.return_value.context_length = 0  # always too small
            dispatcher = SwarmDispatcher()
            provider = dispatcher._make_provider("STANDARD", 100, prompt="hello")
            assert provider is not None
            # Last call should be CRITICAL
            last_call = mock_registry.return_value.get_best_model.call_args_list[-1]
            assert last_call[0] == ("kimi", "CRITICAL")

    def test_make_provider_exception_uses_fallback(self):
        with patch.dict("os.environ", {"KIMI_API_KEY": "test"}), \
             patch("mullet.core.providers.model_registry.get_registry", side_effect=RuntimeError("registry boom")):
            dispatcher = SwarmDispatcher()
            provider = dispatcher._make_provider("STANDARD", 100, prompt="hello")
            assert provider is not None


class TestDispatchOnProgressGap:
    """Cover lines 140-143."""

    def test_on_progress_exception_handled(self):
        def bad_progress(result, completed, total):
            raise ValueError("progress boom")

        async def _test():
            mock_provider = MagicMock()
            mock_provider.generate = AsyncMock(
                return_value=AsyncLLMResponse(text="ok", usage=TokenUsage(), elapsed=0.1)
            )
            dispatcher = SwarmDispatcher(max_retries=0)
            dispatcher._make_provider = lambda tier, prompt_len, provider_name="kimi", prompt=None: mock_provider
            results = await dispatcher.dispatch(
                [SwarmTask(id="t1", instruction="hello")],
                on_progress=bad_progress,
            )
            assert len(results) == 1
            assert results[0].success is True

        asyncio.run(_test())


class TestCancelledErrorGap:
    """Cover line 210."""

    def test_cancelled_error_propagates(self):
        async def _test():
            mock_provider = MagicMock()
            mock_provider.generate = AsyncMock(side_effect=asyncio.CancelledError())
            dispatcher = SwarmDispatcher(max_retries=0)
            dispatcher._make_provider = lambda tier, prompt_len, provider_name="kimi", prompt=None: mock_provider
            with pytest.raises(asyncio.CancelledError):
                await dispatcher._run_task(SwarmTask(id="t1", instruction="hello"))

        asyncio.run(_test())


class TestRetryPaths:
    """Cover lines 220-222 (backoff sleep on retry)."""

    def test_timeout_retry_then_success(self):
        call_count = 0

        async def timeout_then_ok(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.TimeoutError()
            return AsyncLLMResponse(text="ok", usage=TokenUsage(), elapsed=0.1)

        async def _test():
            mock_provider = MagicMock()
            mock_provider.generate = timeout_then_ok
            dispatcher = SwarmDispatcher(task_timeout=10.0, max_retries=1)
            dispatcher._make_provider = lambda tier, prompt_len, provider_name="kimi", prompt=None: mock_provider
            with patch("mullet.core.orchestration.swarm_dispatcher.asyncio.sleep", new=AsyncMock()):
                results = await dispatcher.dispatch([SwarmTask(id="t1", instruction="hello")])
                assert len(results) == 1
                assert results[0].success is True
                assert call_count == 2

        asyncio.run(_test())

    def test_rate_limit_retry_then_success(self):
        call_count = 0

        async def rate_limit_then_ok(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError("too fast")
            return AsyncLLMResponse(text="ok", usage=TokenUsage(), elapsed=0.1)

        async def _test():
            mock_provider = MagicMock()
            mock_provider.generate = rate_limit_then_ok
            dispatcher = SwarmDispatcher(task_timeout=10.0, max_retries=1)
            dispatcher._make_provider = lambda tier, prompt_len, provider_name="kimi", prompt=None: mock_provider
            with patch("mullet.core.orchestration.swarm_dispatcher.asyncio.sleep", new=AsyncMock()):
                results = await dispatcher.dispatch([SwarmTask(id="t1", instruction="hello")])
                assert len(results) == 1
                assert results[0].success is True
                assert call_count == 2

        asyncio.run(_test())

    def test_general_exception_retry_then_success(self):
        call_count = 0

        async def fail_then_ok(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("boom")
            return AsyncLLMResponse(text="ok", usage=TokenUsage(), elapsed=0.1)

        async def _test():
            mock_provider = MagicMock()
            mock_provider.generate = fail_then_ok
            dispatcher = SwarmDispatcher(task_timeout=10.0, max_retries=1)
            dispatcher._make_provider = lambda tier, prompt_len, provider_name="kimi", prompt=None: mock_provider
            with patch("mullet.core.orchestration.swarm_dispatcher.asyncio.sleep", new=AsyncMock()):
                results = await dispatcher.dispatch([SwarmTask(id="t1", instruction="hello")])
                assert len(results) == 1
                assert results[0].success is True
                assert call_count == 2

        asyncio.run(_test())

    def test_rate_limit_final_retry_returns_failure(self):
        async def always_rate_limit(prompt):
            raise RateLimitError("too fast")

        async def _test():
            mock_provider = MagicMock()
            mock_provider.generate = always_rate_limit
            dispatcher = SwarmDispatcher(task_timeout=10.0, max_retries=1)
            dispatcher._make_provider = lambda tier, prompt_len, provider_name="kimi", prompt=None: mock_provider
            with patch("mullet.core.orchestration.swarm_dispatcher.asyncio.sleep", new=AsyncMock()):
                results = await dispatcher.dispatch([SwarmTask(id="t1", instruction="hello")])
                assert len(results) == 1
                assert results[0].success is False
                assert results[0].error == "rate_limit"

        asyncio.run(_test())
