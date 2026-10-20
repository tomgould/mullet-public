import pytest
from unittest.mock import Mock
from mullet.core.providers.fallback import FallbackChain, ProviderHealth, FallbackResult


class MockProvider:
    def __init__(self, name: str, responses: list):
        self.name = name
        self.responses = iter(responses)

    def create(self, messages, tools, **kwargs):
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def test_fallback_chain_success_first_provider():
    # Test 1: First provider succeeds immediately
    provider1 = MockProvider("ProviderA", ["ResponseA"])
    provider2 = MockProvider("ProviderB", ["ResponseB"])
    chain = FallbackChain([provider1, provider2])

    result = chain.create([], [])
    assert result.provider_used == "ProviderA"
    assert result.attempts == 1
    assert result.fallbacks_triggered == 0
    assert result.response == "ResponseA"
    assert chain.health.get_success_rate("ProviderA") == 1.0
    assert chain.health.get_failure_rate("ProviderA") == 0.0


def test_fallback_chain_fallback_to_second_provider():
    # Test 2: First provider fails with retryable errors, second succeeds
    provider1 = MockProvider("ProviderA", [Exception("connection timeout"), Exception("connection timeout"), Exception("connection timeout")])
    provider2 = MockProvider("ProviderB", ["ResponseB"])
    chain = FallbackChain([provider1, provider2], max_retries_per=2)

    result = chain.create([], [])
    assert result.provider_used == "ProviderB"
    assert result.attempts == 4  # 3 attempts on A (retryable), 1 on B
    assert result.fallbacks_triggered == 1
    assert result.response == "ResponseB"
    assert chain.health.get_success_rate("ProviderB") == 1.0
    assert chain.health.get_failure_rate("ProviderA") == 1.0


def test_fallback_chain_non_retryable_skips_to_next():
    # Non-retryable errors skip retries and move to next provider immediately
    provider1 = MockProvider("ProviderA", [Exception("ErrorA")])
    provider2 = MockProvider("ProviderB", ["ResponseB"])
    chain = FallbackChain([provider1, provider2], max_retries_per=2)

    result = chain.create([], [])
    assert result.provider_used == "ProviderB"
    assert result.attempts == 2  # 1 attempt on A (non-retryable), 1 on B
    assert result.fallbacks_triggered == 1


def test_fallback_chain_max_retries_per_provider():
    # Test 3: First provider fails with retryable errors, then second succeeds
    provider1 = MockProvider("ProviderA", [Exception("connection timeout"), Exception("connection timeout"), "ResponseA_Retry"])
    provider2 = MockProvider("ProviderB", ["ResponseB"])
    chain = FallbackChain([provider1, provider2], max_retries_per=1)

    result = chain.create([], [])
    assert result.provider_used == "ProviderB"
    assert result.attempts == 3  # A fails twice (retryable), then B succeeds
    assert result.fallbacks_triggered == 1
    assert result.response == "ResponseB"
    assert chain.health.get_failure_rate("ProviderA") == 1.0
    assert chain.health.get_success_rate("ProviderB") == 1.0


def test_fallback_chain_all_providers_fail():
    # Test 4: All providers fail after all retries (retryable errors)
    provider1 = MockProvider("ProviderA", [Exception("connection timeout"), Exception("connection timeout")])
    provider2 = MockProvider("ProviderB", [Exception("connection timeout"), Exception("connection timeout")])
    chain = FallbackChain([provider1, provider2], max_retries_per=1)

    with pytest.raises(RuntimeError, match="All providers failed"):
        chain.create([], [])
    assert chain.health.get_failure_rate("ProviderA") == 1.0
    assert chain.health.get_failure_rate("ProviderB") == 1.0


def test_provider_health_tracking():
    # Test 5: Verify ProviderHealth tracking
    health = ProviderHealth()
    health.record_success("ProviderX")
    health.record_failure("ProviderX")
    health.record_success("ProviderY")

    assert health.get_success_rate("ProviderX") == 0.5
    assert health.get_failure_rate("ProviderX") == 0.5
    assert health.get_success_rate("ProviderY") == 1.0
    assert health.get_failure_rate("ProviderY") == 0.0
    assert health.get_success_rate("ProviderZ") == 0.0  # Non-existent provider


def test_fallback_result_attributes():
    # Test 6: FallbackResult attributes are correct
    result = FallbackResult("TestProvider", 3, 1, {"text": "Hello"})
    assert result.provider_used == "TestProvider"
    assert result.attempts == 3
    assert result.fallbacks_triggered == 1
    assert result.response == {"text": "Hello"}


def test_fallback_chain_no_providers_init():
    # Test 7: Initialize with no providers raises error
    with pytest.raises(ValueError, match="At least one provider must be provided."):
        FallbackChain([])


def test_fallback_chain_single_provider_success():
    # Test 8: Single provider in the chain succeeds
    provider1 = MockProvider("ProviderA", ["SingleResponse"])
    chain = FallbackChain([provider1])

    result = chain.create([], [])
    assert result.provider_used == "ProviderA"
    assert result.attempts == 1
    assert result.fallbacks_triggered == 0
    assert result.response == "SingleResponse"


def test_fallback_chain_single_provider_failure():
    # Test 9: Single provider in the chain fails after retries
    provider1 = MockProvider("ProviderA", [Exception("E1"), Exception("E2")])
    chain = FallbackChain([provider1], max_retries_per=1)

    with pytest.raises(RuntimeError):
        chain.create([], [])
    assert chain.health.get_failure_rate("ProviderA") == 1.0


def test_fallback_chain_different_kwargs_passed():
    # Test 10: Ensure kwargs are passed to the provider's create method
    mock_create = Mock(return_value="KwargsResponse")
    provider1 = MockProvider("ProviderA", [mock_create])
    provider1.create = mock_create
    chain = FallbackChain([provider1])

    result = chain.create([], [], temperature=0.7, model="gpt-4")
    mock_create.assert_called_once_with([], [], temperature=0.7, model="gpt-4")
    assert result.response == "KwargsResponse"


def test_fallback_chain_with_tools():
    # Test 11: Ensure tools are passed to the provider's create method
    mock_create = Mock(return_value="ToolsResponse")
    provider1 = MockProvider("ProviderA", [mock_create])
    provider1.create = mock_create
    chain = FallbackChain([provider1])
    dummy_tools = [{
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }]

    result = chain.create([], dummy_tools)
    mock_create.assert_called_once_with([], dummy_tools)
    assert result.response == "ToolsResponse"


def test_fallback_chain_complex_retry_pattern():
    # Test 12: A more complex retry scenario — errors must be retryable
    # (e.g. "rate limit", "timeout") for the retry logic to kick in
    provider1 = MockProvider("ProviderA", [Exception("rate limit"), "ResponseA"])
    provider2 = MockProvider("ProviderB", [Exception("timeout"), Exception("timeout"), "ResponseB"])
    provider3 = MockProvider("ProviderC", ["ResponseC"])
    chain = FallbackChain([provider1, provider2, provider3], max_retries_per=1)

    result = chain.create([], [])
    assert result.provider_used == "ProviderA"
    assert result.attempts == 2
    assert result.fallbacks_triggered == 0
    assert result.response == "ResponseA"
    assert chain.health.get_failure_rate("ProviderA") == 0.5
    assert chain.health.get_success_rate("ProviderA") == 0.5
    assert chain.health.get_failure_rate("ProviderB") == 0.0
    assert chain.health.get_success_rate("ProviderB") == 0.0
    assert chain.health.get_failure_rate("ProviderC") == 0.0
    assert chain.health.get_success_rate("ProviderC") == 0.0
