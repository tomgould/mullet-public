"""Tests for mullet.core.providers.kimi_async module."""
import pytest
from unittest.mock import Mock, patch

from mullet.core.providers.kimi_async import KimiAsyncProvider, AsyncLLMResponse
from mullet.core.providers.kimi import KimiProvider
from mullet.core.agent.types import TokenUsage
from mullet.core.providers.base import LLMResponse


@pytest.mark.asyncio
async def test_generate_returns_async_response():
    """Test that generate returns an AsyncLLMResponse with correct text."""
    mock_provider = Mock(spec=KimiProvider)
    mock_provider.create.return_value = LLMResponse(content="hello")
    
    async_provider = KimiAsyncProvider(provider=mock_provider)
    result = await async_provider.generate("test prompt")
    
    assert isinstance(result, AsyncLLMResponse)
    assert result.text == "hello"


@pytest.mark.asyncio
async def test_generate_elapsed_is_positive():
    """Test that elapsed time is positive."""
    mock_provider = Mock(spec=KimiProvider)
    mock_provider.create.return_value = LLMResponse(content="test")
    
    async_provider = KimiAsyncProvider(provider=mock_provider)
    result = await async_provider.generate("test prompt")
    
    assert result.elapsed > 0


@pytest.mark.asyncio
async def test_generate_usage_forwarded():
    """Test that usage information is forwarded from LLMResponse."""
    expected_usage = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)
    mock_provider = Mock(spec=KimiProvider)
    mock_provider.create.return_value = LLMResponse(content="test", usage=expected_usage)
    
    async_provider = KimiAsyncProvider(provider=mock_provider)
    result = await async_provider.generate("test prompt")
    
    assert result.usage == expected_usage
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5
    assert result.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_generate_without_context_refs():
    """Test that messages has only user message when no context_refs provided."""
    mock_provider = Mock(spec=KimiProvider)
    mock_provider.create.return_value = LLMResponse(content="test")
    
    async_provider = KimiAsyncProvider(provider=mock_provider)
    await async_provider.generate("test prompt")
    
    mock_provider.create.assert_called_once()
    messages = mock_provider.create.call_args[1]["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "test prompt"


@pytest.mark.asyncio
async def test_generate_with_context_refs():
    """Test that messages include system message with context_refs when provided."""
    mock_provider = Mock(spec=KimiProvider)
    mock_provider.create.return_value = LLMResponse(content="test")
    
    async_provider = KimiAsyncProvider(provider=mock_provider)
    context_refs = ["/path/to/file1.py", "/path/to/file2.py"]
    await async_provider.generate("test prompt", context_refs=context_refs)
    
    mock_provider.create.assert_called_once()
    messages = mock_provider.create.call_args[1]["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Context files: /path/to/file1.py, /path/to/file2.py"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "test prompt"


@pytest.mark.asyncio
async def test_generate_uses_to_thread():
    """Test that asyncio.to_thread is used to call the sync provider."""
    mock_provider = Mock(spec=KimiProvider)
    mock_provider.create.return_value = LLMResponse(content="test")
    
    # Create a more robust mock for to_thread
    async def mock_to_thread_func(func, *args, **kwargs):
        # Just execute the function directly for this test
        return func(*args, **kwargs)
    
    with patch("asyncio.to_thread", side_effect=mock_to_thread_func) as mock_to_thread:
        async_provider = KimiAsyncProvider(provider=mock_provider)
        await async_provider.generate("test prompt")
        
        mock_to_thread.assert_called_once()
        call_args = mock_to_thread.call_args
        # Check that the sync provider's create method is being passed
        assert call_args[0][0] == mock_provider.create


@pytest.mark.asyncio
async def test_default_provider_created_if_none():
    """Test that KimiProvider is instantiated when provider=None."""
    with patch("mullet.core.providers.kimi_async.KimiProvider") as mock_kimi_class:
        mock_instance = Mock()
        mock_instance.create.return_value = LLMResponse(content="test")
        mock_kimi_class.return_value = mock_instance
        
        async_provider = KimiAsyncProvider(provider=None)
        mock_kimi_class.assert_called_once()
        
        # Verify it can still be used
        await async_provider.generate("test prompt")


@pytest.mark.asyncio
async def test_generate_propagates_exception():
    """Test that exceptions from create are propagated."""
    mock_provider = Mock(spec=KimiProvider)
    mock_provider.create.side_effect = ValueError("Test error")
    
    async_provider = KimiAsyncProvider(provider=mock_provider)
    
    with pytest.raises(ValueError, match="Test error"):
        await async_provider.generate("test prompt")