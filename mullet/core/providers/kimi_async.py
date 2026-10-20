"""Async facade over KimiProvider using asyncio.to_thread.

Uses asyncio.to_thread at default concurrency (≤20 concurrent tasks).
For >50 concurrent tasks, replace with native httpx AsyncClient.
# TODO(native-async): replace with httpx AsyncClient if concurrency exceeds 50.
"""
import asyncio
import time
from dataclasses import dataclass
from typing import Dict
from mullet.core.providers.kimi import KimiProvider
from mullet.core.agent.types import TokenUsage


@dataclass
class AsyncLLMResponse:
    """Async response from the Kimi async provider."""
    text: str
    usage: TokenUsage
    elapsed: float


class KimiAsyncProvider:
    """Async facade over KimiProvider using asyncio.to_thread."""
    
    def __init__(self, provider: KimiProvider | None = None):
        """Initialize the async provider.
        
        Args:
            provider: Optional KimiProvider instance. If None, creates a new one.
        """
        self._sync = provider or KimiProvider()
    
    async def generate(self, prompt: str, context_refs: list[str] | None = None, max_tokens: int = 4096) -> AsyncLLMResponse:
        """Generate response asynchronously.
        
        Args:
            prompt: The user prompt
            context_refs: Optional list of context file paths
            max_tokens: Maximum tokens for response
            
        Returns:
            AsyncLLMResponse with text, usage, and elapsed time
        """
        messages = self._build_messages(prompt, context_refs)
        
        start = time.monotonic()
        response = await asyncio.to_thread(
            self._sync.create,
            messages=messages,
            max_tokens=max_tokens
        )
        elapsed = time.monotonic() - start
        
        return AsyncLLMResponse(
            text=response.content,
            usage=response.usage,
            elapsed=elapsed
        )
    
    def _build_messages(self, prompt: str, context_refs: list[str] | None) -> list[Dict[str, str]]:
        """Build the messages list for the API call.
        
        Args:
            prompt: The user prompt
            context_refs: Optional list of context file paths
            
        Returns:
            List of messages for the API
        """
        messages = []
        
        if context_refs:
            messages.append({
                "role": "system",
                "content": "Context files: " + ", ".join(context_refs)
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        return messages