"""Provider-fallback chain with health tracking + exponential backoff."""
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Protocol

log = logging.getLogger(__name__)


class Provider(Protocol):
    """
    Protocol for AI providers.
    """
    name: str

    def create(self, messages: List[Dict], tools: Optional[List[Dict]], **kwargs) -> Any:
        """Run a chat completion against this provider."""
        ...


class FallbackResult:
    """
    Represents the result of a FallbackChain operation.
    """

    def __init__(self, provider_used: str, attempts: int, fallbacks_triggered: int, response: Any):
        """Capture which provider served the request and how many attempts/fallbacks it took."""
        self.provider_used = provider_used
        self.attempts = attempts
        self.fallbacks_triggered = fallbacks_triggered
        self.response = response


class ProviderHealth:
    """
    Tracks the health (success/failure rates) of individual providers.
    """

    def __init__(self):
        """Initialise empty success/failure counters."""
        self.successes: Dict[str, int] = defaultdict(int)
        self.failures: Dict[str, int] = defaultdict(int)

    def record_success(self, provider_name: str):
        """Increment the success counter for *provider_name*."""
        self.successes[provider_name] += 1

    def record_failure(self, provider_name: str):
        """Increment the failure counter for *provider_name*."""
        self.failures[provider_name] += 1

    def get_success_rate(self, provider_name: str) -> float:
        """Return success rate for *provider_name* (0.0 when no calls recorded)."""
        total = self.successes[provider_name] + self.failures[provider_name]
        return self.successes[provider_name] / total if total > 0 else 0.0

    def get_failure_rate(self, provider_name: str) -> float:
        """Return failure rate for *provider_name* (0.0 when no calls recorded)."""
        total = self.successes[provider_name] + self.failures[provider_name]
        return self.failures[provider_name] / total if total > 0 else 0.0


class FallbackChain:
    """
    A chain of AI providers with fallback logic and health tracking.
    """

    def __init__(self, providers: List[Provider], max_retries_per: int = 2):
        """Build a chain over *providers* with up to *max_retries_per* retries each."""
        if not providers:
            raise ValueError("At least one provider must be provided.")
        self.providers = providers
        self.max_retries_per = max_retries_per
        self.health = ProviderHealth()

    def create(self, messages: List[Dict], tools: Optional[List[Dict]] = None, **kwargs) -> FallbackResult:
        """Try each provider in order, retrying transient failures, and return the first success."""
        total_attempts = 0
        fallbacks_triggered = 0

        for provider in self.providers:
            for attempt in range(self.max_retries_per + 1):  # +1 for the initial attempt
                total_attempts += 1
                try:
                    log.info(f"Attempting with provider {provider.name}, attempt {attempt + 1}")
                    response = provider.create(messages, tools, **kwargs)
                    self.health.record_success(provider.name)
                    return FallbackResult(provider.name, total_attempts, fallbacks_triggered, response)
                except Exception as e:
                    log.warning(f"Provider {provider.name} failed (attempt {attempt + 1}): {e}")
                    self.health.record_failure(provider.name)
                    if attempt < self.max_retries_per and self._is_retryable_error(e):
                        log.info(f"Retrying with {provider.name}...")
                        time.sleep(0.1 * (attempt + 1))  # Simple backoff
                    else:
                        break  # Move to the next provider if not retryable or max retries reached
            fallbacks_triggered += 1

        raise RuntimeError("All providers failed after multiple retries.")

    def _is_retryable_error(self, e: Exception) -> bool:
        """
        Determines if an exception is retryable by delegating to the
        shared retry logic.
        """
        from mullet.core.providers.retry import is_retryable
        return is_retryable(e)
