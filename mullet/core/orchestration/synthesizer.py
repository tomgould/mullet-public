"""Synthesizer for aggregating results from Mullet swarm tasks."""

import logging
from typing import List, Tuple

from mullet.core.agent.types import TokenUsage
from mullet.core.orchestration.swarm_types import TaskResult
from mullet.core.providers.kimi_async import KimiAsyncProvider, AsyncLLMResponse

logger = logging.getLogger(__name__)

SYNTHESIZE_PROMPT = """You are a synthesis agent for the Mullet swarm system. Your task is to combine the results from multiple task executions into a single coherent {output_shape}.

The original goal was: {goal}

There were {n_total} total tasks:
- {n_succeeded} succeeded
- {n_failed} failed

## Results

{results}

{f_failures}

Please synthesize these results into a comprehensive {output_shape} that addresses the original goal. Focus on the successful results and provide insights, analysis, and key findings. If there were failures, note them appropriately without inventing content."""

def _format_results(results: List[TaskResult]) -> str:
    """Format successful results into numbered sections.
    
    Args:
        results: List of successful TaskResult objects
        
    Returns:
        Formatted string with numbered sections for each result
    """
    formatted = []
    for i, result in enumerate(results, 1):
        formatted.append(f"### Result {i} (Task {result.task_id})")
        formatted.append(result.response)
        formatted.append("")  # Add spacing between results
    return "\n".join(formatted).strip()

def _format_failures(failures: List[TaskResult]) -> str:
    """Format failed results into a failures section.
    
    Args:
        failures: List of failed TaskResult objects
        
    Returns:
        Formatted string listing task_id and error for each failure
        Empty string if no failures
    """
    if not failures:
        return ""
    
    formatted = ["## Failures"]
    for failure in failures:
        formatted.append(f"- Task {failure.task_id}: {failure.error}")
    return "\n".join(formatted)

def _format_raw_results(results: List[TaskResult]) -> str:
    """Format all results as fallback when synthesis fails.
    
    Args:
        results: List of all TaskResult objects
        
    Returns:
        Raw concatenation of all responses with headers
    """
    formatted = []
    for result in results:
        status = "SUCCESS" if result.success else "FAILED"
        formatted.append(f"=== Task {result.task_id} ({status}) ===")
        formatted.append(result.response)
        if not result.success and result.error:
            formatted.append(f"Error: {result.error}")
        formatted.append("")
    return "\n".join(formatted).strip()

class Synthesizer:
    """Synthesizes task results into coherent outputs using an LLM provider."""
    
    def __init__(self, provider: KimiAsyncProvider):
        """Initialize the synthesizer with an LLM provider.
        
        Args:
            provider: Async LLM provider for synthesis
        """
        self._provider = provider
    
    async def synthesize(
        self, 
        goal: str, 
        results: List[TaskResult], 
        output_shape: str = "markdown report"
    ) -> Tuple[str, TokenUsage]:
        """Synthesize task results into a coherent output.
        
        Args:
            goal: The original swarm goal
            results: List of task results to synthesize
            output_shape: Desired format/shape of the output
            
        Returns:
            Tuple of (synthesized_text, token_usage)
            Returns ("", TokenUsage()) if no successful results
            Returns raw concatenation if synthesis fails
        """
        if not results:
            return "", TokenUsage()
        
        # Split into succeeded and failed results
        succeeded = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        # If no successful results, return empty
        if not succeeded:
            return "", TokenUsage()
        
        # Format the prompt
        results_str = _format_results(succeeded)
        failures_str = _format_failures(failed)
        
        prompt = SYNTHESIZE_PROMPT.format(
            goal=goal,
            output_shape=output_shape,
            n_total=len(results),
            n_succeeded=len(succeeded),
            n_failed=len(failed),
            results=results_str,
            f_failures=failures_str
        )
        
        try:
            response: AsyncLLMResponse = await self._provider.generate(
                prompt, max_tokens=8192
            )
            return response.text, response.usage
        except Exception as e:
            logger.error(f"Failed to synthesize results: {e}")
            return _format_raw_results(results), TokenUsage()