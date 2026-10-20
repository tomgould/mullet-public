"""Output quality scoring — post-hoc evaluation of agent responses.

Evaluates agent results on multiple dimensions without an LLM call:
- Completeness: did the agent use tools and produce a substantive response?
- Efficiency: token budget usage relative to complexity
- Tool usage: appropriate number of tool calls for the task
- Error rate: ratio of failed tool calls

Scores are 0-100. Used for routing learning feedback and stats.
"""

import logging
from dataclasses import dataclass

from mullet.core.agent.types import AgentResult

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Multi-dimensional quality assessment."""
    overall: float = 0.0
    completeness: float = 0.0
    efficiency: float = 0.0
    tool_usage: float = 0.0
    response_quality: float = 0.0
    notes: list[str] | None = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []


def score_result(result: AgentResult, task: str = "", max_iterations: int = 90) -> QualityScore:
    """Score an agent result on multiple quality dimensions.

    Args:
        result: The completed agent result.
        task: The original task (for context).
        max_iterations: Budget that was available.

    Returns:
        QualityScore with per-dimension scores (0-100).
    """
    notes = []

    # Completeness: did the agent finish and produce output?
    completeness = 0.0
    if result.completed:
        completeness += 50
        if result.final_response and len(result.final_response.strip()) > 10:
            completeness += 30
        if result.final_response and len(result.final_response.strip()) > 100:
            completeness += 20
    else:
        notes.append("Task did not complete")
    completeness = min(100, completeness)

    # Efficiency: how much of the budget was used?
    efficiency = 100.0
    if max_iterations > 0:
        usage_ratio = result.iterations_used / max_iterations
        if usage_ratio > 0.8:
            efficiency = 30  # Used most of budget — likely struggling
            notes.append("Used >80% of iteration budget")
        elif usage_ratio > 0.5:
            efficiency = 60
        elif usage_ratio > 0.2:
            efficiency = 80
        else:
            efficiency = 100  # Quick completion

    # Token efficiency
    if result.tokens_used.total_tokens > 50000:
        efficiency = max(20, efficiency - 30)
        notes.append("High token usage (>50K)")
    elif result.tokens_used.total_tokens > 20000:
        efficiency = max(40, efficiency - 10)

    # Tool usage: did the agent use tools appropriately?
    tool_usage = 50.0  # Default: neutral
    task_lower = task.lower() if task else ""

    # Tasks that should use tools
    tool_words = {"read", "write", "run", "test", "search", "find", "check", "list", "fix", "create", "delete"}
    expects_tools = any(w in task_lower for w in tool_words)

    if expects_tools and result.tool_calls_made > 0:
        tool_usage = 90  # Used tools when expected
    elif expects_tools and result.tool_calls_made == 0:
        tool_usage = 20  # Should have used tools but didn't
        notes.append("Expected tool usage but none occurred")
    elif not expects_tools and result.tool_calls_made == 0:
        tool_usage = 80  # Simple question, no tools needed
    elif result.tool_calls_made > 10:
        tool_usage = max(30, tool_usage - 20)
        notes.append("Excessive tool calls (>10)")

    # Response quality: basic text quality checks
    response_quality = 50.0
    resp = result.final_response.strip() if result.final_response else ""

    if resp:
        # Length appropriateness
        if 20 < len(resp) < 5000:
            response_quality += 20
        elif len(resp) >= 5000:
            response_quality += 10  # Very long might be verbose

        # Contains actual content (not just "I'll" or error messages)
        error_phrases = ["error:", "failed", "cannot", "unable", "not found", "no such file", "permission denied"]
        error_count = sum(1 for phrase in error_phrases if phrase in resp.lower())
        if error_count > 0:
            response_quality -= 10 * error_count
        if resp.count("\n") > 2:
            response_quality += 10  # Structured response

        # Doesn't just repeat the task
        if task and task.lower()[:30] not in resp.lower()[:50]:
            response_quality += 10
    else:
        response_quality = 0
        notes.append("Empty response")

    response_quality = max(0, min(100, response_quality))

    # Overall: weighted average
    overall = (
        completeness * 0.35 +
        efficiency * 0.20 +
        tool_usage * 0.25 +
        response_quality * 0.20
    )

    return QualityScore(
        overall=round(overall, 1),
        completeness=round(completeness, 1),
        efficiency=round(efficiency, 1),
        tool_usage=round(tool_usage, 1),
        response_quality=round(response_quality, 1),
        notes=notes,
    )
