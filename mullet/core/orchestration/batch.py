"""Batch runner — parallel batch processing for bulk tasks.

From Hermes: trajectory generation, bulk evaluation, parallel task execution.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mullet.core.agent.types import AgentResult, TokenUsage

if TYPE_CHECKING:
    from mullet.core.agent.runner import AgentRunner

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result of a batch task execution run."""
    completed: int = 0
    failed: int = 0
    total: int = 0
    results: list[AgentResult] = field(default_factory=list)
    total_tokens: TokenUsage = field(default_factory=TokenUsage)
    errors: list[str] = field(default_factory=list)


def run_batch(
    tasks: list[str],
    agent: "AgentRunner",
    max_parallel: int = 4,
    on_complete=None,
) -> BatchResult:
    """Run multiple tasks in parallel with a thread pool."""
    batch = BatchResult(total=len(tasks))
    logger.info("Batch: executing %d tasks (max_parallel=%d)", len(tasks), max_parallel)

    def _run_one(task: str) -> tuple[str, AgentResult | None, str]:
        try:
            result = agent.run(task)
            return task, result, ""
        except Exception as e:
            return task, None, str(e)

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {pool.submit(_run_one, t): t for t in tasks}
        for future in as_completed(futures):
            task_str, result, error = future.result()
            if result and result.completed:
                batch.completed += 1
                batch.results.append(result)
                batch.total_tokens = batch.total_tokens + result.tokens_used
            else:
                batch.failed += 1
                batch.errors.append(error or f"Task failed: {task_str[:50]}")

            if on_complete:
                on_complete(batch.completed, batch.failed, batch.total)

    logger.info("Batch complete: %d/%d succeeded", batch.completed, batch.total)
    return batch
