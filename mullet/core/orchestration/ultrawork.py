"""Ultrawork \u2014 parallel agent execution.

The parallelism engine. Decomposes a task, spawns N agents via
delegation, routes each subtask via the Router, merges results.
No persistence, no verification loop \u2014 that's ralph's job.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from mullet.core.agent.delegation import delegate_batch
from mullet.core.agent.types import DelegationResult, TokenUsage
from mullet.core.router.decision import Router

if TYPE_CHECKING:
    from mullet.core.agent.runner import AgentRunner

logger = logging.getLogger(__name__)


@dataclass
class UltraworkResult:
    """Result of parallel ultrawork execution."""
    completed: bool = False
    subtask_results: list[DelegationResult] = field(default_factory=list)
    total_cost: float = 0.0
    total_tokens: TokenUsage = field(default_factory=TokenUsage)
    tiers_used: dict[str, int] = field(default_factory=dict)
    total_subtasks: int = 0
    completed_subtasks: int = 0
    failed_subtasks: int = 0
    summary: str = ""


class UltraworkMode:
    """Parallel agent execution \u2014 decompose, route, spawn, merge."""

    def __init__(
        self,
        router: Router | None = None,
        max_workers: int = 3,
        allowed_providers: frozenset[str] | None = None,
    ):
        self.router = router
        self.max_workers = max_workers
        self.allowed_providers = allowed_providers
        self.reinforcement_count = 0  # Track how many times ultrawork was triggered
        self.last_executed_at: float = 0.0

    def execute(
        self,
        task: str,
        parent_agent: AgentRunner,
        subtasks: list[str] | None = None,
    ) -> UltraworkResult:
        """Run subtasks in parallel.

        If subtasks not provided, uses the parent agent to
        decompose the task first.
        """
        import time as _time
        self.reinforcement_count += 1
        self.last_executed_at = _time.time()

        if not subtasks:
            subtasks = self._decompose(task, parent_agent)

        if not subtasks:
            # Can't decompose \u2014 run as single task
            logger.info("No subtasks found, running as single task")
            result = parent_agent.run(task)
            return UltraworkResult(
                completed=result.completed,
                summary=result.final_response,
                total_tokens=result.tokens_used,
            )

        original_subtasks = list(subtasks)
        deduplicated_subtasks, subtask_map = self._deduplicate_subtasks(subtasks)

        logger.info("Ultrawork: executing %d subtasks (deduplicated from %d) in parallel", len(deduplicated_subtasks), len(original_subtasks))

        # Route each subtask and build delegation tasks
        delegation_tasks = []
        provider_override = None
        for st in deduplicated_subtasks:
            task_entry: dict[str, Any] = {
                "goal": st,
                "context": f"This is a subtask of: {task}",
                "max_iterations": 30,
            }
            # Use router to determine tier and provider for this subtask
            if self.router:
                parent_tier = getattr(parent_agent, "tier_used", "STANDARD") or "STANDARD"
                decision = self.router.route_delegation(
                    parent_tier=parent_tier,
                    child_task=st,
                    allowed_providers=self.allowed_providers,
                )
                if decision and hasattr(decision, 'tier'):
                    task_entry["_tier"] = decision.tier.value
                    logger.debug("Subtask '%s' routed to %s tier via %s",
                                 st[:40], decision.tier.value, decision.provider)
                    # Resolve provider override if different from parent
                    if decision.provider != parent_agent.provider.name:
                        try:
                            from mullet.core.router.resolver import ProviderResolver
                            resolver = ProviderResolver(self.router)
                            resolved = resolver.resolve(decision)
                            if resolved:
                                task_entry["_provider_override"] = resolved
                        except Exception as e:
                            logger.debug("Provider resolution failed for subtask: %s", e)
            delegation_tasks.append(task_entry)

        # Execute in parallel via delegation
        # Per-task provider overrides stay in each task dict; delegate_batch
        # reads them individually so heterogeneous sub-tasks route correctly.
        results = delegate_batch(
            tasks=delegation_tasks,
            parent=parent_agent,
            provider_override=None,
            allowed_providers=self.allowed_providers,
        )

        # Merge results
        return self._merge_results(results, original_subtasks, subtask_map)

    def _decompose(self, task: str, agent: AgentRunner) -> list[str]:
        """Use the agent to decompose a task into subtasks.

        Costs one LLM call on the parent's tier.
        """
        decompose_prompt = (
            f"Decompose this task into independent subtasks that can be executed in parallel. "
            f"Return ONLY a numbered list of subtasks, nothing else.\n\n"
            f"Task: {task}"
        )

        result = agent.run(decompose_prompt)
        if not result.completed:
            return []

        # Parse numbered list from response
        subtasks = []
        for line in result.final_response.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):\
                # Strip numbering
                cleaned = line.lstrip("0123456789.-) ").strip()
                if cleaned:
                    subtasks.append(cleaned)

        return subtasks[:self.max_workers * 2]  # Cap at 2x workers

    def _deduplicate_subtasks(self, subtasks: list[str]) -> tuple[list[str], dict[str, str]]:
        """Deduplicate subtasks and return a map from original to deduped subtasks."""
        deduplicated = []
        subtask_map = {}  # Map original subtask to its deduped version
        seen = set()

        for st in subtasks:
            cleaned_st = st.strip().lower() # Simple normalization
            if cleaned_st not in seen:
                deduplicated.append(st)
                seen.add(cleaned_st)
            # Map original subtask to the version that was actually run
            # If 'st' is a duplicate, it maps to the first instance of itself.
            # If 'st' is unique, it maps to itself.
            # This is to handle cases where the original subtask might not be exactly
            # the same as the deduped one if there were subtle differences in casing/whitespace
            # but they were still considered duplicates.
            # For now, we'll just map to the actual string used if it was unique,
            # or to the first instance of the duplicate if it was a duplicate.
            # A more robust solution might involve canonicalizing the subtask strings.
            subtask_map[st] = next((s for s in deduplicated if s.strip().lower() == cleaned_st), st)

        return deduplicated, subtask_map

    def _merge_results(self, results: list[DelegationResult], original_subtasks: list[str], subtask_map: dict[str, str]) -> UltraworkResult:
        """Merge delegation results into a unified UltraworkResult, accounting for deduplication."""
        total_tokens = TokenUsage()
        tiers: dict[str, int] = {}
        
        # Map of deduped subtask string to its DelegationResult
        deduped_results_map = {r.goal: r for r in results}

        # Initialize progress counters
        completed_subtasks = 0
        failed_subtasks = 0
        summaries = []

        for i, original_st in enumerate(original_subtasks):
            deduped_st = subtask_map.get(original_st, original_st) # Get the deduped version
            r = deduped_results_map.get(deduped_st) # Get the result for the deduped task

            status = "PENDING"
            summary_text = f"Subtask '{original_st[:100]}...'"

            if r:
                if r.completed:
                    status = "completed"
                    completed_subtasks += 1
                else:
                    status = "FAILED"
                    failed_subtasks += 1
                
                summary_text = f"Subtask {i+1} ({status}): {r.summary[:200]}"
                if r.agent_result:
                    total_tokens = total_tokens + r.agent_result.tokens_used
                    tier = r.agent_result.tier_used or "unknown"
                    tiers[tier] = tiers.get(tier, 0) + 1
            else:
                summary_text = f"Subtask {i+1} (DEDUPLICATED): Original subtask '{original_st[:100]}...' was a duplicate and its result is merged from an equivalent subtask."
            
            summaries.append(summary_text)

        overall_status = "All subtasks completed successfully." if completed_subtasks == len(original_subtasks) else f"{completed_subtasks} out of {len(original_subtasks)} subtasks completed."
        if failed_subtasks > 0:
            overall_status += f" ({failed_subtasks} failed)."

        summary_parts = [overall_status]
        summary_parts.extend(summaries)

        return UltraworkResult(
            completed=(failed_subtasks == 0),
            subtask_results=results, # This will still contain only the results of the *deduplicated* tasks
            total_tokens=total_tokens,
            tiers_used=tiers,
            total_subtasks=len(original_subtasks),
            completed_subtasks=completed_subtasks,
            failed_subtasks=failed_subtasks,
            summary="\n".join(summary_parts),
        )
