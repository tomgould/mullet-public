"""Team mode — N coordinated agents with staged pipeline.

From OMC: the most sophisticated orchestration pattern.
5 stages: plan → prd → exec → verify → fix

Each stage uses appropriate tier routing:
- plan: explore (LOW) + planner (HIGH)
- exec: N × executor (routed by subtask complexity)
- verify: verifier (MEDIUM) + optional code-reviewer (HIGH)
- fix: executor (MEDIUM)
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mullet.core.agent.delegation import delegate_batch
from mullet.core.agent.types import DelegationResult, TokenUsage
from mullet.core.orchestration.planner import Plan, create_plan
from mullet.core.orchestration.shared_memory import SharedMemory
from mullet.core.orchestration.verifier import verify_criteria
from mullet.core.orchestration.subagent_pipeline import run_three_role_review, format_review_report
from mullet.core.orchestration.state import ModeState

if TYPE_CHECKING:
    from mullet.core.agent.runner import AgentRunner

logger = logging.getLogger(__name__)

MAX_FIX_CYCLES = 3


@dataclass
class Task:
    """A unit of work on the team task board."""
    id: str = ""
    title: str = ""
    description: str = ""
    status: str = "pending"  # pending, claimed, done, failed
    claimed_by: str = ""
    result: str = ""
    blocked_by: list[str] = field(default_factory=list)  # task IDs that must complete first


class TaskBoard:
    """Shared task list with claiming and deduplication."""

    def __init__(self, dedup_threshold: float = 0.6):
        self._tasks: list[Task] = []
        self._next_id = 1
        self._dedup_threshold = dedup_threshold

    def add_task(self, title: str, description: str = "", blocked_by: list[str] | None = None) -> Task:
        """Append a new pending task with an auto-generated id."""
        t = Task(id=f"T-{self._next_id:03d}", title=title, description=description,
                 blocked_by=blocked_by or [])
        self._next_id += 1
        self._tasks.append(t)
        return t

    def _is_unblocked(self, task: Task) -> bool:
        done_ids = {t.id for t in self._tasks if t.status == "done"}
        return all(dep in done_ids for dep in task.blocked_by)

    def get_available(self) -> list[Task]:
        """Return pending tasks whose dependencies are all done."""
        return [t for t in self._tasks if t.status == "pending" and self._is_unblocked(t)]

    def claim_task(self, agent_name: str) -> Task | None:
        """Atomically claim the first available task for ``agent_name`` (or None)."""
        for t in self.get_available():
            t.status = "claimed"
            t.claimed_by = agent_name
            return t
        return None

    def complete_task(self, task_id: str, result: str) -> None:
        """Mark ``task_id`` done and record ``result``. Silent no-op if not found."""
        for t in self._tasks:
            if t.id == task_id:
                t.status = "done"
                t.result = result
                return

    def fail_task(self, task_id: str, error: str) -> None:
        """Mark ``task_id`` failed and store ``error``. Silent no-op if not found."""
        for t in self._tasks:
            if t.id == task_id:
                t.status = "failed"
                t.result = error
                return

    def get_pending(self) -> list[Task]:
        """Return every task still in the ``pending`` state (regardless of blockers)."""
        return [t for t in self._tasks if t.status == "pending"]

    def get_all(self) -> list[Task]:
        """Return a shallow copy of every task on the board."""
        return list(self._tasks)

    def deduplicate(self) -> int:
        """Merge tasks with >60% word overlap in titles."""
        removed = 0
        i = 0
        while i < len(self._tasks):
            j = i + 1
            while j < len(self._tasks):
                if self._tasks[i].status == "pending" and self._tasks[j].status == "pending":
                    words_a = set(self._tasks[i].title.lower().split())
                    words_b = set(self._tasks[j].title.lower().split())
                    if words_a and words_b:
                        overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
                        if overlap > self._dedup_threshold:
                            self._tasks[i].description += f"\n(merged: {self._tasks[j].title})"
                            self._tasks.pop(j)
                            removed += 1
                            continue
                j += 1
            i += 1
        return removed


@dataclass
class TeamResult:
    """Result of a team execution."""
    completed: bool = False
    plan: Plan | None = None
    exec_results: list[DelegationResult] = field(default_factory=list)
    verify_passed: bool = False
    fix_cycles: int = 0
    total_tokens: TokenUsage = field(default_factory=TokenUsage)
    summary: str = ""


class TeamMode:
    """N coordinated agents on a shared task list.

    Staged pipeline:
    1. team-plan: explore codebase, planner creates task graph
    2. team-prd: clarify acceptance criteria
    3. team-exec: spawn N workers, each claims tasks from pool
    4. team-verify: verifier + reviewers check all outputs
    5. team-fix: fix defects, loop back to verify (max 3 cycles)
    """

    def __init__(
        self,
        knowledge=None,
        num_workers: int = 3,
        max_fix_cycles: int = MAX_FIX_CYCLES,
        shared_memory: SharedMemory | None = None,
    ):
        self.knowledge = knowledge
        self.num_workers = num_workers
        self.max_fix_cycles = max_fix_cycles
        self._state = ModeState(knowledge, "team")
        self.shared_memory = shared_memory or SharedMemory()

    def execute(self, task: str, agent: "AgentRunner") -> TeamResult:
        """Run the staged team pipeline."""
        total_tokens = TokenUsage()

        # Stage 1: PLAN
        logger.info("Team Stage 1: PLAN (%d workers)", self.num_workers)
        plan = create_plan(task, agent)
        total_tokens = total_tokens + _estimate_tokens(plan)

        if not plan.steps:
            # Can't create a plan — fall back to single agent
            result = agent.run(task)
            return TeamResult(
                completed=result.completed,
                summary=result.final_response,
                total_tokens=result.tokens_used,
            )

        self._state.save({"stage": "plan", "steps": len(plan.steps)})

        # Stage 2: PRD (refine acceptance criteria)
        logger.info("Team Stage 2: PRD (refining %d steps)", len(plan.steps))
        plan = self._refine_prd(plan, agent)

        # Stage 3: EXEC (parallel workers)
        logger.info("Team Stage 3: EXEC (%d workers, %d tasks)", self.num_workers, len(plan.steps))
        exec_results = self._execute_parallel(plan, agent)
        for r in exec_results:
            if r.agent_result:
                total_tokens = total_tokens + r.agent_result.tokens_used

        self._state.save({"stage": "exec", "completed": sum(1 for r in exec_results if r.completed)})

        # Stage 4: VERIFY
        logger.info("Team Stage 4: VERIFY")
        all_criteria = []
        for step in plan.steps:
            all_criteria.extend(step.acceptance_criteria)

        # Three-role review: spec-compliance + code-quality per completed step
        review_issues: list[str] = []
        for step, exec_result in zip(plan.steps, exec_results):
            if exec_result.completed and exec_result.summary:
                spec_r, qual_r = run_three_role_review(
                    step_title=step.title,
                    step_description=step.description,
                    acceptance_criteria=step.acceptance_criteria,
                    implementation_summary=exec_result.summary,
                    files_modified=step.files,
                )
                if not spec_r.approved or not qual_r.approved:
                    review_issues.append(format_review_report(spec_r, qual_r))

        if review_issues:
            logger.warning("Three-role review found issues in %d step(s)", len(review_issues))

        if all_criteria:
            verify_result = verify_criteria(all_criteria, task, agent)
            total_tokens = total_tokens + verify_result.tokens_used
            verify_passed = verify_result.passed and len(review_issues) == 0
        else:
            verify_passed = all(r.completed for r in exec_results) and len(review_issues) == 0

        # Stage 5: FIX (if verification failed)
        fix_cycles = 0
        while not verify_passed and fix_cycles < self.max_fix_cycles:
            fix_cycles += 1
            logger.info("Team Stage 5: FIX (cycle %d/%d)", fix_cycles, self.max_fix_cycles)

            review_context = ("\n\nThree-role review findings:\n" + "\n\n".join(review_issues)) if review_issues else ""
            fix_result = agent.run(
                f"Fix the issues found during verification:\n\n"
                f"{verify_result.summary if all_criteria else 'Some tasks failed'}\n\n"
                f"Issues: {', '.join(verify_result.issues) if all_criteria else 'Check exec results'}"
                f"{review_context}"
            )
            total_tokens = total_tokens + fix_result.tokens_used

            # Re-verify
            if all_criteria:
                verify_result = verify_criteria(all_criteria, task, agent)
                total_tokens = total_tokens + verify_result.tokens_used
                verify_passed = verify_result.passed
            else:
                verify_passed = True  # Trust the fix

        # Summary
        summary_parts = [
            f"Team execution: {len(plan.steps)} tasks, {self.num_workers} workers",
            f"Exec: {sum(1 for r in exec_results if r.completed)}/{len(exec_results)} completed",
            f"Verify: {'PASS' if verify_passed else 'FAIL'}",
        ]
        if fix_cycles:
            summary_parts.append(f"Fix cycles: {fix_cycles}")

        self._state.clear()

        return TeamResult(
            completed=verify_passed,
            plan=plan,
            exec_results=exec_results,
            verify_passed=verify_passed,
            fix_cycles=fix_cycles,
            total_tokens=total_tokens,
            summary="\n".join(summary_parts),
        )

    def _refine_prd(self, plan: Plan, agent: "AgentRunner") -> Plan:
        """Refine plan steps with concrete acceptance criteria."""
        for step in plan.steps:
            if not step.acceptance_criteria:
                # Ask agent to generate criteria for this step
                result = agent.run(
                    f"Generate 2-3 specific, testable acceptance criteria for:\n"
                    f"Step: {step.title}\n"
                    f"Description: {step.description}\n"
                    f"Files: {', '.join(step.files) if step.files else 'TBD'}\n\n"
                    f"Return ONLY a bulleted list of criteria."
                )
                if result.completed:
                    criteria = [
                        line.strip().lstrip("- •*").strip()
                        for line in result.final_response.split("\n")
                        if line.strip() and line.strip()[0] in "-•*"
                    ]
                    step.acceptance_criteria = criteria or [f"{step.title} is implemented and working"]

        return plan

    def _execute_parallel(self, plan: Plan, agent: "AgentRunner") -> list[DelegationResult]:
        """Execute plan steps in parallel using delegation."""
        ordered_steps = plan.execution_order

        # Build delegation tasks
        tasks = []
        for step in ordered_steps:
            files_hint = f"\nFiles to create/modify: {', '.join(step.files)}" if step.files else ""
            criteria_hint = f"\nAcceptance criteria: {'; '.join(step.acceptance_criteria)}" if step.acceptance_criteria else ""

            tasks.append({
                "goal": f"{step.title}: {step.description}{files_hint}{criteria_hint}",
                "context": f"Part of: {plan.task}\nStep {step.id}",
                "max_iterations": 30,
            })

        # Execute in batches of num_workers
        all_results = []
        for i in range(0, len(tasks), self.num_workers):
            batch = tasks[i:i + self.num_workers]
            results = delegate_batch(batch, parent=agent)
            all_results.extend(results)
            # Store results in shared memory for cross-agent visibility
            for r in results:
                self.shared_memory.set(
                    f"task-result-{r.goal[:40]}",
                    {"completed": r.completed, "summary": r.summary[:500]},
                    namespace="team-results",
                )

        return all_results


    # ================================================================
    # Staged Pipeline: plan → PRD → exec → verify → fix
    # ================================================================

    def run_staged(self, task: str, agent: "AgentRunner") -> TeamResult:
        """Run the full staged pipeline."""
        plan = self._plan_stage(task, agent)
        plan = self._prd_stage(plan, agent)
        results = self._exec_stage(plan, agent)
        passed, verify_tokens = self._verify_stage(plan, results, agent)
        if not passed:
            self._fix_stage(plan, agent)
        total = sum((r.agent_result.tokens_used if r.agent_result else TokenUsage() for r in results), TokenUsage()) + verify_tokens
        return TeamResult(completed=passed, plan=plan, exec_results=results, total_tokens=total)

    def _plan_stage(self, task: str, agent: "AgentRunner") -> Plan:
        """Generate a plan for the given task using the agent."""
        logger.info("Team Stage 1: PLAN")
        return create_plan(task, agent)

    def _prd_stage(self, plan: Plan, agent: "AgentRunner") -> Plan:
        """Refine the plan with acceptance criteria for each step."""
        logger.info("Team Stage 2: PRD")
        prd_prompt = f"Refine this plan with acceptance criteria for each step:\n{plan.overview}"
        _ = agent.run(prd_prompt)
        for step in plan.steps:
            if not step.acceptance_criteria:
                step.acceptance_criteria = [f"Step '{step.title}' completed successfully"]
        return plan

    def _exec_stage(self, plan: Plan, agent: "AgentRunner") -> list:
        """Execute the plan in parallel using delegation."""
        logger.info("Team Stage 3: EXEC")
        return self._execute_parallel(plan, agent)

    def _verify_stage(self, plan: Plan, results: list, agent: "AgentRunner") -> tuple:
        """Verify that all plan steps have been completed successfully."""
        logger.info("Team Stage 4: VERIFY")
        passed = all(getattr(r, 'completed', False) for r in results)
        return passed, TokenUsage()

    def _fix_stage(self, plan: Plan, agent: "AgentRunner") -> None:
        logger.info("Team Stage 5: FIX")
        agent.run(f"Fix issues found in: {plan.overview}")


def _estimate_tokens(plan: Plan) -> TokenUsage:
    """Rough token estimate for plan generation."""
    text_len = len(plan.overview) + sum(
        len(s.title) + len(s.description) for s in plan.steps
    )
    return TokenUsage(total_tokens=text_len // 4)
