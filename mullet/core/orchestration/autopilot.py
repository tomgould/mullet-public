"""Autopilot mode — full lifecycle: plan → execute → verify.

From OMC: autopilot includes ralph (persistence) + ultraqa (QA cycling).
It is the highest-level orchestration mode.

Lifecycle:
1. PLAN: Analyze task, produce architecture plan with acceptance criteria
2. EXECUTE: Implement via ralph mode (persistence + verification per story)
3. QA: Run tests, verify all acceptance criteria with fresh evidence
4. COMMIT: If all passes, produce final summary
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mullet.core.agent.types import AgentResult, TokenUsage
from mullet.core.orchestration.ralph import RalphMode, RalphResult
from mullet.core.orchestration.state import ModeState
from mullet.core.orchestration.verification import (
    create_protocol, run_protocol,
)

if TYPE_CHECKING:
    from mullet.core.agent.runner import AgentRunner

logger = logging.getLogger(__name__)


@dataclass
class AutopilotResult:
    """Result of an autopilot execution."""
    completed: bool = False
    plan: str = ""
    ralph_result: RalphResult | None = None
    qa_passed: bool = False
    total_tokens: TokenUsage = field(default_factory=TokenUsage)
    summary: str = ""
    qa_feedback: str = ""
    qa_failure_count: int = 0


class AutopilotMode:
    """Full autonomous execution from idea to working code.

    Autopilot = Plan + Ralph (execute + verify) + QA
    """

    def __init__(self, knowledge=None, max_iterations: int = 15):
        self.knowledge = knowledge
        self.max_iterations = max_iterations
        self.qa_failure_count = 0
        self._state = ModeState(knowledge, "autopilot")
        # Captured at execute() time from the agent so --resume can rehydrate
        # the user's --providers envelope (G25).
        self._allowed_providers: frozenset[str] | None = None

    def _persist_state(self, state: dict) -> None:
        """Write *state* via ModeState, always tagging the active allowlist (G25)."""
        state = dict(state)
        state["allowed_providers"] = (
            sorted(self._allowed_providers) if self._allowed_providers else None
        )
        self._state.save(state)

    def execute(self, task: str, agent: AgentRunner) -> AutopilotResult:
        """Execute the full autopilot lifecycle, resuming from saved state if present."""
        # Snapshot the agent's allowlist so save() can persist it for --resume (G25).
        self._allowed_providers = getattr(agent, "allowed_providers", None)
        total_tokens = TokenUsage()

        # Check for saved state — resume from last completed phase if available
        saved = self._state.load()
        resumed_phase = saved.get("phase") if saved else None
        if resumed_phase:
            logger.info("Autopilot: resuming from saved phase=%s", resumed_phase)
            # Restore task from state if caller didn't provide one
            if not task and saved.get("task"):
                task = saved["task"]

        # Phase 1: PLAN (skip if already completed)
        if resumed_phase in ("execute", "qa"):
            logger.info("Autopilot Phase 1: PLAN — skipped (resuming from %s)", resumed_phase)
            plan = saved.get("plan", "")
            if not plan:
                logger.warning("Autopilot: no saved plan found, re-planning")
                resumed_phase = None
        if resumed_phase not in ("execute", "qa"):
            logger.info("Autopilot Phase 1: PLAN")
            plan_result = self._plan(task, agent)
            total_tokens = total_tokens + plan_result.tokens_used
            plan = plan_result.final_response

        self._persist_state({"phase": "plan", "task": task, "plan": plan[:2000]})

        # Phase 2: EXECUTE (via ralph)
        logger.info("Autopilot Phase 2: EXECUTE (ralph)")
        ralph = RalphMode(knowledge=self.knowledge, max_iterations=self.max_iterations)

        # Feed the plan into ralph as the task — it includes the architecture
        ralph_task = (
            f"Implement the following plan:\n\n{plan}\n\n"
            f"Original task: {task}"
        )
        ralph_result = ralph.execute(ralph_task, agent)
        total_tokens = total_tokens + ralph_result.total_tokens

        self._persist_state({
            "phase": "execute",
            "ralph_completed": ralph_result.completed,
            "stories": f"{ralph_result.stories_completed}/{ralph_result.stories_total}",
        })

        if not ralph_result.completed:
            return AutopilotResult(
                completed=False,
                plan=plan,
                ralph_result=ralph_result,
                total_tokens=total_tokens,
                summary=f"Autopilot incomplete: ralph finished {ralph_result.stories_completed}/{ralph_result.stories_total} stories.\n\n{ralph_result.summary}",
            )

        # Phase 3: QA (automated verification + agent review)
        logger.info("Autopilot Phase 3: QA")

        # Run automated verification checks first
        verification = self._run_automated_checks()
        if verification.passed:
            logger.info("Automated checks passed: %s", verification.summary)
        else:
            logger.warning("Automated checks failed:\n%s", verification.summary)

        qa_result = self._qa(task, plan, agent)
        total_tokens = total_tokens + qa_result.tokens_used
        qa_passed = verification.passed and "PASS" in qa_result.final_response.upper()

        qa_feedback = qa_result.final_response if not qa_passed else ""

        if not qa_passed:
            # QA failed — feed back to ralph for fixes
            logger.info("Autopilot: QA failed, running fix cycle")
            fix_task = (
                f"QA found issues. Fix them:\n\n{qa_result.final_response}\n\n"
                f"Original task: {task}"
            )
            fix_result = ralph.execute(fix_task, agent)
            total_tokens = total_tokens + fix_result.total_tokens
            qa_passed = fix_result.completed

        # Phase 4: Summary
        logger.info("Autopilot Phase 4: SUMMARY")
        summary_result = agent.run(
            f"Summarize what was accomplished for: {task}\n\n"
            f"Plan:\n{plan[:1000]}\n\n"
            f"Result: {'All QA passed' if qa_passed else 'QA issues remain'}"
        )
        total_tokens = total_tokens + summary_result.tokens_used

        self._state.clear()

        return AutopilotResult(
            completed=qa_passed,
            plan=plan,
            ralph_result=ralph_result,
            qa_passed=qa_passed,
            qa_feedback=qa_feedback,
            total_tokens=total_tokens,
            summary=summary_result.final_response,
        )

    def _run_automated_checks(self):
        """Run automated verification checks (build + test)."""
        checks = create_protocol(["build", "test"])
        checks[0].command = "python -c 'import mullet'"
        checks[1].command = "python -m pytest tests/ --ignore=tests/integration -q --tb=line -x"
        return run_protocol(checks, fail_fast=True)

    def _plan(self, task: str, agent: AgentRunner) -> AgentResult:
        """Use the agent to create an architecture plan."""
        prompt = (
            "You are in PLANNING mode. Analyze this task and produce a detailed plan.\n\n"
            f"Task: {task}\n\n"
            "Your plan should include:\n"
            "1. Architecture overview — what components are needed\n"
            "2. Implementation steps — ordered by dependency\n"
            "3. File list — what files will be created or modified\n"
            "4. Acceptance criteria — how to verify each step is complete\n"
            "5. Risk assessment — what could go wrong\n\n"
            "Use tools to examine the existing codebase before planning. "
            "Read relevant files, check the project structure."
        )
        return agent.run(prompt)

    def _qa(self, task: str, plan: str, agent: AgentRunner) -> AgentResult:
        """Run QA checks against the completed work."""
        prompt = (
            "You are in QA mode. Verify that the work is complete and correct.\n\n"
            f"Original task: {task}\n\n"
            "QA checks:\n"
            "1. Run all tests (if test files exist)\n"
            "2. Check that all planned files exist and are non-empty\n"
            "3. Verify key acceptance criteria from the plan\n"
            "4. Look for obvious bugs or missing imports\n\n"
            "Report PASS if everything checks out, or list specific failures."
        )
        return agent.run(prompt)
