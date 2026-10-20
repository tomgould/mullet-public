"""Ralph mode — persistence loop with PRD-driven verification.

Keeps working until ALL user stories in the PRD pass their acceptance
criteria, verified with fresh evidence. Wraps ultrawork for parallel
execution and adds structured story tracking.

From OMC: ralph = ultrawork + "don't stop until verified"
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from mullet.core.agent.types import AgentResult, TokenUsage
from mullet.core.orchestration.state import ModeState

if TYPE_CHECKING:
    from mullet.core.agent.runner import AgentRunner
    from mullet.core.knowledge.manager import KnowledgeManager

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10
MAX_RETRIES = 3
RETRY_DELAY = 5 # seconds


def _git_head_sha() -> str | None:
    """Return current HEAD commit SHA (short), or None if git unavailable."""
    try:
        import subprocess
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _git_commits_since(sha: str | None) -> list[str]:
    """Return commit messages made since *sha* (exclusive), newest first."""
    if not sha:
        return []
    try:
        import subprocess
        r = subprocess.run(
            ["git", "log", f"{sha}..HEAD", "--oneline", "--no-merges"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0:
            return []
        return [line.strip() for line in r.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def _check_tdd_gates(story: "UserStory", commits: list[str]) -> list[str]:
    """Check commits for RED→GREEN→REFACTOR TDD sequence.

    Returns list of advisory violation strings (empty = no violations).
    Violations are advisory — they do not fail the story.
    """
    if not commits:
        return []

    import re
    violations = []

    # Look for test-first marker (test/fix/refactor commit before feat commit)
    has_test_commit = any(
        re.search(r"^[0-9a-f]+\s+(test|fix|spec|tdd)\b", c, re.IGNORECASE)
        for c in commits
    )
    has_feat_commit = any(
        re.search(r"^[0-9a-f]+\s+(feat|add|implement|create)\b", c, re.IGNORECASE)
        for c in commits
    )

    if has_feat_commit and not has_test_commit:
        violations.append(
            f"TDD advisory [{story.id}]: implementation commits found but no test commit "
            "— consider writing the failing test first (RED phase)"
        )

    # Check commit order: test commits should precede feat commits.
    # commits list is newest-first: index 0 = most recent, higher index = older.
    # Correct TDD: test committed first (older = higher index) before feat (newer = lower index).
    # Violation: test has lower index than feat → test was committed AFTER feat.
    test_indices = [
        i for i, c in enumerate(commits)
        if re.search(r"^[0-9a-f]+\s+(test|fix|spec|tdd)\b", c, re.IGNORECASE)
    ]
    feat_indices = [
        i for i, c in enumerate(commits)
        if re.search(r"^[0-9a-f]+\s+(feat|add|implement|create)\b", c, re.IGNORECASE)
    ]
    if test_indices and feat_indices:
        if min(test_indices) < min(feat_indices):
            violations.append(
                f"TDD advisory [{story.id}]: feat committed before test "
                "— test commit should precede implementation (RED before GREEN)"
            )

    return violations


def _extract_learnings(response: str) -> list[str]:
    """Extract learnings from an implementation response.

    Looks for patterns like "learned that", "discovered", "note:",
    "important:", or bullet points with insights.
    """
    if not response:
        return []

    import re
    learnings = []

    # Look for explicit learning markers
    patterns = [
        r"(?:learned|discovered|found|noticed|realized)\s+(?:that\s+)?(.{20,100})",
        r"(?:note|important|insight|takeaway):\s*(.{20,100})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, response, re.IGNORECASE):
            learnings.append(match.group(1).strip().rstrip("."))

    return learnings[:5]  # Cap at 5 per story


@dataclass
class StoryTask:
    """A sub-task within a user story (ralphthon pattern)."""
    id: str
    title: str
    description: str = ""
    completed: bool = False
    result: str = ""


@dataclass
class UserStory:
    """A single user story in the PRD.

    From OMC ralphthon: stories can contain nested tasks for
    finer-grained tracking within a story.
    """
    id: str
    title: str
    description: str = ""
    acceptance_criteria: list[str] = field(default_factory=list)
    priority: int = 1
    passes: bool = False
    tasks: list[StoryTask] = field(default_factory=list)
    notes: str = ""


@dataclass
class PRD:
    """Product Requirements Document with tracked stories."""
    task: str
    project: str = ""
    branchName: str = ""
    stories: list[UserStory] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize the PRD to a plain dictionary."""
        return {
            "task": self.task,
            "project": self.project,
            "branchName": self.branchName,
            "stories": [
                {
                    "id": s.id,
                    "title": s.title,
                    "description": s.description,
                    "acceptance_criteria": s.acceptance_criteria,
                    "priority": s.priority,
                    "passes": s.passes,
                    "tasks": [
                        {"id": t.id, "title": t.title, "description": t.description,
                         "completed": t.completed, "result": t.result}
                        for t in s.tasks
                    ],
                    "notes": s.notes,
                }
                for s in self.stories
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PRD":
        """Deserialize a PRD from a plain dictionary."""
        if not isinstance(data, dict):
            raise TypeError(f"PRD.from_dict expects a dict, got {type(data).__name__}")
        stories = []
        for s in data.get("stories", []):
            if "id" not in s or "title" not in s:
                raise KeyError(f"UserStory requires 'id' and 'title' keys, got {sorted(s.keys())}")
            tasks = []
            for t in s.get("tasks", []):
                if "id" not in t or "title" not in t:
                    raise KeyError(f"StoryTask requires 'id' and 'title' keys, got {sorted(t.keys())}")
                tasks.append(StoryTask(
                    id=t["id"], title=t["title"],
                    description=t.get("description", ""),
                    completed=t.get("completed", False),
                    result=t.get("result", ""),
                ))
            stories.append(UserStory(
                id=s["id"],
                title=s["title"],
                description=s.get("description", ""),
                acceptance_criteria=s.get("acceptance_criteria", []),
                priority=s.get("priority", 1),
                passes=s.get("passes", False),
                tasks=tasks,
                notes=s.get("notes", ""),
            ))
        # Get project/branch from data or detect from git
        project = data.get("project", "")
        branch = data.get("branchName", "")
        if not project or not branch:
            try:
                import subprocess
                if not project:
                    r = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                                       capture_output=True, text=True, timeout=5)
                    project = r.stdout.strip().split("/")[-1] if r.returncode == 0 else "unknown"
                if not branch:
                    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                       capture_output=True, text=True, timeout=5)
                    branch = r.stdout.strip() if r.returncode == 0 else "unknown"
            except Exception:
                project = project or "unknown"
                branch = branch or "unknown"

        return cls(task=data.get("task", ""), project=project,
                   branchName=branch, stories=stories)

    @property
    def all_pass(self) -> bool:
        """Return True when every story passes its acceptance criteria."""
        return all(s.passes for s in self.stories)

    @property
    def next_story(self) -> UserStory | None:
        """Return the highest-priority story that has not yet passed."""
        for s in sorted(self.stories, key=lambda x: x.priority):
            if not s.passes:
                return s
        return None

    @property
    def progress(self) -> tuple[int, int]:
        """Return the count of passed stories and total stories."""
        done = sum(1 for s in self.stories if s.passes)
        return done, len(self.stories)


@dataclass
class VerificationState:
    """State of the architect verification phase."""
    pending: bool = False
    completion_claim: str = ""
    verification_attempts: int = 0
    max_verification_attempts: int = 3
    architect_feedback: Optional[str] = None
    architect_approved: Optional[bool] = None
    requested_at: Optional[str] = None
    original_task: str = ""
    critic_mode: str = "architect" # e.g., "architect", "critic", "codex"


@dataclass
class RalphResult:
    """Result of a ralph execution."""
    completed: bool = False
    stories_completed: int = 0
    stories_total: int = 0
    iterations_used: int = 0
    total_tokens: TokenUsage = field(default_factory=TokenUsage)
    summary: str = ""
    verification_state: VerificationState = field(default_factory=VerificationState)


@dataclass
class ProgressEntry:
    """A single progress log entry (from OMC ralph progress logging)."""
    iteration: int
    story_id: str
    action: str  # "implement", "verify", "pass", "fail"
    summary: str
    learnings: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    timestamp: float = field(default_factory=time.time)


class RalphMode:
    """Persistence loop with PRD verification.

    Flow:
    1. Generate PRD with stories and acceptance criteria
    2. Pick next incomplete story
    3. Implement the story (with prior learnings context)
    4. Verify each acceptance criterion
    5. Mark story complete if all pass
    6. Log progress with learnings extraction
    7. Loop until all stories pass
    8. Final architect verification
    """

    def __init__(
        self,
        knowledge: "KnowledgeManager | None" = None,
        max_iterations: int = MAX_ITERATIONS,
        critic_mode: str = "architect",  # architect, critic, codex
        agent_profile: str = "default",  # Default agent profile for implementation
        max_tokens: int = 4000,  # Default max tokens for agent calls
    ):
        """Initialize Ralph mode with optional knowledge and iteration limits."""
        self.knowledge = knowledge
        self.max_iterations = max_iterations
        self.critic_mode = critic_mode
        self._state = ModeState(knowledge, "ralph")
        self.total_tokens_used: TokenUsage = TokenUsage()
        self._progress: list[ProgressEntry] = []
        self.agent_profile = agent_profile
        self.max_tokens = max_tokens
        # Captured at execute() time from the agent's allowed_providers so
        # --resume can rehydrate the user's --providers envelope (G25).
        self._allowed_providers: frozenset[str] | None = None

    def _persist_state(self, state: dict) -> None:
        """Write *state* via ModeState, always tagging the active allowlist.

        The allowlist key is included even when None so consumers can
        distinguish 'unrestricted' from 'pre-G25 saved state with no key'.
        """
        state = dict(state)
        state["allowed_providers"] = (
            sorted(self._allowed_providers) if self._allowed_providers else None
        )
        self._state.save(state)

    def execute(self, task: str, agent: "AgentRunner") -> RalphResult:
        """Execute until all PRD stories pass."""
        # Snapshot the agent's allowlist so save() can persist it for --resume (G25).
        self._allowed_providers = getattr(agent, "allowed_providers", None)
        # Check for resumable state
        saved = self._state.load()
        if saved and saved.get("prd"):
            prd = PRD.from_dict(saved["prd"])
            # Restore task from state so --resume works without re-providing the task
            if not task and saved.get("task"):
                task = saved["task"]
            logger.info("Resuming ralph: %d/%d stories complete", *prd.progress)
        else:
            prd = self._generate_prd(task, agent)
            if not prd.stories:
                # Couldn't decompose — run as single task
                result = agent.run(task)
                return RalphResult(
                    completed=result.completed,
                    summary=result.final_response,
                    total_tokens=result.tokens_used,
                )

        total_tokens = TokenUsage()
        iteration = 0

        while not prd.all_pass and iteration < self.max_iterations:
            story = prd.next_story
            if not story:
                break

            iteration += 1
            done, total = prd.progress
            logger.info("Ralph iteration %d: story %s (%d/%d complete)",
                        iteration, story.id, done, total)

            # Reset agent iteration budget for each story (prevents exhaustion across stories)
            from mullet.core.agent.iteration_budget import IterationBudget
            agent.budget = IterationBudget(agent.budget.max_total)

            # Capture HEAD before implementation to detect new commits (TDD gate)
            _head_before = _git_head_sha()

            # Implement the story (with context from previous stories + learnings)
            completed = [s for s in prd.stories if s.passes]
            start_time = time.time()
            impl_result = self._implement_story(
                story, task, agent,
                agent_profile=getattr(self, 'agent_profile', 'default'),
                max_tokens=getattr(agent, 'budget', None) and agent.budget.max_total or 4096,
                completed_stories=completed,
                prior_learnings=self._get_learnings(),
            )
            impl_duration = time.time() - start_time
            total_tokens = total_tokens + impl_result.tokens_used

            # Log progress
            self._progress.append(ProgressEntry(
                iteration=iteration,
                story_id=story.id,
                action="implement",
                summary=impl_result.final_response[:200] if impl_result.final_response else "",
                duration_s=impl_duration,
            ))

            # Verify acceptance criteria using structured verifier
            from mullet.core.orchestration.verifier import verify_criteria
            if story.acceptance_criteria:
                verify_result = verify_criteria(
                    criteria=story.acceptance_criteria,
                    context=f"Story {story.id}: {story.title}",
                    agent=agent,
                )
                total_tokens = total_tokens + verify_result.tokens_used
                if verify_result.passed:
                    story.passes = True
                    # TDD gate: advisory check on commit sequence
                    _new_commits = _git_commits_since(_head_before)
                    _tdd_violations = _check_tdd_gates(story, _new_commits)
                    if _tdd_violations:
                        for v in _tdd_violations:
                            logger.warning(v)
                        tdd_note = " | ".join(_tdd_violations)
                        story.notes = f"{verify_result.summary} [TDD: {tdd_note}]"
                    else:
                        story.notes = verify_result.summary
                    logger.info("Story %s: PASSED (%s)", story.id, story.notes)
                    self._progress.append(ProgressEntry(
                        iteration=iteration, story_id=story.id,
                        action="pass", summary=story.notes,
                        learnings=_extract_learnings(impl_result.final_response),
                    ))
                else:
                    issues = "; ".join(verify_result.issues)
                    logger.info("Story %s: FAILED — %s", story.id, issues)
                    self._progress.append(ProgressEntry(
                        iteration=iteration, story_id=story.id,
                        action="fail", summary=issues,
                        learnings=[f"Failed: {issues}"],
                    ))
            else:
                # No criteria — trust implementation result
                story.passes = True
                logger.info("Story %s: PASSED (no criteria to verify)", story.id)

            # Persist state after each story — include task for resume
            self._persist_state({"prd": prd.to_dict(), "iteration": iteration, "task": task})

        done, total = prd.progress
        completed = prd.all_pass

        # Collect TDD advisory violations from all story notes for summary
        tdd_violations_total = [
            s.notes for s in prd.stories
            if s.notes and "[TDD:" in s.notes
        ]

        # Final verification if all stories pass
        if completed:
            # Architect verification phase
            verification_result = self._architect_verify(prd, agent, total_tokens)
            total_tokens = total_tokens + verification_result.total_tokens
            completed = verification_result.completed
            summary_text = verification_result.summary
            verification_state = verification_result.verification_state
        else:
            summary_text = f"Ralph incomplete: {done}/{total} stories passed after {iteration} iterations."
            verification_state = VerificationState(original_task=task, pending=True, completion_claim=summary_text)

        # Append TDD advisory summary if any violations were recorded
        if tdd_violations_total:
            summary_text += (
                f"\n\n⚠ TDD advisory: {len(tdd_violations_total)} stor"
                f"{'y' if len(tdd_violations_total) == 1 else 'ies'} lacked "
                "test-first commits. See story notes for details."
            )

        # Clean up state
        if completed:
            self._state.clear()

        return RalphResult(
            completed=completed,
            stories_completed=done,
            stories_total=total,
            iterations_used=iteration,
            total_tokens=total_tokens,
            summary=summary_text,
            verification_state=verification_state,
        )

    def _generate_prd(self, task: str, agent: "AgentRunner") -> PRD:
        """Use the agent to generate a PRD with stories and acceptance criteria."""
        prompt = (
            "You are generating a PRD (Product Requirements Document) for a task. "
            "Break the task into discrete user stories. For each story, define:\n"
            "- id: US-001, US-002, etc.\n"
            "- title: short name\n"
            "- description: what to implement\n"
            "- acceptance_criteria: specific, testable criteria (list)\n"
            "- priority: 1 (highest) to N\n"
            "- tasks: sub-tasks within the story (list of {id, title, description})\n\n"
            "Return ONLY valid JSON with this structure:\n"
            '{"stories": [{"id": "US-001", "title": "...", "description": "...", '
            '"acceptance_criteria": ["criterion 1", "criterion 2"], "priority": 1, '
            '"tasks": [{"id": "T-001", "title": "...", "description": "..."}]}]}\n\n'
            f"Task: {task}"
        )

        result = agent.run(prompt)
        if not result.completed:
            return PRD(task=task)

        # Try to parse JSON from response
        prd = self._try_parse_prd(result.final_response, task)
        if prd is not None:
            return prd

        # One retry with explicit correction instruction
        logger.warning("Retrying PRD generation after parse failure")
        retry_prompt = (
            f"{prompt}\n\n"
            f"Your previous response was not valid JSON. "
            f"Return ONLY the JSON object, no prose before or after, no markdown fences."
        )
        result = agent.run(retry_prompt)
        if not result.completed:
            return PRD(task=task)

        prd = self._try_parse_prd(result.final_response, task)
        if prd is not None:
            return prd

        return PRD(task=task)

    def _try_parse_prd(self, text: str, task: str) -> PRD | None:
        """Extract JSON from *text* and return a PRD, or None on failure."""
        try:
            data = self._extract_json(text)
            if data is not None:
                prd = PRD.from_dict(data)
                prd.task = task
                return prd
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to parse PRD: %s", e)
        return None

    def _extract_json(self, text: str) -> dict | None:
        """Extract the first JSON object from *text*, handling markdown fences."""
        if not text:
            return None

        # Strip markdown code fences if present
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            # Drop opening fence (may include language tag like ```json)
            lines = lines[1:]
            # Drop closing fence if present
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines)

        start = stripped.find("{")
        if start < 0:
            return None

        data, _ = json.JSONDecoder().raw_decode(stripped, start)
        if isinstance(data, dict):
            return data
        return None

    def _get_learnings(self) -> list[str]:
        """Extract accumulated learnings from progress entries."""
        learnings = []
        for entry in self._progress:
            learnings.extend(entry.learnings)
        return learnings[-10:]  # Keep last 10 learnings

    def _implement_story(self, story: UserStory, task: str, agent: "AgentRunner",
                         agent_profile: str, max_tokens: int,
                         completed_stories: list[UserStory] | None = None,
                         prior_learnings: list[str] | None = None) -> AgentResult:
        """Implement a single story with context from previous stories."""
        criteria_text = "\n".join(f"  - {c}" for c in story.acceptance_criteria)

        # Build context from previously completed stories
        context_parts = []
        if completed_stories:
            context_parts.append("## Previously completed stories:")
            for cs in completed_stories[-3:]:  # Last 3 to keep context lean
                context_parts.append(f"  - {cs.id}: {cs.title}")
                if cs.notes:
                    context_parts.append(f"    Notes: {cs.notes}")
            context_parts.append("")

        # Include nested tasks if defined
        tasks_text = ""
        if story.tasks:
            task_lines = [f"  {t.id}. {t.title}: {t.description}" for t in story.tasks if not t.completed]
            if task_lines:
                tasks_text = "\nSub-tasks:\n" + "\n".join(task_lines) + "\n"

        # Inject learnings from prior iterations
        if prior_learnings:
            context_parts.append("## Learnings from previous iterations:")
            for learning in prior_learnings[-5:]:
                context_parts.append(f"  - {learning}")
            context_parts.append("")

        context = "\n".join(context_parts) if context_parts else ""

        prompt = (
            f"Implement this user story as part of the task: {task}\n\n"
            f"{context}"
            f"Story: {story.title}\n"
            f"Description: {story.description}\n"
            f"Acceptance Criteria:\n{criteria_text}\n"
            f"{tasks_text}\n"
            "Complete the implementation. Use tools to read, write, and modify files. "
            "Make sure all acceptance criteria will be met."
        )
        result = agent.run(prompt)

        # Mark all tasks as completed on successful implementation
        if result.completed:
            for t in story.tasks:
                t.completed = True
                t.result = "Completed during story implementation"

        return result

    def _verify_story(self, story: UserStory, agent: "AgentRunner") -> AgentResult:
        """Verify a story's acceptance criteria with fresh evidence."""
        criteria_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(story.acceptance_criteria))
        prompt = (
            f"Verify each acceptance criterion for story '{story.title}' with FRESH EVIDENCE.\n\n"
            f"Criteria:\n{criteria_text}\n\n"
            "For each criterion:\n"
            "1. Run a concrete check (read the file, run the test, check the output)\n"
            "2. Report PASS or FAIL with evidence\n\n"
            "At the end, state clearly: ALL CRITERIA PASS or list which failed."
        )
        return agent.run(prompt)

    def _architect_verify(self, prd: PRD, agent: "AgentRunner", initial_tokens: TokenUsage) -> RalphResult:
        """Triggers an architect verification phase for the completed work."""
        verification_state = VerificationState(original_task=prd.task)
        total_tokens = initial_tokens

        while verification_state.verification_attempts < verification_state.max_verification_attempts:
            verification_state.pending = True
            verification_state.verification_attempts += 1
            verification_state.requested_at = time.strftime("%Y-%m-%d %H:%M:%S")
            
            stories_text = "\n".join(
                f"  - {s.id}: {s.title} (PASS)" for s in prd.stories
            )

            prompt = (
                f"All stories for '{prd.task}' have passed verification. Please act as an Architect "
                f"and review the overall task completion. You have {verification_state.max_verification_attempts - verification_state.verification_attempts} attempts remaining.\n\n"
                f"Completed stories:\n{stories_text}\n\n"
                "Review the implemented solution and provide feedback. If the solution is satisfactory, "
                "respond with 'APPROVAL'. If there are issues, describe them and the Ralph mode will continue to address them."
            )

            review_result = agent.run(prompt)
            total_tokens += review_result.tokens_used
            verification_state.completion_claim = review_result.final_response

            if "APPROVAL" in review_result.final_response.upper():
                verification_state.architect_approved = True
                verification_state.pending = False
                verification_state.architect_feedback = "Approved by Architect."
                logger.info("Architect review: APPROVED")
                break
            else:
                verification_state.architect_approved = False
                verification_state.architect_feedback = review_result.final_response
                logger.warning("Architect review: ISSUES FOUND. Ralph will continue to address: %s", review_result.final_response)
                # Save state so Ralph can continue with feedback
                self._persist_state({"prd": prd.to_dict(), "verification_state": verification_state.__dict__})
                return RalphResult(
                    completed=False,
                    stories_completed=sum(1 for s in prd.stories if s.passes),
                    stories_total=len(prd.stories),
                    iterations_used=verification_state.verification_attempts,
                    total_tokens=total_tokens,
                    summary=f"Architect found issues: {review_result.final_response}",
                    verification_state=verification_state,
                )
        
        # If loop completes without approval or max attempts reached
        if not verification_state.architect_approved:
            verification_state.architect_approved = False
            verification_state.pending = False
            verification_state.architect_feedback = "Max verification attempts reached without approval."
            logger.warning("Max architect verification attempts reached without approval.")

        return RalphResult(
            completed=verification_state.architect_approved,
            stories_completed=sum(1 for s in prd.stories if s.passes),
            stories_total=len(prd.stories),
            iterations_used=verification_state.verification_attempts,
            total_tokens=total_tokens,
            summary=verification_state.architect_feedback,
            verification_state=verification_state,
        )

    def _final_summary(self, prd: PRD, agent: "AgentRunner") -> AgentResult:
        """Generate a final summary of all completed work."""
        stories_text = "\n".join(
            f"  - {s.id}: {s.title} (PASS)" for s in prd.stories
        )
        prompt = (
            f"All stories for '{prd.task}' have passed verification.\n\n"
            f"Completed stories:\n{stories_text}\n\n"
            "Provide a concise summary of what was implemented and any important notes."
        )
        return agent.run(prompt)
