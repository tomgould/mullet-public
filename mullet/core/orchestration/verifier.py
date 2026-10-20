"""Verification protocol — verify work with fresh evidence.

Used by ralph for story verification and by autopilot for QA.
The verifier runs concrete checks (read files, run tests, check output)
rather than relying on the agent's self-assessment.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mullet.core.agent.types import TokenUsage

if TYPE_CHECKING:
    from mullet.core.agent.runner import AgentRunner

logger = logging.getLogger(__name__)


@dataclass
class CriterionResult:
    """Result of verifying a single acceptance criterion."""
    criterion: str
    passed: bool = False
    evidence: str = ""
    attempt: int = 1


@dataclass
class VerificationResult:
    """Result of a verification pass."""
    passed: bool = False
    criteria_results: dict[str, bool] = field(default_factory=dict)
    details: list[CriterionResult] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    tokens_used: TokenUsage = field(default_factory=TokenUsage)
    attempts: int = 1

    @property
    def summary(self) -> str:
        """Return a one-line ``STATUS: passed/total criteria met (attempt N)`` summary."""
        total = len(self.criteria_results)
        passed_count = sum(1 for v in self.criteria_results.values() if v)
        status = "PASS" if self.passed else "FAIL"
        return f"{status}: {passed_count}/{total} criteria met (attempt {self.attempts})"

    @property
    def failed_criteria(self) -> list[str]:
        """Return a list of criteria that did not pass."""
        return [k for k, v in self.criteria_results.items() if not v]


def extract_criteria(prd_text: str) -> list[str]:
    """Extract acceptance criteria from a PRD or task description.

    Looks for patterns like:
    - [ ] criterion text
    - criterion text
    1. criterion text
    AC: criterion text
    """
    criteria = []
    for line in prd_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Checkbox items
        m = re.match(r"^[-*]\s*\[[ x]?\]\s*(.+)$", line, re.IGNORECASE)
        if m:
            criteria.append(m.group(1).strip())
            continue
        # Numbered items
        m = re.match(r"^\d+[.)]\s*(.+)$", line)
        if m:
            criteria.append(m.group(1).strip())
            continue
        # AC: prefixed
        m = re.match(r"^(?:AC|Acceptance Criteria?)\s*[:]\s*(.+)$", line, re.IGNORECASE)
        if m:
            criteria.append(m.group(1).strip())
            continue
        # Bullet items (only if they look like criteria, not prose)
        m = re.match(r"^[-*]\s+(.+)$", line)
        if m and len(m.group(1).split()) >= 3:
            criteria.append(m.group(1).strip())
    return criteria


def _parse_criterion_result(response: str, index: int) -> tuple[bool, str]:
    """Parse PASS/FAIL and evidence for a specific criterion from agent response."""
    response_upper = response.upper()
    marker = f"CRITERION {index}:"
    if marker in response_upper:
        idx = response_upper.index(marker)
        section = response[idx:idx + 500]
        if "PASS" in section.upper():
            return True, section.strip()
        elif "FAIL" in section.upper():
            return False, section.strip()
    # Fallback: check for "ALL PASS"
    if "ALL" in response_upper and "PASS" in response_upper:
        return True, "ALL PASS (global)"
    return False, "Not explicitly verified"


def verify_criteria(
    criteria: list[str],
    context: str,
    agent: "AgentRunner",
    max_retries: int = 2,
) -> VerificationResult:
    """Verify acceptance criteria with fresh evidence, retrying failures.

    Args:
        criteria: List of acceptance criteria to verify.
        context: Description of what was implemented.
        agent: AgentRunner to execute verification checks.
        max_retries: Max attempts for failed criteria (1 = no retry).
    """
    all_details: list[CriterionResult] = []
    criteria_results: dict[str, bool] = {}
    all_evidence: list[str] = []
    all_issues: list[str] = []
    total_tokens = TokenUsage()
    remaining = list(enumerate(criteria, 1))

    for attempt in range(1, max_retries + 1):
        if not remaining:
            break

        criteria_text = "\n".join(f"  {i}. {criteria[i-1]}" for i, _ in remaining)
        prompt = (
            f"VERIFICATION MODE (attempt {attempt}/{max_retries}): "
            f"Check each criterion below with FRESH EVIDENCE.\n\n"
            f"Context: {context}\n\n"
            f"Criteria to verify:\n{criteria_text}\n\n"
            "For EACH criterion:\n"
            "1. Run a concrete check (read the file, run the test, check the output)\n"
            "2. Quote the evidence you found\n"
            "3. Mark it PASS or FAIL with a brief explanation\n\n"
            "Format each result as:\n"
            "CRITERION N: PASS|FAIL - [evidence]\n\n"
            "End with a summary line: ALL PASS or list failures."
        )

        result = agent.run(prompt)
        total_tokens = total_tokens + result.tokens_used

        # Parse results for each criterion in this batch
        still_failing = []
        for i, criterion in remaining:
            passed, evidence = _parse_criterion_result(result.final_response, i)
            detail = CriterionResult(
                criterion=criterion,
                passed=passed,
                evidence=evidence,
                attempt=attempt,
            )
            all_details.append(detail)

            if passed:
                criteria_results[criterion] = True
                all_evidence.append(f"Criterion {i}: PASS (attempt {attempt})")
            else:
                criteria_results[criterion] = False
                still_failing.append((i, criterion))
                if attempt == max_retries:
                    all_issues.append(f"Criterion {i}: {criterion}")

        remaining = still_failing
        if not remaining:
            break

    all_passed = all(criteria_results.values()) if criteria_results else False

    return VerificationResult(
        passed=all_passed,
        criteria_results=criteria_results,
        details=all_details,
        evidence=all_evidence,
        issues=all_issues,
        tokens_used=total_tokens,
        attempts=min(max_retries, max((d.attempt for d in all_details), default=1)),
    )


def verify_tests(agent: "AgentRunner") -> VerificationResult:
    """Run the project test suite and verify all pass."""
    result = agent.run(
        "Run the project's test suite. Use the terminal tool to execute:\n"
        "1. Find the test command (pytest, npm test, go test, cargo test, etc.)\n"
        "2. Run it\n"
        "3. Report: total tests, passed, failed, errors\n"
        "4. If tests fail, list the failing test names and errors\n\n"
        "End with: ALL TESTS PASS or list failures."
    )

    passed = "ALL TESTS PASS" in result.final_response.upper() or (
        "PASSED" in result.final_response.upper() and
        "FAILED" not in result.final_response.upper()
    )

    return VerificationResult(
        passed=passed,
        evidence=[result.final_response[:500]],
        issues=[] if passed else ["Test failures detected"],
        tokens_used=result.tokens_used,
    )
