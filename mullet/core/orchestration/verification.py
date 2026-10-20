"""Verification framework for orchestration modes.

Provides standardized checks (BUILD, TEST, LINT, FUNCTIONALITY) that
ralph, autopilot, ultrawork, and team can use for quality assurance.

Ported from OMC's verification module, adapted for Python.
"""

import shlex
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class CheckStatus(Enum):
    """Status enum for a verification check."""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class VerificationCheck:
    """A single verification check."""
    id: str
    name: str
    description: str
    required: bool = True
    command: str | None = None
    status: CheckStatus = CheckStatus.PENDING
    evidence: str = ""
    timestamp: float = 0.0
    duration_s: float = 0.0


@dataclass
class VerificationResult:
    """Result of running a verification protocol."""
    passed: bool
    checks: list[VerificationCheck]
    total_duration_s: float = 0.0
    summary: str = ""

    @property
    def passed_count(self) -> int:
        """Return the number of passed checks."""
        return sum(1 for c in self.checks if c.status == CheckStatus.PASSED)

    @property
    def failed_count(self) -> int:
        """Return the number of failed checks."""
        return sum(1 for c in self.checks if c.status == CheckStatus.FAILED)

    @property
    def required_failures(self) -> list[VerificationCheck]:
        """Return failed checks that are marked as required."""
        return [c for c in self.checks
                if c.required and c.status == CheckStatus.FAILED]


# Standard checks reusable across workflows
STANDARD_CHECKS = {
    "build": VerificationCheck(
        id="build", name="Build Success",
        description="Code compiles/imports without errors",
    ),
    "test": VerificationCheck(
        id="test", name="Tests Pass",
        description="All tests pass without errors",
    ),
    "lint": VerificationCheck(
        id="lint", name="Lint Clean",
        description="No linting errors",
    ),
    "functionality": VerificationCheck(
        id="functionality", name="Functionality Verified",
        description="All requested features work as described",
    ),
    "error_free": VerificationCheck(
        id="error_free", name="Error Free",
        description="Zero unaddressed errors in output",
    ),
}


def get_standard_check(check_id: str) -> VerificationCheck:
    """Get a copy of a standard check by ID."""
    template = STANDARD_CHECKS.get(check_id)
    if not template:
        raise ValueError(f"Unknown standard check: {check_id}")
    return VerificationCheck(
        id=template.id,
        name=template.name,
        description=template.description,
        required=template.required,
        command=template.command,
    )


def create_protocol(
    check_ids: list[str] | None = None,
    custom_checks: list[VerificationCheck] | None = None,
) -> list[VerificationCheck]:
    """Create a verification protocol from standard + custom checks.

    Args:
        check_ids: List of standard check IDs to include.
        custom_checks: Additional custom checks to append.

    Returns:
        List of VerificationCheck instances ready to run.
    """
    checks = []
    if check_ids:
        for cid in check_ids:
            checks.append(get_standard_check(cid))
    if custom_checks:
        checks.extend(custom_checks)
    return checks


def run_check(check: VerificationCheck, cwd: str | None = None) -> VerificationCheck:
    """Run a single verification check.

    If the check has a command, runs it as a subprocess.
    Otherwise marks it as needing manual verification.
    """
    check.timestamp = time.time()
    start = time.monotonic()

    if check.command:
        try:
            result = subprocess.run(
                shlex.split(check.command), shell=False, capture_output=True,
                text=True, timeout=300, cwd=cwd,
            )
            check.duration_s = time.monotonic() - start
            if result.returncode == 0:
                check.status = CheckStatus.PASSED
                check.evidence = result.stdout[-2000:] if result.stdout else "Exit code 0"
            else:
                check.status = CheckStatus.FAILED
                output = result.stderr or result.stdout or ""
                check.evidence = f"Exit code {result.returncode}: {output[-2000:]}"
        except subprocess.TimeoutExpired:
            check.duration_s = time.monotonic() - start
            check.status = CheckStatus.FAILED
            check.evidence = "Command timed out (300s)"
        except Exception as e:
            check.duration_s = time.monotonic() - start
            check.status = CheckStatus.FAILED
            check.evidence = f"Error: {e}"
    else:
        # No command — requires manual/agent verification
        check.duration_s = 0.0
        check.status = CheckStatus.SKIPPED
        check.evidence = "No automated command; requires manual verification"

    return check


def run_protocol(
    checks: list[VerificationCheck],
    cwd: str | None = None,
    fail_fast: bool = False,
    on_check: Callable[[VerificationCheck], None] | None = None,
) -> VerificationResult:
    """Run a full verification protocol.

    Args:
        checks: List of checks to execute.
        cwd: Working directory for commands.
        fail_fast: Stop on first required failure.
        on_check: Callback after each check completes.

    Returns:
        VerificationResult with all check outcomes.
    """
    start = time.monotonic()

    for check in checks:
        run_check(check, cwd=cwd)
        if on_check:
            on_check(check)
        if fail_fast and check.required and check.status == CheckStatus.FAILED:
            # Mark remaining as skipped
            for remaining in checks[checks.index(check) + 1:]:
                remaining.status = CheckStatus.SKIPPED
                remaining.evidence = "Skipped (fail-fast after previous failure)"
            break

    total_duration = time.monotonic() - start

    # Protocol passes if no required checks failed
    passed = all(
        c.status != CheckStatus.FAILED
        for c in checks if c.required
    )

    # Build summary
    parts = []
    for c in checks:
        icon = {"passed": "+", "failed": "!", "skipped": "~", "pending": "?"}[c.status.value]
        parts.append(f"  [{icon}] {c.name}: {c.status.value}")
    summary = "\n".join(parts)

    return VerificationResult(
        passed=passed,
        checks=checks,
        total_duration_s=total_duration,
        summary=summary,
    )


def quick_verify(cwd: str | None = None) -> VerificationResult:
    """Run a quick verification: import check + tests.

    Convenience function for common verification scenarios.
    """
    checks = create_protocol(["build", "test"])

    # Auto-detect commands
    checks[0].command = "python -c 'import mullet'"
    checks[1].command = "python -m pytest tests/ --ignore=tests/integration -q --tb=short"

    return run_protocol(checks, cwd=cwd)


# ================================================================
# LLM-as-Judge + Eval Dataset Framework
# ================================================================

def llm_as_judge(response: str, criteria: list[str], agent) -> tuple[float, str]:
    """Use an LLM to evaluate response quality against criteria.

    Returns (score 0.0-1.0, feedback text).
    """
    criteria_text = "\n".join(f"- {c}" for c in criteria)
    prompt = (
        f"Evaluate this response against the criteria below.\n"
        f"Score 0.0 (fail) to 1.0 (perfect). Be strict.\n\n"
        f"Criteria:\n{criteria_text}\n\n"
        f"Response:\n{response[:2000]}\n\n"
        f"Reply ONLY with: SCORE: <number>\nFEEDBACK: <text>"
    )
    result = agent.run(prompt)
    text = result.final_response
    score = 0.5  # default
    feedback = text
    import re
    m = re.search(r"SCORE:\s*([\d.]+)", text)
    if m:
        score = min(1.0, max(0.0, float(m.group(1))))
    m2 = re.search(r"FEEDBACK:\s*(.+)", text, re.DOTALL)
    if m2:
        feedback = m2.group(1).strip()
    return score, feedback


def eval_dataset_check(result: str, expected: str, fuzzy: bool = True) -> tuple[bool, float]:
    """Compare result against expected output for eval dataset runs.

    Returns (passed: bool, similarity: float 0.0-1.0).
    """
    if result.strip() == expected.strip():
        return True, 1.0
    if not fuzzy:
        return False, 0.0
    # Simple word overlap similarity
    result_words = set(result.lower().split())
    expected_words = set(expected.lower().split())
    if not expected_words:
        return False, 0.0
    overlap = len(result_words & expected_words)
    similarity = overlap / len(expected_words)
    return similarity >= 0.8, similarity
