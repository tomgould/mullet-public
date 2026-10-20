"""Three-role review pipeline for subagent dispatch.

Provides structured two-stage review after each implementation:
1. Spec-compliance reviewer: does output match the spec step?
2. Code-quality reviewer: bugs, security, style.

Designed to be called from TeamMode._verify_stage() or standalone.
"""
import logging
import re
from dataclasses import dataclass, field
from typing import List, Tuple

logger = logging.getLogger(__name__)

@dataclass
class ReviewFinding:
    """A single review observation — role, severity, description, optional file:line."""

    role: str          # "spec-compliance" or "code-quality"
    severity: str      # "critical", "major", "minor", "suggestion"
    description: str
    file: str = ""
    line: int = 0

@dataclass
class ReviewResult:
    """Aggregate review outcome — list of findings + summary + auto-approved flag."""

    findings: List[ReviewFinding]
    summary: str
    approved: bool = field(init=False)

    def __post_init__(self):
        self.approved = not any(f.severity in ["critical", "major"] for f in self.findings)
        if not self.summary:
            if self.approved:
                self.summary = "Approved: No critical or major findings."
            else:
                critical_count = sum(1 for f in self.findings if f.severity == "critical")
                major_count = sum(1 for f in self.findings if f.severity == "major")
                self.summary = f"Rejected: {critical_count} critical, {major_count} major findings."


def spec_compliance_review(
    step_title: str,
    step_description: str,
    acceptance_criteria: List[str],
    implementation_summary: str
) -> ReviewResult:
    """Checks if the implementation_summary covers all acceptance_criteria."""
    findings: List[ReviewFinding] = []
    


    for criterion in acceptance_criteria:
        # Split criterion into keywords and check if all are present in the summary
        keywords = [word for word in criterion.lower().split() if len(word) > 2]
        all_keywords_found = True
        for keyword in keywords:
            if keyword not in implementation_summary.lower():
                all_keywords_found = False
                break

        if not all_keywords_found:
            findings.append(ReviewFinding(
                role="spec-compliance",
                severity="critical",
                description=f"Acceptance criterion not met: '{criterion}'"
            ))
        else:
            pass
    
    return ReviewResult(findings=findings, summary="")


def code_quality_review(
    implementation_summary: str,
    files_modified: List[str] = None
) -> ReviewResult:
    """Scans implementation_summary for red flags."""
    findings: List[ReviewFinding] = []

    # Critical findings
    critical_patterns = [
        (r"rm -rf", "Potential dangerous command 'rm -rf' detected."),
        (r"eval\(", "Use of 'eval()' detected, potential security risk."),
        (r"exec\(", "Use of 'exec()' detected, potential security risk."),
        (r"(password|credential|secret|token)\s*=\s*[\"'][^\"']+[\"']", "Hard-coded credentials detected."),
    ]
    for pattern, desc in critical_patterns:
        if re.search(pattern, implementation_summary, re.IGNORECASE):
            findings.append(ReviewFinding(role="code-quality", severity="critical", description=desc))

    # Major findings
    major_patterns = [
        (r"TODO", "TODO comment left in implementation summary, indicating incomplete work."),
        (r"FIXME", "FIXME comment left in implementation summary, indicating incomplete work."),
        (r"pass\s*$", "Placeholder 'pass' statement found, indicating incomplete logic."),
        (r"raise NotImplementedError", "NotImplementedError found, indicating incomplete logic."),
    ]
    for pattern, desc in major_patterns:
        if re.search(pattern, implementation_summary, re.IGNORECASE):
            findings.append(ReviewFinding(role="code-quality", severity="major", description=desc))

    return ReviewResult(findings=findings, summary="")


def run_three_role_review(
    step_title: str,
    step_description: str,
    acceptance_criteria: List[str],
    implementation_summary: str,
    files_modified: List[str] = None
) -> Tuple[ReviewResult, ReviewResult]:
    """Runs both reviews, returns (spec_result, quality_result)."""
    spec_result = spec_compliance_review(step_title, step_description, acceptance_criteria, implementation_summary)
    quality_result = code_quality_review(implementation_summary, files_modified)

    for finding in spec_result.findings:
        logger.warning(f"SPEC REVIEW ({finding.severity.upper()}): {finding.description}")
    for finding in quality_result.findings:
        logger.warning(f"QUALITY REVIEW ({finding.severity.upper()}): {finding.description}")

    return spec_result, quality_result


def format_review_report(spec_result: ReviewResult, quality_result: ReviewResult) -> str:
    """Returns a formatted multi-line string suitable for including in fix prompts."""
    report_lines: List[str] = []

    emoji_map = {
        "critical": "🔴",
        "major": "🟡",
        "minor": "🟢",
        "suggestion": "🔵",
    }

    report_lines.append("--- Review Report ---")
    
    if spec_result.findings:
        report_lines.append("\\nSPEC COMPLIANCE REVIEW:")
        for finding in spec_result.findings:
            emoji = emoji_map.get(finding.severity, "⚪")
            report_lines.append(f"{emoji} {finding.severity.upper()}: {finding.description}")
    else:
        report_lines.append("\\nSPEC COMPLIANCE REVIEW: All acceptance criteria met. ✅")

    if quality_result.findings:
        report_lines.append("\\nCODE QUALITY REVIEW:")
        for finding in quality_result.findings:
            emoji = emoji_map.get(finding.severity, "⚪")
            report_lines.append(f"{emoji} {finding.severity.upper()}: {finding.description}")
    else:
        report_lines.append("\\nCODE QUALITY REVIEW: No critical or major code quality issues found. ✅")

    report_lines.append("\\n--- End Review Report ---")
    return "\\n".join(report_lines)

__all__ = [
    "ReviewFinding",
    "ReviewResult",
    "code_quality_review",
    "format_review_report",
    "run_three_role_review",
    "spec_compliance_review",
]
