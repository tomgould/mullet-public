"""Structured delegation prompt builder.

Adapts the 7-section delegation format from ``jarrodwatts/claude-delegator``
(``rules/delegation-format.md``) into a Mullet helper so delegates receive
prompts with consistent structure instead of a flat task string.

Also exposes named role constants (architect, plan_reviewer, scope_analyst,
code_reviewer, security_analyst, generic) so callers can tag a delegation with
a role. The role is embedded as a ROLE header in the rendered prompt and is
intended to be picked up by routing priors via ``RoutingOutcome.tags`` (future
work — this module only produces the prompt string).

Usage::

    from mullet.core.delegation.prompt_builder import (
        SevenSectionPrompt, Role, build_seven_section_prompt,
    )

    prompt = build_seven_section_prompt(
        task="Fix the calculate_total function to handle tax",
        expected_outcome="All tax tests pass",
        context="src/utils.py has calculate_total; fails tax_test.py::test_vat",
        must_do=["Preserve existing rounding", "Update calculate_total only"],
        must_not_do=["Rename the function", "Add new dependencies"],
        output_format="EDIT blocks only — no prose",
        role=Role.CODE_REVIEWER,
    )

The helper is intentionally pure text — no provider dispatch, no I/O — so it
can be unit-tested and composed with any message builder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class Role(str, Enum):
    """Named delegation role, orthogonal to routing tier.

    Values are lower-case snake_case strings so they can be used directly as
    tags on :class:`mullet.core.routing_history.RoutingOutcome` and as keys in
    future role-aware prior tables.
    """

    ARCHITECTURE = "architect"
    PLAN_REVIEWER = "plan_reviewer"
    SCOPE_ANALYST = "scope_analyst"
    CODE_REVIEWER = "code_reviewer"
    SECURITY_ANALYST = "security_analyst"
    GENERIC = "generic"


# Short persona headers lifted in spirit from claude-delegator's prompts/*.md.
# Kept to one line so the prompt stays compact; fuller role prompts can land in
# mullet/core/delegation/roles/*.md in a follow-up commit.
_ROLE_PERSONA: dict[Role, str] = {
    Role.ARCHITECTURE: (
        "You are a software architect. Apply pragmatic minimalism: simplest "
        "solution that meets the requirement. Tag effort as Quick/Short/Medium/Large."
    ),
    Role.PLAN_REVIEWER: (
        "You are a plan reviewer. Your verdict is APPROVE or REJECT with "
        "justification covering Clarity, Verifiability, Completeness, Big Picture."
    ),
    Role.SCOPE_ANALYST: (
        "You are a pre-planning scope analyst. Classify intent, surface hidden "
        "requirements and risks, and list questions that need answers before work begins."
    ),
    Role.CODE_REVIEWER: (
        "You are a senior code reviewer. Prioritise Correctness > Security > "
        "Performance > Maintainability. Skip style nitpicks."
    ),
    Role.SECURITY_ANALYST: (
        "You are a security engineer. Apply an attacker's mindset and the OWASP Top 10. "
        "Provide practical remediation, not theoretical concerns."
    ),
    Role.GENERIC: (
        "You are a careful implementation assistant. Follow the CONSTRAINTS and "
        "MUST DO / MUST NOT DO lists precisely."
    ),
}


@dataclass
class SevenSectionPrompt:
    """A delegation prompt rendered in the 7-section format."""

    task: str
    expected_outcome: str = ""
    context: str = ""
    constraints: str = ""
    must_do: list[str] = field(default_factory=list)
    must_not_do: list[str] = field(default_factory=list)
    output_format: str = ""
    role: Role = Role.GENERIC

    def render(self) -> str:
        """Render as a single string ready to pass as the delegation ``task``."""
        return _render(self)


def build_seven_section_prompt(
    task: str,
    *,
    expected_outcome: str = "",
    context: str = "",
    constraints: str = "",
    must_do: Iterable[str] | None = None,
    must_not_do: Iterable[str] | None = None,
    output_format: str = "",
    role: Role | str = Role.GENERIC,
) -> str:
    """Build a 7-section prompt string.

    All fields except ``task`` are optional — empty sections are omitted so a
    minimal call (``build_seven_section_prompt("do X")``) degrades to a
    one-line prompt with just a ROLE header and a TASK line.
    """
    if not task or not task.strip():
        raise ValueError("task must be a non-empty string")

    if isinstance(role, str):
        try:
            role = Role(role)
        except ValueError as exc:
            valid = ", ".join(r.value for r in Role)
            raise ValueError(
                f"Unknown role {role!r}. Valid roles: {valid}"
            ) from exc

    prompt = SevenSectionPrompt(
        task=task.strip(),
        expected_outcome=expected_outcome.strip(),
        context=context.strip(),
        constraints=constraints.strip(),
        must_do=[s.strip() for s in (must_do or []) if s and s.strip()],
        must_not_do=[s.strip() for s in (must_not_do or []) if s and s.strip()],
        output_format=output_format.strip(),
        role=role,
    )
    return _render(prompt)


def _render(p: SevenSectionPrompt) -> str:
    lines: list[str] = []
    persona = _ROLE_PERSONA[p.role]
    lines.append(f"ROLE: {p.role.value}")
    lines.append(persona)
    lines.append("")
    lines.append(f"1. TASK: {p.task}")
    if p.expected_outcome:
        lines.append("")
        lines.append(f"2. EXPECTED OUTCOME: {p.expected_outcome}")
    if p.context:
        lines.append("")
        lines.append("3. CONTEXT:")
        lines.append(p.context)
    if p.constraints:
        lines.append("")
        lines.append("4. CONSTRAINTS:")
        lines.append(p.constraints)
    if p.must_do:
        lines.append("")
        lines.append("5. MUST DO:")
        lines.extend(f"   - {item}" for item in p.must_do)
    if p.must_not_do:
        lines.append("")
        lines.append("6. MUST NOT DO:")
        lines.extend(f"   - {item}" for item in p.must_not_do)
    if p.output_format:
        lines.append("")
        lines.append(f"7. OUTPUT FORMAT: {p.output_format}")
    return "\n".join(lines)


__all__ = [
    "Role",
    "SevenSectionPrompt",
    "build_seven_section_prompt",
]
