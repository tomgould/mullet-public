"""Tests for the 7-section delegation prompt builder."""

from __future__ import annotations

import pytest

from mullet.core.delegation.prompt_builder import (
    Role,
    SevenSectionPrompt,
    build_seven_section_prompt,
)


def test_minimal_prompt_has_role_header_and_task():
    out = build_seven_section_prompt("do the thing")
    assert "ROLE: generic" in out
    assert "1. TASK: do the thing" in out
    # Optional sections must not render when empty — check against the
    # numbered section headers specifically so the persona line (which
    # happens to mention CONSTRAINTS / MUST NOT DO in prose) doesn't
    # trip the assertion.
    for marker in ("2. EXPECTED OUTCOME", "3. CONTEXT", "4. CONSTRAINTS",
                   "5. MUST DO", "6. MUST NOT DO", "7. OUTPUT FORMAT"):
        assert marker not in out


def test_full_prompt_renders_all_seven_sections_in_order():
    out = build_seven_section_prompt(
        task="fix calc",
        expected_outcome="tests green",
        context="src/foo.py line 10",
        constraints="python 3.10",
        must_do=["preserve signature", "add docstring"],
        must_not_do=["rename function"],
        output_format="EDIT blocks only",
        role=Role.CODE_REVIEWER,
    )
    # Ordering: section 1 precedes section 2 precedes ... section 7.
    positions = [out.index(f"{n}. ") for n in range(1, 8)]
    assert positions == sorted(positions)
    assert "ROLE: code_reviewer" in out
    # Bullet rendering.
    assert "   - preserve signature" in out
    assert "   - add docstring" in out
    assert "   - rename function" in out


def test_empty_task_raises():
    with pytest.raises(ValueError):
        build_seven_section_prompt("")
    with pytest.raises(ValueError):
        build_seven_section_prompt("   \n  ")


def test_string_role_is_accepted():
    out = build_seven_section_prompt("x", role="architect")
    assert "ROLE: architect" in out
    # Architect persona header present.
    assert "software architect" in out.lower()


def test_unknown_role_raises():
    with pytest.raises(ValueError) as excinfo:
        build_seven_section_prompt("x", role="wizard")
    # Error message lists valid roles so humans/tests can recover.
    msg = str(excinfo.value)
    for role in Role:
        assert role.value in msg


def test_whitespace_only_bullets_are_dropped():
    out = build_seven_section_prompt(
        "x", must_do=["", "  ", "real item"], must_not_do=["   "]
    )
    assert "- real item" in out
    # Because all must_not_do entries were whitespace, the numbered section
    # header must be skipped (the persona line may still mention "MUST NOT DO"
    # in prose, which is fine).
    assert "6. MUST NOT DO" not in out


def test_all_roles_have_a_persona_header():
    # Safety net: if a new Role enum member is added, the lookup table must
    # be updated too or render() will KeyError.
    for role in Role:
        out = build_seven_section_prompt("task", role=role)
        assert f"ROLE: {role.value}" in out
        # Persona line is non-empty — two lines after "ROLE:" header.
        lines = out.splitlines()
        assert lines[0] == f"ROLE: {role.value}"
        assert len(lines[1]) > 10


def test_dataclass_render_matches_builder():
    built = build_seven_section_prompt(
        "t", expected_outcome="eo", role=Role.SECURITY_ANALYST
    )
    direct = SevenSectionPrompt(
        task="t", expected_outcome="eo", role=Role.SECURITY_ANALYST
    ).render()
    assert built == direct
