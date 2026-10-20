"""Tests for mullet.core.delegation.prompt_builder — pure string rendering."""

import pytest
from mullet.core.delegation.prompt_builder import (
    Role,
    SevenSectionPrompt,
    build_seven_section_prompt,
)


# ---------------------------------------------------------------------------
# Role enum
# ---------------------------------------------------------------------------

class TestRole:
    def test_all_roles_exist(self):
        expected = {"architect", "plan_reviewer", "scope_analyst",
                    "code_reviewer", "security_analyst", "generic"}
        assert {r.value for r in Role} == expected

    def test_role_values_are_strings(self):
        for r in Role:
            assert isinstance(r.value, str)

    def test_role_from_string(self):
        assert Role("code_reviewer") == Role.CODE_REVIEWER

    def test_role_is_str_subclass(self):
        assert isinstance(Role.GENERIC, str)


# ---------------------------------------------------------------------------
# SevenSectionPrompt.render
# ---------------------------------------------------------------------------

class TestSevenSectionPromptRender:
    def test_minimal_render_contains_task(self):
        p = SevenSectionPrompt(task="Fix the bug")
        out = p.render()
        assert "Fix the bug" in out

    def test_minimal_render_has_role_header(self):
        p = SevenSectionPrompt(task="Do something", role=Role.GENERIC)
        out = p.render()
        assert "ROLE: generic" in out

    def test_minimal_render_has_task_label(self):
        p = SevenSectionPrompt(task="Do something")
        out = p.render()
        assert "1. TASK:" in out

    def test_empty_optional_sections_omitted(self):
        p = SevenSectionPrompt(task="X")
        out = p.render()
        assert "2. EXPECTED OUTCOME" not in out
        assert "3. CONTEXT:" not in out
        assert "4. CONSTRAINTS:" not in out
        assert "5. MUST DO:" not in out
        assert "6. MUST NOT DO:" not in out
        assert "7. OUTPUT FORMAT:" not in out

    def test_expected_outcome_included(self):
        p = SevenSectionPrompt(task="Do X", expected_outcome="Tests pass")
        out = p.render()
        assert "2. EXPECTED OUTCOME:" in out
        assert "Tests pass" in out

    def test_context_included(self):
        p = SevenSectionPrompt(task="X", context="src/utils.py line 42")
        out = p.render()
        assert "3. CONTEXT:" in out
        assert "src/utils.py line 42" in out

    def test_constraints_included(self):
        p = SevenSectionPrompt(task="X", constraints="No global state")
        out = p.render()
        assert "4. CONSTRAINTS:" in out
        assert "No global state" in out

    def test_must_do_items_rendered_as_bullets(self):
        p = SevenSectionPrompt(task="X", must_do=["Preserve tests", "Use type hints"])
        out = p.render()
        assert "5. MUST DO:" in out
        assert "   - Preserve tests" in out
        assert "   - Use type hints" in out

    def test_must_not_do_items_rendered_as_bullets(self):
        p = SevenSectionPrompt(task="X", must_not_do=["Rename functions", "Add deps"])
        out = p.render()
        assert "6. MUST NOT DO:" in out
        assert "   - Rename functions" in out
        assert "   - Add deps" in out

    def test_output_format_included(self):
        p = SevenSectionPrompt(task="X", output_format="EDIT blocks only")
        out = p.render()
        assert "7. OUTPUT FORMAT: EDIT blocks only" in out

    def test_persona_text_present(self):
        p = SevenSectionPrompt(task="X", role=Role.CODE_REVIEWER)
        out = p.render()
        assert "Correctness" in out

    def test_architecture_role_persona(self):
        p = SevenSectionPrompt(task="X", role=Role.ARCHITECTURE)
        out = p.render()
        assert "architect" in out.lower()

    def test_security_analyst_role_persona(self):
        p = SevenSectionPrompt(task="X", role=Role.SECURITY_ANALYST)
        out = p.render()
        assert "OWASP" in out

    def test_section_order(self):
        p = SevenSectionPrompt(
            task="T",
            expected_outcome="E",
            context="C",
            constraints="CON",
            must_do=["M"],
            must_not_do=["N"],
            output_format="O",
        )
        out = p.render()
        positions = {
            "TASK": out.index("1. TASK:"),
            "OUTCOME": out.index("2. EXPECTED OUTCOME:"),
            "CONTEXT": out.index("3. CONTEXT:"),
            "CONSTRAINTS": out.index("4. CONSTRAINTS:"),
            "MUST_DO": out.index("5. MUST DO:"),
            "MUST_NOT": out.index("6. MUST NOT DO:"),
            "OUTPUT": out.index("7. OUTPUT FORMAT:"),
        }
        keys = list(positions.keys())
        for i in range(len(keys) - 1):
            assert positions[keys[i]] < positions[keys[i + 1]]


# ---------------------------------------------------------------------------
# build_seven_section_prompt
# ---------------------------------------------------------------------------

class TestBuildSevenSectionPrompt:
    def test_minimal_call(self):
        out = build_seven_section_prompt("Fix auth")
        assert "Fix auth" in out
        assert "ROLE: generic" in out

    def test_empty_task_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_seven_section_prompt("")

    def test_whitespace_only_task_raises(self):
        with pytest.raises(ValueError):
            build_seven_section_prompt("   ")

    def test_task_stripped(self):
        out = build_seven_section_prompt("  stripped task  ")
        assert "stripped task" in out

    def test_string_role_resolved(self):
        out = build_seven_section_prompt("X", role="code_reviewer")
        assert "ROLE: code_reviewer" in out

    def test_invalid_string_role_raises(self):
        with pytest.raises(ValueError, match="Unknown role"):
            build_seven_section_prompt("X", role="invalid_role")

    def test_enum_role_accepted(self):
        out = build_seven_section_prompt("X", role=Role.PLAN_REVIEWER)
        assert "ROLE: plan_reviewer" in out

    def test_must_do_whitespace_stripped(self):
        out = build_seven_section_prompt("X", must_do=["  item one  ", "  item two  "])
        assert "   - item one" in out
        assert "   - item two" in out

    def test_empty_must_do_items_filtered(self):
        out = build_seven_section_prompt("X", must_do=["", "  ", "keep this"])
        assert "keep this" in out
        assert "MUST DO:" in out
        # Empty/whitespace items stripped but "keep this" still listed
        lines = [l.strip() for l in out.splitlines() if "- " in l]
        assert "- keep this" in lines

    def test_none_must_do_treated_as_empty(self):
        out = build_seven_section_prompt("X", must_do=None)
        assert "MUST DO:" not in out

    def test_all_sections_present_when_set(self):
        out = build_seven_section_prompt(
            "task",
            expected_outcome="outcome",
            context="ctx",
            constraints="no state",
            must_do=["do this"],
            must_not_do=["not that"],
            output_format="JSON",
        )
        for section in ["1. TASK:", "2. EXPECTED", "3. CONTEXT:", "4. CONSTRAINTS:",
                        "5. MUST DO:", "6. MUST NOT DO:", "7. OUTPUT FORMAT:"]:
            assert section in out

    def test_returns_string(self):
        assert isinstance(build_seven_section_prompt("X"), str)

    def test_output_contains_persona(self):
        out = build_seven_section_prompt("X", role=Role.SCOPE_ANALYST)
        assert "pre-planning" in out or "scope" in out.lower()
