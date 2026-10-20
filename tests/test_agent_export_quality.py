"""100% coverage tests for agent export and quality modules."""

from __future__ import annotations

import json

import pytest

from mullet.core.agent.export import export_json, export_markdown
from mullet.core.agent.quality import QualityScore, score_result
from mullet.core.agent.types import AgentResult, TokenUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    *,
    completed: bool = True,
    final_response: str = "done",
    messages: list[dict] | None = None,
    tool_calls_made: int = 0,
    iterations_used: int = 1,
    tokens_used: TokenUsage | None = None,
    provider_used: str = "openrouter",
    tier_used: str = "cheap",
) -> AgentResult:
    return AgentResult(
        completed=completed,
        final_response=final_response,
        messages=messages or [],
        tool_calls_made=tool_calls_made,
        iterations_used=iterations_used,
        tokens_used=tokens_used or TokenUsage(),
        provider_used=provider_used,
        tier_used=tier_used,
    )


# ---------------------------------------------------------------------------
# export_json
# ---------------------------------------------------------------------------

class TestExportJson:
    """Tests for export_json."""

    def test_without_output_path(self):
        result = _make_result(
            completed=True,
            final_response="hello",
            messages=[{"role": "user", "content": "hi"}],
            tool_calls_made=2,
            iterations_used=3,
            tokens_used=TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30),
            provider_used="kimi",
            tier_used="cheap",
        )
        json_str = export_json(result, task="test task")
        data = json.loads(json_str)

        assert isinstance(data["mullet_version"], str)
        assert isinstance(data["exported_at"], str)
        assert data["task"] == "test task"
        assert data["completed"] is True
        assert data["provider"] == "kimi"
        assert data["tier"] == "cheap"
        assert data["tokens"] == {"input": 10, "output": 20, "total": 30}
        assert data["tool_calls"] == 2
        assert data["iterations"] == 3
        assert data["messages"] == [{"role": "user", "content": "hi"}]
        assert data["final_response"] == "hello"

    def test_with_output_path(self, tmp_path):
        out = tmp_path / "out.json"
        result = _make_result(completed=False, final_response="nope")
        json_str = export_json(result, task="t2", output_path=str(out))

        assert out.read_text() == json_str
        data = json.loads(json_str)
        assert data["completed"] is False
        assert data["final_response"] == "nope"


# ---------------------------------------------------------------------------
# export_markdown
# ---------------------------------------------------------------------------

class TestExportMarkdown:
    """Tests for export_markdown covering every role branch."""

    def test_without_output_path(self):
        result = _make_result(
            completed=True,
            final_response="final",
            messages=[{"role": "user", "content": "hello"}],
            provider_used="p",
            tier_used="t",
        )
        md = export_markdown(result, task="my task")
        assert "# Mullet Conversation Export" in md
        assert "**Task:** my task" in md
        assert "**Status:** Completed" in md
        assert "### User" in md
        assert "hello" in md
        assert "### Final Response" in md
        assert "final" in md

    def test_with_output_path(self, tmp_path):
        out = tmp_path / "out.md"
        result = _make_result(completed=False, final_response="incomplete")
        md = export_markdown(result, task="t", output_path=str(out))
        assert out.read_text() == md
        assert "**Status:** Incomplete" in md

    # -- system role --

    def test_system_short_content(self):
        result = _make_result(messages=[{"role": "system", "content": "short"}])
        md = export_markdown(result, task="t")
        assert "### System Prompt" in md
        assert "```\nshort\n```" in md
        assert "..." not in md.split("System Prompt")[1].split("```")[1]

    def test_system_long_content(self):
        long_content = "x" * 250
        result = _make_result(messages=[{"role": "system", "content": long_content}])
        md = export_markdown(result, task="t")
        assert "### System Prompt" in md
        assert f"```\n{'x' * 200}...\n```" in md

    # -- user role --

    def test_user_role(self):
        result = _make_result(messages=[{"role": "user", "content": "user msg"}])
        md = export_markdown(result, task="t")
        assert "### User" in md
        assert "user msg" in md

    # -- assistant role --

    def test_assistant_content_only(self):
        result = _make_result(messages=[{"role": "assistant", "content": "ans"}])
        md = export_markdown(result, task="t")
        assert "### Assistant" in md
        assert "ans" in md
        assert "**Tool call: `" not in md

    def test_assistant_tool_calls_short_args(self):
        result = _make_result(
            messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "foo", "arguments": "short args"}}
                    ],
                }
            ]
        )
        md = export_markdown(result, task="t")
        assert "**Tool call: `foo`**" in md
        assert "```json\nshort args\n```" in md
        assert "..." not in md.split("**Tool call: `foo`**")[1]

    def test_assistant_tool_calls_long_args(self):
        long_args = "y" * 250
        result = _make_result(
            messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "bar", "arguments": long_args}}
                    ],
                }
            ]
        )
        md = export_markdown(result, task="t")
        assert "**Tool call: `bar`**" in md
        assert f"```json\n{'y' * 200}...\n```" in md

    # -- tool role --

    def test_tool_short_content(self):
        result = _make_result(messages=[{"role": "tool", "content": "ok"}])
        md = export_markdown(result, task="t")
        assert "### Tool Result" in md
        assert "```\nok\n```" in md
        assert "..." not in md.split("Tool Result")[1].split("```")[1]

    def test_tool_long_content(self):
        long_content = "z" * 600
        result = _make_result(messages=[{"role": "tool", "content": long_content}])
        md = export_markdown(result, task="t")
        assert "### Tool Result" in md
        assert f"```\n{'z' * 500}...\n```" in md

    # -- unknown role --

    def test_unknown_role_skipped_silently(self):
        result = _make_result(messages=[{"role": "unknown", "content": "skip me"}])
        md = export_markdown(result, task="t")
        assert "skip me" not in md
        assert "### Unknown" not in md


# ---------------------------------------------------------------------------
# QualityScore
# ---------------------------------------------------------------------------

class TestQualityScore:
    """Tests for QualityScore dataclass."""

    def test_post_init_notes_none(self):
        qs = QualityScore(notes=None)
        assert qs.notes == []

    def test_post_init_notes_provided(self):
        qs = QualityScore(notes=["a"])
        assert qs.notes == ["a"]


# ---------------------------------------------------------------------------
# score_result — completeness
# ---------------------------------------------------------------------------

class TestScoreResultCompleteness:
    """Completeness dimension tests."""

    def test_not_completed(self):
        result = _make_result(completed=False, final_response="")
        qs = score_result(result)
        assert qs.completeness == 0.0
        assert "Task did not complete" in qs.notes

    def test_completed_short_response(self):
        result = _make_result(completed=True, final_response="x" * 10)
        qs = score_result(result)
        assert qs.completeness == 50.0

    def test_completed_medium_response(self):
        result = _make_result(completed=True, final_response="x" * 11)
        qs = score_result(result)
        assert qs.completeness == 80.0

    def test_completed_long_response(self):
        result = _make_result(completed=True, final_response="x" * 101)
        qs = score_result(result)
        assert qs.completeness == 100.0


# ---------------------------------------------------------------------------
# score_result — efficiency
# ---------------------------------------------------------------------------

class TestScoreResultEfficiency:
    """Efficiency dimension tests."""

    def test_ratio_above_0_8(self):
        result = _make_result(iterations_used=81, tokens_used=TokenUsage(total_tokens=100))
        qs = score_result(result, max_iterations=100)
        assert qs.efficiency == 30.0
        assert "Used >80% of iteration budget" in qs.notes

    def test_ratio_above_0_5(self):
        result = _make_result(iterations_used=51, tokens_used=TokenUsage(total_tokens=100))
        qs = score_result(result, max_iterations=100)
        assert qs.efficiency == 60.0

    def test_ratio_above_0_2(self):
        result = _make_result(iterations_used=21, tokens_used=TokenUsage(total_tokens=100))
        qs = score_result(result, max_iterations=100)
        assert qs.efficiency == 80.0

    def test_ratio_at_or_below_0_2(self):
        result = _make_result(iterations_used=20, tokens_used=TokenUsage(total_tokens=100))
        qs = score_result(result, max_iterations=100)
        assert qs.efficiency == 100.0

    def test_max_iterations_zero(self):
        result = _make_result(iterations_used=0, tokens_used=TokenUsage(total_tokens=100))
        qs = score_result(result, max_iterations=0)
        assert qs.efficiency == 100.0

    # -- token penalties --

    def test_tokens_above_50k(self):
        result = _make_result(iterations_used=20, tokens_used=TokenUsage(total_tokens=60000))
        qs = score_result(result, max_iterations=100)
        assert qs.efficiency == 70.0  # 100 - 30 = 70, max(20,70)=70
        assert "High token usage (>50K)" in qs.notes

    def test_tokens_above_50k_with_low_base(self):
        result = _make_result(iterations_used=81, tokens_used=TokenUsage(total_tokens=60000))
        qs = score_result(result, max_iterations=100)
        assert qs.efficiency == 20.0  # 30 - 30 = 0, max(20,0)=20
        assert "High token usage (>50K)" in qs.notes

    def test_tokens_above_20k(self):
        result = _make_result(iterations_used=20, tokens_used=TokenUsage(total_tokens=25000))
        qs = score_result(result, max_iterations=100)
        assert qs.efficiency == 90.0  # 100 - 10 = 90, max(40,90)=90

    def test_tokens_above_20k_with_mid_base(self):
        result = _make_result(iterations_used=51, tokens_used=TokenUsage(total_tokens=25000))
        qs = score_result(result, max_iterations=100)
        assert qs.efficiency == 50.0  # 60 - 10 = 50, max(40,50)=50

    def test_tokens_no_penalty(self):
        result = _make_result(iterations_used=20, tokens_used=TokenUsage(total_tokens=20000))
        qs = score_result(result, max_iterations=100)
        assert qs.efficiency == 100.0


# ---------------------------------------------------------------------------
# score_result — tool_usage
# ---------------------------------------------------------------------------

class TestScoreResultToolUsage:
    """Tool usage dimension tests."""

    def test_task_expects_tools_and_used(self):
        result = _make_result(tool_calls_made=5)
        qs = score_result(result, task="read the file")
        assert qs.tool_usage == 90.0

    def test_task_expects_tools_but_none(self):
        result = _make_result(tool_calls_made=0)
        qs = score_result(result, task="read the file")
        assert qs.tool_usage == 20.0
        assert "Expected tool usage but none occurred" in qs.notes

    def test_task_no_tool_words_no_calls(self):
        result = _make_result(tool_calls_made=0)
        qs = score_result(result, task="hello world")
        assert qs.tool_usage == 80.0

    def test_excessive_tool_calls(self):
        result = _make_result(tool_calls_made=15)
        qs = score_result(result, task="hello world")
        assert qs.tool_usage == 30.0
        assert "Excessive tool calls (>10)" in qs.notes


# ---------------------------------------------------------------------------
# score_result — response_quality
# ---------------------------------------------------------------------------

class TestScoreResultResponseQuality:
    """Response quality dimension tests."""

    def test_empty_response(self):
        result = _make_result(final_response="")
        qs = score_result(result)
        assert qs.response_quality == 0.0
        assert "Empty response" in qs.notes

    def test_short_response(self):
        result = _make_result(final_response="x" * 20)
        qs = score_result(result, task="")
        assert qs.response_quality == 50.0

    def test_medium_response(self):
        result = _make_result(final_response="x" * 21)
        qs = score_result(result, task="")
        assert qs.response_quality == 70.0

    def test_long_response(self):
        result = _make_result(final_response="x" * 5000)
        qs = score_result(result, task="")
        assert qs.response_quality == 60.0

    def test_error_phrases_reduce_score(self):
        result = _make_result(final_response="error: failed now")
        qs = score_result(result, task="")
        # base 50, short (<=20) no length bonus, 2 error phrases -> -20
        assert qs.response_quality == 30.0

    def test_multiline_bonus(self):
        result = _make_result(final_response="x" * 21 + "\nline2\nline3\nline4")
        qs = score_result(result, task="")
        # base 50 + 20 (medium) + 10 (multiline) = 80
        assert qs.response_quality == 80.0

    def test_task_prefix_no_bonus(self):
        result = _make_result(final_response="read the file and report")
        qs = score_result(result, task="read the file")
        # base 50 + 20 (medium) = 70, no +10 because task prefix is in resp
        assert qs.response_quality == 70.0

    def test_no_task_prefix_bonus(self):
        result = _make_result(final_response="here is the result for you")
        qs = score_result(result, task="read the file")
        # base 50 + 20 (medium) + 10 (no task prefix) = 80
        assert qs.response_quality == 80.0

    def test_combined_medium_multiline_no_prefix(self):
        result = _make_result(final_response="a" * 21 + "\n\n\nextra")
        qs = score_result(result, task="something else")
        # 50 + 20 + 10 + 10 = 90
        assert qs.response_quality == 90.0


# ---------------------------------------------------------------------------
# score_result — overall
# ---------------------------------------------------------------------------

class TestScoreResultOverall:
    """Overall weighted average tests."""

    def test_overall_weighted_average(self):
        result = _make_result(
            completed=True,
            final_response="x" * 101,  # completeness=100
            iterations_used=20,         # efficiency=100
            tokens_used=TokenUsage(total_tokens=100),
            tool_calls_made=5,          # tool_usage=90 (task expects tools)
        )
        qs = score_result(result, task="read file", max_iterations=100)
        # response_quality: medium (>20), no errors, no multiline, no task prefix bonus
        # resp = "x" * 101 -> medium, no task -> +10
        # response_quality = 50 + 20 + 10 = 80
        expected = (
            100.0 * 0.35 +
            100.0 * 0.20 +
            90.0 * 0.25 +
            80.0 * 0.20
        )
        assert qs.overall == round(expected, 1)
        assert qs.completeness == 100.0
        assert qs.efficiency == 100.0
        assert qs.tool_usage == 90.0
        assert qs.response_quality == 80.0

    def test_overall_with_incomplete(self):
        result = _make_result(
            completed=False,
            final_response="",
            iterations_used=0,
            tool_calls_made=0,
        )
        qs = score_result(result, task="")
        # completeness=0, efficiency=100, tool_usage=80, response_quality=0
        expected = 0.0 * 0.35 + 100.0 * 0.20 + 80.0 * 0.25 + 0.0 * 0.20
        assert qs.overall == round(expected, 1)
