"""Tests for output quality scoring."""

from mullet.core.agent.quality import score_result
from mullet.core.agent.types import AgentResult, TokenUsage


def _result(completed=True, response="Done.", tool_calls=1, iterations=2, tokens=3000):
    return AgentResult(
        completed=completed,
        final_response=response,
        tool_calls_made=tool_calls,
        iterations_used=iterations,
        tokens_used=TokenUsage(total_tokens=tokens),
        provider_used="local",
    )


def test_good_result():
    r = _result(response="The function has been updated and tests pass.", tool_calls=3, iterations=4)
    score = score_result(r, task="fix the auth bug")
    assert score.overall > 60
    assert score.completeness >= 80
    assert score.notes == [] or all("error" not in n.lower() for n in score.notes)


def test_incomplete_result():
    r = _result(completed=False, response="", tool_calls=0)
    score = score_result(r, task="fix the bug")
    assert score.overall < 30
    assert score.completeness == 0
    assert any("not complete" in n.lower() for n in score.notes)


def test_no_tools_when_expected():
    r = _result(tool_calls=0, response="I can help with that.")
    score = score_result(r, task="read the file and fix it")
    assert score.tool_usage < 50
    assert any("tool usage" in n.lower() for n in score.notes)


def test_simple_question_no_tools_ok():
    r = _result(tool_calls=0, response="Python is a programming language.", iterations=1)
    score = score_result(r, task="what is python?")
    assert score.tool_usage >= 50  # No tools expected for questions


def test_excessive_iterations():
    r = _result(iterations=80)
    score = score_result(r, task="simple task", max_iterations=90)
    assert score.efficiency < 50
    assert any("budget" in n.lower() for n in score.notes)


def test_efficient_completion():
    r = _result(iterations=2, tokens=1000)
    score = score_result(r, task="quick task")
    assert score.efficiency >= 80


def test_high_token_usage():
    r = _result(tokens=60000)
    score = score_result(r)
    assert score.efficiency < 80  # Penalized but not severely


def test_empty_response():
    r = _result(response="")
    score = score_result(r)
    assert score.response_quality == 0


def test_error_response():
    r = _result(response="Error: something went wrong and I cannot proceed.")
    score = score_result(r)
    assert score.response_quality < 60  # Penalized for error content


def test_structured_response():
    r = _result(response="Here are the results:\n1. File created\n2. Tests pass\n3. Deployed")
    score = score_result(r, task="deploy the app")
    assert score.response_quality > 50


def test_quality_score_range():
    """All scores should be 0-100."""
    for completed in [True, False]:
        for tools in [0, 5, 15]:
            r = _result(completed=completed, tool_calls=tools)
            score = score_result(r)
            assert 0 <= score.overall <= 100
            assert 0 <= score.completeness <= 100
            assert 0 <= score.efficiency <= 100
            assert 0 <= score.tool_usage <= 100
            assert 0 <= score.response_quality <= 100
