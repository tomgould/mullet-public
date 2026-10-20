"""Tests for knowledge context building and token estimation."""

from mullet.core.knowledge.context import (
    build_system_prompt,
    estimate_tokens,
    truncate_to_budget,
)
from mullet.core.knowledge.manager import KnowledgeContext


def test_build_prompt_no_context():
    result = build_system_prompt("You are Mullet.")
    assert result == "You are Mullet."


def test_build_prompt_with_context():
    ctx = KnowledgeContext(markdown="## Results\nSome code context")
    result = build_system_prompt("Base prompt.", ctx, "my-project")
    assert "Base prompt" in result
    assert "Project: my-project" in result
    assert "Some code context" in result


def test_build_prompt_empty_context():
    ctx = KnowledgeContext(markdown="")
    result = build_system_prompt("Base prompt.", ctx)
    assert result == "Base prompt."


def test_build_prompt_none_context():
    result = build_system_prompt("Base prompt.", None)
    assert result == "Base prompt."


def test_estimate_tokens():
    assert estimate_tokens("hello world") == 2  # 11 chars / 4
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 400) == 100


def test_truncate_within_budget():
    text = "Short text"
    assert truncate_to_budget(text, 100) == text


def test_truncate_over_budget():
    text = "x" * 1000
    result = truncate_to_budget(text, 10)  # 10 tokens = 40 chars
    assert len(result) < len(text)
    assert "truncated" in result
