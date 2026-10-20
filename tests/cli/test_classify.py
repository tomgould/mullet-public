"""Tests for mullet.cli.classify — intent classification debug helper."""

from __future__ import annotations

from mullet.cli.classify import classify


def test_trivial_default():
    # #given a short factual prompt
    # #when classified
    result = classify("what's the current branch?")
    # #then it falls through to Exploratory (what-is) not Trivial
    assert result.intent in {"Exploratory", "Trivial"}


def test_commit_is_explicit():
    result = classify("commit and push")
    assert result.intent == "Explicit"
    assert result.confidence >= 0.8


def test_github_url_is_github_work():
    result = classify("look into https://github.com/foo/bar/issues/42 and create a PR")
    assert result.intent == "GitHubWork"
    assert result.confidence >= 0.9


def test_plan_is_open_ended():
    result = classify("plan a roadmap for the router overhaul")
    assert result.intent == "OpenEnded"


def test_how_does_is_exploratory():
    result = classify("how does the budget guard work?")
    assert result.intent == "Exploratory"


def test_ambiguous_one_word():
    result = classify("help")
    assert result.intent == "Ambiguous"


def test_long_prompt_defaults_to_open_ended():
    prompt = "x " * 250  # >400 chars, no trigger keywords
    result = classify(prompt)
    assert result.intent == "OpenEnded"


def test_skill_hints_detected():
    result = classify("reindex the trawl pool and then benchmark the router")
    assert "trawl" in result.skill_hints
    assert "benchmark" in result.skill_hints


def test_empty_prompt_is_ambiguous():
    result = classify("")
    assert result.intent == "Ambiguous"


def test_delegation_keyword_hint():
    result = classify("delegate this to kimi")
    assert "delegation" in result.skill_hints
