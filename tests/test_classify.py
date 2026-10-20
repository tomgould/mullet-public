"""Tests for mullet.cli.classify."""
from __future__ import annotations

import io
import json
import sys
from dataclasses import asdict
from unittest.mock import patch

import pytest

from mullet.cli.classify import classify, main, Classification


# ---------------------------------------------------------------------------
# classify() unit tests
# ---------------------------------------------------------------------------

class TestClassify:
    def test_empty_prompt_returns_ambiguous(self) -> None:
        result = classify("")
        assert result.intent == "Ambiguous"
        assert result.confidence == 0.5

    def test_whitespace_only_returns_ambiguous(self) -> None:
        result = classify("   ")
        assert result.intent == "Ambiguous"

    def test_github_url_returns_github_work(self) -> None:
        result = classify("review github.com/org/repo/pull/123 and merge it")
        assert result.intent == "GitHubWork"
        assert result.confidence >= 0.90

    def test_pr_number_returns_github_work(self) -> None:
        result = classify("look at PR #456 and summarise changes")
        assert result.intent == "GitHubWork"

    def test_create_pull_request_returns_github_work(self) -> None:
        result = classify("create a pull request for the auth feature")
        assert result.intent == "GitHubWork"

    def test_commit_returns_explicit(self) -> None:
        result = classify("commit the staged changes")
        assert result.intent == "Explicit"
        assert result.confidence >= 0.80

    def test_run_tests_returns_explicit(self) -> None:
        result = classify("execute tests for this module")
        assert result.intent == "Explicit"

    def test_reindex_returns_explicit(self) -> None:
        result = classify("reindex the mullet-repos pool")
        assert result.intent == "Explicit"

    def test_plan_returns_open_ended(self) -> None:
        result = classify("plan the new auth subsystem")
        assert result.intent == "OpenEnded"

    def test_refactor_returns_open_ended(self) -> None:
        result = classify("refactor the routing layer")
        assert result.intent == "OpenEnded"

    def test_analyze_returns_open_ended(self) -> None:
        result = classify("analyze performance bottlenecks in the API")
        assert result.intent == "OpenEnded"

    def test_how_does_returns_exploratory(self) -> None:
        result = classify("how does the routing decision work?")
        assert result.intent == "Exploratory"

    def test_find_returns_exploratory(self) -> None:
        result = classify("find where sessions are stored")
        assert result.intent == "Exploratory"

    def test_help_returns_ambiguous(self) -> None:
        result = classify("help")
        assert result.intent == "Ambiguous"

    def test_trivial_short_prompt(self) -> None:
        result = classify("add a comment here")
        assert result.intent == "Trivial"

    def test_long_unmatched_prompt_is_open_ended(self) -> None:
        long_prompt = "x " * 210  # > 400 chars, no intent keyword
        result = classify(long_prompt)
        assert result.intent == "OpenEnded"
        assert "length>400" in result.matched_triggers

    def test_highest_confidence_wins(self) -> None:
        # GitHub URL (0.95) + commit (0.85) → GitHubWork wins
        result = classify("commit the change at github.com/org/repo/pull/1")
        assert result.intent == "GitHubWork"
        assert result.confidence >= 0.95

    def test_skill_hints_delegation(self) -> None:
        result = classify("delegate this to kimi and wait")
        assert "delegation" in result.skill_hints

    def test_skill_hints_trawl(self) -> None:
        result = classify("use trawl to search the codebase")
        assert "trawl" in result.skill_hints

    def test_skill_hints_empty_for_trivial(self) -> None:
        result = classify("add a comment")
        # no skill keywords → empty hints
        assert result.skill_hints == []

    def test_suggested_action_populated(self) -> None:
        result = classify("plan the migration")
        assert result.suggested_action  # non-empty string

    def test_returns_classification_dataclass(self) -> None:
        result = classify("review the changes")
        assert isinstance(result, Classification)
        d = asdict(result)
        assert "intent" in d
        assert "confidence" in d
        assert "matched_triggers" in d
        assert "suggested_action" in d
        assert "skill_hints" in d


# ---------------------------------------------------------------------------
# main() / CLI tests
# ---------------------------------------------------------------------------

class TestMain:
    def test_human_readable_output(self, capsys: pytest.CaptureFixture) -> None:
        rc = main(["commit the staged changes"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "intent" in out
        assert "confidence" in out

    def test_json_output_is_valid_json(self, capsys: pytest.CaptureFixture) -> None:
        rc = main(["--json", "plan the auth subsystem"])
        assert rc == 0
        raw = capsys.readouterr().out
        parsed = json.loads(raw)
        assert "intent" in parsed
        assert "confidence" in parsed

    def test_json_contains_correct_intent(self, capsys: pytest.CaptureFixture) -> None:
        main(["--json", "reindex the pool"])
        raw = capsys.readouterr().out
        parsed = json.loads(raw)
        assert parsed["intent"] == "Explicit"

    def test_stdin_mode(self, capsys: pytest.CaptureFixture) -> None:
        with patch.object(sys, "stdin", io.StringIO("commit and push")):
            rc = main(["-"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Explicit" in out

    def test_no_prompt_reads_stdin(self, capsys: pytest.CaptureFixture) -> None:
        with patch.object(sys, "stdin", io.StringIO("plan something")):
            rc = main([])
        assert rc == 0
