"""Tests for rich_errors, error_log, contextualize, and test_runner pure helpers."""

import re
import os
import json

import pytest

from mullet.cli.rich_errors import format_error
from mullet.core.agent.error_log import get_error_log_path
from mullet.knowledge.engine.search.contextualize import query_contextualize, _FOLLOWUP_RE
from mullet.core.tools.test_runner import (
    _detect_test_framework,
    TEST_RUNNER_SCHEMA,
    run_tests_tool,
)


# ---------------------------------------------------------------------------
# format_error
# ---------------------------------------------------------------------------

class TestFormatErrorReturns:
    def test_returns_string(self):
        result = format_error(ValueError("bad value"))
        assert isinstance(result, str)

    def test_contains_error_type(self):
        result = format_error(ValueError("bad"))
        assert "ValueError" in result

    def test_contains_error_message(self):
        result = format_error(ValueError("specific problem"))
        assert "specific problem" in result


class TestFormatErrorSuggestions:
    def test_import_error_suggests_pip(self):
        err = ImportError("No module named 'foo'")
        result = format_error(err)
        assert "pip install" in result

    def test_import_error_includes_module_name(self):
        err = ImportError("No module named 'requests'")
        result = format_error(err)
        assert "requests" in result

    def test_file_not_found_suggests_path_check(self):
        result = format_error(FileNotFoundError("no such file"))
        assert "file" in result.lower() or "path" in result.lower()

    def test_connection_error_suggests_network(self):
        result = format_error(ConnectionError("refused"))
        assert "network" in result.lower() or "connection" in result.lower() or "server" in result.lower()

    def test_value_error_suggests_input_review(self):
        result = format_error(ValueError("bad value"))
        assert "input" in result.lower() or "value" in result.lower() or "argument" in result.lower()

    def test_type_error_suggests_type_check(self):
        result = format_error(TypeError("wrong type"))
        assert "type" in result.lower() or "argument" in result.lower()

    def test_unknown_error_no_crash(self):
        result = format_error(RuntimeError("unexpected error"))
        assert isinstance(result, str)
        assert "RuntimeError" in result


class TestFormatErrorContext:
    def test_context_included_in_output(self):
        result = format_error(ValueError("bad"), context="while processing")
        assert "while processing" in result

    def test_no_context_does_not_crash(self):
        result = format_error(ValueError("bad"), context=None)
        assert "ValueError" in result


# ---------------------------------------------------------------------------
# get_error_log_path
# ---------------------------------------------------------------------------

class TestGetErrorLogPath:
    def test_returns_string(self):
        result = get_error_log_path()
        assert isinstance(result, str)

    def test_ends_with_dot_log(self):
        assert get_error_log_path().endswith(".log")

    def test_under_home_dir(self):
        home = os.path.expanduser("~")
        assert get_error_log_path().startswith(home)

    def test_contains_mullet_logs(self):
        path = get_error_log_path()
        assert ".mullet" in path and "logs" in path

    def test_directory_created(self):
        path = get_error_log_path()
        assert os.path.isdir(os.path.dirname(path))


# ---------------------------------------------------------------------------
# _FOLLOWUP_RE
# ---------------------------------------------------------------------------

class TestFollowupRe:
    def test_what_about_matches(self):
        assert _FOLLOWUP_RE.search("what about the tests")

    def test_how_about_matches(self):
        assert _FOLLOWUP_RE.search("how about the options")

    def test_also_matches(self):
        assert _FOLLOWUP_RE.search("also, what does it do")

    def test_same_matches(self):
        assert _FOLLOWUP_RE.search("same for Python")

    def test_self_contained_question_no_match(self):
        assert not _FOLLOWUP_RE.search("How do I install the package?")

    def test_short_question_matches(self):
        assert _FOLLOWUP_RE.search("tests?")

    def test_medium_length_word_matches(self):
        assert _FOLLOWUP_RE.search("more?")


# ---------------------------------------------------------------------------
# query_contextualize
# ---------------------------------------------------------------------------

class TestQueryContextualizePassthrough:
    def test_empty_history_returns_query(self):
        result = query_contextualize("How do I install Python?", [])
        assert result == "How do I install Python?"

    def test_empty_query_returned_unchanged(self):
        result = query_contextualize("", ["previous"])
        assert result == ""

    def test_self_contained_query_unchanged(self):
        result = query_contextualize("How do I configure logging?", ["earlier query"])
        assert result == "How do I configure logging?"

    def test_returns_string(self):
        result = query_contextualize("some query", [])
        assert isinstance(result, str)


class TestQueryContextualizeFollowUp:
    def test_followup_prefixed_with_prior_context(self):
        result = query_contextualize(
            "What about the tests?",
            ["How do I set up the database?"],
        )
        assert "How do I set up the database?" in result
        assert "What about the tests?" in result

    def test_followup_separated_by_dash(self):
        result = query_contextualize(
            "What about the tests?",
            ["How do I set up the database?"],
        )
        assert "—" in result

    def test_short_question_gets_context(self):
        result = query_contextualize("tests?", ["How do I configure the database?"])
        assert "configure the database" in result

    def test_also_query_gets_context(self):
        result = query_contextualize(
            "also, what is the timeout?",
            ["How does the retry logic work?"],
        )
        assert "retry logic" in result

    def test_skips_short_prior_queries(self):
        result = query_contextualize(
            "What about tests?",
            ["hi", "How do I configure the system logging?"],
        )
        assert "configure" in result

    def test_skips_prior_followup_queries(self):
        result = query_contextualize(
            "what about output?",
            ["also?", "How do I run the tests properly?"],
        )
        assert "run the tests" in result

    def test_no_suitable_prior_returns_original(self):
        result = query_contextualize("what about it?", ["hi", "ok", "yes"])
        assert result == "what about it?"


# ---------------------------------------------------------------------------
# _detect_test_framework
# ---------------------------------------------------------------------------

class TestDetectTestFramework:
    def test_returns_string(self):
        result = _detect_test_framework()
        assert isinstance(result, str)

    def test_returns_pytest_in_this_repo(self):
        result = _detect_test_framework()
        assert result == "pytest"

    def test_valid_framework_name(self):
        result = _detect_test_framework()
        assert result in ("pytest", "npm")


# ---------------------------------------------------------------------------
# TEST_RUNNER_SCHEMA
# ---------------------------------------------------------------------------

class TestRunnerSchema:
    def test_is_dict(self):
        assert isinstance(TEST_RUNNER_SCHEMA, dict)

    def test_has_description(self):
        assert "description" in TEST_RUNNER_SCHEMA

    def test_has_parameters(self):
        assert "parameters" in TEST_RUNNER_SCHEMA

    def test_parameters_has_path(self):
        props = TEST_RUNNER_SCHEMA["parameters"]["properties"]
        assert "path" in props

    def test_parameters_has_filter(self):
        props = TEST_RUNNER_SCHEMA["parameters"]["properties"]
        assert "filter" in props


# ---------------------------------------------------------------------------
# run_tests_tool — pure parsing / schema tests (no subprocess)
# ---------------------------------------------------------------------------

class TestRunTestsTool:
    def test_invalid_json_returns_error(self):
        result = run_tests_tool("{invalid json here")
        data = json.loads(result)
        assert "error" in data

    def test_kwargs_path_accepted(self):
        # Should not crash — may return error if test fails, but returns valid JSON
        result = run_tests_tool('{"path": "nonexistent_path_xyz"}')
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_pytest_summary_parsing(self):
        # Test the regex used inside _run_pytest directly
        output = "3 passed, 2 failed in 1.23s"
        m = re.search(
            r"(\d+)\s+passed(?:.*?(\d+)\s+failed)?(?:.*?(\d+)\s+error)?(?:.*?in\s+([\d.]+)s)?",
            output,
        )
        assert m is not None
        assert int(m.group(1)) == 3
        assert int(m.group(2)) == 2
        assert float(m.group(4)) == pytest.approx(1.23)

    def test_pytest_summary_no_failures(self):
        output = "10 passed in 0.5s"
        m = re.search(
            r"(\d+)\s+passed(?:.*?(\d+)\s+failed)?(?:.*?(\d+)\s+error)?(?:.*?in\s+([\d.]+)s)?",
            output,
        )
        assert m is not None
        assert int(m.group(1)) == 10
        assert m.group(2) is None

    def test_failed_lines_parsed(self):
        output = "FAILED tests/test_foo.py::TestBar::test_baz\n1 failed in 0.1s"
        failures = re.findall(r"FAILED\s+(\S+)", output)
        assert len(failures) == 1
        assert "test_foo" in failures[0]
