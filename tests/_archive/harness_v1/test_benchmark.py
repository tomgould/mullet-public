"""Tests for the benchmark harness."""

from __future__ import annotations

import re
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from mullet.harness.benchmark.suite import (
    BenchmarkTask,
    BenchmarkResult,
    DEFAULT_SUITE,
    _grade,
    _estimate_cost,
    run_suite,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class _FakeLLMResponse:
    content: str = ""
    usage: _FakeUsage | None = None


def _make_provider(response_text: str = "", tokens_in: int = 10, tokens_out: int = 5) -> MagicMock:
    provider = MagicMock()
    provider.name = "mock"
    provider.model = "mock-model"
    provider.create.return_value = _FakeLLMResponse(
        content=response_text,
        usage=_FakeUsage(input_tokens=tokens_in, output_tokens=tokens_out),
    )
    return provider


def _make_failing_provider(exc: Exception = RuntimeError("boom")) -> MagicMock:
    provider = MagicMock()
    provider.name = "mock"
    provider.model = "mock-model"
    provider.create.side_effect = exc
    return provider


# ---------------------------------------------------------------------------
# _grade
# ---------------------------------------------------------------------------

def test_grade_substring_match():
    task = BenchmarkTask(name="t", prompt="p", expected_substrings=["hello", "world"])
    assert _grade(task, "hello world") is True
    assert _grade(task, "hello") is False


def test_grade_regex_match():
    task = BenchmarkTask(name="t", prompt="p", expected_regex=r"^\d+$")
    assert _grade(task, "123") is True
    assert _grade(task, "abc") is False


def test_grade_both_criteria():
    task = BenchmarkTask(name="t", prompt="p", expected_substrings=["a"], expected_regex=r"b")
    assert _grade(task, "ab") is True
    assert _grade(task, "a") is False


def test_grade_no_criteria():
    task = BenchmarkTask(name="t", prompt="p")
    assert _grade(task, "anything") is True


# ---------------------------------------------------------------------------
# _estimate_cost
# ---------------------------------------------------------------------------

def test_estimate_cost_known_model():
    # kimi-k2.6 pricing from model_info: 0.0006 / 1K in, 0.0025 / 1K out
    cost = _estimate_cost("kimi-k2.6", 1000, 1000)
    assert cost == pytest.approx(0.0006 + 0.0025, rel=1e-6)


def test_estimate_cost_unknown_model():
    cost = _estimate_cost("nonexistent-model-xyz", 1000, 1000)
    assert cost == 0.0


# ---------------------------------------------------------------------------
# run_suite
# ---------------------------------------------------------------------------

def test_run_suite_passes_with_mock_provider():
    provider = _make_provider(response_text="HELLO_WORLD")
    factory = lambda name, model: provider  # noqa: E731

    results = run_suite("mock", tasks=DEFAULT_SUITE[:1], engine_factory=factory)

    assert len(results) == 1
    assert results[0].success is True
    assert results[0].task_name == "hello"
    provider.create.assert_called_once()


def test_run_suite_grades_substring_match():
    provider = _make_provider(response_text="here is def is_palindrome and return")
    factory = lambda name, model: provider  # noqa: E731

    task = BenchmarkTask(name="code", prompt="write code", expected_substrings=["def is_palindrome", "return"])
    results = run_suite("mock", tasks=[task], engine_factory=factory)

    assert results[0].success is True


def test_run_suite_grades_regex_match():
    provider = _make_provider(response_text="apple\nbanana\ncherry")
    factory = lambda name, model: provider  # noqa: E731

    task = BenchmarkTask(name="lines", prompt="list", expected_regex=r"^[a-z]+\n[a-z]+\n[a-z]+")
    results = run_suite("mock", tasks=[task], engine_factory=factory)

    assert results[0].success is True


def test_run_suite_handles_provider_error():
    provider = _make_failing_provider(RuntimeError("network timeout"))
    factory = lambda name, model: provider  # noqa: E731

    results = run_suite("mock", tasks=DEFAULT_SUITE[:1], engine_factory=factory)

    assert len(results) == 1
    assert results[0].success is False
    assert "network timeout" in (results[0].error or "")


def test_run_suite_returns_cost_estimate():
    provider = _make_provider(response_text="HELLO_WORLD", tokens_in=2000, tokens_out=1000)
    factory = lambda name, model: provider  # noqa: E731

    # Patch get_pricing so we get predictable numbers
    with patch("mullet.harness.benchmark.suite.get_pricing", return_value=(0.001, 0.002)):
        results = run_suite("mock", tasks=DEFAULT_SUITE[:1], engine_factory=factory)

    assert results[0].cost_usd == pytest.approx(2.0 * 0.001 + 1.0 * 0.002, rel=1e-6)


def test_run_suite_unresolvable_provider():
    factory = lambda name, model: None  # noqa: E731

    results = run_suite("ghost", tasks=DEFAULT_SUITE[:2], engine_factory=factory)

    assert len(results) == 2
    for r in results:
        assert r.success is False
        assert "could not be resolved" in (r.error or "")


def test_run_suite_model_attribute_fallback():
    """Provider with no 'model' attr falls back to the passed model."""
    provider = MagicMock()
    provider.name = "fallback"
    # No model attribute
    del provider.model
    provider.create.return_value = _FakeLLMResponse(content="ok", usage=_FakeUsage(1, 1))
    factory = lambda name, model: provider  # noqa: E731

    results = run_suite("fallback", model="my-model", tasks=DEFAULT_SUITE[:1], engine_factory=factory)
    assert results[0].model == "my-model"


# ---------------------------------------------------------------------------
# DEFAULT_SUITE sanity
# ---------------------------------------------------------------------------

def test_default_suite_has_unique_names():
    names = [t.name for t in DEFAULT_SUITE]
    assert len(names) == len(set(names))


def test_default_suite_instruction_follow_regex():
    task = next(t for t in DEFAULT_SUITE if t.name == "instruction_follow")
    assert task.expected_regex is not None
    # Should match three lowercase words on separate lines
    assert re.search(task.expected_regex, "apple\nbanana\ncherry", re.MULTILINE)
    assert not re.search(task.expected_regex, "Apple\nBanana\nCherry", re.MULTILINE)


# ---------------------------------------------------------------------------
# BenchmarkResult defaults
# ---------------------------------------------------------------------------

def test_benchmark_result_defaults():
    r = BenchmarkResult(task_name="t", provider="p", model="m", success=True, response="", tokens_in=0, tokens_out=0, cost_usd=0.0, duration_seconds=0.0)
    assert r.error is None
