"""Tests for the shootout CLI."""

from __future__ import annotations

import json
from unittest.mock import patch

from mullet.cli.shootout_cli import cmd_shootout
from mullet.harness.benchmark.suite import BenchmarkResult


def _make_result(
    task_name: str,
    success: bool,
    error: str | None = None,
    duration: float = 0.5,
    cost: float = 0.001,
) -> BenchmarkResult:
    return BenchmarkResult(
        task_name=task_name,
        provider="mock",
        model="mock-model",
        success=success,
        response="ok",
        tokens_in=10,
        tokens_out=5,
        cost_usd=cost,
        duration_seconds=duration,
        error=error,
    )


def test_shootout_runs_multiple_providers(capsys):
    def mock_run_suite(provider_name, tasks=None, **kwargs):
        return [_make_result(t.name, True) for t in (tasks or [])]

    with patch("mullet.cli.shootout_cli.run_suite", side_effect=mock_run_suite):
        rc = cmd_shootout(["--providers", "kimi,gemini", "--task", "hello"])

    assert rc == 0
    captured = capsys.readouterr().out
    assert "kimi" in captured
    assert "gemini" in captured
    assert "hello" in captured


def test_shootout_table_format(capsys):
    def mock_run_suite(provider_name, tasks=None, **kwargs):
        return [_make_result(t.name, provider_name == "kimi") for t in (tasks or [])]

    with patch("mullet.cli.shootout_cli.run_suite", side_effect=mock_run_suite):
        rc = cmd_shootout(["--providers", "kimi,gemini", "--task", "hello"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "kimi" in out
    assert "gemini" in out
    assert "hello" in out


def test_shootout_one_provider_failure_doesnt_break_others():
    call_count = 0

    def mock_run_suite(provider_name, tasks=None, **kwargs):
        nonlocal call_count
        call_count += 1
        if provider_name == "bad":
            raise RuntimeError("boom")
        return [_make_result(t.name, True) for t in (tasks or [])]

    with patch("mullet.cli.shootout_cli.run_suite", side_effect=mock_run_suite):
        rc = cmd_shootout(["--providers", "good,bad", "--task", "hello"])

    assert rc == 0
    assert call_count == 2


def test_shootout_json_output(capsys):
    def mock_run_suite(provider_name, tasks=None, **kwargs):
        return [_make_result(t.name, True) for t in (tasks or [])]

    with patch("mullet.cli.shootout_cli.run_suite", side_effect=mock_run_suite):
        rc = cmd_shootout(["--providers", "a", "--task", "hello", "--json"])

    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "a" in data
    assert data["a"][0]["task_name"] == "hello"
    assert data["a"][0]["success"] is True


def test_shootout_default_providers():
    def mock_run_suite(provider_name, tasks=None, **kwargs):
        return [_make_result(t.name, True) for t in (tasks or [])]

    with patch("mullet.cli.shootout_cli._default_providers", return_value=["kimi", "gemini"]):
        with patch("mullet.cli.shootout_cli.run_suite", side_effect=mock_run_suite):
            rc = cmd_shootout(["--task", "hello"])

    assert rc == 0
