"""Tests for `mullet shell --dry-run` preview mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from mullet.cli.harness_dry_run import build_dry_run_report, dry_run_preview


pytestmark = pytest.mark.harness


def _make_args(**overrides) -> argparse.Namespace:
    defaults: dict = {
        "prompt": None,
        "session_id": None,
        "provider": None,
        "model": None,
        "output_format": "text",
        "list_sessions": False,
        "dry_run": True,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_dry_run_returns_ready_with_full_config(
    isolated_profiles_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All defaults + KIMI_API_KEY set → verdict is 'ready'."""
    monkeypatch.setenv("KIMI_API_KEY", "sk-test-token")
    monkeypatch.chdir(tmp_path)
    args = _make_args(provider="kimi")
    report = build_dry_run_report(args, cwd=tmp_path)
    assert report.verdict == "ready", f"got {report.verdict}: {report.blockers}"
    assert report.profile["name"] == "kimi"
    assert report.auth["present"] is True
    assert report.next_steps  # non-empty next-step suggestion


def test_dry_run_returns_blocked_when_active_profile_missing_auth(
    isolated_profiles_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty KIMI_API_KEY → verdict is 'blocked' with auth blocker."""
    monkeypatch.setenv("KIMI_API_KEY", "")
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    args = _make_args(provider="kimi")
    report = build_dry_run_report(args, cwd=tmp_path)
    assert report.verdict == "blocked"
    assert any("auth missing" in b or "KIMI_API_KEY" in b for b in report.blockers)
    assert report.next_steps


def test_dry_run_returns_warning_for_missing_mcp_server(
    isolated_profiles_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A .mcp.json that points at a non-existent command should produce a warning."""
    monkeypatch.setenv("KIMI_API_KEY", "sk-test-token")
    monkeypatch.chdir(tmp_path)
    mcp_path = tmp_path / ".mcp.json"
    mcp_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "ghost": {
                        "command": "/nonexistent/path/ghost-mcp",
                        "args": [],
                    }
                }
            }
        )
    )
    args = _make_args(provider="kimi")
    report = build_dry_run_report(args, cwd=tmp_path)
    assert report.verdict == "warning"
    assert any("ghost" in w or "command not found" in w for w in report.warnings)


def test_dry_run_json_output_shape(
    isolated_profiles_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--output-format json produces valid JSON with expected top-level keys."""
    monkeypatch.setenv("KIMI_API_KEY", "sk-test-token")
    monkeypatch.chdir(tmp_path)
    args = _make_args(provider="kimi", output_format="json")
    rc = dry_run_preview(args)
    assert rc == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["verdict"] in {"ready", "warning", "blocked"}
    for key in (
        "verdict",
        "profile",
        "auth",
        "permissions",
        "skills",
        "plugins",
        "mcp",
        "session",
        "warnings",
        "blockers",
        "next_steps",
    ):
        assert key in data, f"missing top-level key: {key}"


def test_dry_run_blocked_returns_exit_1(
    isolated_profiles_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A blocked verdict must produce exit code 1."""
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    args = _make_args(provider="kimi")
    rc = dry_run_preview(args)
    captured = capsys.readouterr()
    assert "BLOCKED" in captured.out.upper()
    assert rc == 1


def test_dry_run_with_prompt_predicts_skills(
    isolated_profiles_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prompt mentioning 'commit' should match the commit built-in skill."""
    monkeypatch.setenv("KIMI_API_KEY", "sk-test-token")
    monkeypatch.chdir(tmp_path)
    args = _make_args(
        prompt="please commit the staged changes with a clear message",
        provider="kimi",
    )
    report = build_dry_run_report(args, cwd=tmp_path)
    pp = report.prompt_prediction
    assert pp, "prompt_prediction should be populated when prompt is provided"
    assert "commit" in pp.get("matched_skills", [])
    assert pp["prompt_preview"].startswith("please commit")
