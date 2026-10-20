"""Tests for the ``mullet provider`` CLI dispatcher."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mullet.cli.provider_cli import cmd_provider
from mullet.harness.profiles.builtins import BUILTIN_PROFILE_NAMES
from mullet.harness.profiles.manager import ProfileManager


pytestmark = pytest.mark.harness


# ----- list ----------------------------------------------------------------


def test_list_human_output_shows_builtins(isolated_profiles_dir: Path, capsys, monkeypatch) -> None:
    # Strip auth env vars so the status string is deterministic.
    for var in ("KIMI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                "GEMINI_API_KEY", "OPENROUTER_API_KEY", "AWS_ACCESS_KEY_ID"):
        monkeypatch.delenv(var, raising=False)

    rc = cmd_provider(["list"])
    captured = capsys.readouterr()
    assert rc == 0
    out = captured.out
    # Each built-in profile should appear in the output
    for name in BUILTIN_PROFILE_NAMES:
        assert name in out
    # And each line includes a status tag
    assert "[ready]" in out or "[missing auth]" in out or "[unavailable]" in out


def test_list_marks_active_with_asterisk(isolated_profiles_dir: Path, capsys) -> None:
    rc = cmd_provider(["list"])
    captured = capsys.readouterr()
    assert rc == 0
    # Default active profile is "kimi"
    assert "* kimi:" in captured.out
    # Non-active profiles have two-space indent rather than "* "
    assert "  ollama:" in captured.out


def test_list_json_output_valid(isolated_profiles_dir: Path, capsys) -> None:
    rc = cmd_provider(["list", "--json"])
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert "active" in payload
    assert "profiles" in payload
    assert payload["active"] == "kimi"
    assert isinstance(payload["profiles"], list)
    assert len(payload["profiles"]) >= 5
    sample = payload["profiles"][0]
    for key in ("name", "label", "status", "auth_source", "model", "base_url"):
        assert key in sample


# ----- add -----------------------------------------------------------------


def test_add_minimal(isolated_profiles_dir: Path, capsys) -> None:
    rc = cmd_provider([
        "add", "my-test",
        "--label", "T",
        "--provider", "kimi",
        "--api-format", "openai",
        "--auth-source", "FOO",
        "--model", "bar",
    ])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Saved provider profile: my-test" in captured.out

    mgr = ProfileManager()
    config = mgr.load()
    profile = next((p for p in config.profiles if p.name == "my-test"), None)
    assert profile is not None
    assert profile.label == "T"
    assert profile.provider_type == "kimi"
    assert profile.api_format == "openai"
    assert profile.auth_source == "FOO"
    assert profile.model == "bar"


def test_add_with_base_url(isolated_profiles_dir: Path, capsys) -> None:
    rc = cmd_provider([
        "add", "my-test",
        "--label", "T",
        "--provider", "openai_compat",
        "--api-format", "openai",
        "--auth-source", "FOO",
        "--model", "bar",
        "--base-url", "https://custom.example.com/v1",
    ])
    assert rc == 0

    mgr = ProfileManager()
    profile = next(p for p in mgr.load().profiles if p.name == "my-test")
    assert profile.base_url == "https://custom.example.com/v1"


def test_add_with_headers(isolated_profiles_dir: Path, capsys) -> None:
    rc = cmd_provider([
        "add", "my-test",
        "--label", "T",
        "--provider", "openai_compat",
        "--api-format", "openai",
        "--auth-source", "FOO",
        "--model", "bar",
        "--header", "K1=V1",
        "--header", "K2=V2",
    ])
    assert rc == 0

    mgr = ProfileManager()
    profile = next(p for p in mgr.load().profiles if p.name == "my-test")
    assert profile.extra_headers == {"K1": "V1", "K2": "V2"}


def test_add_invalid_api_format(isolated_profiles_dir: Path, capsys) -> None:
    rc = cmd_provider([
        "add", "my-test",
        "--label", "T",
        "--provider", "kimi",
        "--api-format", "bogus",
        "--auth-source", "FOO",
        "--model", "bar",
    ])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.err  # argparse writes a "invalid choice" message to stderr


def test_add_duplicate(isolated_profiles_dir: Path, capsys) -> None:
    args = [
        "add", "my-test",
        "--label", "T",
        "--provider", "kimi",
        "--api-format", "openai",
        "--auth-source", "FOO",
        "--model", "bar",
    ]
    rc1 = cmd_provider(args)
    assert rc1 == 0
    capsys.readouterr()  # drain
    rc2 = cmd_provider(args)
    captured = capsys.readouterr()
    assert rc2 == 1
    assert "already exists" in captured.err


# ----- use -----------------------------------------------------------------


def test_use_existing(isolated_profiles_dir: Path, capsys, monkeypatch) -> None:
    # Provide an env var so no warning is emitted
    monkeypatch.setenv("OPENAI_API_KEY", "abc123")
    rc = cmd_provider(["use", "openai"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Activated provider profile: openai" in captured.out

    mgr = ProfileManager()
    assert mgr.load().active == "openai"


def test_use_unknown(isolated_profiles_dir: Path, capsys) -> None:
    rc = cmd_provider(["use", "nope-not-real"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "unknown profile" in captured.err


def test_use_warns_on_missing_auth_env_var(
    isolated_profiles_dir: Path, capsys, monkeypatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    rc = cmd_provider(["use", "openai"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "warning" in captured.err.lower()
    # Activation still happens
    mgr = ProfileManager()
    assert mgr.load().active == "openai"


# ----- remove --------------------------------------------------------------


def test_remove_user_profile_succeeds(isolated_profiles_dir: Path, capsys) -> None:
    cmd_provider([
        "add", "my-test",
        "--label", "T",
        "--provider", "kimi",
        "--api-format", "openai",
        "--auth-source", "FOO",
        "--model", "bar",
    ])
    capsys.readouterr()
    rc = cmd_provider(["remove", "my-test"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Removed provider profile: my-test" in captured.out

    mgr = ProfileManager()
    assert all(p.name != "my-test" for p in mgr.load().profiles)


def test_remove_active_without_force(isolated_profiles_dir: Path, capsys) -> None:
    # The default active profile is "kimi" — also a built-in.
    rc = cmd_provider(["remove", "kimi"])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.err


def test_remove_active_with_force(isolated_profiles_dir: Path, capsys) -> None:
    rc = cmd_provider(["remove", "kimi", "--force"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Removed provider profile: kimi" in captured.out

    mgr = ProfileManager()
    config = mgr.load()
    assert all(p.name != "kimi" for p in config.profiles)
    assert config.active != "kimi"


def test_remove_builtin_without_force(isolated_profiles_dir: Path, capsys) -> None:
    # "ollama" is a built-in but NOT the active profile. Should still refuse.
    rc = cmd_provider(["remove", "ollama"])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.err


def test_remove_builtin_with_force(isolated_profiles_dir: Path, capsys) -> None:
    rc = cmd_provider(["remove", "ollama", "--force"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Removed provider profile: ollama" in captured.out


# ----- show ----------------------------------------------------------------


def test_show_existing(isolated_profiles_dir: Path, capsys) -> None:
    rc = cmd_provider(["show", "kimi"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "name: kimi" in captured.out
    assert "label:" in captured.out
    assert "auth_source:" in captured.out
    assert "model:" in captured.out


def test_show_unknown(isolated_profiles_dir: Path, capsys) -> None:
    rc = cmd_provider(["show", "no-such-profile"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "unknown profile" in captured.err


# ----- dispatcher ----------------------------------------------------------


def test_no_subaction_prints_help_returns_0(isolated_profiles_dir: Path, capsys) -> None:
    rc = cmd_provider([])
    captured = capsys.readouterr()
    assert rc == 0
    assert "mullet provider" in captured.out
    assert "add" in captured.out and "list" in captured.out


def test_unknown_subaction(isolated_profiles_dir: Path, capsys) -> None:
    rc = cmd_provider(["frobnicate"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "unknown subaction" in captured.err


def test_help_flag_prints_usage(isolated_profiles_dir: Path, capsys) -> None:
    rc = cmd_provider(["--help"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "mullet provider" in captured.out
