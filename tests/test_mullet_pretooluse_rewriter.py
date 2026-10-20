"""Tests for the PreToolUse rewriter hook (P2-1 bridge integration seam)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

HOOK = Path(__file__).resolve().parent.parent / "scripts" / "hooks" / "mullet_pretooluse_rewriter.py"


def _load_hook_module():
    """Load the hook module fresh from its file path.

    Avoids any sys.modules pollution when the full test suite runs other
    tests that happen to touch the `scripts.*` namespace package.
    """
    spec = importlib.util.spec_from_file_location(
        "_mullet_pretooluse_rewriter_under_test", str(HOOK)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def hook_mod():
    return _load_hook_module()


def _run(payload: dict, extra_env: dict | None = None) -> tuple[int, str, str]:
    """Run the hook with payload on stdin. Returns (exit_code, stdout, stderr)."""
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        cwd=str(HOOK.parent.parent.parent),
    )
    return proc.returncode, proc.stdout, proc.stderr


def _parse_output(stdout: str) -> dict | None:
    stdout = stdout.strip()
    if not stdout:
        return None
    return json.loads(stdout)


class TestPassThrough:
    def test_non_mullet_tool_passes_through(self):
        rc, out, _ = _run({"tool_name": "Bash", "tool_input": {"command": "ls"}})
        assert rc == 0
        assert out.strip() == ""

    def test_unknown_mullet_tool_passes_through(self):
        # e.g. trawl_search is not a dispatch target
        rc, out, _ = _run({"tool_name": "mcp__mullet__trawl_search", "tool_input": {"query": "x"}})
        assert rc == 0
        assert out.strip() == ""

    def test_malformed_input_exits_zero(self):
        rc, out, err = _run({})
        assert rc == 0

    def test_empty_stdin_exits_zero(self):
        proc = subprocess.run(
            [sys.executable, str(HOOK)],
            input="",
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0

    def test_disabled_via_env_passes_through(self):
        rc, out, _ = _run(
            {"tool_name": "mcp__mullet__mullet_run", "tool_input": {"task": "x"}},
            extra_env={"MULLET_DISABLE_REWRITER": "1"},
        )
        assert rc == 0
        assert out.strip() == ""


class TestRewriter:
    """Unit tests via direct import — avoid subprocess env drag."""

    def test_pressure_mapping(self, hook_mod):
        _pressure_to_tier = hook_mod._pressure_to_tier
        assert _pressure_to_tier(0.1) == "ADVANCED"
        assert _pressure_to_tier(0.4) == "STANDARD"
        assert _pressure_to_tier(0.7) == "LOW"
        assert _pressure_to_tier(0.95) == "LOW"

    def test_explicit_tier_not_overwritten(self, hook_mod):
        _rewrite_input = hook_mod._rewrite_input
        with patch.object(hook_mod, "_get_session_override", return_value=None), \
             patch.object(hook_mod, "_get_budget_pressure", return_value=0.1):
            out = _rewrite_input(
                "mcp__mullet__mullet_run",
                {"task": "x", "tier": "CRITICAL"},
            )
            assert out is None  # user tier wins, no rewrite

    def test_autofill_tier_when_low_pressure(self, hook_mod):
        _rewrite_input = hook_mod._rewrite_input
        with patch.object(hook_mod, "_get_session_override", return_value=None), \
             patch.object(hook_mod, "_get_budget_pressure", return_value=0.15):
            out = _rewrite_input(
                "mcp__mullet__mullet_run",
                {"task": "x"},
            )
            assert out is not None
            assert out["tier"] == "ADVANCED"
            assert out["task"] == "x"  # original preserved

    def test_autofill_tier_when_high_pressure(self, hook_mod):
        _rewrite_input = hook_mod._rewrite_input
        with patch.object(hook_mod, "_get_session_override", return_value=None), \
             patch.object(hook_mod, "_get_budget_pressure", return_value=0.85):
            out = _rewrite_input(
                "mcp__mullet__mullet_run",
                {"task": "x"},
            )
            assert out is not None
            assert out["tier"] == "LOW"

    def test_override_pins_premium_and_providers(self, hook_mod):
        _rewrite_input = hook_mod._rewrite_input
        with patch.object(
            hook_mod, "_get_session_override",
            return_value={"provider": "opus-4-7", "model": "claude-opus-4-7"},
        ):
            out = _rewrite_input(
                "mcp__mullet__mullet_run",
                {"task": "x"},
            )
            assert out is not None
            assert out["tier"] == "PREMIUM"
            assert out["providers"].startswith("opus-4-7,")

    def test_override_respects_explicit_providers(self, hook_mod):
        """If user set providers themselves, the override shouldn't overwrite it."""
        _rewrite_input = hook_mod._rewrite_input
        with patch.object(
            hook_mod, "_get_session_override",
            return_value={"provider": "opus-4-7", "model": "claude-opus-4-7"},
        ):
            out = _rewrite_input(
                "mcp__mullet__mullet_run",
                {"task": "x", "providers": "kimi,gemini"},
            )
            # providers preserved; tier can still be filled
            assert out is not None
            assert out["providers"] == "kimi,gemini"
            assert out["tier"] == "PREMIUM"

    def test_override_respects_explicit_tier(self, hook_mod):
        _rewrite_input = hook_mod._rewrite_input
        with patch.object(
            hook_mod, "_get_session_override",
            return_value={"provider": "opus-4-7", "model": "claude-opus-4-7"},
        ):
            out = _rewrite_input(
                "mcp__mullet__mullet_run",
                {"task": "x", "tier": "STANDARD"},
            )
            # tier preserved; providers can still be filled
            assert out is not None
            assert out["tier"] == "STANDARD"
            assert "opus-4-7" in out["providers"]

    def test_override_all_fields_user_set_passthrough(self, hook_mod):
        """If caller set both providers AND tier, nothing to do."""
        _rewrite_input = hook_mod._rewrite_input
        with patch.object(
            hook_mod, "_get_session_override",
            return_value={"provider": "opus-4-7", "model": "claude-opus-4-7"},
        ):
            out = _rewrite_input(
                "mcp__mullet__mullet_run",
                {"task": "x", "providers": "kimi", "tier": "CRITICAL"},
            )
            assert out is None

    def test_budget_unavailable_passes_through(self, hook_mod):
        _rewrite_input = hook_mod._rewrite_input
        with patch.object(hook_mod, "_get_session_override", return_value=None), \
             patch.object(hook_mod, "_get_budget_pressure", return_value=None):
            out = _rewrite_input(
                "mcp__mullet__mullet_run",
                {"task": "x"},
            )
            assert out is None

    def test_delegate_tool_also_rewritten(self, hook_mod):
        _rewrite_input = hook_mod._rewrite_input
        with patch.object(hook_mod, "_get_session_override", return_value=None), \
             patch.object(hook_mod, "_get_budget_pressure", return_value=0.2):
            out = _rewrite_input(
                "mcp__mullet__mullet_delegate",
                {"task": "y"},
            )
            assert out is not None
            assert out["tier"] == "ADVANCED"

    def test_swarm_tool_also_rewritten(self, hook_mod):
        _rewrite_input = hook_mod._rewrite_input
        with patch.object(hook_mod, "_get_session_override", return_value=None), \
             patch.object(hook_mod, "_get_budget_pressure", return_value=0.2):
            out = _rewrite_input(
                "mcp__mullet__mullet_swarm",
                {"task": "z"},
            )
            assert out is not None
            assert out["tier"] == "ADVANCED"

    def test_non_dict_input_returns_none(self, hook_mod):
        _rewrite_input = hook_mod._rewrite_input
        assert _rewrite_input("mcp__mullet__mullet_run", None) is None
        assert _rewrite_input("mcp__mullet__mullet_run", "string") is None


class TestOutputShape:
    """Verify the emitted JSON matches the Claude Code PreToolUse hook contract."""

    def test_output_has_hook_specific_output_wrapper(self):
        # Stub session_state to force a rewrite, then run the full hook via subprocess.
        # We rely on the hook's import of session_state finding an override present.
        from mullet.core.session_state import set_override, clear_override
        clear_override()
        try:
            set_override(provider="opus-4-7", model="claude-opus-4-7")
            rc, out, _ = _run({
                "tool_name": "mcp__mullet__mullet_run",
                "tool_input": {"task": "do a thing"},
            })
            assert rc == 0
            parsed = _parse_output(out)
            assert parsed is not None
            hso = parsed["hookSpecificOutput"]
            assert hso["hookEventName"] == "PreToolUse"
            assert hso["permissionDecision"] == "allow"
            ui = hso["updatedInput"]
            assert ui["tier"] == "PREMIUM"
            assert "opus-4-7" in ui["providers"]
            assert ui["task"] == "do a thing"
        finally:
            clear_override()

    def test_no_output_when_no_rewrite(self):
        from mullet.core.session_state import clear_override
        clear_override()
        # Explicit tier + low-impact tool → no rewrite expected
        rc, out, _ = _run({
            "tool_name": "mcp__mullet__mullet_run",
            "tool_input": {"task": "x", "tier": "STANDARD"},
        })
        assert rc == 0
        assert out.strip() == ""
