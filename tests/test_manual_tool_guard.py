"""Tests for mullet/hooks/manual_tool_guard.py and the escalation extension
in mullet/hooks/delegation_reminder.py."""

import importlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_HOOK_DIR = Path(__file__).resolve().parent.parent / "mullet" / "hooks"


def _load(name: str):
    """Load a hook module by filename without requiring package structure."""
    path = _HOOK_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Helpers to invoke the hook as a function rather than a subprocess
# ---------------------------------------------------------------------------


def _run_guard(
    tool_name: str,
    tool_input: dict | None = None,
    session_id: str = "test-session",
    bypass: bool = False,
    *,
    state_dir: str,
    budget_warranted: bool = True,
) -> tuple[int, str, str]:
    """Run the manual_tool_guard main() and return (exit_code, stdout, stderr).

    Patches sys.stdin, sys.exit, os.getppid, and /tmp so the state file is
    written under *state_dir* instead of the real /tmp.

    *budget_warranted* stubs the delegation budget gate. It defaults to True
    because these tests are about what the block says once it is justified; the
    gate itself is covered in test_delegation_budget_gate.py.
    """
    # Patch the module to write state files into our temp dir
    guard = _load("manual_tool_guard")
    guard.delegation_warranted = lambda *a, **k: (budget_warranted, "stubbed")

    payload = {"tool_name": tool_name, "session_id": session_id}
    if tool_input:
        payload["tool_input"] = tool_input

    fake_stdin = io.StringIO(json.dumps(payload))
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    exit_code = 0

    def fake_exit(code: int = 0) -> None:
        nonlocal exit_code
        exit_code = code
        raise SystemExit(code)

    # Redirect the state file path into our temp dir
    original_session_file = guard._session_file

    def patched_session_file(data: dict) -> Path:
        sid = data.get("session_id") or "ppid"
        return Path(state_dir) / f"claude_session_{sid}.json"

    env_patch = {"MULLET_GUARD_BYPASS": "1"} if bypass else {}
    # Ensure bypass env is clean when not requested
    clean_env = {k: v for k, v in os.environ.items() if k != "MULLET_GUARD_BYPASS"}
    clean_env.update(env_patch)

    with (
        patch.object(sys, "stdin", fake_stdin),
        patch.object(sys, "stdout", stdout_buf),
        patch.object(sys, "stderr", stderr_buf),
        patch.object(guard, "_session_file", patched_session_file),
        patch.dict(os.environ, clean_env, clear=True),
    ):
        try:
            result = guard.main()
            if result is not None:
                exit_code = int(result)
        except SystemExit as exc:
            exit_code = int(exc.code) if exc.code is not None else 0

    return exit_code, stdout_buf.getvalue(), stderr_buf.getvalue()


def _state_for(state_dir: str, session_id: str = "test-session") -> dict:
    p = Path(state_dir) / f"claude_session_{session_id}.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}
    return {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestManualToolGuard(unittest.TestCase):

    def setUp(self):
        # Fresh temp dir per test — isolated state
        self._tmp = tempfile.TemporaryDirectory()
        self.td = self._tmp.name
        # Force reimport so module-level state is clean
        guard = _load("manual_tool_guard")
        guard = _load("manual_tool_guard")

    def tearDown(self):
        self._tmp.cleanup()

    # -- first call: counter goes 0 → 1, not blocked -----------------------

    def test_first_call_allowed(self):
        code, _out, _err = _run_guard("Read", state_dir=self.td)
        self.assertEqual(code, 0)
        state = _state_for(self.td)
        self.assertEqual(state["manual_tools"], 1)

    # -- 6th call: counter reaches 6, still allowed -------------------------

    def test_sixth_call_allowed(self):
        for _ in range(5):
            _run_guard("Grep", state_dir=self.td)
        code, _out, _err = _run_guard("Read", state_dir=self.td)
        self.assertEqual(code, 0)
        state = _state_for(self.td)
        self.assertEqual(state["manual_tools"], 6)

    # -- 7th call: hard block -----------------------------------------------

    def test_seventh_call_blocked(self):
        for _ in range(6):
            _run_guard("Bash", tool_input={"command": "cat file.txt"}, state_dir=self.td)
        code, _out, err = _run_guard("Glob", state_dir=self.td)
        self.assertEqual(code, 2)
        self.assertIn("[manual-tool guard] Blocked", err)
        # Message branch (mcp__mullet__ vs Agent()) depends on MCP availability —
        # covered explicitly in test_mcp_*_message_used_when_mcp_*. Here just
        # assert that the block fires and names some delegation path.
        self.assertTrue(
            "mcp__mullet__mullet_run" in err or "Agent(" in err,
            f"Block message should recommend a delegation path, got: {err!r}",
        )

    # -- delegation resets the counter --------------------------------------

    def test_mullet_run_resets_counter(self):
        for _ in range(4):
            _run_guard("Read", state_dir=self.td)
        self.assertEqual(_state_for(self.td)["manual_tools"], 4)

        _run_guard("mcp__mullet__mullet_run", state_dir=self.td)
        self.assertEqual(_state_for(self.td)["manual_tools"], 0)

        # counter starts fresh
        _run_guard("Grep", state_dir=self.td)
        self.assertEqual(_state_for(self.td)["manual_tools"], 1)

    # -- BYPASS env prevents block ------------------------------------------

    def test_bypass_prevents_block(self):
        for _ in range(6):
            _run_guard("Read", state_dir=self.td)
        code, _out, err = _run_guard("Glob", bypass=True, state_dir=self.td)
        self.assertEqual(code, 0)
        self.assertIn("BYPASS", err)

    # -- allow-listed Bash commands never increment -------------------------

    def test_git_command_allowlisted(self):
        code, _out, _err = _run_guard(
            "Bash", tool_input={"command": "git status"}, state_dir=self.td
        )
        self.assertEqual(code, 0)
        state = _state_for(self.td)
        self.assertEqual(state.get("manual_tools", 0), 0)

    def test_pytest_command_allowlisted(self):
        code, _out, _err = _run_guard(
            "Bash",
            tool_input={"command": "python3 -m pytest tests/ -q"},
            state_dir=self.td,
        )
        self.assertEqual(code, 0)
        self.assertEqual(_state_for(self.td).get("manual_tools", 0), 0)

    def test_non_allowlisted_bash_counted(self):
        _run_guard("Bash", tool_input={"command": "cat somefile"}, state_dir=self.td)
        self.assertEqual(_state_for(self.td)["manual_tools"], 1)

    # -- non-counted tools are ignored --------------------------------------

    def test_uncounted_tool_ignored(self):
        _run_guard("Edit", state_dir=self.td)
        self.assertEqual(_state_for(self.td).get("manual_tools", 0), 0)

    # -- Task tool resets counter (counts as delegation) --------------------

    def test_task_resets_counter(self):
        for _ in range(3):
            _run_guard("Read", state_dir=self.td)
        _run_guard("Task", state_dir=self.td)
        self.assertEqual(_state_for(self.td)["manual_tools"], 0)

    # -- MCP-aware block message --------------------------------------------

    def test_mcp_fallback_message_used_when_mcp_unavailable(self):
        """When MCP isn't available, the guard should point at the Agent tool."""
        guard = _load("manual_tool_guard")

        payload = {"tool_name": "Glob", "session_id": "mcp-off"}
        fake_stdin = io.StringIO(json.dumps(payload))
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        def patched_sf(data: dict) -> Path:
            return Path(self.td) / f"claude_session_{data['session_id']}.json"

        # Pre-populate state so next call trips the block
        (Path(self.td) / "claude_session_mcp-off.json").write_text(
            json.dumps({"manual_tools": 6, "history": [], "last_delegation_ts": 0})
        )

        clean_env = {k: v for k, v in os.environ.items()
                     if k not in ("MULLET_GUARD_BYPASS", "MULLET_MCP_ENABLED")}

        with (
            patch.object(sys, "stdin", fake_stdin),
            patch.object(sys, "stdout", stdout_buf),
            patch.object(sys, "stderr", stderr_buf),
            patch.object(guard, "_session_file", patched_sf),
            patch.object(guard, "_is_mcp_available", lambda: False),
            patch.object(guard, "delegation_warranted", lambda *a, **k: (True, "stubbed")),
            patch.dict(os.environ, clean_env, clear=True),
        ):
            exit_code = guard.main()

        self.assertEqual(exit_code, 2)
        err = stderr_buf.getvalue()
        self.assertIn("Agent(", err)
        self.assertIn("subagent_type", err)
        self.assertNotIn("mcp__mullet__mullet_run", err)

    def test_mcp_message_used_when_mcp_available(self):
        """When MCP IS available, the guard should still recommend mullet_run."""
        guard = _load("manual_tool_guard")

        payload = {"tool_name": "Glob", "session_id": "mcp-on"}
        fake_stdin = io.StringIO(json.dumps(payload))
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        def patched_sf(data: dict) -> Path:
            return Path(self.td) / f"claude_session_{data['session_id']}.json"

        (Path(self.td) / "claude_session_mcp-on.json").write_text(
            json.dumps({"manual_tools": 6, "history": [], "last_delegation_ts": 0})
        )

        clean_env = {k: v for k, v in os.environ.items()
                     if k not in ("MULLET_GUARD_BYPASS",)}

        with (
            patch.object(sys, "stdin", fake_stdin),
            patch.object(sys, "stdout", stdout_buf),
            patch.object(sys, "stderr", stderr_buf),
            patch.object(guard, "_session_file", patched_sf),
            patch.object(guard, "_is_mcp_available", lambda: True),
            patch.object(guard, "delegation_warranted", lambda *a, **k: (True, "stubbed")),
            patch.dict(os.environ, clean_env, clear=True),
        ):
            exit_code = guard.main()

        self.assertEqual(exit_code, 2)
        err = stderr_buf.getvalue()
        self.assertIn("mcp__mullet__mullet_run", err)
        self.assertNotIn("Agent(", err)

    def test_is_mcp_available_env_var(self):
        """MULLET_MCP_ENABLED=1 should force availability to True."""
        guard = _load("manual_tool_guard")
        with patch.dict(os.environ, {"MULLET_MCP_ENABLED": "1"}, clear=False):
            self.assertTrue(guard._is_mcp_available())

    def test_is_mcp_available_default_false_when_no_signals(self):
        """No env, no PID marker → probe returns False, regardless of mcp
        package being importable. Package import is not a valid signal —
        it's installed almost everywhere even when MCP server is off."""
        guard = _load("manual_tool_guard")
        clean_env = {k: v for k, v in os.environ.items()
                     if k != "MULLET_MCP_ENABLED"}
        # Point the marker probe at a directory we guarantee is empty
        with patch.dict(os.environ, clean_env, clear=True), \
             patch.object(guard.Path, "home", return_value=Path(self.td)):
            self.assertFalse(guard._is_mcp_available())

    def test_mcp_seen_recently_detects_mullet_mcp_tool_in_history(self):
        """History containing an mcp__mullet__ tool call within the window
        means MCP is live, regardless of env/marker."""
        guard = _load("manual_tool_guard")
        import time as _t
        state = {
            "history": [
                {"tool": "mcp__mullet__mullet_run", "ts": int(_t.time()) - 60}
            ]
        }
        self.assertTrue(guard._mcp_seen_recently(state))

    def test_mcp_seen_recently_ignores_stale_entries(self):
        guard = _load("manual_tool_guard")
        import time as _t
        state = {
            "history": [
                {"tool": "mcp__mullet__mullet_run", "ts": int(_t.time()) - 3 * 3600}
            ]
        }
        self.assertFalse(guard._mcp_seen_recently(state))


# ---------------------------------------------------------------------------
# Tests for delegation_reminder escalation
# ---------------------------------------------------------------------------

def _run_reminder(
    prompt: str,
    session_id: str = "test-session",
    *,
    state_dir: str,
    manual_tools: int = 0,
    budget_warranted: bool = True,
) -> tuple[int, str, str]:
    dr = _load("delegation_reminder")
    # Same as above: default to "budget is under pressure" so these tests
    # exercise the message content. The gate is covered separately.
    dr.delegation_warranted = lambda *a, **k: (budget_warranted, "stubbed")

    # Pre-populate session state if needed
    if manual_tools > 0:
        sf = Path(state_dir) / f"claude_session_{session_id}.json"
        sf.write_text(json.dumps({"manual_tools": manual_tools, "last_delegation_ts": 0, "history": []}))

    payload = {"prompt": prompt, "session_id": session_id}
    fake_stdin = io.StringIO(json.dumps(payload))
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    exit_code = 0

    original_sf = dr._session_file

    def patched_sf(data: dict) -> Path:
        sid = data.get("session_id") or "ppid"
        return Path(state_dir) / f"claude_session_{sid}.json"

    with (
        patch.object(sys, "stdin", fake_stdin),
        patch.object(sys, "stdout", stdout_buf),
        patch.object(sys, "stderr", stderr_buf),
        patch.object(dr, "_session_file", patched_sf),
    ):
        try:
            result = dr.main()
            if result is not None:
                exit_code = int(result)
        except SystemExit as exc:
            exit_code = int(exc.code) if exc.code is not None else 0

    return exit_code, stdout_buf.getvalue(), stderr_buf.getvalue()


class TestDelegationReminderEscalation(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.td = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_escalated_reminder_when_count_high(self):
        """Non-matching prompt + manual_tools=6 → FINAL WARNING reminder with opencode."""
        _code, out, _err = _run_reminder(
            "Just a normal message with no keywords",
            manual_tools=6,
            state_dir=self.td,
        )
        self.assertIn("FINAL WARNING", out)
        self.assertIn("mullet_opencode", out)
        self.assertIn("6", out)

    def test_escalated_at_threshold_5(self):
        """Escalation triggers at exactly ESCALATION_THRESHOLD (5)."""
        _code, out, _err = _run_reminder(
            "nothing special",
            manual_tools=5,
            state_dir=self.td,
        )
        self.assertIn("ESCALATED", out)

    def test_no_escalation_below_threshold(self):
        """Below threshold + non-matching prompt → no output."""
        _code, out, _err = _run_reminder(
            "nothing special",
            manual_tools=4,
            state_dir=self.td,
        )
        self.assertNotIn("ESCALATED", out)
        self.assertNotIn("delegation reminder", out)

    def test_soft_reminder_still_fires(self):
        """Pattern match with low counter → soft reminder."""
        _code, out, _err = _run_reminder(
            "please plan out the roadmap for us",
            manual_tools=0,
            state_dir=self.td,
        )
        self.assertIn("[delegation suggestion", out)
        self.assertNotIn("ESCALATED", out)

    def test_escalation_message_at_count_5_says_two_more(self):
        """At count=5, two manual calls remain before block — message must say so."""
        _code, out, _err = _run_reminder(
            "nothing special",
            manual_tools=5,
            state_dir=self.td,
        )
        self.assertIn("ESCALATED", out)
        self.assertIn("Two more", out)
        self.assertNotIn("FINAL WARNING", out)

    def test_escalation_message_at_count_6_says_one_more(self):
        """At count=6, only ONE manual call remains — FINAL WARNING template."""
        _code, out, _err = _run_reminder(
            "nothing special",
            manual_tools=6,
            state_dir=self.td,
        )
        self.assertIn("FINAL WARNING", out)
        self.assertIn("One more", out)
        self.assertNotIn("Two more", out)

    def test_escalation_noop_at_count_7(self):
        """At count>=7 the guard has already blocked — reminder stays silent."""
        _code, out, _err = _run_reminder(
            "nothing special",
            manual_tools=7,
            state_dir=self.td,
        )
        self.assertNotIn("ESCALATED", out)
        self.assertNotIn("FINAL WARNING", out)
        self.assertNotIn("delegation reminder", out)


if __name__ == "__main__":
    unittest.main()
