"""Unit tests for mullet.tools.opencode_tool main functions.

Covers _mullet_run_fallback and opencode_run (dispatches via
`opencode run --format json`, a one-shot subprocess per attempt — not the
ACP/JSON-RPC session protocol, which silently ignores per-call model
overrides; see opencode_tool.py's module docstring).
"""

import json
import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

import mullet.tools.opencode_tool as _oc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_proc(events, stderr_lines=None, exit_code=0):
    """Build a mock Popen simulating `opencode run --format json` output.

    events: list of dicts (JSON event lines, e.g. {"type": "text", ...}) or
    plain strings (raw lines, e.g. to test malformed-JSON handling). An EOF
    marker ("") is appended automatically.
    """
    proc = MagicMock()
    encoded = []
    for e in events:
        if isinstance(e, dict):
            encoded.append(json.dumps(e) + "\n")
        elif e == "":
            encoded.append("")
        else:
            encoded.append(str(e) + "\n")
    encoded.append("")  # final EOF
    proc.stdout.readline.side_effect = encoded
    if stderr_lines:
        proc.stderr.readline.side_effect = [l + "\n" for l in stderr_lines] + [""]
    else:
        proc.stderr.readline.return_value = ""
    proc.poll.side_effect = [None] * 50 + [exit_code]
    proc.wait.return_value = exit_code
    proc.kill.return_value = None
    return proc


def _model_from_argv(args) -> str:
    """Extract the --model value from a Popen call's positional argv arg."""
    argv = args[0]
    return argv[argv.index("--model") + 1]


def _text_event(text, session_id="s1"):
    return {"type": "text", "sessionID": session_id, "part": {"text": text}}


def _step_finish_event(total=0, input=0, output=0, cost=0.0, session_id="s1"):
    return {
        "type": "step_finish", "sessionID": session_id,
        "part": {"tokens": {"total": total, "input": input, "output": output}, "cost": cost},
    }


def _error_event(message, session_id="s1"):
    return {"type": "error", "sessionID": session_id, "error": {"data": {"message": message}}}


def _tool_use_event(tool="edit", status="completed", title="file.py", session_id="s1"):
    return {
        "type": "tool_use", "sessionID": session_id,
        "part": {"tool": tool, "title": title, "state": {"status": status}},
    }


# ---------------------------------------------------------------------------
# _mullet_run_fallback
# ---------------------------------------------------------------------------

def test_mullet_run_fallback_success():
    with patch("mullet.server.mcp_handlers.handle_mullet_run", return_value={"response": "done", "duration": 1.5}):
        result = _oc._mullet_run_fallback("task", "/tmp")
    assert result["completed"] is True
    assert result["response"] == "done"
    assert result["fallback"] == "mullet_run/kimi"
    assert result["duration"] == 1.5


def test_mullet_run_fallback_error_key():
    with patch("mullet.server.mcp_handlers.handle_mullet_run", return_value={"error": "msg"}):
        result = _oc._mullet_run_fallback("task", "/tmp")
    assert result["completed"] is True
    assert result["response"] == "msg"


def test_mullet_run_fallback_exception():
    with patch("mullet.server.mcp_handlers.handle_mullet_run", side_effect=RuntimeError("kimi down")):
        result = _oc._mullet_run_fallback("task", "/tmp")
    assert "error" in result
    assert "install_hint" in result
    assert "kimi down" in result["error"]


# ---------------------------------------------------------------------------
# opencode_run — high-level paths
# ---------------------------------------------------------------------------

def test_opencode_not_installed_uses_fallback(monkeypatch):
    fallback = {"completed": True, "response": "ok", "fallback": "mullet_run/kimi", "warning": "w"}
    monkeypatch.setattr(_oc, "_mullet_run_fallback", lambda task, cwd: fallback)
    with patch("mullet.tools.opencode_tool.shutil.which", return_value=None):
        result = _oc.opencode_run("task", "/tmp")
    assert result == fallback


def test_popen_raises_error():
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", side_effect=FileNotFoundError("not found")):
        result = _oc.opencode_run("task", "/tmp")
    assert result["error"].startswith("failed to spawn opencode")


def test_dispatch_argv_shape(monkeypatch):
    """opencode_run must spawn `opencode run <task> --model X --format json
    --dir Y --auto`, non-interactively (no stdin pipe needed)."""
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    proc = _make_mock_proc([_text_event("hi"), _step_finish_event(total=5)])
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", return_value=proc) as mock_popen, \
         patch("mullet.tools.opencode_tool.select.select", return_value=([proc.stdout], [], [])), \
         patch("mullet.core.routing_history.make_outcome"), \
         patch("mullet.core.routing_history.record"):
        _oc.opencode_run("do the thing", "/some/dir", model="openrouter/deepseek/deepseek-v4-flash")
    args, kwargs = mock_popen.call_args
    argv = args[0]
    assert "do the thing" in argv
    assert "--model" in argv and argv[argv.index("--model") + 1] == "openrouter/deepseek/deepseek-v4-flash"
    assert "--format" in argv and argv[argv.index("--format") + 1] == "json"
    assert "--dir" in argv and argv[argv.index("--dir") + 1] == "/some/dir"
    assert "--auto" in argv
    assert kwargs["stdin"] == subprocess.DEVNULL


# ---------------------------------------------------------------------------
# opencode_run — full success path
# ---------------------------------------------------------------------------

def test_full_success_path(monkeypatch):
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    events = [
        {"type": "step_start", "sessionID": "sess-abc"},
        _tool_use_event(),
        _step_finish_event(total=20, input=15, output=5, cost=0.0, session_id="sess-abc"),
        _text_event("Hello world", session_id="sess-abc"),
        _step_finish_event(total=50, input=20, output=30, cost=0.001, session_id="sess-abc"),
    ]
    proc = _make_mock_proc(events)
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", return_value=proc), \
         patch("mullet.tools.opencode_tool.select.select", return_value=([proc.stdout], [], [])), \
         patch("mullet.core.routing_history.make_outcome"), \
         patch("mullet.core.routing_history.record"):
        result = _oc.opencode_run("task", "/tmp")
    assert result["completed"] is True
    assert result["response"] == "Hello world"
    assert result["session_id"] == "sess-abc"
    # Tokens/cost summed across BOTH step_finish events, not just the last.
    assert result["usage"]["total_tokens"] == 70
    assert result["usage"]["input_tokens"] == 35
    assert result["usage"]["output_tokens"] == 35
    assert result["usage"]["cost_usd"] == 0.001


def test_success_with_multiple_text_chunks(monkeypatch):
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    events = [_text_event("Hello"), _text_event(" world"), _step_finish_event(total=10)]
    proc = _make_mock_proc(events)
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", return_value=proc), \
         patch("mullet.tools.opencode_tool.select.select", return_value=([proc.stdout], [], [])), \
         patch("mullet.core.routing_history.make_outcome"), \
         patch("mullet.core.routing_history.record"):
        result = _oc.opencode_run("task", "/tmp")
    assert result["response"] == "Hello world"
    assert result["usage"]["total_tokens"] == 10


def test_success_with_empty_text(monkeypatch):
    """No text event at all (e.g. a pure tool-call task) -> placeholder text."""
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    proc = _make_mock_proc([_step_finish_event(total=5)])
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", return_value=proc), \
         patch("mullet.tools.opencode_tool.select.select", return_value=([proc.stdout], [], [])), \
         patch("mullet.core.routing_history.make_outcome"), \
         patch("mullet.core.routing_history.record"):
        result = _oc.opencode_run("task", "/tmp")
    assert result["response"] == "(no text output)"


def test_success_missing_tokens_key_defaults_to_zero(monkeypatch):
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    events = [_text_event("ok"), {"type": "step_finish", "sessionID": "s1", "part": {}}]
    proc = _make_mock_proc(events)
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", return_value=proc), \
         patch("mullet.tools.opencode_tool.select.select", return_value=([proc.stdout], [], [])), \
         patch("mullet.core.routing_history.make_outcome"), \
         patch("mullet.core.routing_history.record"):
        result = _oc.opencode_run("task", "/tmp")
    assert result["completed"] is True
    assert result["usage"]["total_tokens"] == 0
    assert result["usage"]["cost_usd"] == 0


def test_session_id_captured_from_any_event(monkeypatch):
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    events = [{"type": "step_start", "sessionID": "sess-alt"}, _step_finish_event(session_id="sess-alt")]
    proc = _make_mock_proc(events)
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", return_value=proc), \
         patch("mullet.tools.opencode_tool.select.select", return_value=([proc.stdout], [], [])), \
         patch("mullet.core.routing_history.make_outcome"), \
         patch("mullet.core.routing_history.record"):
        result = _oc.opencode_run("task", "/tmp")
    assert result["session_id"] == "sess-alt"


# ---------------------------------------------------------------------------
# opencode_run — event-stream edge cases
# ---------------------------------------------------------------------------

def test_json_decode_error_skipped(monkeypatch):
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    events = [_text_event("ok"), "not json", _step_finish_event(total=1)]
    proc = _make_mock_proc(events)
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", return_value=proc), \
         patch("mullet.tools.opencode_tool.select.select", return_value=([proc.stdout], [], [])), \
         patch("mullet.core.routing_history.make_outcome"), \
         patch("mullet.core.routing_history.record"):
        result = _oc.opencode_run("task", "/tmp")
    assert result["completed"] is True
    assert result["response"] == "ok"


def test_text_event_no_text_key_ignored(monkeypatch):
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    events = [{"type": "text", "sessionID": "s1", "part": {}}, _step_finish_event()]
    proc = _make_mock_proc(events)
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", return_value=proc), \
         patch("mullet.tools.opencode_tool.select.select", return_value=([proc.stdout], [], [])), \
         patch("mullet.core.routing_history.make_outcome"), \
         patch("mullet.core.routing_history.record"):
        result = _oc.opencode_run("task", "/tmp")
    assert result["response"] == "(no text output)"


def test_tool_use_event_logged_not_crash(monkeypatch):
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    events = [_tool_use_event(tool="read"), _tool_use_event(tool="edit"), _text_event("Done"), _step_finish_event(total=1)]
    proc = _make_mock_proc(events)
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", return_value=proc), \
         patch("mullet.tools.opencode_tool.select.select", return_value=([proc.stdout], [], [])), \
         patch("mullet.core.routing_history.make_outcome"), \
         patch("mullet.core.routing_history.record"):
        result = _oc.opencode_run("task", "/tmp")
    assert result["completed"] is True
    assert result["response"] == "Done"


def test_unknown_event_type_ignored(monkeypatch):
    """Event types outside the known set (text/tool_use/step_finish/error)
    must be safely ignored, not crash — the schema may grow."""
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    events = [{"type": "step_start", "sessionID": "s1"}, {"type": "some_future_event", "sessionID": "s1"}, _step_finish_event(total=1)]
    proc = _make_mock_proc(events)
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", return_value=proc), \
         patch("mullet.tools.opencode_tool.select.select", return_value=([proc.stdout], [], [])), \
         patch("mullet.core.routing_history.make_outcome"), \
         patch("mullet.core.routing_history.record"):
        result = _oc.opencode_run("task", "/tmp")
    assert result["completed"] is True


# ---------------------------------------------------------------------------
# opencode_run — error event fails fast (not silent-timeout)
# ---------------------------------------------------------------------------

def test_error_event_fails_fast_not_silent_timeout(monkeypatch):
    """A real opencode `error` event (bad params, context overflow, auth,
    etc.) must be surfaced immediately, not silently discarded until
    SILENT_TIMEOUT_S elapses with a misleading 'API may be overloaded'."""
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    procs = []
    def _factory(*a, **k):
        p = _make_mock_proc([_error_event("Internal error: n_keep: 10842 >= n_ctx: 4096")])
        procs.append(p)
        return p
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", side_effect=_factory), \
         patch("mullet.tools.opencode_tool.select.select", side_effect=lambda *a: ([procs[-1].stdout], [], [])):
        result = _oc.opencode_run("task", "/tmp")
    # Fails fast on BOTH attempts (the error is deterministic, re-triggered
    # identically on the escalated model too) -- but critically, the error
    # message must be the REAL opencode error, not a timeout/overload string.
    assert "n_keep: 10842" in result["error"]
    assert "overloaded" not in result["error"]
    assert "no output for" not in result["error"]
    assert result["error_reason"] == "context_overflow"


def test_error_event_kills_still_running_process(monkeypatch):
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    procs = []
    def _factory(*a, **k):
        p = _make_mock_proc([_error_event("boom")])
        p.poll.side_effect = None
        p.poll.return_value = None  # still "running" when the error event arrives
        procs.append(p)
        return p
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", side_effect=_factory), \
         patch("mullet.tools.opencode_tool.select.select", side_effect=lambda *a: ([procs[-1].stdout], [], [])):
        _oc.opencode_run("task", "/tmp")
    assert procs[0].kill.called


def test_transient_error_reason_classified(monkeypatch):
    """A non-context error (e.g. a genuine rate-limit/overload) must be
    classified as 'transient', not 'context_overflow'."""
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    procs = []
    def _factory(*a, **k):
        p = _make_mock_proc([_error_event("429 rate limit exceeded, try again later")])
        procs.append(p)
        return p
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", side_effect=_factory), \
         patch("mullet.tools.opencode_tool.select.select", side_effect=lambda *a: ([procs[-1].stdout], [], [])):
        result = _oc.opencode_run("task", "/tmp")
    assert result["error_reason"] == "transient"


def test_mid_dispatch_context_overflow_escalates_to_largest_not_next_rung(monkeypatch):
    """A context_overflow error discovered DURING dispatch (not caught by the
    pre-flight guard, e.g. opencode's own system-prompt overhead) must also
    use the largest-context escalation, not the normal small-step chain —
    which would otherwise send openrouter/deepseek-v4-flash ->
    moonshot/kimi-k2.7-code (262K, SMALLER), but the largest remaining
    context_overflow candidate is qwen3-coder (1,048,576)."""
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    responses = [
        [_error_event("n_keep: 5000 >= n_ctx: 4096")],
        [_text_event("ok"), _step_finish_event(total=1)],
    ]
    procs = []
    models_used = []
    def _factory(*a, **k):
        models_used.append(_model_from_argv(a))
        p = _make_mock_proc(responses[len(procs)])
        procs.append(p)
        return p
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", side_effect=_factory), \
         patch("mullet.tools.opencode_tool.select.select", side_effect=lambda *a: ([procs[-1].stdout], [], [])), \
         patch("mullet.core.routing_history.make_outcome"), \
         patch("mullet.core.routing_history.record"):
        result = _oc.opencode_run("task", "/tmp", model="openrouter/deepseek/deepseek-v4-flash")
    assert models_used == ["openrouter/deepseek/deepseek-v4-flash", "openrouter/qwen/qwen3-coder"]
    assert result["completed"] is True


def test_final_failure_records_real_error_message(monkeypatch):
    """The routing-history record on final failure must carry the REAL
    error message (error_message), not just the exception class name
    (error_class) — the whole point of the fix is that this be diagnosable
    from the log without live reproduction."""
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    procs = []
    def _factory(*a, **k):
        p = _make_mock_proc([_error_event("n_keep: 10842 >= n_ctx: 4096")])
        procs.append(p)
        return p
    captured = {}
    def _capture_outcome(*a, **kw):
        captured.update(kw)
        return MagicMock()
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", side_effect=_factory), \
         patch("mullet.tools.opencode_tool.select.select", side_effect=lambda *a: ([procs[-1].stdout], [], [])), \
         patch("mullet.core.routing_history.make_outcome", side_effect=_capture_outcome), \
         patch("mullet.core.routing_history.record"):
        _oc.opencode_run("task", "/tmp")
    assert captured["error_class"] == "RuntimeError"
    assert "n_keep: 10842" in captured["error_message"]


# ---------------------------------------------------------------------------
# opencode_run — retry / failure paths
# ---------------------------------------------------------------------------

def test_no_output_and_nonzero_exit_retries_then_fails(monkeypatch):
    """Process exits (EOF) with a non-zero code and no events at all (e.g.
    crashed before producing output) -> treated as a failure, retried."""
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    procs = []
    def _factory(*a, **k):
        p = _make_mock_proc([], exit_code=1, stderr_lines=["fatal: something broke"])
        procs.append(p)
        return p
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", side_effect=_factory), \
         patch("mullet.tools.opencode_tool.select.select", side_effect=lambda *a: ([procs[-1].stdout], [], [])):
        result = _oc.opencode_run("task", "/tmp")
    assert len(procs) == 2
    assert "failed after 2 attempts" in result["error"]
    assert "exited with code 1" in result["error"]


def test_silent_timeout_kills_then_retries(monkeypatch):
    monkeypatch.setenv("OPENCODE_SILENT_TIMEOUT_S", "0")
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    ticks = iter(range(100))
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=lambda: next(ticks)))
    procs = []
    select_after_popen = 0
    def _factory(*a, **k):
        nonlocal select_after_popen
        select_after_popen = 0
        # Two harmless events so the first two "ready" select ticks are
        # consumed by real reads, not immediate EOF -- otherwise the loop
        # exits before ever reaching the silent-timeout check below.
        p = _make_mock_proc([{"type": "step_start", "sessionID": "s1"}, {"type": "step_start", "sessionID": "s1"}])
        procs.append(p)
        return p
    def _select(*a):
        nonlocal select_after_popen
        select_after_popen += 1
        if select_after_popen <= 2:
            return ([procs[-1].stdout], [], [])
        return ([], [], [])
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", side_effect=_factory), \
         patch("mullet.tools.opencode_tool.select.select", side_effect=_select):
        result = _oc.opencode_run("task", "/tmp")
    assert len(procs) == 2
    assert "failed after 2 attempts" in result["error"]
    assert procs[0].kill.called
    assert procs[1].kill.called


def test_model_escalation_on_retry(monkeypatch):
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    procs = []
    models_used = []
    def _factory(*a, **k):
        models_used.append(_model_from_argv(a))
        p = _make_mock_proc([], exit_code=1)
        procs.append(p)
        return p
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", side_effect=_factory), \
         patch("mullet.tools.opencode_tool.select.select", side_effect=lambda *a: ([procs[-1].stdout], [], [])):
        result = _oc.opencode_run("task", "/tmp", model="local/my-model")
    assert len(procs) == 2
    assert models_used == ["local/my-model", "openrouter/deepseek/deepseek-v4-flash"]
    assert "failed after 2 attempts" in result["error"]


def test_openrouter_model_escalates_to_kimi_on_retry(monkeypatch):
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    procs = []
    models_used = []
    def _factory(*a, **k):
        models_used.append(_model_from_argv(a))
        p = _make_mock_proc([], exit_code=1)
        procs.append(p)
        return p
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", side_effect=_factory), \
         patch("mullet.tools.opencode_tool.select.select", side_effect=lambda *a: ([procs[-1].stdout], [], [])):
        _oc.opencode_run("task", "/tmp", model="openrouter/deepseek/deepseek-v4-flash")
    assert models_used == ["openrouter/deepseek/deepseek-v4-flash", "moonshot/kimi-k2.7-code"]


def test_generic_exception_no_retry():
    """An unexpected low-level I/O error (not part of the normal
    RuntimeError/TimeoutError retry path) must surface immediately, and any
    still-running process must be killed."""
    proc = _make_mock_proc([])
    proc.stdout.readline.side_effect = OSError("broken pipe")
    proc.poll.return_value = None
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", return_value=proc), \
         patch("mullet.tools.opencode_tool.select.select", return_value=([proc.stdout], [], [])):
        result = _oc.opencode_run("task", "/tmp")
    assert result["error"].startswith("opencode run error")
    assert "broken pipe" in result["error"]
    assert proc.kill.called


def test_second_attempt_popen_raises(monkeypatch):
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    procs = []
    def _factory(*a, **k):
        if not procs:
            p = _make_mock_proc([], exit_code=1)
            procs.append(p)
            return p
        raise FileNotFoundError("gone")
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", side_effect=_factory), \
         patch("mullet.tools.opencode_tool.select.select", side_effect=lambda *a: ([procs[0].stdout], [], [])):
        result = _oc.opencode_run("task", "/tmp")
    assert "failed to spawn opencode" in result["error"]


def test_wait_timeout_expired_then_killed(monkeypatch):
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    proc = _make_mock_proc([_text_event("ok"), _step_finish_event(total=1)])
    proc.wait.side_effect = [subprocess.TimeoutExpired("opencode", 5), 0]
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", return_value=proc), \
         patch("mullet.tools.opencode_tool.select.select", return_value=([proc.stdout], [], [])), \
         patch("mullet.core.routing_history.make_outcome"), \
         patch("mullet.core.routing_history.record"):
        result = _oc.opencode_run("task", "/tmp")
    assert result["completed"] is True
    assert proc.kill.called


def test_routing_history_exception_swallowed(monkeypatch):
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    proc = _make_mock_proc([_text_event("ok"), _step_finish_event(total=1)])
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", return_value=proc), \
         patch("mullet.tools.opencode_tool.select.select", return_value=([proc.stdout], [], [])), \
         patch("mullet.core.routing_history.make_outcome", side_effect=RuntimeError("boom")), \
         patch("mullet.core.routing_history.record"):
        result = _oc.opencode_run("task", "/tmp")
    assert result["completed"] is True


# ---------------------------------------------------------------------------
# opencode_run — pre-flight context guard (delegation-safety gap fix)
# ---------------------------------------------------------------------------

def test_preflight_overflow_skips_dispatch_and_escalates(monkeypatch):
    """A task too big for the starting model must never spawn a subprocess
    for that model — it escalates straight to the LARGEST-context candidate
    instead of burning a doomed 240s attempt (context_overflow escalation,
    not the normal small-step _escalate_model chain)."""
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    monkeypatch.setattr(_oc, "estimate_tokens", lambda task: 1000)
    _ctx = {
        "local/my-model": 100,
        "openrouter/qwen/qwen3-coder": 1_048_576,
        "openrouter/deepseek/deepseek-v4-flash": 1_000_000,
        "moonshot/kimi-k2.7-code": 262_144,
    }
    monkeypatch.setattr(_oc, "get_context_length", lambda model: _ctx.get(model, 10_000_000))
    procs = []
    models_used = []
    def _factory(*a, **k):
        models_used.append(_model_from_argv(a))
        p = _make_mock_proc([_text_event("ok"), _step_finish_event(total=1)])
        procs.append(p)
        return p
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", side_effect=_factory), \
         patch("mullet.tools.opencode_tool.select.select", side_effect=lambda *a: ([procs[-1].stdout], [], [])), \
         patch("mullet.core.routing_history.make_outcome"), \
         patch("mullet.core.routing_history.record"):
        result = _oc.opencode_run("task", "/tmp", model="local/my-model")
    # Exactly one subprocess spawned: attempt 1 (local/my-model, 100 ctx) was
    # skipped entirely by the pre-flight check; attempt 2 escalated straight
    # to the LARGEST-context candidate (qwen3-coder, 1,048,576) and
    # succeeded there — not deepseek-v4-flash, which is smaller.
    assert len(procs) == 1
    assert models_used == ["openrouter/qwen/qwen3-coder"]
    assert result["completed"] is True


def test_preflight_overflow_at_top_of_chain_fails_fast(monkeypatch):
    """Already on Kimi (top of the escalation chain) with an oversized task:
    fail immediately without spawning any subprocess at all."""
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    monkeypatch.setattr(_oc, "estimate_tokens", lambda task: 1000)
    monkeypatch.setattr(_oc, "get_context_length", lambda model: 100)
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen") as mock_popen:
        result = _oc.opencode_run("task", "/tmp", model="moonshot/kimi-k2.7-code")
    mock_popen.assert_not_called()
    assert "exceeds" in result["error"]


def test_default_model_escalates_on_retry(monkeypatch):
    """opencode/big-pickle (the true no-model-specified default) previously
    matched neither escalation branch and silently retried the same model on
    failure. It must now escalate like any other starting point."""
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    procs = []
    models_used = []
    def _factory(*a, **k):
        models_used.append(_model_from_argv(a))
        p = _make_mock_proc([], exit_code=1)
        procs.append(p)
        return p
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", side_effect=_factory), \
         patch("mullet.tools.opencode_tool.select.select", side_effect=lambda *a: ([procs[-1].stdout], [], [])):
        result = _oc.opencode_run("task", "/tmp", model="opencode/big-pickle")
    assert len(procs) == 2
    assert models_used == ["opencode/big-pickle", "openrouter/deepseek/deepseek-v4-flash"]
    assert "failed after 2 attempts" in result["error"]


# ---------------------------------------------------------------------------
# opencode_run — files_changed scope report
# ---------------------------------------------------------------------------

def test_files_changed_reported_on_success(monkeypatch):
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    statuses = iter([{"a.py"}, {"a.py", "tsconfig.json"}])
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: next(statuses))
    proc = _make_mock_proc([_text_event("ok"), _step_finish_event(total=1)])
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", return_value=proc), \
         patch("mullet.tools.opencode_tool.select.select", return_value=([proc.stdout], [], [])), \
         patch("mullet.core.routing_history.make_outcome"), \
         patch("mullet.core.routing_history.record"):
        result = _oc.opencode_run("task", "/tmp")
    assert result["files_changed"] == ["tsconfig.json"]


def test_files_changed_none_when_git_status_unavailable(monkeypatch):
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    proc = _make_mock_proc([_text_event("ok"), _step_finish_event(total=1)])
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", return_value=proc), \
         patch("mullet.tools.opencode_tool.select.select", return_value=([proc.stdout], [], [])), \
         patch("mullet.core.routing_history.make_outcome"), \
         patch("mullet.core.routing_history.record"):
        result = _oc.opencode_run("task", "/tmp")
    assert result["files_changed"] is None


def test_files_changed_reported_on_final_failure(monkeypatch):
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    statuses = iter([{"a.py"}, {"a.py", "stray.json"}])
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: next(statuses))
    procs = []
    def _factory(*a, **k):
        p = _make_mock_proc([], exit_code=1)
        procs.append(p)
        return p
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", side_effect=_factory), \
         patch("mullet.tools.opencode_tool.select.select", side_effect=lambda *a: ([procs[-1].stdout], [], [])):
        result = _oc.opencode_run("task", "/tmp")
    assert result["files_changed"] == ["stray.json"]


# ---------------------------------------------------------------------------
# _escalate_model / _git_status_snapshot — unit tests
# ---------------------------------------------------------------------------

def test_escalate_model_chain():
    assert _oc._escalate_model("local/my-model") == "openrouter/deepseek/deepseek-v4-flash"
    assert _oc._escalate_model("lmstudio/qwen") == "openrouter/deepseek/deepseek-v4-flash"
    assert _oc._escalate_model("openrouter/deepseek/deepseek-v4-flash") == "moonshot/kimi-k2.7-code"
    assert _oc._escalate_model("moonshot/kimi-k2.7-code") is None
    assert _oc._escalate_model("kimi-k2.6") is None
    assert _oc._escalate_model("opencode/big-pickle") == "openrouter/deepseek/deepseek-v4-flash"


# ---------------------------------------------------------------------------
# _classify_error / _escalate_for_context_overflow
# ---------------------------------------------------------------------------

def test_classify_error_context_overflow():
    assert _oc._classify_error("n_keep: 10842 >= n_ctx: 4096") == "context_overflow"
    assert _oc._classify_error("Context window exceeded") == "context_overflow"
    assert _oc._classify_error("Too many tokens in prompt") == "context_overflow"


def test_classify_error_auth():
    assert _oc._classify_error("401 Unauthorized") == "auth"
    assert _oc._classify_error("Invalid API key provided") == "auth"
    assert _oc._classify_error("403 Forbidden") == "auth"


def test_classify_error_transient():
    assert _oc._classify_error("no output for 240s — API may be overloaded") == "transient"
    assert _oc._classify_error("Request timed out") == "transient"
    assert _oc._classify_error("429 rate limit exceeded") == "transient"
    assert _oc._classify_error("opencode process exited unexpectedly") == "transient"


def test_classify_error_unknown_fallback():
    assert _oc._classify_error("something completely unrecognised happened") == "unknown"


def test_escalate_for_context_overflow_picks_largest():
    # qwen3-coder (1,048,576) > deepseek-v4-flash (1,000,000) > kimi (262,144)
    assert _oc._escalate_for_context_overflow("local/my-model") == "openrouter/qwen/qwen3-coder"


def test_escalate_for_context_overflow_skips_current_model():
    # Already on the largest candidate — must pick the next-largest, not itself.
    assert _oc._escalate_for_context_overflow("openrouter/qwen/qwen3-coder") == "openrouter/deepseek/deepseek-v4-flash"


def test_escalate_for_context_overflow_none_when_no_candidates_left(monkeypatch):
    monkeypatch.setattr(_oc, "_CONTEXT_OVERFLOW_ESCALATION_CANDIDATES", ("only/one",))
    assert _oc._escalate_for_context_overflow("only/one") is None


# ---------------------------------------------------------------------------
# known-bad-model cache (_mark_model_bad / _known_bad_reason / _load_known_bad_models)
# ---------------------------------------------------------------------------

def test_mark_and_check_known_bad_model(tmp_path, monkeypatch):
    monkeypatch.setattr(_oc, "_KNOWN_BAD_MODELS_PATH", tmp_path / "known_bad.json")
    assert _oc._known_bad_reason("lmstudio/qwen2.5-coder-14b-instruct") is None
    _oc._mark_model_bad("lmstudio/qwen2.5-coder-14b-instruct", "context_overflow", "n_keep: 10842 >= n_ctx: 4096")
    assert _oc._known_bad_reason("lmstudio/qwen2.5-coder-14b-instruct") == "context_overflow"
    # A different model is unaffected.
    assert _oc._known_bad_reason("openrouter/deepseek/deepseek-v4-flash") is None


def test_known_bad_expires_after_ttl(tmp_path, monkeypatch):
    monkeypatch.setattr(_oc, "_KNOWN_BAD_MODELS_PATH", tmp_path / "known_bad.json")
    monkeypatch.setattr(_oc, "_KNOWN_BAD_TTL_S", 60)
    _oc._mark_model_bad("local/x", "context_overflow", "boom")
    # Still fresh.
    assert _oc._known_bad_reason("local/x") == "context_overflow"
    # Simulate the entry aging past the TTL by rewriting its timestamp directly.
    data = json.loads((tmp_path / "known_bad.json").read_text())
    data["local/x"]["ts"] = time.time() - 120
    (tmp_path / "known_bad.json").write_text(json.dumps(data))
    assert _oc._known_bad_reason("local/x") is None


def test_load_known_bad_models_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(_oc, "_KNOWN_BAD_MODELS_PATH", tmp_path / "does-not-exist.json")
    assert _oc._load_known_bad_models() == {}


def test_load_known_bad_models_corrupt_file_returns_empty(tmp_path, monkeypatch):
    p = tmp_path / "known_bad.json"
    p.write_text("{not json")
    monkeypatch.setattr(_oc, "_KNOWN_BAD_MODELS_PATH", p)
    assert _oc._load_known_bad_models() == {}


def test_mark_model_bad_swallows_write_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(_oc, "_KNOWN_BAD_MODELS_PATH", tmp_path / "no" / "such" / "dir" / "known_bad.json")
    with patch("mullet.tools.opencode_tool.tempfile.mkstemp", side_effect=OSError("disk full")):
        _oc._mark_model_bad("local/x", "context_overflow", "boom")  # must not raise


def test_known_bad_model_skipped_without_dispatch(monkeypatch):
    """A model already marked context_overflow-bad must be skipped entirely
    (no subprocess spawned for it) and escalate immediately."""
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    monkeypatch.setattr(_oc, "_known_bad_reason", lambda model: "context_overflow" if model == "lmstudio/qwen2.5-coder-14b-instruct" else None)
    procs = []
    models_used = []
    def _factory(*a, **k):
        models_used.append(_model_from_argv(a))
        p = _make_mock_proc([_text_event("ok"), _step_finish_event(total=1)])
        procs.append(p)
        return p
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", side_effect=_factory), \
         patch("mullet.tools.opencode_tool.select.select", side_effect=lambda *a: ([procs[-1].stdout], [], [])), \
         patch("mullet.core.routing_history.make_outcome"), \
         patch("mullet.core.routing_history.record"):
        result = _oc.opencode_run("task", "/tmp", model="lmstudio/qwen2.5-coder-14b-instruct")
    assert len(procs) == 1
    assert models_used == ["openrouter/qwen/qwen3-coder"]
    assert result["completed"] is True


def test_real_context_overflow_failure_marks_model_bad(tmp_path, monkeypatch):
    """A REAL (mid-dispatch, not pre-flight) context_overflow error must
    persist to the known-bad cache so the NEXT call skips it proactively —
    this is what the static pre-flight guard can't do on its own, since it
    only knows model_info.py's ASSUMED context, not what's actually loaded
    (exactly the incident this fix was written for: LM Studio loaded at
    4096 tokens while model_info.py/opencode.json assumed 32768)."""
    monkeypatch.setattr(_oc, "_KNOWN_BAD_MODELS_PATH", tmp_path / "known_bad.json")
    # monotonic frozen for deterministic silence-timeout math; .time() left
    # as a real wall-clock float since _mark_model_bad persists it to disk
    # (a JSON-serialisable MagicMock is not, and the write is soft-failed).
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0), time=time.time))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    procs = []
    def _factory(*a, **k):
        p = _make_mock_proc([_error_event("n_keep: 10842 >= n_ctx: 4096")])
        procs.append(p)
        return p
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", side_effect=_factory), \
         patch("mullet.tools.opencode_tool.select.select", side_effect=lambda *a: ([procs[-1].stdout], [], [])):
        _oc.opencode_run("task", "/tmp", model="lmstudio/qwen2.5-coder-14b-instruct")
    assert _oc._known_bad_reason("lmstudio/qwen2.5-coder-14b-instruct") == "context_overflow"


def test_git_status_snapshot_parses_short_status(tmp_path):
    import subprocess as _subprocess
    _subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "new_file.txt").write_text("hi")
    result = _oc._git_status_snapshot(str(tmp_path))
    assert result == {"new_file.txt"}


def test_git_status_snapshot_returns_none_on_error(tmp_path):
    result = _oc._git_status_snapshot(str(tmp_path / "does-not-exist"))
    assert result is None


# ---------------------------------------------------------------------------
# opencode_run — HUD accuracy on escalation (agent_id threading)
# ---------------------------------------------------------------------------

def test_escalation_corrects_hud_model(monkeypatch):
    """On retry-escalation, the HUD entry must be corrected to the model that
    is ACTUALLY about to run — not left showing the failed attempt's model."""
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    calls = []
    monkeypatch.setattr(
        "mullet.hud.agent_state.update_agent_model",
        lambda agent_id, provider, model: calls.append((agent_id, provider, model)),
    )
    procs = []
    def _factory(*a, **k):
        p = _make_mock_proc([], exit_code=1)
        procs.append(p)
        return p
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", side_effect=_factory), \
         patch("mullet.tools.opencode_tool.select.select", side_effect=lambda *a: ([procs[-1].stdout], [], [])):
        _oc.opencode_run("task", "/tmp", model="local/my-model", agent_id="opencode-abc123")
    assert calls == [("opencode-abc123", "openrouter", "openrouter/deepseek/deepseek-v4-flash")]


def test_no_hud_update_when_agent_id_missing(monkeypatch):
    """Without an agent_id, opencode_run must never touch HUD state at all."""
    monkeypatch.setattr(_oc, "time", MagicMock(monotonic=MagicMock(return_value=0)))
    monkeypatch.setattr(_oc, "_git_status_snapshot", lambda cwd: None)
    calls = []
    monkeypatch.setattr(
        "mullet.hud.agent_state.update_agent_model",
        lambda *a, **k: calls.append((a, k)),
    )
    procs = []
    def _factory(*a, **k):
        p = _make_mock_proc([], exit_code=1)
        procs.append(p)
        return p
    with patch("mullet.tools.opencode_tool.shutil.which", return_value="/usr/bin/opencode"), \
         patch("mullet.tools.opencode_tool.subprocess.Popen", side_effect=_factory), \
         patch("mullet.tools.opencode_tool.select.select", side_effect=lambda *a: ([procs[-1].stdout], [], [])):
        _oc.opencode_run("task", "/tmp", model="local/my-model")
    assert calls == []


def test_update_hud_model_swallows_exceptions(monkeypatch):
    """A HUD write failure during escalation must never break the delegation."""
    monkeypatch.setattr(
        "mullet.hud.agent_state.update_agent_model",
        MagicMock(side_effect=RuntimeError("disk full")),
    )
    _oc._update_hud_model("agent-1", "openrouter/deepseek/deepseek-v4-flash")  # must not raise
