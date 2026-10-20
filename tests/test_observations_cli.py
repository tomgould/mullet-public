"""Additional tests for mullet.cli.observations_cmd to cover missing lines."""
import io
import sys
from unittest.mock import patch

import pytest

from mullet.core.observations import Observation
from mullet.cli.observations_cmd import run
from mullet.core.observation_summary import SessionSummary


def _make_obs(sid, tool, preview, ts="2026-05-02T12:34:56Z"):
    return Observation(
        timestamp=ts,
        session_id=sid,
        tool_name=tool,
        tool_input_preview=preview,
        tool_output_preview="out",
        cwd="/tmp",
    )


def _capture_run(args, monkeypatch, tmp_path):
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    try:
        code = cmd_mod.run(args)
    except SystemExit as e:
        code = int(e.code) if e.code is not None else 0
    return code, buf.getvalue()


# ---------------------------------------------------------------------------
# Tail follow mode — covers lines 91-96 and 106-125
# ---------------------------------------------------------------------------

def test_tail_follow_non_session_new_rows(monkeypatch, tmp_path):
    """Follow loop (non-session) detects and prints new rows."""
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)

    obs1 = _make_obs("s1", "Read", "file1")
    obs2 = _make_obs("s1", "Read", "file2")

    call_count = 0

    def fake_iter_recent(limit, base):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return iter([obs1])
        return iter([obs2, obs1])

    monkeypatch.setattr(cmd_mod, "iter_recent", fake_iter_recent)

    sleep_count = 0

    def fake_sleep(s):
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count >= 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(cmd_mod.time, "sleep", fake_sleep)

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    code = cmd_mod.run(["tail", "--follow", "--limit", "1"])
    assert code == 0
    out = buf.getvalue()
    assert "file1" in out


def test_tail_follow_session_new_rows(monkeypatch, tmp_path):
    """Follow loop (session) detects and prints new rows."""
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)

    obs1 = _make_obs("ses-a", "Read", "file1")
    obs2 = _make_obs("ses-a", "Read", "file2")

    call_count = 0

    def fake_iter_session(session, base):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return iter([obs1])
        return iter([obs1, obs2])

    monkeypatch.setattr(cmd_mod, "iter_session", fake_iter_session)

    sleep_count = 0

    def fake_sleep(s):
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count >= 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(cmd_mod.time, "sleep", fake_sleep)

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    code = cmd_mod.run(["tail", "--session", "ses-a", "--follow", "--limit", "1"])
    assert code == 0
    out = buf.getvalue()
    assert "file1" in out


def test_tail_initial_session(monkeypatch, tmp_path):
    """Tail without follow but with session uses the session initial path."""
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)

    obs = [_make_obs("ses-b", "Read", f"file{i}") for i in range(5)]
    monkeypatch.setattr(cmd_mod, "iter_session", lambda session, base: iter(obs))

    code, out = _capture_run(["tail", "--session", "ses-b", "--limit", "3"], monkeypatch, tmp_path)
    assert code == 0
    lines = [l for l in out.strip().splitlines() if l.strip()]
    assert len(lines) == 3


# ---------------------------------------------------------------------------
# Summarise — covers default window, provider choices, and save edge cases
# ---------------------------------------------------------------------------

def test_summarise_default_window(monkeypatch, tmp_path):
    """Summarise with no --window defaults to 60 minutes."""
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)
    summary = SessionSummary(
        session_id="window-60m",
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T01:00:00Z",
        tool_counts={"Read": 2},
        files_touched=[],
        summary="default window",
    )
    with patch("mullet.core.observation_summary.summarise_window", return_value=summary) as mock_sw:
        buf = io.StringIO()
        monkeypatch.setattr(sys, "stdout", buf)
        code = cmd_mod.run(["summarise"])
    assert code == 0
    mock_sw.assert_called_once_with(minutes=60, provider="auto", base=tmp_path)


def test_summarise_provider_local(monkeypatch, tmp_path):
    """Summarise respects --provider local."""
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)
    summary = SessionSummary(
        session_id="s1",
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T01:00:00Z",
        tool_counts={"Read": 1},
        files_touched=[],
        summary="local provider",
    )
    with patch("mullet.core.observation_summary.summarise_session", return_value=summary) as mock_ss:
        buf = io.StringIO()
        monkeypatch.setattr(sys, "stdout", buf)
        code = cmd_mod.run(["summarise", "--session", "s1", "--provider", "local"])
    assert code == 0
    mock_ss.assert_called_once_with("s1", provider="local", base=tmp_path)


def test_summarise_provider_haiku(monkeypatch, tmp_path):
    """Summarise respects --provider haiku."""
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)
    summary = SessionSummary(
        session_id="s1",
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T01:00:00Z",
        tool_counts={"Read": 1},
        files_touched=[],
        summary="haiku provider",
    )
    with patch("mullet.core.observation_summary.summarise_session", return_value=summary) as mock_ss:
        buf = io.StringIO()
        monkeypatch.setattr(sys, "stdout", buf)
        code = cmd_mod.run(["summarise", "--session", "s1", "--provider", "haiku"])
    assert code == 0
    mock_ss.assert_called_once_with("s1", provider="haiku", base=tmp_path)


def test_summarise_save_memory_sync_fails_open(monkeypatch, tmp_path):
    """Save continues even when memory_sync module is missing."""
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)
    summary = SessionSummary(
        session_id="s1",
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T01:00:00Z",
        tool_counts={"Read": 1},
        files_touched=[],
        summary="save test",
    )
    with patch("mullet.core.observation_summary.summarise_session", return_value=summary):
        with patch("mullet.core.memory_sync.sync_observation_summary", side_effect=ImportError("no module")):
            buf = io.StringIO()
            monkeypatch.setattr(sys, "stdout", buf)
            code = cmd_mod.run(["summarise", "--session", "s1", "--save"])
    assert code == 0
    out = buf.getvalue()
    assert "Saved:" in out


# ---------------------------------------------------------------------------
# Error / edge paths
# ---------------------------------------------------------------------------

def test_search_missing_query(monkeypatch, tmp_path):
    """Search without a query causes argparse to exit."""
    code, out = _capture_run(["search"], monkeypatch, tmp_path)
    assert code != 0


# ---------------------------------------------------------------------------
# No-color import path — covers lines 22-23, 28, 45
# ---------------------------------------------------------------------------

def test_no_color_import_path():
    """Patch mullet.hud.colors to raise on import, exercising _HAS_COLOR=False."""
    import importlib
    import sys

    import mullet.cli

    # Save current module states
    orig_cmd = sys.modules.get("mullet.cli.observations_cmd")
    orig_colors = sys.modules.get("mullet.hud.colors")
    orig_attr = getattr(mullet.cli, "observations_cmd", None)

    class FakeColors:
        def __getattr__(self, name):
            raise ImportError("no colors")

    # Force re-import of observations_cmd with broken colors
    sys.modules["mullet.hud.colors"] = FakeColors()
    sys.modules.pop("mullet.cli.observations_cmd", None)

    try:
        import mullet.cli.observations_cmd as nocolor_mod
        assert nocolor_mod._HAS_COLOR is False
        # Line 28: _tool_color returns unchanged when _HAS_COLOR is False
        assert nocolor_mod._tool_color("Read") == "Read"
        # Line 45: _fmt_row without dim formatting when _HAS_COLOR is False
        obs = _make_obs("s1", "Read", "file1")
        row = nocolor_mod._fmt_row(obs)
        assert "2026-05-02 12:34" in row
        assert "Read" in row
        assert "file1" in row
    finally:
        # Restore original modules and parent-package attribute
        if orig_cmd is not None:
            sys.modules["mullet.cli.observations_cmd"] = orig_cmd
        else:
            sys.modules.pop("mullet.cli.observations_cmd", None)
        if orig_colors is not None:
            sys.modules["mullet.hud.colors"] = orig_colors
        else:
            sys.modules.pop("mullet.hud.colors", None)
        if orig_attr is not None:
            sys.modules["mullet.cli"].observations_cmd = orig_attr
        importlib.import_module("mullet.cli.observations_cmd")


# ---------------------------------------------------------------------------
# _project_root — covers lines 50-54
# ---------------------------------------------------------------------------

def test_project_root_finds_parent_mullet(monkeypatch, tmp_path):
    """_project_root walks upward and finds .mullet dir in a parent."""
    import mullet.cli.observations_cmd as cmd_mod
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / ".mullet").mkdir()
    subdir = parent / "child" / "grandchild"
    subdir.mkdir(parents=True)
    monkeypatch.chdir(subdir)
    assert cmd_mod._project_root() == parent


def test_project_root_fallback_cwd(monkeypatch, tmp_path):
    """_project_root returns cwd when no .mullet dir is found anywhere upward."""
    import mullet.cli.observations_cmd as cmd_mod
    empty = tmp_path / "empty"
    empty.mkdir()

    class FakePath:
        def __init__(self, p):
            self._p = p
        @classmethod
        def cwd(cls):
            return cls(empty)
        def __truediv__(self, other):
            return FakePath(self._p / other)
        def is_dir(self):
            return False
        @property
        def parents(self):
            return []
        def __eq__(self, other):
            return self._p == other

    monkeypatch.setattr(cmd_mod, "Path", FakePath)
    assert cmd_mod._project_root() == empty


# ---------------------------------------------------------------------------
# _run_list — covers lines 58-77
# ---------------------------------------------------------------------------

def test_list_with_session(monkeypatch, tmp_path):
    """list with --session calls iter_session and prints formatted rows."""
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)
    obs = [_make_obs("ses1", "Read", f"file{i}") for i in range(2)]
    monkeypatch.setattr(cmd_mod, "iter_session", lambda sid, base: iter(obs))
    code, out = _capture_run(["list", "--session", "ses1"], monkeypatch, tmp_path)
    assert code == 0
    assert "file0" in out
    assert "file1" in out


def test_list_without_session(monkeypatch, tmp_path):
    """list without --session calls iter_recent and prints formatted rows."""
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)
    obs = [_make_obs("s1", "Read", f"file{i}") for i in range(2)]
    monkeypatch.setattr(cmd_mod, "iter_recent", lambda limit, base: iter(obs))
    code, out = _capture_run(["list", "--limit", "5"], monkeypatch, tmp_path)
    assert code == 0
    assert "file0" in out


def test_list_no_observations(monkeypatch, tmp_path):
    """list with no rows prints 'No observations found.'"""
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(cmd_mod, "iter_recent", lambda limit, base: iter([]))
    code, out = _capture_run(["list"], monkeypatch, tmp_path)
    assert code == 0
    assert "No observations found." in out


# ---------------------------------------------------------------------------
# _get_rows inner function — covers lines 91-96
# ---------------------------------------------------------------------------

def test_get_rows_inner_function(monkeypatch, tmp_path):
    """Directly exercise the _get_rows closure defined inside _run_tail."""
    import types

    import mullet.cli.observations_cmd as cmd_mod

    inner_code = None
    for const in cmd_mod._run_tail.__code__.co_consts:
        if isinstance(const, types.CodeType) and const.co_name == "_get_rows":
            inner_code = const
            break
    assert inner_code is not None

    def _make_cell(val):
        return (lambda: val).__closure__[0]

    # Non-session path (lines 94-96)
    class FakeNS:
        session = None
        limit = 2

    g = dict(cmd_mod.__dict__)
    obs1 = _make_obs("s1", "Read", "a")
    obs2 = _make_obs("s1", "Read", "b")
    g["iter_recent"] = lambda limit, base: [obs1, obs2]
    g["iter_session"] = lambda session, base: []

    _get_rows = types.FunctionType(
        inner_code, g, "_get_rows", (), (_make_cell(tmp_path), _make_cell(FakeNS()))
    )
    assert _get_rows(0) == [obs1, obs2]
    assert _get_rows(1) == [obs2]

    # Session path (lines 91-93)
    class FakeNS2:
        session = "ses1"
        limit = 2

    g2 = dict(cmd_mod.__dict__)
    obs3 = _make_obs("ses1", "Read", "c")
    g2["iter_recent"] = lambda limit, base: []
    g2["iter_session"] = lambda session, base: [obs3]

    _get_rows2 = types.FunctionType(
        inner_code, g2, "_get_rows", (), (_make_cell(tmp_path), _make_cell(FakeNS2()))
    )
    assert _get_rows2(0) == [obs3]


# ---------------------------------------------------------------------------
# Follow loop body — covers lines 110-122
# ---------------------------------------------------------------------------

def test_follow_loop_body_non_session(monkeypatch, tmp_path):
    """Follow loop (non-session) executes body and prints new rows."""
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)

    obs1 = _make_obs("s1", "Read", "file1")
    obs2 = _make_obs("s1", "Read", "file2")

    call_count = 0

    def fake_iter_recent(limit, base):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [obs1]
        return [obs2, obs1]

    monkeypatch.setattr(cmd_mod, "iter_recent", fake_iter_recent)

    sleep_count = 0

    def fake_sleep(s):
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count >= 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(cmd_mod.time, "sleep", fake_sleep)

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    code = cmd_mod.run(["tail", "--follow", "--limit", "1"])
    assert code == 0
    out = buf.getvalue()
    assert "file2" in out


def test_follow_loop_body_session(monkeypatch, tmp_path):
    """Follow loop (session) executes body and prints new rows."""
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)

    obs1 = _make_obs("ses-a", "Read", "file1")
    obs2 = _make_obs("ses-a", "Read", "file2")

    call_count = 0

    def fake_iter_session(session, base):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [obs1]
        return [obs1, obs2]

    monkeypatch.setattr(cmd_mod, "iter_session", fake_iter_session)

    sleep_count = 0

    def fake_sleep(s):
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count >= 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(cmd_mod.time, "sleep", fake_sleep)

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    code = cmd_mod.run(["tail", "--session", "ses-a", "--follow", "--limit", "1"])
    assert code == 0
    out = buf.getvalue()
    assert "file1" in out


# ---------------------------------------------------------------------------
# _run_search — covers lines 136-148
# ---------------------------------------------------------------------------

def test_search_with_results(monkeypatch, tmp_path):
    """search with results prints formatted rows."""
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)
    obs = [_make_obs("s1", "Read", "file1")]
    monkeypatch.setattr(cmd_mod, "obs_search", lambda query, limit, base: obs)
    code, out = _capture_run(["search", "file1"], monkeypatch, tmp_path)
    assert code == 0
    assert "file1" in out


def test_search_with_session_filter(monkeypatch, tmp_path):
    """search with --session filters results by session_id."""
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)
    obs = [_make_obs("s1", "Read", "file1"), _make_obs("s2", "Read", "file2")]
    monkeypatch.setattr(cmd_mod, "obs_search", lambda query, limit, base: obs)
    code, out = _capture_run(["search", "file", "--session", "s1"], monkeypatch, tmp_path)
    assert code == 0
    assert "file1" in out
    assert "file2" not in out


def test_search_no_results(monkeypatch, tmp_path):
    """search with no matching results prints the empty message."""
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(cmd_mod, "obs_search", lambda query, limit, base: [])
    code, out = _capture_run(["search", "xyz"], monkeypatch, tmp_path)
    assert code == 0
    assert "No observations matching" in out


# ---------------------------------------------------------------------------
# _run_summarise edge cases — covers lines 178-179, 188, 192-194, 196-198
# ---------------------------------------------------------------------------

def test_summarise_none_summary(monkeypatch, tmp_path):
    """summarise with summary=None prints error and returns 1."""
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)
    with patch("mullet.core.observation_summary.summarise_window", return_value=None):
        code, out = _capture_run(["summarise"], monkeypatch, tmp_path)
    assert code == 1
    assert "No observations found" in out


def test_summarise_files_touched(monkeypatch, tmp_path):
    """summarise prints Files line when files_touched is non-empty."""
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)
    summary = SessionSummary(
        session_id="s1",
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T01:00:00Z",
        tool_counts={"Read": 1},
        files_touched=["/path/to/file.py"],
        summary="has files",
        bullet_points=[],
        open_questions=[],
    )
    with patch("mullet.core.observation_summary.summarise_session", return_value=summary):
        code, out = _capture_run(["summarise", "--session", "s1"], monkeypatch, tmp_path)
    assert code == 0
    assert "Files:" in out
    assert "/path/to/file.py" in out


def test_summarise_bullet_points(monkeypatch, tmp_path):
    """summarise prints 'What happened:' when bullet_points is non-empty."""
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)
    summary = SessionSummary(
        session_id="s1",
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T01:00:00Z",
        tool_counts={"Read": 1},
        files_touched=[],
        summary="has bullets",
        bullet_points=["point 1", "point 2"],
        open_questions=[],
    )
    with patch("mullet.core.observation_summary.summarise_session", return_value=summary):
        code, out = _capture_run(["summarise", "--session", "s1"], monkeypatch, tmp_path)
    assert code == 0
    assert "What happened:" in out
    assert "point 1" in out


def test_summarise_open_questions(monkeypatch, tmp_path):
    """summarise prints 'Open questions:' when open_questions is non-empty."""
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)
    summary = SessionSummary(
        session_id="s1",
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T01:00:00Z",
        tool_counts={"Read": 1},
        files_touched=[],
        summary="has questions",
        bullet_points=[],
        open_questions=["q1?"],
    )
    with patch("mullet.core.observation_summary.summarise_session", return_value=summary):
        code, out = _capture_run(["summarise", "--session", "s1"], monkeypatch, tmp_path)
    assert code == 0
    assert "Open questions:" in out
    assert "q1?" in out


# ---------------------------------------------------------------------------
# _print_help and run dispatch — covers lines 222-231, 237-238, 242, 250-251
# ---------------------------------------------------------------------------

def test_print_help():
    """_print_help outputs usage text."""
    import mullet.cli.observations_cmd as cmd_mod
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        cmd_mod._print_help()
    out = buf.getvalue()
    assert "Usage:" in out
    assert "list" in out
    assert "tail" in out
    assert "search" in out
    assert "summarise" in out


def test_run_empty_args():
    """run([]) calls _print_help and returns 1."""
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        code = run([])
    assert code == 1
    assert "Usage:" in buf.getvalue()


def test_run_help():
    """run(['--help']) returns 0."""
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        code = run(["--help"])
    assert code == 0
    assert "Usage:" in buf.getvalue()


def test_run_list_action(monkeypatch, tmp_path):
    """run(['list']) dispatches to _run_list."""
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(cmd_mod, "iter_recent", lambda limit, base: iter([]))
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        code = cmd_mod.run(["list"])
    assert code == 0
    assert "No observations found." in buf.getvalue()


def test_run_unknown_action():
    """run(['unknown']) prints error and returns 1."""
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        code = run(["unknown"])
    assert code == 1
    assert "Unknown observations action" in buf.getvalue()
