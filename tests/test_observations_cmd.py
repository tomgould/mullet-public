"""Tests for mullet.cli.observations_cmd — list, tail, search actions."""
import io
import sys
from pathlib import Path


from mullet.core.observations import record, Observation
from mullet.cli.observations_cmd import run, _tool_color, _fmt_row, _project_root, _HAS_COLOR


def _seed(tmp_path, n=5, sid="cmd-ses"):
    for i in range(n):
        record(
            tool_name="Read" if i % 2 == 0 else "Bash",
            tool_input={"file": f"/tmp/file{i}.py"},
            tool_output=f"output line {i}" + (" special" if i == 2 else ""),
            session_id=sid,
            base=tmp_path,
        )


def _capture_run(args, monkeypatch, tmp_path):
    """Run the CLI and return (exit_code, stdout_text)."""
    # Monkey-patch observations_cmd._project_root to return tmp_path
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    try:
        code = cmd_mod.run(args)
    except SystemExit as e:
        code = int(e.code) if e.code is not None else 0
    return code, buf.getvalue()


def test_list_no_observations(monkeypatch, tmp_path):
    code, out = _capture_run(["list"], monkeypatch, tmp_path)
    assert code == 0
    assert "No observations" in out


def test_list_shows_rows(monkeypatch, tmp_path):
    _seed(tmp_path)
    code, out = _capture_run(["list"], monkeypatch, tmp_path)
    assert code == 0
    lines = [l for l in out.strip().splitlines() if l.strip()]
    assert len(lines) >= 5


def test_list_by_session(monkeypatch, tmp_path):
    _seed(tmp_path, n=3, sid="ses-a")
    _seed(tmp_path, n=2, sid="ses-b")
    code, out = _capture_run(["list", "--session", "ses-a"], monkeypatch, tmp_path)
    assert code == 0
    lines = [l for l in out.strip().splitlines() if l.strip()]
    assert len(lines) == 3


def test_list_limit(monkeypatch, tmp_path):
    _seed(tmp_path, n=10)
    code, out = _capture_run(["list", "--limit", "3"], monkeypatch, tmp_path)
    assert code == 0
    lines = [l for l in out.strip().splitlines() if l.strip()]
    assert len(lines) == 3


def test_tail_returns_last_n(monkeypatch, tmp_path):
    _seed(tmp_path, n=8)
    code, out = _capture_run(["tail", "--limit", "4"], monkeypatch, tmp_path)
    assert code == 0
    lines = [l for l in out.strip().splitlines() if l.strip()]
    assert len(lines) == 4


def test_tail_no_follow_exits(monkeypatch, tmp_path):
    _seed(tmp_path, n=3)
    # Without --follow should return without blocking
    code, out = _capture_run(["tail"], monkeypatch, tmp_path)
    assert code == 0


def test_search_finds_result(monkeypatch, tmp_path):
    # Search matches on tool_input_preview too — seed a file with a unique path fragment
    record("Read", {"file": "/tmp/special_needle.py"}, "output ok", "cmd-ses", base=tmp_path)
    code, out = _capture_run(["search", "special_needle"], monkeypatch, tmp_path)
    assert code == 0
    # At least one matching row should appear
    assert len([l for l in out.strip().splitlines() if l.strip()]) >= 1


def test_search_no_match(monkeypatch, tmp_path):
    _seed(tmp_path)
    code, out = _capture_run(["search", "xyzzy_never_matches"], monkeypatch, tmp_path)
    assert code == 0
    assert "No observations" in out


def test_search_with_session_filter(monkeypatch, tmp_path):
    _seed(tmp_path, n=3, sid="alpha")
    record("Read", {"file": "/special/needle.py"}, "needle here", "beta", base=tmp_path)
    code, out = _capture_run(["search", "needle", "--session", "alpha"], monkeypatch, tmp_path)
    assert code == 0
    assert "No observations" in out  # needle only in beta session


def test_unknown_action(monkeypatch, tmp_path):
    code, out = _capture_run(["frobnicate"], monkeypatch, tmp_path)
    assert code != 0


def test_no_args(monkeypatch, tmp_path):
    code, out = _capture_run([], monkeypatch, tmp_path)
    assert code != 0


# ---------------------------------------------------------------------------
# Pure helper tests
# ---------------------------------------------------------------------------

class TestToolColor:
    def test_no_color_returns_plain(self, monkeypatch):
        monkeypatch.setattr("mullet.cli.observations_cmd._HAS_COLOR", False)
        assert _tool_color("Read") == "Read"

    def test_known_tool_colored(self, monkeypatch):
        monkeypatch.setattr("mullet.cli.observations_cmd._HAS_COLOR", True)
        out = _tool_color("Read")
        assert "Read" in out
        if _HAS_COLOR:
            assert "\033[" in out

    def test_unknown_tool_uses_default_color(self, monkeypatch):
        monkeypatch.setattr("mullet.cli.observations_cmd._HAS_COLOR", True)
        out = _tool_color("Unknown")
        assert "Unknown" in out
        if _HAS_COLOR:
            assert "\033[" in out


class TestFmtRow:
    def test_fmt_row_no_color(self, monkeypatch):
        monkeypatch.setattr("mullet.cli.observations_cmd._HAS_COLOR", False)
        obs = Observation(
            timestamp="2026-05-02T12:34:56Z",
            session_id="s1",
            tool_name="Read",
            tool_input_preview="some file content",
            tool_output_preview="output",
            cwd="/tmp",
        )
        row = _fmt_row(obs)
        assert "2026-05-02 12:34" in row
        assert "Read" in row
        assert "some file content" in row

    def test_fmt_row_with_color(self, monkeypatch):
        monkeypatch.setattr("mullet.cli.observations_cmd._HAS_COLOR", True)
        obs = Observation(
            timestamp="2026-05-02T12:34:56Z",
            session_id="s1",
            tool_name="Read",
            tool_input_preview="some file content",
            tool_output_preview="output",
            cwd="/tmp",
        )
        row = _fmt_row(obs)
        assert "some file content" in row
        if _HAS_COLOR:
            assert "\033[" in row


class TestProjectRoot:
    def test_finds_dot_mullet(self, monkeypatch, tmp_path):
        (tmp_path / ".mullet").mkdir()
        monkeypatch.chdir(tmp_path)
        assert _project_root() == tmp_path

    def test_finds_mullet_package(self, monkeypatch, tmp_path):
        (tmp_path / "mullet").mkdir()
        monkeypatch.chdir(tmp_path)
        assert _project_root() == tmp_path

    def test_fallback_to_cwd(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        # Prevent parent dirs (e.g. /tmp) that contain .mullet from shadowing cwd.
        orig_is_dir = Path.is_dir

        def capped_is_dir(self):
            if str(self).startswith(str(tmp_path)):
                return orig_is_dir(self)
            return False

        monkeypatch.setattr(Path, "is_dir", capped_is_dir)
        assert _project_root() == tmp_path


import builtins
import json
import sys
from unittest.mock import patch

from mullet.core.observation_summary import SessionSummary


def test_color_import_failure():
    real_import = builtins.__import__
    orig_cmd = sys.modules.get("mullet.cli.observations_cmd")
    orig_colors = sys.modules.get("mullet.hud.colors")

    def fake_import(name, *args, **kwargs):
        if name == "mullet.hud.colors":
            raise ImportError("no colors")
        return real_import(name, *args, **kwargs)

    try:
        builtins.__import__ = fake_import
        for mod in list(sys.modules.keys()):
            if mod.startswith("mullet.cli.observations_cmd") or mod == "mullet.hud.colors":
                del sys.modules[mod]
        import mullet.cli.observations_cmd as reloaded
        assert reloaded._HAS_COLOR is False
    finally:
        builtins.__import__ = real_import
        for mod in list(sys.modules.keys()):
            if mod.startswith("mullet.cli.observations_cmd") or mod == "mullet.hud.colors":
                if mod in sys.modules:
                    del sys.modules[mod]
        if orig_cmd is not None:
            sys.modules["mullet.cli.observations_cmd"] = orig_cmd
        else:
            sys.modules.pop("mullet.cli.observations_cmd", None)
        if orig_colors is not None:
            sys.modules["mullet.hud.colors"] = orig_colors
        else:
            sys.modules.pop("mullet.hud.colors", None)


def test_tail_with_session(monkeypatch, tmp_path):
    _seed(tmp_path, n=5, sid="tail-ses")
    code, out = _capture_run(["tail", "--session", "tail-ses", "--limit", "3"], monkeypatch, tmp_path)
    assert code == 0
    lines = [l for l in out.strip().splitlines() if l.strip()]
    assert len(lines) == 3


def test_follow_keyboard_interrupt(monkeypatch, tmp_path):
    _seed(tmp_path, n=2)
    monkeypatch.setattr(
        "mullet.cli.observations_cmd.time.sleep",
        lambda s: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    code, out = _capture_run(["tail", "--follow"], monkeypatch, tmp_path)
    assert code == 0
    lines = [l for l in out.strip().splitlines() if l.strip()]
    assert len(lines) >= 2


def _make_summary(session_id="s1"):
    return SessionSummary(
        session_id=session_id,
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T01:00:00Z",
        tool_counts={"Read": 5},
        files_touched=["/tmp/a.py"],
        summary="Did some work",
        bullet_points=["read files"],
        open_questions=["what next?"],
    )


def test_summarise_session(monkeypatch, tmp_path):
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)
    summary = _make_summary("ses-1")
    with patch("mullet.core.observation_summary.summarise_session", return_value=summary) as mock_ss:
        buf = io.StringIO()
        monkeypatch.setattr(sys, "stdout", buf)
        code = cmd_mod.run(["summarise", "--session", "ses-1"])
    assert code == 0
    out = buf.getvalue()
    assert "Did some work" in out
    assert "read files" in out
    assert "what next?" in out
    mock_ss.assert_called_once_with("ses-1", provider="auto", base=tmp_path)


def test_summarise_window(monkeypatch, tmp_path):
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)
    summary = _make_summary()
    with patch("mullet.core.observation_summary.summarise_window", return_value=summary) as mock_sw:
        buf = io.StringIO()
        monkeypatch.setattr(sys, "stdout", buf)
        code = cmd_mod.run(["summarise", "--window", "30"])
    assert code == 0
    out = buf.getvalue()
    assert "Did some work" in out
    mock_sw.assert_called_once_with(minutes=30, provider="auto", base=tmp_path)


def test_summarise_no_observations(monkeypatch, tmp_path):
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)
    with patch("mullet.core.observation_summary.summarise_window", return_value=None):
        buf = io.StringIO()
        monkeypatch.setattr(sys, "stdout", buf)
        code = cmd_mod.run(["summarise"])
    assert code == 1
    out = buf.getvalue()
    assert "No observations found" in out


def test_summarise_save(monkeypatch, tmp_path):
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)
    summary = _make_summary("ses-save")
    with patch("mullet.core.observation_summary.summarise_session", return_value=summary):
        with patch("mullet.core.memory_sync.sync_observation_summary") as mock_sync:
            buf = io.StringIO()
            monkeypatch.setattr(sys, "stdout", buf)
            code = cmd_mod.run(["summarise", "--session", "ses-save", "--save"])
    assert code == 0
    out = buf.getvalue()
    assert "Saved:" in out
    save_dir = tmp_path / ".mullet" / "observations" / "summaries"
    assert save_dir.exists()
    files = list(save_dir.iterdir())
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["session_id"] == "ses-save"
    mock_sync.assert_called_once()


def test_summarise_save_memory_sync_exception(monkeypatch, tmp_path):
    import mullet.cli.observations_cmd as cmd_mod
    monkeypatch.setattr(cmd_mod, "_project_root", lambda: tmp_path)
    summary = _make_summary("ses-save2")
    with patch("mullet.core.observation_summary.summarise_session", return_value=summary):
        with patch("mullet.core.memory_sync.sync_observation_summary", side_effect=RuntimeError("sync fail")):
            buf = io.StringIO()
            monkeypatch.setattr(sys, "stdout", buf)
            code = cmd_mod.run(["summarise", "--session", "ses-save2", "--save"])
    assert code == 0
    out = buf.getvalue()
    assert "Saved:" in out


