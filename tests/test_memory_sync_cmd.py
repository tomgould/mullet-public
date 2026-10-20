"""Tests for mullet.cli.memory_sync_cmd."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from unittest.mock import patch


from mullet.core.observation_summary import SessionSummary
from mullet.core.decisions import Decision
from mullet.cli.memory_sync_cmd import _dry_run_observations, _dry_run_decisions, _project_root


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_summary(session_id: str = "sess-cmd") -> SessionSummary:
    return SessionSummary(
        session_id=session_id,
        started_at="2026-04-24T08:00:00Z",
        ended_at="2026-04-24T09:00:00Z",
        tool_counts={"Read": 4, "Bash": 2},
        files_touched=["/tmp/foo.py"],
        summary="Developer worked on memory integration.",
        bullet_points=["Wired memory sync", "Added tests"],
        open_questions=["Any edge cases?"],
    )


def _write_summaries(base: Path, count: int = 2) -> None:
    summaries_dir = base / ".mullet" / "observations" / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        s = _make_summary(f"sess-{i}")
        p = summaries_dir / f"sess-{i}-20260424T{i:02d}0000Z.json"
        p.write_text(json.dumps(dataclasses.asdict(s)))


def _write_decisions(base: Path, count: int = 3) -> None:
    path = base / ".mullet" / "decisions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for i in range(count):
            d = Decision(
                id=f"id{i:016d}",
                source_path="docs/README.md",
                source_type="docs",
                line_number=i + 1,
                timestamp="2026-04-24T08:00:00Z",
                excerpt=f"decided to use approach {i} because it is simpler",
                kind="decision",
                keywords=["decided to"],
            )
            fh.write(json.dumps(d.to_dict()) + "\n")


# ---------------------------------------------------------------------------
# --observations flag
# ---------------------------------------------------------------------------


def test_observations_flag(tmp_path, capsys):
    _write_summaries(tmp_path, 2)

    with patch("mullet.core.memory_sync._get_store", return_value=None), \
         patch("mullet.cli.memory_sync_cmd._project_root", return_value=tmp_path):
        from mullet.cli.memory_sync_cmd import run

        rc = run(["--observations", "--base", str(tmp_path)])

    out = capsys.readouterr().out
    assert rc == 0
    assert "observation" in out.lower()
    assert "2" in out


# ---------------------------------------------------------------------------
# --decisions flag
# ---------------------------------------------------------------------------


def test_decisions_flag(tmp_path, capsys):
    _write_decisions(tmp_path, 3)

    with patch("mullet.core.memory_sync._get_store", return_value=None), \
         patch("mullet.cli.memory_sync_cmd._project_root", return_value=tmp_path):
        from mullet.cli.memory_sync_cmd import run

        rc = run(["--decisions", "--base", str(tmp_path)])

    out = capsys.readouterr().out
    assert rc == 0
    assert "decision" in out.lower()
    assert "3" in out


# ---------------------------------------------------------------------------
# --all flag
# ---------------------------------------------------------------------------


def test_all_flag(tmp_path, capsys):
    _write_summaries(tmp_path, 1)
    _write_decisions(tmp_path, 2)

    with patch("mullet.core.memory_sync._get_store", return_value=None), \
         patch("mullet.cli.memory_sync_cmd._project_root", return_value=tmp_path):
        from mullet.cli.memory_sync_cmd import run

        rc = run(["--all", "--base", str(tmp_path)])

    out = capsys.readouterr().out
    assert rc == 0
    assert "1" in out   # observations
    assert "2" in out   # decisions


# ---------------------------------------------------------------------------
# --dry-run flag
# ---------------------------------------------------------------------------


def test_dry_run_no_writes(tmp_path, capsys):
    """--dry-run must not write anything to disk."""
    _write_summaries(tmp_path, 2)
    _write_decisions(tmp_path, 3)

    before_files = set(tmp_path.rglob("memory_sync.jsonl"))

    with patch("mullet.core.memory_sync._get_store", return_value=None), \
         patch("mullet.cli.memory_sync_cmd._project_root", return_value=tmp_path):
        from mullet.cli.memory_sync_cmd import run

        rc = run(["--dry-run", "--base", str(tmp_path)])

    after_files = set(tmp_path.rglob("memory_sync.jsonl"))
    assert rc == 0
    # No new JSONL files should have been created
    assert before_files == after_files

    out = capsys.readouterr().out
    assert "dry-run" in out.lower()
    assert "2" in out   # would sync 2 observations
    assert "3" in out   # would sync 3 decisions


# ---------------------------------------------------------------------------
# default (no flags) runs --all
# ---------------------------------------------------------------------------


def test_default_runs_all(tmp_path, capsys):
    """Calling with no mode flags should sync everything."""
    _write_summaries(tmp_path, 1)
    _write_decisions(tmp_path, 1)

    with patch("mullet.core.memory_sync._get_store", return_value=None), \
         patch("mullet.cli.memory_sync_cmd._project_root", return_value=tmp_path):
        from mullet.cli.memory_sync_cmd import run

        rc = run(["--base", str(tmp_path)])

    out = capsys.readouterr().out
    assert rc == 0
    assert "memory-sync complete" in out


# ---------------------------------------------------------------------------
# Pure helper tests
# ---------------------------------------------------------------------------

class TestDryRunObservations:
    def test_empty_dir_returns_zero(self, tmp_path, capsys):
        assert _dry_run_observations(tmp_path) == 0
        assert capsys.readouterr().out == ""

    def test_counts_json_files(self, tmp_path, capsys):
        summaries = tmp_path / ".mullet" / "observations" / "summaries"
        summaries.mkdir(parents=True)
        for i in range(3):
            (summaries / f"summary{i}.json").write_text("{}")
        assert _dry_run_observations(tmp_path) == 3
        out = capsys.readouterr().out
        assert "Would sync 3 observation summary file(s)" in out

    def test_truncates_after_five(self, tmp_path, capsys):
        summaries = tmp_path / ".mullet" / "observations" / "summaries"
        summaries.mkdir(parents=True)
        for i in range(7):
            (summaries / f"summary{i}.json").write_text("{}")
        assert _dry_run_observations(tmp_path) == 7
        out = capsys.readouterr().out
        assert "and 2 more" in out


class TestDryRunDecisions:
    def test_missing_file_returns_zero(self, tmp_path, capsys):
        assert _dry_run_decisions(tmp_path) == 0
        assert capsys.readouterr().out == ""

    def test_counts_lines(self, tmp_path, capsys):
        path = tmp_path / ".mullet" / "decisions.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("\n".join([f'{{"kind":"decision","excerpt":"excerpt {i}"}}' for i in range(4)]))
        assert _dry_run_decisions(tmp_path) == 4
        out = capsys.readouterr().out
        assert "Would sync 4 decision(s)" in out

    def test_invalid_json_graceful(self, tmp_path, capsys):
        path = tmp_path / ".mullet" / "decisions.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("not json\n")
        assert _dry_run_decisions(tmp_path) == 1
        out = capsys.readouterr().out
        assert "Would sync 1 decision(s)" in out


class TestProjectRoot:
    def test_finds_dot_mullet(self, monkeypatch, tmp_path):
        (tmp_path / ".mullet").mkdir()
        monkeypatch.chdir(tmp_path)
        assert _project_root() == tmp_path



    def test_fallback_to_here_when_no_mullet_dir(self, monkeypatch, tmp_path):
        """Line 24: no .mullet anywhere in hierarchy → return here (cwd)."""
        monkeypatch.chdir(tmp_path)
        with patch("pathlib.Path.is_dir", return_value=False):
            result = _project_root()
        assert result == tmp_path


class TestDryRunEmpty:
    """Lines 109 and 114: dry-run with no data → 'No X found.' messages."""

    def test_dry_run_no_observations_prints_message(self, tmp_path, capsys):
        """Line 109: _dry_run_observations returns 0 → 'No observation summaries found.'"""
        from mullet.cli.memory_sync_cmd import run
        with patch("mullet.cli.memory_sync_cmd._project_root", return_value=tmp_path):
            rc = run(["--observations", "--dry-run", "--base", str(tmp_path)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "No observation summaries found" in out

    def test_dry_run_no_decisions_prints_message(self, tmp_path, capsys):
        """Line 114: _dry_run_decisions returns 0 → 'No decisions found.'"""
        from mullet.cli.memory_sync_cmd import run
        with patch("mullet.cli.memory_sync_cmd._project_root", return_value=tmp_path):
            rc = run(["--decisions", "--dry-run", "--base", str(tmp_path)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "No decisions found" in out


class TestDryRunDecisionsReadError:
    """Lines 53-54: except Exception in _dry_run_decisions when read fails."""

    def test_unreadable_file_swallowed(self, tmp_path, capsys):
        path = tmp_path / ".mullet" / "decisions.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text('{"kind":"x","excerpt":"y"}\n')
        path.chmod(0o000)
        try:
            count = _dry_run_decisions(tmp_path)
            assert count == 0  # read failed → exception caught → count stays 0
        finally:
            path.chmod(0o644)


class TestObservationsSyncException:
    """Lines 150-152: sync_observation_summary raises → Warning: skipped."""

    def test_skips_bad_summary_file(self, tmp_path, capsys):
        _write_summaries(tmp_path, 1)
        with patch("mullet.core.memory_sync.sync_observation_summary",
                   side_effect=RuntimeError("store down")), \
             patch("mullet.cli.memory_sync_cmd._project_root", return_value=tmp_path):
            from mullet.cli.memory_sync_cmd import run
            rc = run(["--observations", "--base", str(tmp_path)])
        err = capsys.readouterr().err
        assert rc == 0
        assert "Warning: skipped" in err


class TestSampleOutputExceptions:
    """Lines 171-172, 185-186: except Exception: pass when sample read fails."""

    def test_obs_sample_unreadable_swallowed(self, tmp_path, capsys):
        """Lines 171-172: last summary file has bad JSON → json.loads raises → except pass."""
        import dataclasses
        summaries_dir = tmp_path / ".mullet" / "observations" / "summaries"
        summaries_dir.mkdir(parents=True)
        # File A: valid — syncs successfully, obs_synced = 1
        s = _make_summary("sess-aa")
        (summaries_dir / "a_valid.json").write_text(json.dumps(dataclasses.asdict(s)))
        # File B: corrupt — synced fails (Warning), but sorted LAST → sample read fails
        (summaries_dir / "z_corrupt.json").write_text("{ bad json")

        with patch("mullet.core.memory_sync._get_store", return_value=None), \
             patch("mullet.cli.memory_sync_cmd._project_root", return_value=tmp_path):
            from mullet.cli.memory_sync_cmd import run
            rc = run(["--observations", "--base", str(tmp_path)])
        assert rc == 0  # no crash; except pass swallowed

    def test_dec_sample_corrupt_swallowed(self, tmp_path, capsys):
        """Lines 185-186: sync_decisions returns 1 but file has corrupt JSON → except pass."""
        path = tmp_path / ".mullet" / "decisions.jsonl"
        path.parent.mkdir(parents=True)
        # Write a line with bad JSON so json.loads(lines[-1]) raises inside sample block
        path.write_text("not valid json\n")

        # Patch sync_decisions inside mullet.core.memory_sync so run() sees dec_synced=1
        with patch("mullet.core.memory_sync.sync_decisions", return_value=1), \
             patch("mullet.cli.memory_sync_cmd._project_root", return_value=tmp_path):
            from mullet.cli.memory_sync_cmd import run
            rc = run(["--decisions", "--base", str(tmp_path)])
        assert rc == 0
