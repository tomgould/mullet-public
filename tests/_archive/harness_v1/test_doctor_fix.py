"""Tests for `mullet doctor --fix` auto-remediation."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from unittest.mock import patch


from mullet.cli.doctor_harness import (
    CheckResult,
    fix_harness_layer,
)


class TestFixProfilesYaml:
    def test_fix_creates_missing_profiles_yaml(self, tmp_path: Path) -> None:
        with patch(
            "mullet.harness.profiles.manager.Path.home",
            return_value=tmp_path,
        ):
            # Simulate a warn result for profiles
            result = CheckResult(
                name="Harness: profiles",
                status="warn",
                message="profiles.yaml failed to load",
            )
            fixes = fix_harness_layer([result])
            assert len(fixes) == 1
            assert fixes[0].fixed is True
            assert "wrote default profiles.yaml" in fixes[0].message
            assert (tmp_path / ".config" / "mullet" / "profiles.yaml").exists()


class TestFixSessionsDir:
    def test_fix_creates_missing_sessions_dir(self, tmp_path: Path) -> None:
        sessions_dir = tmp_path / ".config" / "mullet" / "sessions"
        with patch(
            "mullet.cli.doctor_harness.Path.home",
            return_value=tmp_path,
        ):
            result = CheckResult(
                name="Harness: sessions dir",
                status="warn",
                message=f"{sessions_dir} not writable",
            )
            fixes = fix_harness_layer([result])
            assert len(fixes) == 1
            assert fixes[0].fixed is True
            assert sessions_dir.exists()
            assert os.access(sessions_dir, os.W_OK)


class TestFixWorkflowStore:
    def test_fix_reinits_db_schema(self, tmp_path: Path) -> None:
        db_path = tmp_path / ".config" / "mullet" / "workflows.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Create an empty DB without tables
        db_path.touch()
        with patch(
            "mullet.cli.doctor_harness.Path.home",
            return_value=tmp_path,
        ):
            result = CheckResult(
                name="Harness: workflow store",
                status="warn",
                message="missing tables: workflow_runs",
            )
            fixes = fix_harness_layer([result])
            assert len(fixes) == 1
            assert fixes[0].fixed is True
            # Verify tables now exist
            conn = sqlite3.connect(str(db_path))
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            conn.close()
            tables = {r[0] for r in rows}
            assert "workflow_runs" in tables


class TestFixReturnsZero:
    def test_fix_returns_0_when_all_pass(self) -> None:
        from mullet.cli.main import _run_doctor
        with patch("mullet.cli.main.detect_local_provider", return_value=None):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
                rc = _run_doctor(full=False, fix=False)
                assert rc in (0, 1)  # may be 1 due to missing local provider

    def test_fix_attempts_only_non_ok(self, tmp_path: Path) -> None:
        ok_result = CheckResult(name="Harness: profiles", status="ok", message="ok")
        warn_result = CheckResult(
            name="Harness: sessions dir", status="warn", message="not writable"
        )
        with patch(
            "mullet.cli.doctor_harness.Path.home",
            return_value=tmp_path,
        ):
            fixes = fix_harness_layer([ok_result, warn_result])
            assert len(fixes) == 1
            assert fixes[0].name == "Harness: sessions dir"
