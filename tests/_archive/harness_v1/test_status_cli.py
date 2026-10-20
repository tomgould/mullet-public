"""Tests for `mullet status` dashboard CLI."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from mullet.cli.status_cli import cmd_status


class TestStatusCLI:
    @pytest.fixture(autouse=True)
    def _patch_deps(self, monkeypatch: pytest.MonkeyPatch):
        """Patch expensive/network dependencies for every test in this class."""
        # Budget
        budget = Mock()
        budget.status = Mock(value="go")
        budget.five_hour_pct = 13.0
        budget.weekly_pct = 15.0
        budget.pressure = 0.03
        budget.recommended_provider = "sonnet"
        monkeypatch.setattr("mullet.hud.guard.check_budget", lambda: budget)

        # Profile
        prof = Mock(auth_source="KIMI_API_KEY", provider_type="kimi", model="kimi-k2.6")
        prof.name = "kimi"
        cfg = Mock(active="kimi", profiles=[prof])
        pm = Mock()
        pm.load.return_value = cfg
        monkeypatch.setattr("mullet.harness.profiles.manager.ProfileManager", lambda: pm)

        # Permissions
        perm = Mock(mode="default", path_rules=[])
        monkeypatch.setattr("mullet.harness.permissions.checker.load_permissions", lambda _cwd=None: perm)

        # Skills
        skill = Mock(scope="global")
        monkeypatch.setattr("mullet.harness.skills.loader.load_skills", lambda: [skill])

        # Sessions
        meta = Mock(id="sess-abc123", created_at=time.time() - 300, message_count=4, project_hash="abc123")
        msg = Mock(role="user", content="Hello world")
        store = Mock()
        store.list_recent.return_value = [meta]
        store.load.return_value = (meta, [msg], [])
        monkeypatch.setattr("mullet.harness.sessions.store.SessionStore", lambda: store)

        # Workflows
        run = Mock(run_id="deadbeef1234", status="completed", workflow_type="agent.run", created_at=time.time() - 600)
        wf = Mock()
        wf.list_runs.return_value = [run]
        monkeypatch.setattr("mullet.harness.workflows.api.WorkflowAPI", lambda: wf)

        # MCP
        mcp = Mock()
        mcp.load_config.return_value = [Mock(), Mock()]
        mcp.list_servers.return_value = [Mock()]
        monkeypatch.setattr("mullet.harness.mcp.manager.MCPManager", lambda: mcp)

    def test_status_returns_0(self, capsys, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "sk-test")
        rc = cmd_status(["--no-color"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Mullet Status" in captured.out

    def test_status_json_format_valid(self, capsys, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "sk-test")
        rc = cmd_status(["--json"])
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "budget" in data
        assert "profile" in data
        assert "sessions" in data
        assert "workflows" in data
        assert "mcp" in data

    def test_status_shows_active_profile(self, capsys, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "sk-test")
        rc = cmd_status(["--no-color"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "kimi" in captured.out
        assert "ready" in captured.out

    def test_status_shows_recent_sessions(self, capsys, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "sk-test")
        rc = cmd_status(["--no-color"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "sess-abc123" in captured.out
        assert "Hello world" in captured.out

    def test_status_shows_budget_info(self, capsys, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "sk-test")
        rc = cmd_status(["--no-color"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "5h: 13%" in captured.out
        assert "sonnet" in captured.out

    def test_status_watchdog_alert(self, capsys, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "sk-test")
        warn_path = Path("/tmp/mullet_budget_warn")
        try:
            warn_path.write_text("")
            rc = cmd_status(["--no-color"])
            assert rc == 0
            captured = capsys.readouterr()
            assert "budget warning" in captured.out
        finally:
            if warn_path.exists():
                warn_path.unlink()
