"""Full permission-system tests for the Mullet harness.

Replaces the Phase 1 stub coverage. Verifies that
:class:`mullet.harness.permissions.checker.PermissionChecker` honours
default/auto/plan modes, path rules, denied commands, and tool whitelists.
Also verifies that :class:`mullet.harness.engine.loop.HarnessEngine`
auto-registers a permission ``PreToolUse`` hook when constructed with no
explicit registry.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mullet.harness.engine.loop import HarnessEngine
from mullet.harness.permissions.checker import (
    Allow,
    Deny,
    EXEC_TOOLS,
    PathRule,
    PermissionChecker,
    PermissionsConfig,
    Prompt,
    WRITE_TOOLS,
    load_permissions,
)


@pytest.mark.harness
class TestDefaultMode:
    def test_default_mode_reads_allow(self):
        chk = PermissionChecker(PermissionsConfig(mode="default"))
        assert chk.check("Read", {"file_path": "/tmp/foo"}) is Allow

    def test_default_mode_writes_prompt(self):
        chk = PermissionChecker(PermissionsConfig(mode="default"))
        assert chk.check("Write", {"file_path": "/tmp/foo"}) is Prompt

    def test_default_mode_exec_prompt(self):
        chk = PermissionChecker(PermissionsConfig(mode="default"))
        assert chk.check("Bash", {"command": "echo hi"}) is Prompt


@pytest.mark.harness
class TestAutoMode:
    def test_auto_mode_writes_allow(self):
        chk = PermissionChecker(PermissionsConfig(mode="auto"))
        assert chk.check("Write", {"file_path": "/tmp/foo"}) is Allow

    def test_auto_mode_exec_allow(self):
        chk = PermissionChecker(PermissionsConfig(mode="auto"))
        assert chk.check("Bash", {"command": "ls"}) is Allow


@pytest.mark.harness
class TestPlanMode:
    def test_plan_mode_writes_deny(self):
        chk = PermissionChecker(PermissionsConfig(mode="plan"))
        assert chk.check("Write", {"file_path": "/tmp/foo"}) is Deny

    def test_plan_mode_exec_deny(self):
        chk = PermissionChecker(PermissionsConfig(mode="plan"))
        assert chk.check("Bash", {"command": "ls"}) is Deny

    def test_plan_mode_reads_allow(self):
        chk = PermissionChecker(PermissionsConfig(mode="plan"))
        assert chk.check("Read", {"file_path": "/tmp/foo"}) is Allow


@pytest.mark.harness
class TestPathRules:
    def test_path_rule_deny(self):
        cfg = PermissionsConfig(
            mode="default",
            path_rules=[PathRule(pattern="/etc/*", allow=False)],
        )
        chk = PermissionChecker(cfg)
        assert chk.check("Edit", {"file_path": "/etc/hosts"}) is Deny

    def test_path_rule_allow(self):
        # Default mode would prompt for a write — explicit allow rule overrides.
        cfg = PermissionsConfig(
            mode="default",
            path_rules=[PathRule(pattern="/tmp/*", allow=True)],
        )
        chk = PermissionChecker(cfg)
        assert chk.check("Write", {"file_path": "/tmp/foo"}) is Allow


@pytest.mark.harness
class TestCommandRules:
    def test_denied_command(self):
        cfg = PermissionsConfig(
            mode="auto",
            denied_commands=["rm -rf /"],
        )
        chk = PermissionChecker(cfg)
        # Substring match: command embeds the denied fragment.
        assert chk.check("Bash", {"command": "echo hi && rm -rf /"}) is Deny

    def test_denied_command_clean_passes(self):
        cfg = PermissionsConfig(
            mode="auto",
            denied_commands=["rm -rf /"],
        )
        chk = PermissionChecker(cfg)
        assert chk.check("Bash", {"command": "echo hi"}) is Allow


@pytest.mark.harness
class TestToolFilters:
    def test_denied_tool(self):
        cfg = PermissionsConfig(denied_tools=["WebSearch"])
        chk = PermissionChecker(cfg)
        assert chk.check("WebSearch", {}) is Deny

    def test_allowed_tools_whitelist(self):
        cfg = PermissionsConfig(allowed_tools=["Read", "Bash"])
        chk = PermissionChecker(cfg)
        # Edit is not in the whitelist.
        assert chk.check("Edit", {"file_path": "/tmp/foo"}) is Deny
        # Read is whitelisted.
        assert chk.check("Read", {"file_path": "/tmp/foo"}) is Allow


@pytest.mark.harness
class TestLoadPermissions:
    def test_load_permissions_from_yaml(self, tmp_path: Path):
        cfg_dir = tmp_path / ".mullet"
        cfg_dir.mkdir()
        (cfg_dir / "permissions.yaml").write_text(
            yaml.safe_dump(
                {
                    "mode": "plan",
                    "path_rules": [{"pattern": "/etc/*", "allow": False}],
                    "denied_commands": ["rm -rf /"],
                    "denied_tools": ["WebSearch"],
                }
            )
        )
        cfg = load_permissions(tmp_path)
        assert cfg.mode == "plan"
        assert cfg.path_rules == [PathRule(pattern="/etc/*", allow=False)]
        assert cfg.denied_commands == ["rm -rf /"]
        assert cfg.denied_tools == ["WebSearch"]

    def test_load_permissions_missing_returns_defaults(self, tmp_path: Path):
        cfg = load_permissions(tmp_path)
        assert cfg.mode == "default"
        assert cfg.path_rules == []
        assert cfg.denied_commands == []
        assert cfg.allowed_tools == []
        assert cfg.denied_tools == []


@pytest.mark.harness
class TestExplain:
    def test_explain_returns_string(self):
        chk = PermissionChecker(PermissionsConfig(mode="default"))
        msg = chk.explain("Write", {"file_path": "/tmp/foo"})
        assert isinstance(msg, str)
        assert "Write" in msg


@pytest.mark.harness
class TestEngineDefaultHook:
    def test_engine_registers_permission_hook_on_default_init(self, tmp_path: Path):
        # Construct with default args (no hooks supplied) and a tmp store
        # to avoid touching the user's real session directory.
        from mullet.harness.sessions.store import SessionStore

        engine = HarnessEngine(store=SessionStore(base_dir=tmp_path / "sessions"))
        # Default engine wires permission hook + 4 guard hooks on PreToolUse.
        assert engine.hooks.count("PreToolUse") >= 1

    def test_engine_does_not_register_on_custom_hooks(self, tmp_path: Path):
        from mullet.harness.hooks.registry import HookRegistry
        from mullet.harness.sessions.store import SessionStore

        registry = HookRegistry()
        engine = HarnessEngine(
            store=SessionStore(base_dir=tmp_path / "sessions"),
            hooks=registry,
        )
        # Caller-supplied registry stays clean.
        assert engine.hooks.count("PreToolUse") == 0


@pytest.mark.harness
class TestConstants:
    def test_write_tools_membership(self):
        assert "Write" in WRITE_TOOLS
        assert "Edit" in WRITE_TOOLS
        assert "MultiEdit" in WRITE_TOOLS
        assert "NotebookEdit" in WRITE_TOOLS

    def test_exec_tools_membership(self):
        assert "Bash" in EXEC_TOOLS
        assert "Terminal" in EXEC_TOOLS
        assert "RunTests" in EXEC_TOOLS


@pytest.mark.harness
class TestModeOverride:
    def test_mode_override_replaces_config_mode(self):
        chk = PermissionChecker(PermissionsConfig(mode="auto"), mode_override="plan")
        assert chk.mode == "plan"
        assert chk.check("Write", {"file_path": "/tmp/foo"}) is Deny

    def test_invalid_mode_override_rejected(self):
        with pytest.raises(ValueError, match="Invalid permission mode"):
            PermissionChecker(PermissionsConfig(), mode_override="bogus")

    def test_set_mode_validates(self):
        chk = PermissionChecker(PermissionsConfig())
        with pytest.raises(ValueError, match="Invalid permission mode"):
            chk.set_mode("foo")
