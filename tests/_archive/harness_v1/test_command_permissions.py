"""Tests for mullet.harness.commands.permissions."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from mullet.harness.commands.permissions import handle_permissions_command
from mullet.harness.permissions.checker import PermissionChecker, PermissionsConfig

pytestmark = pytest.mark.harness


@pytest.fixture
def perms_path(tmp_path: Path) -> Path:
    return tmp_path / ".mullet" / "permissions.yaml"


def _seed_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_list_outputs_current_config(perms_path: Path) -> None:
    _seed_yaml(
        perms_path,
        {
            "mode": "default",
            "path_rules": [
                {"pattern": "/tmp/*", "allow": True},
                {"pattern": "/etc/*", "allow": False},
            ],
            "denied_commands": ["rm -rf /"],
            "denied_tools": [],
        },
    )
    out = handle_permissions_command("list", perms_config_path=perms_path)
    assert "Mode: default" in out
    assert "Path rules: 2" in out
    assert "/tmp/*" in out and "(allow)" in out
    assert "/etc/*" in out and "(deny)" in out
    assert "Denied commands: 1" in out
    assert "rm -rf /" in out
    assert "Denied tools: 0" in out


def test_mode_change_persists_to_yaml(perms_path: Path) -> None:
    out = handle_permissions_command("mode plan", perms_config_path=perms_path)
    assert "plan" in out
    on_disk = yaml.safe_load(perms_path.read_text())
    assert on_disk["mode"] == "plan"


def test_mode_change_applies_to_engine(perms_path: Path) -> None:
    perms = PermissionChecker(PermissionsConfig(mode="default"))
    engine = SimpleNamespace(perms=perms)
    handle_permissions_command(
        "mode plan", perms_config_path=perms_path, engine=engine
    )
    assert engine.perms.mode == "plan"


def test_allow_appends_path_rule(perms_path: Path) -> None:
    handle_permissions_command("allow /tmp/work/*", perms_config_path=perms_path)
    on_disk = yaml.safe_load(perms_path.read_text())
    assert any(
        r["pattern"] == "/tmp/work/*" and r["allow"] is True
        for r in on_disk["path_rules"]
    )


def test_deny_appends_path_rule(perms_path: Path) -> None:
    handle_permissions_command("deny /etc/secrets/*", perms_config_path=perms_path)
    on_disk = yaml.safe_load(perms_path.read_text())
    assert any(
        r["pattern"] == "/etc/secrets/*" and r["allow"] is False
        for r in on_disk["path_rules"]
    )


def test_deny_cmd_appends(perms_path: Path) -> None:
    handle_permissions_command("deny-cmd rm -rf /", perms_config_path=perms_path)
    on_disk = yaml.safe_load(perms_path.read_text())
    assert "rm -rf /" in on_disk["denied_commands"]


def test_deny_tool_appends(perms_path: Path) -> None:
    handle_permissions_command("deny-tool Bash", perms_config_path=perms_path)
    on_disk = yaml.safe_load(perms_path.read_text())
    assert "Bash" in on_disk["denied_tools"]
