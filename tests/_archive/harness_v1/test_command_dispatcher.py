"""Tests for mullet.harness.commands dispatcher."""

from __future__ import annotations

from pathlib import Path

import pytest

from mullet.harness.commands import COMMAND_HANDLERS, dispatch_command
from mullet.harness.hooks.registry import HookRegistry

pytestmark = pytest.mark.harness


def test_dispatch_routes_to_skill() -> None:
    out = dispatch_command("/skill list")
    assert "commit" in out


def test_dispatch_routes_to_commit() -> None:
    reg = HookRegistry()
    out = dispatch_command("/commit fix bug", registry=reg)
    assert "/commit ready" in out
    assert reg.count("Message") == 1


def test_dispatch_unknown_returns_friendly_message() -> None:
    out = dispatch_command("/nope hello")
    assert "Unknown command" in out
    assert "/nope" in out
    assert "/help" in out


def test_dispatch_strips_leading_slash() -> None:
    # Without leading slash should still dispatch (helper strips it defensively).
    out = dispatch_command("skill list")
    assert "commit" in out


def test_dispatcher_filters_unknown_kwargs(tmp_path: Path) -> None:
    """The dispatcher should drop kwargs the handler doesn't accept.

    /skill doesn't take ``perms_config_path`` but /permissions does — passing
    both must not raise.
    """
    out = dispatch_command(
        "/skill list",
        perms_config_path=tmp_path / "permissions.yaml",
        store=None,
        engine=None,
    )
    assert "commit" in out


def test_dispatcher_registry_contains_expected_handlers() -> None:
    assert set(COMMAND_HANDLERS) >= {
        "skill",
        "commit",
        "plan",
        "resume",
        "permissions",
    }
