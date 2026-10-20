"""Legacy permission-checker tests (formerly the Phase 1 stub coverage).

The stub returned ``Allow`` for every tool call and emitted a warning. With the
P2.8 full implementation the same call sites now return mode-aware results
(reads ``Allow``, writes ``Prompt`` in default mode, ``Deny`` in plan mode).
These tests have been adapted to the new behaviour while preserving their
historical intent — they continue to exercise the public ``PermissionChecker``
API, mode validation, and the ``default_checker`` singleton.

See ``tests/harness/test_permissions_full.py`` for the full coverage suite.
"""

from __future__ import annotations

import pytest

from mullet.harness.permissions.checker import (
    Allow,
    Deny,
    PermissionChecker,
    PermissionsConfig,
    Prompt,
    default_checker,
)


@pytest.mark.harness
class TestPermissionLegacyStub:
    def test_default_mode_write_now_prompts(self):
        """Phase 1 stub allowed everything; full impl prompts on writes."""
        chk = PermissionChecker(PermissionsConfig(mode="default"))
        assert chk.check("Write", tool_args={"path": "/tmp/foo"}) is Prompt

    def test_auto_mode_allows(self):
        chk = PermissionChecker(PermissionsConfig(mode="auto"))
        assert chk.check("Bash", tool_args={"command": "echo hi"}) is Allow

    def test_plan_mode_now_denies_writes(self):
        """Plan mode used to allow in the stub; the full impl denies all writes."""
        chk = PermissionChecker(PermissionsConfig(mode="plan"))
        assert chk.check("Edit", tool_args={"path": "src/main.py"}) is Deny

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="Invalid permission mode"):
            PermissionChecker(mode_override="bogus")

    def test_set_mode_validates(self):
        chk = PermissionChecker()
        with pytest.raises(ValueError, match="Invalid permission mode"):
            chk.set_mode("foo")

    def test_default_checker_singleton(self):
        # Default mode + read-only tool → Allow.
        assert default_checker.mode == "default"
        assert default_checker.check("AnyTool") is Allow
