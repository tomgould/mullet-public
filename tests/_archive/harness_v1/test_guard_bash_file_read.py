"""Tests for bash_file_read_guard."""


from mullet.harness.hooks.guards.bash_file_read_guard import make_bash_file_read_guard
from mullet.harness.hooks.registry import Block
from mullet.harness.hooks.types import PreToolUsePayload


class TestBashFileReadGuard:
    def test_detects_cat_command_warns(self) -> None:
        guard = make_bash_file_read_guard(strict=False)
        payload = PreToolUsePayload(
            tool_name="Bash",
            tool_args={"command": "cat foo.py"},
            session_id="s1",
        )
        result = guard(payload)
        assert result is payload
        assert "warning" in payload.metadata
        assert "Read tool" in payload.metadata["warning"]

    def test_strict_mode_blocks(self) -> None:
        guard = make_bash_file_read_guard(strict=True)
        payload = PreToolUsePayload(
            tool_name="Bash",
            tool_args={"command": "head -n 10 bar.py"},
            session_id="s1",
        )
        result = guard(payload)
        assert isinstance(result, Block)
        assert "Read tool" in result.reason

    def test_allows_unrelated_bash(self) -> None:
        guard = make_bash_file_read_guard(strict=False)
        payload = PreToolUsePayload(
            tool_name="Bash",
            tool_args={"command": "npm install"},
            session_id="s1",
        )
        assert guard(payload) is None
