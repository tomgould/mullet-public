"""Tests for secret_redaction_guard and exporter integration."""

from types import SimpleNamespace

from mullet.harness.hooks.guards.secret_redaction_guard import make_secret_redaction_guard
from mullet.harness.hooks.types import PreToolUsePayload
from mullet.harness.sessions.exporter import export_json, export_markdown


class TestSecretRedactionGuard:
    def test_pretool_guard_redacts_args(self) -> None:
        guard = make_secret_redaction_guard()
        payload = PreToolUsePayload(
            tool_name="Bash",
            tool_args={"command": "echo sk-" + "x" * 32},
            session_id="s1",
        )
        result = guard(payload)
        assert result is payload
        assert "[REDACTED]" in payload.tool_args["command"]
        assert "sk-" + "x" * 32 not in payload.tool_args["command"]

    def test_pretool_guard_no_match_returns_none(self) -> None:
        guard = make_secret_redaction_guard()
        payload = PreToolUsePayload(
            tool_name="Bash",
            tool_args={"command": "echo hello"},
            session_id="s1",
        )
        assert guard(payload) is None

    def test_pretool_guard_preserves_non_string_args(self) -> None:
        guard = make_secret_redaction_guard()
        payload = PreToolUsePayload(
            tool_name="Write",
            tool_args={"file_path": "/tmp/foo.py", "offset": 10, "content": "sk-" + "y" * 32},
            session_id="s1",
        )
        guard(payload)
        assert payload.tool_args["offset"] == 10
        assert "[REDACTED]" in payload.tool_args["content"]


class TestExporterRedaction:
    def test_export_redacts_messages(self) -> None:
        meta = SimpleNamespace(
            id="test-session",
            created_at=0.0,
            updated_at=1.0,
            project_hash="abc",
            cwd="/home/user/project",
            model="gpt-4",
            profile="openai",
            message_count=1,
            status="done",
        )
        msg = SimpleNamespace(
            role="user",
            content="My key is sk-" + "z" * 32,
            timestamp=0.0,
        )
        md = export_markdown(meta, [msg], [])
        assert "[REDACTED]" in md
        assert "sk-" + "z" * 32 not in md

    def test_export_json_redacts_content(self) -> None:
        meta = SimpleNamespace(
            id="test-session",
            created_at=0.0,
            updated_at=1.0,
            project_hash="abc",
            cwd="/home/user/project",
            model="gpt-4",
            profile="openai",
            message_count=1,
            status="done",
        )
        msg = SimpleNamespace(
            role="assistant",
            content="Token: ghp_" + "a" * 36,
            timestamp=0.0,
        )
        result = export_json(meta, [msg], [])
        assert "[REDACTED]" in result["messages"][0]["content"]
        assert "ghp_" + "a" * 36 not in result["messages"][0]["content"]
