"""Tests for directory_agents_injector."""

import os
import tempfile


from mullet.harness.hooks.guards.directory_agents_injector import (
    make_directory_agents_injector,
)
from mullet.harness.hooks.types import PreToolUsePayload


class TestDirectoryAgentsInjector:
    def test_finds_agents_md_in_target_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_path = os.path.join(tmpdir, "AGENTS.md")
            with open(agents_path, "w") as f:
                f.write("context here")

            injector = make_directory_agents_injector(max_chars=100)
            payload = PreToolUsePayload(
                tool_name="Read",
                tool_args={"path": os.path.join(tmpdir, "foo.py")},
                session_id="s1",
            )
            result = injector(payload)
            assert result is payload
            assert payload.metadata.get("directory_context") == "context here"
            assert payload.metadata.get("directory_context_source") == agents_path

    def test_walks_up_to_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "a", "b", "c")
            os.makedirs(subdir)
            agents_path = os.path.join(tmpdir, "CLAUDE.md")
            with open(agents_path, "w") as f:
                f.write("top level context")

            injector = make_directory_agents_injector(max_chars=100)
            payload = PreToolUsePayload(
                tool_name="Read",
                tool_args={"path": os.path.join(subdir, "foo.py")},
                session_id="s1",
            )
            result = injector(payload)
            assert result is payload
            assert "top level context" in payload.metadata.get("directory_context", "")

    def test_returns_unchanged_when_no_agents_md(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            injector = make_directory_agents_injector(max_chars=100)
            payload = PreToolUsePayload(
                tool_name="Read",
                tool_args={"path": os.path.join(tmpdir, "foo.py")},
                session_id="s1",
            )
            result = injector(payload)
            assert result is None
