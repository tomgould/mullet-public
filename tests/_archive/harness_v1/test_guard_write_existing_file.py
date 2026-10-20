"""Tests for write_existing_file_guard."""

import os
import tempfile


from mullet.harness.hooks.guards.write_existing_file_guard import (
    make_write_existing_file_guard,
)
from mullet.harness.hooks.registry import Block
from mullet.harness.hooks.types import PostToolUsePayload, PreToolUsePayload


class TestWriteExistingFileGuard:
    def test_block_write_to_existing_unread_file(self) -> None:
        tracker: dict = {}
        guard = make_write_existing_file_guard(session_read_tracker=tracker)

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("hello")
            path = f.name

        try:
            pre = PreToolUsePayload(
                tool_name="Write",
                tool_args={"path": path},
                session_id="sess-1",
            )
            result = guard(pre)
            assert isinstance(result, Block)
            assert "Read this file first" in result.reason
        finally:
            os.unlink(path)

    def test_allow_write_to_existing_after_read(self) -> None:
        tracker: dict = {}
        guard = make_write_existing_file_guard(session_read_tracker=tracker)

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("hello")
            path = f.name

        try:
            # Simulate read
            post = PostToolUsePayload(
                tool_name="Read",
                tool_args={"path": path},
                tool_result="hello",
                tool_error=None,
                session_id="sess-1",
            )
            assert guard(post) is None
            assert path in tracker["sess-1"]

            # Now write should pass
            pre = PreToolUsePayload(
                tool_name="Write",
                tool_args={"path": path},
                session_id="sess-1",
            )
            assert guard(pre) is None
        finally:
            os.unlink(path)

    def test_allow_write_to_new_file(self) -> None:
        tracker: dict = {}
        guard = make_write_existing_file_guard(session_read_tracker=tracker)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "new_file.txt")
            pre = PreToolUsePayload(
                tool_name="Write",
                tool_args={"path": path},
                session_id="sess-1",
            )
            assert guard(pre) is None
