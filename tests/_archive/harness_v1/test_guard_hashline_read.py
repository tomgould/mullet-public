"""Tests for hashline_read_enhancer and hashline_edit_validator."""

import os
import tempfile


from mullet.harness.hooks.guards.hashline_read_enhancer import (
    make_hashline_edit_validator,
    make_hashline_read_enhancer,
)
from mullet.harness.hooks.registry import Block
from mullet.harness.hooks.types import PostToolUsePayload, PreToolUsePayload


class TestHashlineReadEnhancer:
    def test_read_output_gets_hashlines(self) -> None:
        enhancer = make_hashline_read_enhancer()
        payload = PostToolUsePayload(
            tool_name="Read",
            tool_args={"path": "/tmp/foo.py"},
            tool_result="line one\n\nline two\n",
            tool_error=None,
            session_id="s1",
        )
        result = enhancer(payload)
        assert result is payload
        assert "LINE#" in payload.tool_result
        assert payload.tool_result.startswith("LINE#")
        # Blank line should stay blank
        lines = payload.tool_result.splitlines()
        assert any(line.strip() == "" for line in lines)


class TestHashlineEditValidator:
    def test_edit_with_matching_hash_passes(self) -> None:
        validator = make_hashline_edit_validator()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("alpha\nbeta\n")
            path = f.name

        try:
            # Compute expected hash for "alpha"
            import hashlib

            expected = hashlib.sha1(b"alpha").hexdigest()[:6]
            pre = PreToolUsePayload(
                tool_name="Edit",
                tool_args={
                    "path": path,
                    "oldString": f"LINE#{expected}: alpha",
                    "newString": "gamma",
                },
                session_id="s1",
            )
            assert validator(pre) is None
        finally:
            os.unlink(path)

    def test_edit_with_mismatching_hash_blocks(self) -> None:
        validator = make_hashline_edit_validator()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("alpha\nbeta\n")
            path = f.name

        try:
            pre = PreToolUsePayload(
                tool_name="Edit",
                tool_args={
                    "path": path,
                    "oldString": "LINE#000000: alpha",
                    "newString": "gamma",
                },
                session_id="s1",
            )
            result = validator(pre)
            assert isinstance(result, Block)
            assert "File changed since last Read." in result.reason
        finally:
            os.unlink(path)
