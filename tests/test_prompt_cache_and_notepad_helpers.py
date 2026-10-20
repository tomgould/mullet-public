"""Tests for prompt_cache pure functions and notepad pure helpers."""

import copy

from mullet.core.providers.prompt_cache import (
    _apply_cache_marker,
    apply_anthropic_cache_control,
)
from mullet.core.tools.notepad import (
    _sanitize_key,
    NOTEPAD_SCHEMA,
)


# ---------------------------------------------------------------------------
# _apply_cache_marker
# ---------------------------------------------------------------------------

MARKER = {"type": "ephemeral"}


class TestApplyCacheMarker:
    def test_string_content_wrapped_in_list(self):
        msg = {"role": "user", "content": "hello"}
        _apply_cache_marker(msg, MARKER)
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["cache_control"] == MARKER

    def test_list_content_last_block_marked(self):
        msg = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        _apply_cache_marker(msg, MARKER)
        assert msg["content"][-1]["cache_control"] == MARKER

    def test_none_content_adds_marker(self):
        msg = {"role": "user", "content": None}
        _apply_cache_marker(msg, MARKER)
        assert msg.get("cache_control") == MARKER

    def test_empty_string_content_adds_marker(self):
        msg = {"role": "user", "content": ""}
        _apply_cache_marker(msg, MARKER)
        assert msg.get("cache_control") == MARKER

    def test_tool_role_without_native_no_marker(self):
        msg = {"role": "tool", "content": "tool result"}
        _apply_cache_marker(msg, MARKER, native_anthropic=False)
        assert "cache_control" not in msg

    def test_tool_role_native_anthropic_gets_marker(self):
        msg = {"role": "tool", "content": "tool result"}
        _apply_cache_marker(msg, MARKER, native_anthropic=True)
        assert msg.get("cache_control") == MARKER

    def test_does_not_mutate_original_text(self):
        msg = {"role": "user", "content": "original text"}
        _apply_cache_marker(msg, MARKER)
        # Text should be preserved in the new list structure
        assert msg["content"][0]["text"] == "original text"


# ---------------------------------------------------------------------------
# apply_anthropic_cache_control
# ---------------------------------------------------------------------------

class TestApplyAnthropicCacheControl:
    def test_returns_list(self):
        result = apply_anthropic_cache_control([])
        assert isinstance(result, list)

    def test_empty_messages_returns_empty(self):
        result = apply_anthropic_cache_control([])
        assert result == []

    def test_original_not_mutated(self):
        messages = [{"role": "user", "content": "hello"}]
        original_copy = copy.deepcopy(messages)
        apply_anthropic_cache_control(messages)
        assert messages == original_copy

    def test_returns_deep_copy(self):
        messages = [{"role": "user", "content": "hello"}]
        result = apply_anthropic_cache_control(messages)
        assert result is not messages

    def test_single_user_message_gets_marker(self):
        messages = [{"role": "user", "content": "hello"}]
        result = apply_anthropic_cache_control(messages)
        content = result[0].get("content")
        if isinstance(content, list):
            has_marker = any("cache_control" in block for block in content)
        else:
            has_marker = "cache_control" in result[0]
        assert has_marker

    def test_system_message_gets_marker(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hello"},
        ]
        result = apply_anthropic_cache_control(messages)
        system = result[0]
        if isinstance(system.get("content"), list):
            has_sys_marker = any("cache_control" in b for b in system["content"])
        else:
            has_sys_marker = "cache_control" in system
        assert has_sys_marker

    def test_max_four_breakpoints(self):
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        result = apply_anthropic_cache_control(messages)
        marked_count = 0
        for msg in result:
            content = msg.get("content")
            if isinstance(content, list):
                if any("cache_control" in b for b in content if isinstance(b, dict)):
                    marked_count += 1
            elif isinstance(content, dict) and "cache_control" in content or "cache_control" in msg:
                marked_count += 1
        assert marked_count <= 4


# ---------------------------------------------------------------------------
# _sanitize_key
# ---------------------------------------------------------------------------

class TestSanitizeKey:
    def test_alphanumeric_unchanged(self):
        assert _sanitize_key("mykey") == "mykey"

    def test_dashes_preserved(self):
        assert _sanitize_key("my-key") == "my-key"

    def test_underscores_preserved(self):
        assert _sanitize_key("my_key") == "my_key"

    def test_slash_replaced(self):
        result = _sanitize_key("has/slash")
        assert "/" not in result

    def test_space_replaced(self):
        result = _sanitize_key("has space")
        assert " " not in result

    def test_empty_string_becomes_default(self):
        assert _sanitize_key("") == "default"

    def test_long_key_truncated_to_64(self):
        result = _sanitize_key("a" * 100)
        assert len(result) == 64

    def test_returns_string(self):
        assert isinstance(_sanitize_key("key"), str)


# ---------------------------------------------------------------------------
# NOTEPAD_SCHEMA
# ---------------------------------------------------------------------------

class TestNotepadSchema:
    def test_has_description(self):
        assert "description" in NOTEPAD_SCHEMA

    def test_has_parameters(self):
        assert "parameters" in NOTEPAD_SCHEMA

    def test_action_has_enum(self):
        props = NOTEPAD_SCHEMA["parameters"]["properties"]
        assert "action" in props
        assert "enum" in props["action"]

    def test_expected_actions_in_enum(self):
        enum = NOTEPAD_SCHEMA["parameters"]["properties"]["action"]["enum"]
        assert "read" in enum
        assert "write" in enum
        assert "append" in enum
