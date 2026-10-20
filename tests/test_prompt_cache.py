"""Tests for mullet.core.providers.prompt_cache."""
from __future__ import annotations



from mullet.core.providers.prompt_cache import (
    _apply_cache_marker,
    apply_anthropic_cache_control,
)

_MARKER = {"type": "ephemeral"}


class TestApplyCacheMarker:
    def test_string_content_converted_to_list(self) -> None:
        msg = {"role": "user", "content": "Hello"}
        _apply_cache_marker(msg, _MARKER)
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["text"] == "Hello"
        assert msg["content"][0]["cache_control"] == _MARKER

    def test_list_content_last_item_gets_marker(self) -> None:
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "First"},
                {"type": "text", "text": "Last"},
            ],
        }
        _apply_cache_marker(msg, _MARKER)
        assert msg["content"][-1]["cache_control"] == _MARKER
        assert "cache_control" not in msg["content"][0]

    def test_none_content_sets_marker_on_message(self) -> None:
        msg = {"role": "user", "content": None}
        _apply_cache_marker(msg, _MARKER)
        assert msg["cache_control"] == _MARKER

    def test_empty_content_sets_marker_on_message(self) -> None:
        msg = {"role": "user", "content": ""}
        _apply_cache_marker(msg, _MARKER)
        assert msg["cache_control"] == _MARKER

    def test_tool_role_native_anthropic(self) -> None:
        msg = {"role": "tool", "content": "result"}
        _apply_cache_marker(msg, _MARKER, native_anthropic=True)
        assert msg["cache_control"] == _MARKER

    def test_tool_role_not_native_returns_early(self) -> None:
        msg = {"role": "tool", "content": "result"}
        _apply_cache_marker(msg, _MARKER, native_anthropic=False)
        assert "cache_control" not in msg


class TestApplyAnthropicCacheControl:
    def test_empty_messages_returns_empty(self) -> None:
        result = apply_anthropic_cache_control([])
        assert result == []

    def test_returns_deep_copy(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        result = apply_anthropic_cache_control(msgs)
        result[0]["role"] = "mutated"
        assert msgs[0]["role"] == "user"

    def test_system_prompt_gets_marker(self) -> None:
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = apply_anthropic_cache_control(msgs)
        sys_content = result[0]["content"]
        assert isinstance(sys_content, list)
        assert sys_content[0].get("cache_control") == _MARKER

    def test_last_3_non_system_get_markers(self) -> None:
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
        ]
        result = apply_anthropic_cache_control(msgs)
        for idx in [3, 4, 5]:
            content = result[idx]["content"]
            assert isinstance(content, list)
            assert content[-1].get("cache_control") == _MARKER

    def test_no_system_prompt_caches_last_4(self) -> None:
        msgs = [{"role": "user", "content": f"msg{i}"} for i in range(5)]
        result = apply_anthropic_cache_control(msgs)
        for idx in [1, 2, 3, 4]:
            content = result[idx]["content"]
            assert isinstance(content, list)
            assert content[-1].get("cache_control") == _MARKER

    def test_1h_ttl_adds_ttl_field(self) -> None:
        msgs = [{"role": "user", "content": "hello"}]
        result = apply_anthropic_cache_control(msgs, cache_ttl="1h")
        content = result[0]["content"]
        marker = content[-1]["cache_control"]
        assert marker.get("ttl") == "1h"

    def test_default_5m_ttl_no_ttl_field(self) -> None:
        msgs = [{"role": "user", "content": "hello"}]
        result = apply_anthropic_cache_control(msgs, cache_ttl="5m")
        content = result[0]["content"]
        marker = content[-1]["cache_control"]
        assert "ttl" not in marker

    def test_single_message_gets_marker(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        result = apply_anthropic_cache_control(msgs)
        assert result[0]["content"][0].get("cache_control") == _MARKER
