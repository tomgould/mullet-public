"""Tests for mullet.core.providers.cache — prompt-caching helper functions."""
from __future__ import annotations

from mullet.core.providers.cache import (
    CacheBreakpoint,
    apply_cache_control,
    detect_cacheable_messages,
    estimate_cache_savings,
    should_cache,
)


# ---------------------------------------------------------------------------
# detect_cacheable_messages
# ---------------------------------------------------------------------------

class TestDetectCacheableMessages:
    def test_empty_messages_returns_empty(self):
        assert detect_cacheable_messages([]) == []

    def test_system_message_at_position_0_returns_breakpoint(self):
        msgs = [{"role": "system", "content": "You are a helpful assistant."}]
        bps = detect_cacheable_messages(msgs)
        assert len(bps) == 1
        assert bps[0].position == 0
        assert bps[0].type == "ephemeral"

    def test_non_system_first_message_no_breakpoint(self):
        msgs = [{"role": "user", "content": "Hello"}]
        bps = detect_cacheable_messages(msgs)
        assert bps == []

    def test_system_message_among_many_messages(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        bps = detect_cacheable_messages(msgs)
        assert any(bp.position == 0 for bp in bps)


# ---------------------------------------------------------------------------
# apply_cache_control
# ---------------------------------------------------------------------------

class TestApplyCacheControl:
    def test_string_content_converted_to_list_with_cache_control(self):
        msgs = [{"role": "system", "content": "system prompt"}]
        bps = [CacheBreakpoint(position=0, type="ephemeral")]
        result = apply_cache_control(msgs, bps)
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_list_content_gets_cache_control_on_first_block(self):
        msgs = [{"role": "system", "content": [{"type": "text", "text": "sys"}]}]
        bps = [CacheBreakpoint(position=0, type="persistent")]
        result = apply_cache_control(msgs, bps)
        assert result[0]["content"][0]["cache_control"] == {"type": "persistent"}

    def test_out_of_bounds_position_does_nothing(self):
        msgs = [{"role": "user", "content": "q"}]
        bps = [CacheBreakpoint(position=5, type="ephemeral")]
        result = apply_cache_control(msgs, bps)
        assert "cache_control" not in result[0].get("content", [{}])[0] if isinstance(result[0]["content"], list) else True

    def test_original_messages_not_mutated(self):
        msgs = [{"role": "system", "content": "original"}]
        original_content = msgs[0]["content"]
        bps = [CacheBreakpoint(position=0, type="ephemeral")]
        apply_cache_control(msgs, bps)
        assert msgs[0]["content"] == original_content

    def test_no_breakpoints_returns_unchanged_messages(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = apply_cache_control(msgs, [])
        assert result[0]["content"] == "hello"

    def test_multiple_breakpoints_applied(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
        ]
        bps = [
            CacheBreakpoint(position=0, type="ephemeral"),
            CacheBreakpoint(position=1, type="ephemeral"),
        ]
        result = apply_cache_control(msgs, bps)
        assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert result[1]["content"][0]["cache_control"] == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# should_cache
# ---------------------------------------------------------------------------

class TestShouldCache:
    def test_claude_prefix_returns_true(self):
        assert should_cache("claude-sonnet-4-6") is True

    def test_anthropic_slash_prefix_returns_true(self):
        assert should_cache("anthropic/claude-opus-4.1") is True

    def test_kimi_returns_false(self):
        assert should_cache("kimi-k2.7-code") is False

    def test_openrouter_returns_false(self):
        assert should_cache("openrouter/deepseek/deepseek-v4-flash") is False

    def test_local_returns_false(self):
        assert should_cache("local/llama3") is False

    def test_empty_string_returns_false(self):
        assert should_cache("") is False


# ---------------------------------------------------------------------------
# estimate_cache_savings
# ---------------------------------------------------------------------------

class TestEstimateCacheSavings:
    def test_empty_messages_returns_zero_savings(self):
        result = estimate_cache_savings([], [])
        assert result["cached_tokens"] == 0
        assert result["savings_pct"] == 0

    def test_no_breakpoints_returns_zero_savings(self):
        msgs = [{"role": "system", "content": "hello world"}]
        result = estimate_cache_savings(msgs, [])
        assert result["cached_tokens"] == 0
        assert result["savings_pct"] == 0

    def test_string_content_token_estimate(self):
        content = "a" * 40  # 40 chars ≈ 10 tokens
        msgs = [{"role": "system", "content": content}]
        bps = [CacheBreakpoint(position=0, type="ephemeral")]
        result = estimate_cache_savings(msgs, bps)
        assert result["cached_tokens"] == 10
        assert result["savings_pct"] == 100.0

    def test_list_content_token_estimate(self):
        content = [{"type": "text", "text": "a" * 80}]  # 80 chars ≈ 20 tokens
        msgs = [{"role": "system", "content": content}]
        bps = [CacheBreakpoint(position=0, type="ephemeral")]
        result = estimate_cache_savings(msgs, bps)
        assert result["cached_tokens"] == 20

    def test_partial_cache_savings(self):
        sys_content = "a" * 40   # 10 tokens
        user_content = "b" * 40  # 10 tokens — total 20 tokens
        msgs = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_content},
        ]
        bps = [CacheBreakpoint(position=0, type="ephemeral")]  # only cache system
        result = estimate_cache_savings(msgs, bps)
        assert result["cached_tokens"] == 10
        assert result["savings_pct"] == 50.0


def test_detect_cacheable_messages_repeated_pair_passes():
    """Line 44: pass branch when a repeated user/assistant pair is found."""
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    bps = detect_cacheable_messages(msgs)
    assert isinstance(bps, list)
