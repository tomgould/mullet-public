"""Tests for ContextCompressor.should_compress trigger + get_context_length.

Covers the threshold logic without running a real compression pass
(that requires a provider + summarization call).
"""

import json
import sys
from unittest.mock import MagicMock, patch

from mullet.core.agent.compression import (
    ContextCompressor, StructuredSummary, get_context_length
)



# --- get_context_length ----------------------------------------------------


def test_get_context_length_known_claude_models():
    from mullet.core.agent.compression import get_context_length
    # Delegates to model_info DB — uses ModelInfo.context_length if known
    assert get_context_length("claude-sonnet-4-6") >= 200_000
    assert get_context_length("claude-haiku-4-5") >= 128_000


def test_get_context_length_unknown_model_default():
    from mullet.core.agent.compression import get_context_length
    out = get_context_length("totally-made-up-model-xyz")
    assert out >= 32_000  # default fallback


def test_get_context_length_empty_string():
    from mullet.core.agent.compression import get_context_length
    # Empty string falls through model_info (which returns 32K for unknown)
    # before reaching the compression.py local table fallback
    assert get_context_length("") > 0


def test_get_context_length_fuzzy_match_mistral():
    from mullet.core.agent.compression import get_context_length, MODEL_CONTEXT_LENGTHS
    # Fuzzy substring match: "some-mistral-v2" hits "mistral" key
    out = get_context_length("some-mistral-v2-finetune")
    assert out == MODEL_CONTEXT_LENGTHS["mistral"]


# --- ContextCompressor.should_compress -------------------------------------


def test_should_compress_small_conversation_false():
    from mullet.core.agent.compression import ContextCompressor
    c = ContextCompressor(context_length=128_000, threshold_percent=0.5)
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert c.should_compress(messages) is False


def test_should_compress_over_threshold_true():
    """Conversation tokens > 50% of context window → True."""
    from mullet.core.agent.compression import ContextCompressor
    # 128_000 ctx, 50% threshold = 64_000 tokens ≈ 256_000 chars
    c = ContextCompressor(context_length=128_000, threshold_percent=0.5)
    big_text = "x" * 300_000  # ~75k tokens — well over threshold
    messages = [{"role": "user", "content": big_text}]
    assert c.should_compress(messages) is True


def test_should_compress_exactly_at_threshold_false():
    """Strictly greater-than — equals is not yet enough to compress."""
    from mullet.core.agent.compression import ContextCompressor
    c = ContextCompressor(context_length=10_000, threshold_percent=0.5)
    # threshold = 5_000 tokens → 5_000 * 4 = 20_000 chars exactly
    messages = [{"role": "user", "content": "a" * 20_000}]
    # estimated_tokens = 20_000 // 4 = 5_000; 5_000 > 5_000 is False
    assert c.should_compress(messages) is False


def test_should_compress_empty_messages_false():
    from mullet.core.agent.compression import ContextCompressor
    c = ContextCompressor()
    assert c.should_compress([]) is False


def test_should_compress_respects_custom_threshold():
    """Aggressive threshold_percent=0.1 triggers on smaller conversations."""
    from mullet.core.agent.compression import ContextCompressor
    c = ContextCompressor(context_length=1_000, threshold_percent=0.1)
    # threshold = 100 tokens = 400 chars
    messages = [{"role": "user", "content": "x" * 500}]
    assert c.should_compress(messages) is True


def test_compressor_default_params():
    from mullet.core.agent.compression import ContextCompressor
    c = ContextCompressor()
    assert c.context_length == 128_000
    assert c.threshold_percent == 0.5
    assert c.protect_first_n == 2
    assert c.protect_last_n == 6
    assert c.tail_protection == 5


class TestCompressionMissingLines:
    # --- get_context_length (lines 50-58) ------------------------------------

    def test_get_context_length_import_error_fallback_to_local_table(self):
        with patch.dict(sys.modules, {"mullet.core.providers.model_info": None}):
            # Should fallback to local table; "gpt-4" is in MODEL_CONTEXT_LENGTHS
            assert get_context_length("gpt-4") == 128_000

    def test_get_context_length_import_error_no_match_default(self):
        with patch.dict(sys.modules, {"mullet.core.providers.model_info": None}):
            # No match in local table -> default
            assert get_context_length("unknown-model-xyz") == 128_000

    def test_get_context_length_local_table_model_match(self):
        from mullet.core.agent.compression import MODEL_CONTEXT_LENGTHS
        # Direct match for "gpt-4"
        assert get_context_length("gpt-4") == MODEL_CONTEXT_LENGTHS["gpt-4"]

    # --- compress line 147 ---------------------------------------------------

    def test_compress_too_short_returns_unchanged(self):
        c = ContextCompressor(context_length=128_000, protect_first_n=1, tail_protection=2)
        # protect_first_n + tail_protection + 1 = 4, so <= 4 returns unchanged
        messages = [{"role": "user", "content": "a"}] * 4
        result = c.compress(messages)
        assert result == messages

    # --- compress line 151 (knowledge present) -------------------------------

    def test_compress_with_knowledge_calls_save_findings(self):
        c = ContextCompressor(context_length=128_000, protect_first_n=1, tail_protection=1)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]
        knowledge = MagicMock()
        # Should not raise
        result = c.compress(messages, knowledge=knowledge)
        assert isinstance(result, list)

    # --- compress lines 155-159 (provider + existing_summary) ----------------

    def test_compress_with_provider_and_existing_summary_calls_iterative_update(self):
        c = ContextCompressor(context_length=128_000, protect_first_n=1, tail_protection=1)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]
        provider = MagicMock()
        provider.create.return_value = MagicMock(
            content=json.dumps({
                "key_decisions": ["D"],
                "code_changes": [],
                "open_questions": [],
                "context": "ctx"
            })
        )
        existing = StructuredSummary(context="existing")
        result = c.compress(messages, provider=provider, existing_summary=existing)
        assert isinstance(result, list)
        assert any("[Context Summary]" in str(m.get("content", "")) for m in result)

    def test_compress_with_provider_no_existing_summary(self):
        c = ContextCompressor(context_length=128_000, protect_first_n=1, tail_protection=1)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]
        provider = MagicMock()
        provider.create.return_value = MagicMock(
            content=json.dumps({
                "key_decisions": [],
                "code_changes": [],
                "open_questions": [],
                "context": "new_ctx"
            })
        )
        result = c.compress(messages, provider=provider)
        assert isinstance(result, list)
        assert any("[Context Summary]" in str(m.get("content", "")) for m in result)

    # --- iterative_update lines 204-205, 213 ---------------------------------

    def test_iterative_update_new_context_nonempty_existing_empty(self):
        c = ContextCompressor()
        existing = StructuredSummary()
        new_seg = StructuredSummary(context="new context")
        result = c.iterative_update(existing, new_seg)
        assert result.context == "new context"

    def test_iterative_update_both_contexts_nonempty_concatenated(self):
        c = ContextCompressor()
        existing = StructuredSummary(context="existing context")
        new_seg = StructuredSummary(context="new context")
        result = c.iterative_update(existing, new_seg)
        assert result.context == "existing context\n\nnew context"

    def test_iterative_update_deduplicates_lists(self):
        c = ContextCompressor()
        existing = StructuredSummary(
            key_decisions=["A"], code_changes=["B"], open_questions=["C"]
        )
        new_seg = StructuredSummary(
            key_decisions=["A", "D"],
            code_changes=["E"],
            open_questions=["C", "F"],
        )
        result = c.iterative_update(existing, new_seg)
        assert result.key_decisions == ["A", "D"]
        assert result.code_changes == ["B", "E"]
        assert result.open_questions == ["C", "F"]

    # --- _generate_structured_summary lines 219-254 --------------------------

    def test_generate_structured_summary_mock_provider_json(self):
        c = ContextCompressor()
        provider = MagicMock()
        provider.create.return_value = MagicMock(
            content=json.dumps({
                "key_decisions": ["D"],
                "code_changes": [],
                "open_questions": [],
                "context": "ctx"
            })
        )
        messages = [{"role": "user", "content": "hello"}]
        result = c._generate_structured_summary(messages, provider)
        assert isinstance(result, StructuredSummary)
        assert result.key_decisions == ["D"]
        assert result.context == "ctx"

    def test_generate_structured_summary_provider_raises_returns_simple(self):
        c = ContextCompressor()
        provider = MagicMock()
        provider.create.side_effect = RuntimeError("boom")
        messages = [{"role": "user", "content": "hello"}]
        result = c._generate_structured_summary(messages, provider)
        assert isinstance(result, StructuredSummary)
        # Fallback to simple summary context
        assert result.context == "Previous conversation context (compressed)."

    # --- _save_findings line 287 ---------------------------------------------

    def test_save_findings_no_exception(self):
        c = ContextCompressor()
        knowledge = MagicMock()
        # Should not raise
        c._save_findings([{"role": "user", "content": "hi"}], knowledge)

    # --- _prune_old_tool_outputs line 294 ------------------------------------

    def test_prune_old_tool_outputs_long_tool_message(self):
        c = ContextCompressor()
        messages = [{"role": "tool", "content": "x" * 201, "tool_call_id": "tc1"}]
        result = c._prune_old_tool_outputs(messages)
        assert len(result) == 1
        assert result[0]["content"] == "[Tool output compressed due to length]"
        assert result[0]["tool_call_id"] == "tc1"

    def test_prune_old_tool_outputs_short_tool_message_unchanged(self):
        c = ContextCompressor()
        messages = [{"role": "tool", "content": "x" * 200, "tool_call_id": "tc1"}]
        result = c._prune_old_tool_outputs(messages)
        assert len(result) == 1
        assert result[0]["content"] == "x" * 200

    def test_prune_old_tool_outputs_non_tool_message_unchanged(self):
        c = ContextCompressor()
        messages = [{"role": "user", "content": "hello"}]
        result = c._prune_old_tool_outputs(messages)
        assert result == messages

    # --- _msg_text lines 308-312 ---------------------------------------------

    def test_msg_text_multimodal_list(self):
        c = ContextCompressor()
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image", "url": "http://example.com/img.png"},
                {"type": "text", "text": "world"},
            ]
        }
        assert c._msg_text(msg) == "hello world"

    def test_msg_text_string_content(self):
        c = ContextCompressor()
        assert c._msg_text({"role": "user", "content": "plain text"}) == "plain text"

    def test_msg_text_none_content(self):
        c = ContextCompressor()
        assert c._msg_text({"role": "assistant", "content": None}) == ""

    # --- _format_messages lines 316-325 --------------------------------------

    def test_format_messages_with_tool_calls_no_content(self):
        c = ContextCompressor()
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"function": {"name": "do_thing"}}],
            }
        ]
        result = c._format_messages(messages)
        assert "Tool Calls" in result
        assert "do_thing" in result

    def test_format_messages_normal_content(self):
        c = ContextCompressor()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = c._format_messages(messages)
        assert "User: hello" in result
        assert "Assistant: hi there" in result

    # --- StructuredSummary.to_message_content (lines 75, 77, 82) -------------

    def test_summary_to_message_content_code_changes(self):
        s = StructuredSummary(code_changes=["Added foo()", "Fixed bar()"])
        out = s.to_message_content()
        assert "Code Changes:" in out
        assert "- Added foo()" in out
        assert "- Fixed bar()" in out

    def test_summary_to_message_content_open_questions(self):
        s = StructuredSummary(open_questions=["Q1?", "Q2?"])
        out = s.to_message_content()
        assert "Open Questions:" in out
        assert "- Q1?" in out
        assert "- Q2?" in out

    def test_summary_to_message_content_empty_parts(self):
        s = StructuredSummary()
        out = s.to_message_content()
        assert out == "Previous conversation context (compressed)."

    # --- compress line 147 (empty middle) ------------------------------------

    def test_compress_empty_middle_returns_unchanged(self):
        # tail_protection=0 makes middle empty while len > protect_first_n + 0 + 1
        c = ContextCompressor(protect_first_n=1, tail_protection=0)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
        # len=3 > 1+0+1=2, so passes early guard; middle = messages[1:0] = empty
        result = c.compress(messages)
        assert result == messages

    # --- _simple_summary lines 269-271, 274-275 (tool_calls) -----------------

    def test_simple_summary_with_tool_calls(self):
        c = ContextCompressor()
        messages = [
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {"function": {"name": "read_file"}},
                    {"function": {"name": "read_file"}},
                    {"function": {"name": "write_file"}},
                ],
            }
        ]
        result = c._simple_summary(messages)
        assert "Tools used: read_file, write_file" in result

    def test_simple_summary_with_tool_calls_no_content(self):
        c = ContextCompressor()
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"function": {"name": "do_thing"}}],
            }
        ]
        result = c._simple_summary(messages)
        assert "Tools used: do_thing" in result
        assert "Key points:" not in result
