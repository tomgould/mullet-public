"""Tests for interaction_log pure helpers."""


from mullet.core.logging.interaction_log import _extract_preview, _extract_last_user_message


# ---------------------------------------------------------------------------
# _extract_preview
# ---------------------------------------------------------------------------

class TestExtractPreview:
    def test_short_text_returned_as_is(self):
        assert _extract_preview("hello") == "hello"

    def test_none_returns_empty_string(self):
        assert _extract_preview(None) == ""

    def test_exact_length_not_truncated(self):
        text = "a" * 500
        result = _extract_preview(text, max_length=500)
        assert result == text

    def test_over_length_truncated_with_ellipsis(self):
        text = "a" * 600
        result = _extract_preview(text, max_length=500)
        assert result.endswith("...")
        assert len(result) == 500

    def test_custom_max_length(self):
        result = _extract_preview("hello world", max_length=5)
        assert len(result) == 5
        assert result.endswith("...")

    def test_returns_string(self):
        assert isinstance(_extract_preview("test"), str)


# ---------------------------------------------------------------------------
# _extract_last_user_message
# ---------------------------------------------------------------------------

class TestExtractLastUserMessage:
    def test_empty_messages_returns_empty(self):
        assert _extract_last_user_message([]) == ""

    def test_single_user_message(self):
        messages = [{"role": "user", "content": "Hello!"}]
        result = _extract_last_user_message(messages)
        assert result == "Hello!"

    def test_returns_last_user_message(self):
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Reply"},
            {"role": "user", "content": "Second"},
        ]
        result = _extract_last_user_message(messages)
        assert result == "Second"

    def test_skips_assistant_messages(self):
        messages = [
            {"role": "assistant", "content": "Assistant text"},
        ]
        result = _extract_last_user_message(messages)
        assert result == ""

    def test_content_as_list_of_text_blocks(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello from list"}]}
        ]
        result = _extract_last_user_message(messages)
        assert result == "Hello from list"

    def test_content_as_mixed_list(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First part"},
                    {"type": "image", "url": "img.png"},
                    {"type": "text", "text": "Second part"},
                ]
            }
        ]
        result = _extract_last_user_message(messages)
        assert "First part" in result
        assert "Second part" in result

    def test_content_as_string_in_list(self):
        messages = [{"role": "user", "content": ["plain string"]}]
        result = _extract_last_user_message(messages)
        assert "plain string" in result

    def test_no_user_messages_returns_empty(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "reply"},
        ]
        assert _extract_last_user_message(messages) == ""
