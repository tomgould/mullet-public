"""Tests for context compression."""

from mullet.core.agent.compression import ContextCompressor


def test_should_not_compress_short():
    comp = ContextCompressor(context_length=100_000)
    messages = [
        {"role": "system", "content": "You are a test agent."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    assert comp.should_compress(messages) is False


def test_should_compress_long():
    comp = ContextCompressor(context_length=1000, threshold_percent=0.5)
    messages = [
        {"role": "system", "content": "You are a test agent."},
    ]
    # Add enough messages to exceed 500 tokens (~2000 chars)
    for i in range(50):
        messages.append({"role": "user", "content": f"Message {i} " + "x" * 100})
        messages.append({"role": "assistant", "content": f"Response {i} " + "y" * 100})
    assert comp.should_compress(messages) is True


def test_compress_preserves_head_and_tail():
    comp = ContextCompressor(
        context_length=1000,
        threshold_percent=0.1,
        protect_first_n=2,
        protect_last_n=2,
        tail_protection=2,
    )
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "first user message"},
        {"role": "assistant", "content": "middle 1"},
        {"role": "user", "content": "middle 2"},
        {"role": "assistant", "content": "middle 3"},
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
    ]

    compressed = comp.compress(messages)

    # Head preserved
    assert compressed[0]["content"] == "system prompt"
    assert compressed[1]["content"] == "first user message"

    # Summary inserted
    assert "Summary" in compressed[2]["content"]

    # Tail preserved
    assert compressed[-1]["content"] == "recent answer"
    assert compressed[-2]["content"] == "recent question"

    # Should be shorter
    assert len(compressed) < len(messages)


def test_compress_too_short():
    comp = ContextCompressor(protect_first_n=2, protect_last_n=2)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    # Too short to compress — should return unchanged
    result = comp.compress(messages)
    assert result == messages


def test_simple_summary_extracts_tool_calls():
    comp = ContextCompressor()
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"function": {"name": "terminal", "arguments": '{"command": "ls"}'}, "id": "1"},
            {"function": {"name": "read_file", "arguments": '{"path": "foo.py"}'}, "id": "2"},
        ]},
        {"role": "tool", "content": "file1.py\nfile2.py", "tool_call_id": "1"},
        {"role": "assistant", "content": "Found the issue in the auth module."},
    ]

    summary = comp._simple_summary(messages)
    assert "terminal" in summary
    assert "read_file" in summary
    assert "auth module" in summary


# ---------------------------------------------------------------------------
# StructuredSummary.to_message_content
# ---------------------------------------------------------------------------

from mullet.core.agent.compression import StructuredSummary


class TestToMessageContent:
    def test_empty_returns_placeholder(self):
        s = StructuredSummary()
        assert s.to_message_content() == "Previous conversation context (compressed)."

    def test_key_decisions_rendered(self):
        s = StructuredSummary(key_decisions=["Use SQLite", "No async"])
        text = s.to_message_content()
        assert "Key Decisions:" in text
        assert "Use SQLite" in text
        assert "No async" in text

    def test_code_changes_rendered(self):
        s = StructuredSummary(code_changes=["Added tests", "Fixed auth bug"])
        text = s.to_message_content()
        assert "Code Changes:" in text
        assert "Added tests" in text

    def test_open_questions_rendered(self):
        s = StructuredSummary(open_questions=["Should we cache?"])
        text = s.to_message_content()
        assert "Open Questions:" in text
        assert "Should we cache?" in text

    def test_context_rendered(self):
        s = StructuredSummary(context="Working on auth module refactor.")
        text = s.to_message_content()
        assert "Context:" in text
        assert "auth module" in text

    def test_all_sections_present(self):
        s = StructuredSummary(
            key_decisions=["A"],
            code_changes=["B"],
            open_questions=["C"],
            context="D",
        )
        text = s.to_message_content()
        for marker in ("Key Decisions:", "Code Changes:", "Open Questions:", "Context:"):
            assert marker in text

    def test_sections_separated_by_double_newline(self):
        s = StructuredSummary(key_decisions=["A"], code_changes=["B"])
        text = s.to_message_content()
        assert "\n\n" in text


# ---------------------------------------------------------------------------
# ContextCompressor.iterative_update
# ---------------------------------------------------------------------------

class TestIterativeUpdate:
    def test_appends_new_decisions(self):
        comp = ContextCompressor()
        existing = StructuredSummary(key_decisions=["Use SQL"])
        new_seg = StructuredSummary(key_decisions=["Add caching"])
        result = comp.iterative_update(existing, new_seg)
        assert "Use SQL" in result.key_decisions
        assert "Add caching" in result.key_decisions

    def test_deduplicates_decisions(self):
        comp = ContextCompressor()
        existing = StructuredSummary(key_decisions=["Use SQL"])
        new_seg = StructuredSummary(key_decisions=["Use SQL"])
        result = comp.iterative_update(existing, new_seg)
        assert result.key_decisions.count("Use SQL") == 1

    def test_appends_code_changes(self):
        comp = ContextCompressor()
        existing = StructuredSummary(code_changes=["Fixed auth"])
        new_seg = StructuredSummary(code_changes=["Added tests"])
        result = comp.iterative_update(existing, new_seg)
        assert "Fixed auth" in result.code_changes
        assert "Added tests" in result.code_changes

    def test_context_concatenated(self):
        comp = ContextCompressor()
        existing = StructuredSummary(context="Phase 1 done.")
        new_seg = StructuredSummary(context="Phase 2 done.")
        result = comp.iterative_update(existing, new_seg)
        assert "Phase 1 done." in result.context
        assert "Phase 2 done." in result.context

    def test_empty_new_context_not_appended(self):
        comp = ContextCompressor()
        existing = StructuredSummary(context="Original context.")
        new_seg = StructuredSummary()  # no context
        result = comp.iterative_update(existing, new_seg)
        assert result.context == "Original context."

    def test_does_not_mutate_existing(self):
        comp = ContextCompressor()
        original_decisions = ["Decision A"]
        existing = StructuredSummary(key_decisions=list(original_decisions))
        new_seg = StructuredSummary(key_decisions=["Decision B"])
        comp.iterative_update(existing, new_seg)
        assert existing.key_decisions == original_decisions  # original unchanged
