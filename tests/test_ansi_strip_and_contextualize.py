"""Tests for ansi_strip utility and query contextualization helper."""
from __future__ import annotations

from mullet.core.utils.ansi_strip import strip_ansi
from mullet.knowledge.engine.search.contextualize import query_contextualize


# ---------------------------------------------------------------------------
# strip_ansi
# ---------------------------------------------------------------------------

class TestStripAnsi:
    def test_plain_text_unchanged(self) -> None:
        assert strip_ansi("hello world") == "hello world"

    def test_removes_color_code(self) -> None:
        assert strip_ansi("\x1b[31mred text\x1b[0m") == "red text"

    def test_removes_bold(self) -> None:
        assert strip_ansi("\x1b[1mbold\x1b[0m") == "bold"

    def test_removes_multi_param_sequence(self) -> None:
        result = strip_ansi("\x1b[1;32mgreen bold\x1b[0m")
        assert result == "green bold"

    def test_empty_string_unchanged(self) -> None:
        assert strip_ansi("") == ""

    def test_mixed_content(self) -> None:
        result = strip_ansi("normal \x1b[33myellow\x1b[0m and back")
        assert result == "normal yellow and back"

    def test_osc_sequence_removed(self) -> None:
        # OSC: \x1b] ... \x07
        result = strip_ansi("\x1b]0;window title\x07plain")
        assert result == "plain"

    def test_multiple_sequences(self) -> None:
        result = strip_ansi("\x1b[31mA\x1b[0m \x1b[32mB\x1b[0m")
        assert result == "A B"

    def test_cursor_movement_removed(self) -> None:
        # Cursor up: \x1b[1A
        result = strip_ansi("\x1b[1Asome text")
        assert result == "some text"

    def test_no_escape_code_in_output(self) -> None:
        text = "\x1b[0m\x1b[1m\x1b[32m"
        assert strip_ansi(text) == ""


# ---------------------------------------------------------------------------
# query_contextualize
# ---------------------------------------------------------------------------

class TestQueryContextualize:
    def test_standalone_query_unchanged(self) -> None:
        result = query_contextualize(
            "How does authentication work?",
            ["previous context"]
        )
        assert result == "How does authentication work?"

    def test_empty_history_returns_query(self) -> None:
        result = query_contextualize("what about tests?", [])
        assert result == "what about tests?"

    def test_empty_query_returns_empty(self) -> None:
        result = query_contextualize("", ["some history"])
        assert result == ""

    def test_what_about_triggers_followup(self) -> None:
        history = ["How does the authentication system work?"]
        result = query_contextualize("what about the tests?", history)
        assert "authentication system" in result
        assert "what about the tests?" in result

    def test_how_about_triggers_followup(self) -> None:
        history = ["Explain the database schema"]
        result = query_contextualize("how about the indexes?", history)
        assert "database schema" in result

    def test_also_triggers_followup(self) -> None:
        history = ["Show me the login flow"]
        result = query_contextualize("also the logout flow", history)
        assert "login flow" in result

    def test_short_question_triggers_followup(self) -> None:
        # Very short question like "tests?" — pattern r"^\w{1,5}\?$"
        history = ["How do I run the integration tests for the auth module?"]
        result = query_contextualize("tests?", history)
        assert "integration tests" in result

    def test_skips_short_history_entries(self) -> None:
        # History entries ≤10 chars are skipped
        history = ["short", "How does the cache invalidation work in the system?"]
        result = query_contextualize("what about eviction?", history)
        assert "cache invalidation" in result

    def test_skips_followup_history_entries(self) -> None:
        # Previous follow-ups are skipped; looks for substantive entry
        history = [
            "How does authentication work?",
            "what about tests?",  # itself a follow-up
        ]
        result = query_contextualize("also logging?", history)
        # Most recent NON-followup is "How does authentication work?"
        assert "authentication" in result

    def test_separator_format(self) -> None:
        history = ["Explain the payment processing flow"]
        result = query_contextualize("what about refunds?", history)
        assert " — " in result

    def test_furthermore_triggers_followup(self) -> None:
        history = ["Describe the microservice architecture"]
        result = query_contextualize("furthermore the deployment pipeline", history)
        assert "microservice" in result

    def test_same_triggers_followup(self) -> None:
        history = ["How do I configure the database pool?"]
        result = query_contextualize("same for redis?", history)
        assert "database pool" in result

    def test_returns_query_when_no_substantive_history(self) -> None:
        # All history entries are follow-ups themselves
        history = ["what about that?", "also this?"]
        original = "more details?"
        result = query_contextualize(original, history)
        assert result == original
