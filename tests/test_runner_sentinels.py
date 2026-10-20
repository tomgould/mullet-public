"""Tests for agent runner sentinel functions."""

from mullet.core.agent.runner import _is_intermediate_ack, _preview_args


class TestIsIntermediateAck:
    def test_planning_statement_detected(self):
        assert _is_intermediate_ack("I'll check the file for you") is True

    def test_let_me_run_detected(self):
        assert _is_intermediate_ack("Let me run the tests now") is True

    def test_result_answer_not_ack(self):
        assert _is_intermediate_ack("Here is the answer: 42") is False

    def test_empty_string_false(self):
        assert _is_intermediate_ack("") is False

    def test_too_long_returns_false(self):
        # Strings >800 chars are not acks
        assert _is_intermediate_ack("I'll check " + "x" * 800) is False

    def test_no_future_phrase_false(self):
        assert _is_intermediate_ack("The file contains some data") is False

    def test_no_action_word_false(self):
        assert _is_intermediate_ack("I'll be happy to assist") is False


class TestPreviewArgs:
    def test_truncates_long_string_arg(self):
        result = _preview_args("x" * 200, max_len=50)
        assert len(result) <= 50

    def test_known_tool_extracts_primary(self):
        result = _preview_args({"command": "ls -la"}, tool_name="terminal")
        assert result == "ls -la"

    def test_unknown_tool_fallback(self):
        result = _preview_args({"key": "value"}, tool_name="unknown_tool")
        assert "value" in result

    def test_empty_args_returns_empty(self):
        assert _preview_args({}) == ""

    def test_json_string_parsed(self):
        import json
        result = _preview_args(json.dumps({"command": "pwd"}), tool_name="terminal")
        assert result == "pwd"

    def test_invalid_json_string_truncated(self):
        result = _preview_args("not json at all " * 10, max_len=20)
        assert len(result) <= 20

    def test_deeply_nested_args_truncated(self):
        large = {"data": "x" * 500}
        result = _preview_args(large, max_len=50)
        assert len(result) <= 53  # +3 for "..."
