"""Tests for runner.py internal helpers: _is_intermediate_ack + _preview_args."""


from mullet.core.agent.runner import _is_intermediate_ack, _preview_args


# --- _is_intermediate_ack --------------------------------------------------


def test_ack_empty_returns_false():
    assert _is_intermediate_ack("") is False
    assert _is_intermediate_ack(None) is False  # type: ignore[arg-type]


def test_ack_too_long_returns_false():
    content = "x" * 900
    assert _is_intermediate_ack(content) is False


def test_ack_no_future_phrase_returns_false():
    """Plain answers without a future-action phrase aren't acks."""
    assert _is_intermediate_ack("The answer is 42.") is False
    assert _is_intermediate_ack("Here's the result.") is False


def test_ack_classic_positive():
    """Classic LLM planning utterances fire."""
    assert _is_intermediate_ack("I'll check the file for errors.") is True
    assert _is_intermediate_ack("Let me run the tests.") is True
    assert _is_intermediate_ack("I need to look at the config first.") is True


def test_ack_no_action_word_returns_false():
    """Future phrase without an action word doesn't fire."""
    assert _is_intermediate_ack("I'll be right back.") is False
    assert _is_intermediate_ack("Let me think.") is False


def test_ack_with_result_returns_false():
    """If the message already has an answer, it's not an ack."""
    assert (
        _is_intermediate_ack(
            "I'll look at this — the answer is 42."
        )
        is False
    )
    assert (
        _is_intermediate_ack(
            "Let me check. Here is the output: OK."
        )
        is False
    )


# --- _preview_args ---------------------------------------------------------


def test_preview_args_short_dict():
    out = _preview_args({"key": "value"})
    assert "key" in out


def test_preview_args_long_string_truncates():
    long = "x" * 500
    out = _preview_args(long, max_len=80)
    assert len(out) <= 80


def test_preview_args_invalid_json_string_still_truncates():
    out = _preview_args("not valid json {", max_len=50)
    assert len(out) <= 50


def test_preview_args_empty_dict():
    out = _preview_args({})
    assert out is not None  # should not crash


def test_preview_args_json_string_decodes_to_str():
    """JSON-encoded string literal must not crash primary-key lookup."""
    out = _preview_args('"npm run dev"', tool_name="terminal")
    assert out == "npm run dev"


def test_preview_args_json_list_decodes_to_list():
    out = _preview_args('["a", "b"]', tool_name="terminal")
    assert out == "['a', 'b']"


def test_preview_args_json_int_decodes_to_int():
    out = _preview_args("42", tool_name="terminal")
    assert out == "42"


def test_preview_args_non_dict_input_directly():
    out = _preview_args(["x", "y"], tool_name="terminal")
    assert out == "['x', 'y']"
