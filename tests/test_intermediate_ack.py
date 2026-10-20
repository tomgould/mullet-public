"""Tests for intermediate ack detection — preventing premature stops."""

from mullet.core.agent.runner import _is_intermediate_ack


def test_planning_message():
    assert _is_intermediate_ack("I'll check the file and see what's wrong.")


def test_let_me_message():
    assert _is_intermediate_ack("Let me read the configuration file.")


def test_going_to_message():
    assert _is_intermediate_ack("I'm going to run the tests to verify.")


def test_actual_answer_not_ack():
    assert not _is_intermediate_ack("The function returns 42.")


def test_result_message_not_ack():
    assert not _is_intermediate_ack("Here is the output from the command.")


def test_long_response_not_ack():
    assert not _is_intermediate_ack("I'll check " + "x" * 1000)


def test_empty_not_ack():
    assert not _is_intermediate_ack("")
    assert not _is_intermediate_ack(None)


def test_no_action_word():
    assert not _is_intermediate_ack("I'll think about this problem.")


def test_no_future_phrase():
    assert not _is_intermediate_ack("Check the file for errors.")


def test_found_result_not_ack():
    assert not _is_intermediate_ack("I found the issue in the config file.")
