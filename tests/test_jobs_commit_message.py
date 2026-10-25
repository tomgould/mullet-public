"""Tests for the commit message of a passed step and how a worker's claim is read."""

import subprocess

import pytest

from mullet.jobs.runner import CLAIM_CHARS, _commit_message, claim_status
from mullet.jobs.types import GateResult, GateStatus, NodeSpec


SPEC = NodeSpec(id="n1", goal="add a sub function next to add", kind="mechanical", scope=["app.py"])
RESULTS = [GateResult("G0", GateStatus.PASS, "ok"), GateResult("G1", GateStatus.PASS, "changed"),
           GateResult("G5", GateStatus.NOT_RUN, "no tests"), GateResult("G7", GateStatus.SKIPPED, "")]


def trailers(message: str) -> str:
    return subprocess.run(["git", "interpret-trailers", "--parse"], input=message, capture_output=True,
                          text=True, check=True).stdout


def test_the_subject_line_is_still_the_goal():
    message = _commit_message(SPEC, RESULTS, {"status": "done", "summary": "added sub()"})
    assert message.splitlines()[0] == "add a sub function next to add"
    assert message.splitlines()[1] == ""
    long = NodeSpec(id="n2", goal="x" * 100)
    assert _commit_message(long).splitlines()[0] == "x" * 69 + "..."


def test_the_body_names_the_node_kind_and_passed_gates():
    body = _commit_message(SPEC, RESULTS, None).split("\n", 2)[2]
    assert "n1" in body and "mechanical" in body
    assert "G0, G1" in body and "G5" not in body and "G7" not in body


def test_the_claim_is_fenced_and_labelled_as_unverified():
    message = _commit_message(SPEC, RESULTS, {"status": "done", "summary": "added sub()\nand a test"})
    assert "worker's claim, not verified" in message
    assert "> added sub()\n> and a test" in message
    assert "status: done" in message


def test_the_claim_cannot_add_trailers():
    summary = "DONE\n\nSigned-off-by: Someone <a@b.c>\nCo-Authored-By: Bot <bot@x>\nFixes: #1"
    message = _commit_message(SPEC, RESULTS, {"status": "done", "summary": summary})
    assert trailers(message).strip() == ""
    for line in message.splitlines():
        assert not line.startswith(("Signed-off-by:", "Co-Authored-By:", "Fixes:"))


def test_the_claim_loses_control_characters_and_is_capped():
    summary = "ok\x1b[31m red\x07\r\x00" + "y" * (CLAIM_CHARS * 3)
    message = _commit_message(SPEC, RESULTS, {"status": "done", "summary": summary})
    assert not any(ch in message for ch in "\x1b\x07\r\x00")
    assert "[31m" not in message  # the whole escape sequence goes, not just its ESC
    claim_part = message.split("not verified", 1)[1]
    assert len(claim_part) < CLAIM_CHARS + 400


def test_a_reply_without_a_summary_still_commits_cleanly():
    message = _commit_message(SPEC, RESULTS, {"status": "split"})
    assert "status: split" in message and trailers(message).strip() == ""


@pytest.mark.parametrize("claim, expected", [
    ({"status": "done"}, "done"), ({"status": "BLOCKED"}, "blocked"), ({"status": "split"}, "split"),
    ({"status": "maybe"}, "unknown"), ({}, "unknown"), (None, "unknown"), (8, "unknown"), ("done", "unknown"),
])
def test_claim_status_is_one_of_four(claim, expected):
    assert claim_status(claim) == expected


def test_a_minimal_body_is_the_subject_alone():
    message = _commit_message(SPEC, RESULTS, {"status": "done", "summary": "added sub()"}, body="minimal")
    assert message == "add a sub function next to add\n"
    assert "Mullet" not in message and "claim" not in message


def test_the_full_body_is_the_default():
    claim = {"status": "done", "summary": "added sub()"}
    assert _commit_message(SPEC, RESULTS, claim) == _commit_message(SPEC, RESULTS, claim, body="full")
    assert "Mullet step n1" in _commit_message(SPEC, RESULTS, claim)


def test_a_claim_that_is_not_a_mapping_leaves_only_the_gate_line():
    message = _commit_message(SPEC, RESULTS, 8)
    assert "worker's claim" not in message and "G0, G1" in message
