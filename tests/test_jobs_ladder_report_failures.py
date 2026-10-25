"""The "Why attempts failed" section of the ladder report, from a hand-built ledger."""

from __future__ import annotations

import pytest

from mullet.jobs.ladder import LadderConfig
from mullet.jobs.ladder_report import build, render_json, render_text
from mullet.jobs.ledger import Ledger
from mullet.jobs.types import GateResult, GateStatus, JobStatus, NodeSpec

GATES = ("G1", "G2", "G3", "G4", "G5", "G6", "G7")


def _fail_at(ledger: Ledger, job_id: str, gate: str, summary: str) -> None:
    """Finish one attempt whose gates passed until `gate` failed."""
    attempt_id = ledger.start_attempt(job_id, "a", "opencode", "base")
    ledger.record_executor_result(attempt_id, 0, {"rung": "free"}, None, None)
    for earlier in GATES:
        if earlier == gate:
            break
        ledger.record_gate(attempt_id, GateResult(gate=earlier, status=GateStatus.PASS, summary="ok"))
    ledger.record_gate(attempt_id, GateResult(gate=gate, status=GateStatus.FAIL, summary=summary))
    ledger.finish_attempt(attempt_id, f"end-{attempt_id}")


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / "ledger.db") as led:
        job_id = led.create_job("/tmp/repo", "g", "main", "abc123", {}, budget={"ladder": True}, max_attempts=20)
        led.set_plan(job_id, [(NodeSpec(id="a", goal="one", kind="mechanical"), None)])
        led.set_job_status(job_id, JobStatus.AWAITING_APPROVAL)
        led.approve(job_id)
        led.set_job_status(job_id, JobStatus.RUNNING)
        for _ in range(3):
            _fail_at(led, job_id, "G7", "reviewer found a missing edge case")
        _fail_at(led, job_id, "G7", "reviewer rejected the change")
        for _ in range(2):
            _fail_at(led, job_id, "G3", "tests failed: " + "x" * 400)
        _fail_at(led, job_id, "G2", "lint errors")
        yield led


def test_failures_are_counted_by_first_failing_gate(ledger):
    report = build(ledger, config=LadderConfig())
    assert report.failures_by_gate == {"G7": 4, "G3": 2, "G2": 1}
    assert report.failed_attempts == 7
    assert report.failure_reasons["G7"] == "reviewer found a missing edge case"


def test_text_section_truncates_the_summary(ledger):
    text = render_text(build(ledger, config=LadderConfig()))
    section = text.split("Why attempts failed", 1)[1].split("\n\n", 1)[0]
    assert "G7" in section and "4 of 7" in section
    assert "x" * 200 not in section
    assert "..." in section


def test_json_has_the_same_numbers(ledger):
    data = render_json(build(ledger, config=LadderConfig()))["why_attempts_failed"]
    assert data["failed_attempts"] == 7
    assert [g["gate"] for g in data["gates"]] == ["G7", "G3", "G2"]
    assert data["gates"][0] == {"gate": "G7", "attempts": 4, "reason": "reviewer found a missing edge case"}
    assert len(data["gates"][1]["reason"]) <= 123


def test_no_failures_says_so(tmp_path):
    with Ledger(tmp_path / "ledger.db") as led:
        job_id = led.create_job("/tmp/repo", "g", "main", "abc123", {}, budget={"ladder": True})
        led.set_plan(job_id, [(NodeSpec(id="a", goal="one", kind="mechanical"), None)])
        report = build(led, config=LadderConfig())
    assert "no failed attempts" in render_text(report)
    assert render_json(report)["why_attempts_failed"] == {"failed_attempts": 0, "gates": []}
