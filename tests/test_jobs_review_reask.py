"""Tests for asking the G7 reviewer again, in the same session, when it gives no verdict."""

import json
import stat
from types import SimpleNamespace

import pytest

from mullet.jobs.clone import ChangedFile, Snapshot
from mullet.jobs.executors.base import Budget, ExecResult
from mullet.jobs.executors.claude_code import ClaudeCodeExecutor
from mullet.jobs.gates import GatePaused
from mullet.jobs.review import REASK_PROMPT, REVIEW_SCHEMA, make_reviewer
from mullet.jobs.types import GateStatus, NodeSpec

from tests.jobs_helpers import FAKE_CLAUDE

APPROVE = {"verdict": "approve", "findings": [], "reasons": []}
# A session that finished on its own but left no structured verdict: the only kind asked again.
NO_VERDICT = ExecResult(exit_code=0, session_id="sess-9")
# Sessions that were cut off mid-review: asking them to "answer now" would get a verdict on partial reading.
TURN_LIMIT = ExecResult(exit_code=1, error="error_max_turns", session_id="sess-9")
TIMED_OUT = ExecResult(exit_code=None, error="timed out", session_id="sess-9")


def ctx(tmp_path):
    node = NodeSpec(id="n1", goal="add a greeting", scope=["app.py"])
    snap = Snapshot(tree="t", base="b", diff="diff --git a/app.py b/app.py\n+y\n",
                    files=[ChangedFile("app.py", "M", "100644", "100644")])
    return SimpleNamespace(node=node, snapshot=snap, clone=SimpleNamespace(work=tmp_path))


class Script:
    """A fake reviewer session: `ask` and `reask` return the next scripted result and record the call."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def ask(self, prompt, cwd, schema, budget):
        self.calls.append(("ask", None, prompt))
        return self.results.pop(0)

    def reask(self, prompt, session, cwd, schema, budget):
        assert schema is REVIEW_SCHEMA and cwd is not None
        self.calls.append(("reask", session, prompt))
        return self.results.pop(0)


def test_no_verdict_then_a_verdict_in_the_same_session_passes(tmp_path):
    script = Script(NO_VERDICT, ExecResult(exit_code=0, claim=APPROVE, session_id="sess-9"))
    result = make_reviewer("Goal", script.ask, reask=script.reask, reask_attempts=1)(ctx(tmp_path))
    assert result.status is GateStatus.PASS
    assert [(kind, session) for kind, session, _ in script.calls] == [("ask", None), ("reask", "sess-9")]
    assert script.calls[1][2] == REASK_PROMPT
    assert result.details["reasked"] == 1


def test_the_reask_prompt_asks_only_for_the_verdict():
    assert "only the JSON verdict" in REASK_PROMPT
    assert "Don't review again" in REASK_PROMPT


def test_a_block_after_asking_again_fails_with_its_findings(tmp_path):
    block = {"verdict": "block", "reasons": [], "findings": [{"path": "app.py", "line": 2, "problem": "no test"}]}
    script = Script(NO_VERDICT, ExecResult(exit_code=0, claim=block))
    result = make_reviewer("Goal", script.ask, reask=script.reask, reask_attempts=1)(ctx(tmp_path))
    assert result.status is GateStatus.FAIL and result.details["problems"] == ["app.py:2: no test"]


def test_with_reask_attempts_zero_no_verdict_is_not_run_as_before(tmp_path):
    script = Script(NO_VERDICT)
    result = make_reviewer("Goal", script.ask, reask=script.reask, reask_attempts=0)(ctx(tmp_path))
    assert result.status is GateStatus.NOT_RUN
    assert [kind for kind, _, _ in script.calls] == ["ask"]
    assert "reasked" not in result.details


def test_it_gives_up_after_the_configured_number_of_reasks(tmp_path):
    script = Script(NO_VERDICT, NO_VERDICT, NO_VERDICT, ExecResult(exit_code=0, claim=APPROVE))
    result = make_reviewer("Goal", script.ask, reask=script.reask, reask_attempts=2)(ctx(tmp_path))
    assert result.status is GateStatus.NOT_RUN
    assert [kind for kind, _, _ in script.calls] == ["ask", "reask", "reask"]
    assert result.details["reasked"] == 2
    assert "asked again 2 times" in result.summary


def test_without_a_session_to_resume_it_does_not_reask(tmp_path):
    script = Script(ExecResult(exit_code=1, error="no verdict"))
    result = make_reviewer("Goal", script.ask, reask=script.reask, reask_attempts=1)(ctx(tmp_path))
    assert result.status is GateStatus.NOT_RUN and len(script.calls) == 1


@pytest.mark.parametrize("cut_off", [TURN_LIMIT, TIMED_OUT], ids=["turn-limit", "timeout"])
def test_a_review_cut_off_by_its_turn_limit_or_clock_is_not_asked_again(tmp_path, cut_off):
    script = Script(cut_off, ExecResult(exit_code=0, claim=APPROVE, session_id="sess-9"))
    result = make_reviewer("Goal", script.ask, reask=script.reask, reask_attempts=1)(ctx(tmp_path))
    assert result.status is GateStatus.NOT_RUN
    assert [kind for kind, _, _ in script.calls] == ["ask"]
    assert "reasked" not in result.details


def test_a_reask_that_is_itself_cut_off_is_not_asked_again(tmp_path):
    script = Script(NO_VERDICT, TURN_LIMIT, ExecResult(exit_code=0, claim=APPROVE))
    result = make_reviewer("Goal", script.ask, reask=script.reask, reask_attempts=3)(ctx(tmp_path))
    assert result.status is GateStatus.NOT_RUN
    assert [kind for kind, _, _ in script.calls] == ["ask", "reask"]


def test_a_verdict_first_time_never_reasks(tmp_path):
    script = Script(ExecResult(exit_code=0, claim=APPROVE, session_id="sess-1"))
    make_reviewer("Goal", script.ask, reask=script.reask, reask_attempts=1)(ctx(tmp_path))
    assert len(script.calls) == 1


def test_a_usage_limit_on_the_reask_pauses_and_reports_usage(tmp_path):
    usage = []
    script = Script(NO_VERDICT, ExecResult(exit_code=1, paused_until=99.0, error="limit", usage={"n": 2}))
    reviewer = make_reviewer("Goal", script.ask, reask=script.reask, reask_attempts=1, on_usage=usage.append)
    with pytest.raises(GatePaused) as paused:
        reviewer(ctx(tmp_path))
    assert paused.value.until == 99.0
    assert usage == [{}, {"n": 2}]


# The executor side: a reviewer session can be kept and resumed.


@pytest.fixture
def fake(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "claude"
    script.write_text(FAKE_CLAUDE)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    (bin_dir / "scenario.json").write_text(json.dumps({"calls": [{}, {"structured": APPROVE}]}))
    return {"binary": str(script), "calls": bin_dir / "calls.jsonl"}


def test_the_claude_executor_can_keep_and_resume_a_read_only_session(tmp_path, fake):
    work = tmp_path / "work"
    work.mkdir()
    executor = ClaudeCodeExecutor(tmp_path / "logs", binary=fake["binary"])
    first = executor.ask("review this", work, REVIEW_SCHEMA, Budget(), persist=True)
    assert first.claim is None and first.session_id == "sess-1"
    second = executor.ask(REASK_PROMPT, work, REVIEW_SCHEMA, Budget(), persist=True, resume=first.session_id)
    assert second.claim == APPROVE
    calls = [json.loads(line) for line in fake["calls"].read_text().splitlines()]
    assert "--no-session-persistence" not in calls[0]["argv"]
    assert calls[1]["argv"][calls[1]["argv"].index("--resume") + 1] == "sess-1"
    assert "Read,Glob,Grep" in calls[1]["argv"]  # still read-only
