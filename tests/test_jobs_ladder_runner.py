"""Ladder jobs end to end: real clones, real gates in the sandbox and a real ledger.

Each rung's executor is a scripted fake and the ladder's outside lookups
(Claude's budget, LM Studio, GitHub, keys) are replaced, so every outcome
still comes from the gates and every rung decision from the policy.
"""

import time
from pathlib import Path

import pytest

from mullet.jobs import machine, sandbox
from mullet.jobs.clone import resolve_commit
from mullet.jobs.executors.base import ExecResult, ExecutorUnavailable
from mullet.jobs.executors.fake import FakeExecutor
from mullet.jobs.ladder import BudgetState, Ladder, MachineState, PublicCheck, parse_config
from mullet.jobs.ledger import Ledger
from mullet.jobs.night import NightState
from mullet.jobs.profile import Checks, Profile
from mullet.jobs.report import build_report
from mullet.jobs.runner import JobRunner, RunnerOptions
from mullet.jobs.types import AttemptOutcome, JobStatus, NodeSpec, NodeStatus

from tests.jobs_helpers import good_change, make_source_repo

pytestmark = pytest.mark.skipif(not sandbox.check().ok, reason=sandbox.check().reason)

CHECKS = Checks(test_full="python3 runtests.py", test_targeted="python3 runtests.py {tests}", junit=True)
MECH = NodeSpec(id="n1", goal="add sub()", kind="mechanical")
KIMI = "moonshotai/kimi-k2.7-code"
QWEN = "openrouter/qwen/qwen3-coder"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setattr(machine.paths, "state_dir", lambda: tmp_path / "state")
    source = make_source_repo(tmp_path / "src")
    ledger = Ledger(tmp_path / "jobs" / "ledger.db")
    yield {"tmp": tmp_path, "source": source, "ledger": ledger}
    ledger.close()


def make_job(env, nodes, *, max_attempts=3, max_spend=2.0, **profile_kwargs) -> str:
    ledger = env["ledger"]
    profile_kwargs.setdefault("allowed_executors", ["claude-code", "opencode"])
    profile = Profile(repo_id="fixture", checks=CHECKS, env={"PYTHONDONTWRITEBYTECODE": "1"}, **profile_kwargs)
    job_id = ledger.create_job(str(env["source"]), "improve the app", "main", resolve_commit(env["source"], "HEAD"),
                               profile.model_dump(), max_attempts=max_attempts,
                               budget={"max_turns": 5, "wall_clock_s": 60, "ladder": True, "max_spend": max_spend})
    ledger.set_plan(job_id, [n if isinstance(n, tuple) else (n, None) for n in nodes])
    ledger.set_job_status(job_id, JobStatus.AWAITING_APPROVAL)
    ledger.approve(job_id)
    return job_id


def make_ladder(config=None, *, level="medium", machine_ok=False, public=False, public_check=None,
                night=None):
    ladder = Ladder(
        config=parse_config(config if config is not None else {"kimi": KIMI, "openrouter": [QWEN]}),
        usage=lambda: None,
        machine=lambda c, wait_s=0: MachineState(machine_ok, "fake machine"),
        public_check=public_check or (lambda repo, sha, profile: PublicCheck(public, "fake public check", "o/r",
                                                                             "main", time.time())),
        has_key=lambda rung: True,
        night=night if night is not None else (lambda: NightState()),
    )
    ladder.budget = lambda: BudgetState(level, {"low": 0.2, "medium": 0.6, "high": 0.9}[level])
    return ladder


def break_add(work: Path, task=None):
    (work / "app.py").write_text((work / "app.py").read_text().replace("return a + b", "return a - b"))


def good(work, task):
    good_change(work)


def bad(work, task):
    break_add(work)


class Notes:
    def __init__(self):
        self.sent = []

    def __call__(self, event, job_id, goal, detail=""):
        self.sent.append((event, detail))


def run(env, job_id, fakes, ladder=None, **kwargs):
    notes = kwargs.pop("notify", Notes())
    runner = JobRunner(env["ledger"], job_id, ladder=ladder or make_ladder(),
                       rung_executors=lambda rung: fakes[rung.name], notify=notes, **kwargs)
    return runner.run(), notes


def rungs_tried(env, job_id, node_id="n1"):
    return [((a.usage or {}).get("rung"), a.outcome) for a in env["ledger"].attempts(job_id, node_id)]


def test_a_mechanical_step_gets_one_attempt_per_rung_until_one_passes(env):
    fakes = {"kimi": FakeExecutor([bad]), "openrouter": FakeExecutor([good]), "claude": FakeExecutor([])}
    job_id = make_job(env, [MECH])
    status, notes = run(env, job_id, fakes)
    assert status is JobStatus.DONE
    ledger = env["ledger"]
    assert rungs_tried(env, job_id) == [("kimi", AttemptOutcome.FAILED), ("openrouter", AttemptOutcome.PASSED)]
    attempts = ledger.attempts(job_id, "n1")
    assert [a.executor for a in attempts] == ["opencode", "opencode"]
    assert [(a.usage["rung_key"], a.usage["model"]) for a in attempts] == [
        (f"kimi:{KIMI}", KIMI), (f"openrouter:{QWEN}", QWEN)]
    assert fakes["claude"].calls == []
    assert any("G5 failed" in f for f in fakes["openrouter"].calls[0].feedback)  # the gate output went up too
    choices = [e.data for e in ledger.events(job_id) if e.kind == "ladder_choice"]
    assert [c["rung"] for c in choices] == [f"kimi:{KIMI}", f"openrouter:{QWEN}"]
    assert choices[0]["rungs"] == [f"kimi:{KIMI}", f"openrouter:{QWEN}", "claude:"]
    assert ledger.audit(job_id) == []
    assert notes.sent == [("finished", "1 of 1 steps passed")]

    report = build_report(ledger, job_id, config=parse_config({}))
    assert f"Attempt 1 (opencode, rung kimi, {KIMI}): failed" in report
    assert f"Attempt 2 (opencode, rung openrouter, {QWEN}): passed" in report
    assert "Ladder: 1 of 1 steps passed below Claude; spent $0.00; Claude used on 0 steps" in report


@pytest.mark.parametrize("failure", [
    ExecResult(exit_code=None, aborted=True, provider_error=True, error="opencode reported an error: 429"),
    ExecResult(exit_code=3, aborted=True, error="opencode exited 3"),
])
def test_a_provider_error_or_broken_run_moves_up_without_using_an_attempt(env, failure):
    fakes = {"kimi": FakeExecutor([lambda w, t: failure]), "openrouter": FakeExecutor([good]),
             "claude": FakeExecutor([])}
    job_id = make_job(env, [MECH])
    assert run(env, job_id, fakes)[0] is JobStatus.DONE
    ledger = env["ledger"]
    assert rungs_tried(env, job_id) == [("kimi", AttemptOutcome.ABORTED), ("openrouter", AttemptOutcome.PASSED)]
    first = ledger.attempts(job_id, "n1")[0]
    assert first.usage["moved_up"] is True and first.usage["provider_error"] is failure.provider_error
    assert first.reason.startswith("provider error on kimi" if failure.provider_error else "executor failed on kimi")
    assert ledger.attempt_count(job_id, "n1") == 1  # only the passing attempt counted
    assert [e.data["rung"] for e in ledger.events(job_id) if e.kind == "rung_moved_up"] == [f"kimi:{KIMI}"]


def test_a_rung_whose_executor_cant_be_made_moves_up(env):
    claude = FakeExecutor([good])

    def factory(rung):
        if rung.name == "claude":
            return claude
        raise ExecutorUnavailable(f"no model {rung.model}")

    job_id = make_job(env, [MECH])
    runner = JobRunner(env["ledger"], job_id, ladder=make_ladder(), rung_executors=factory, notify=Notes())
    assert runner.run() is JobStatus.DONE
    assert rungs_tried(env, job_id) == [("kimi", AttemptOutcome.ABORTED), ("openrouter", AttemptOutcome.ABORTED),
                                        ("claude", AttemptOutcome.PASSED)]
    assert "no model" in env["ledger"].attempts(job_id, "n1")[0].reason


def costly_failure(cost):
    def step(work, task):
        break_add(work)
        return ExecResult(exit_code=0, claim={"status": "done"}, usage={
            "total_cost_usd": cost, "cost_reported": True, "model": KIMI, "tokens": {"input": 1000, "output": 10}})
    return step


def test_the_spending_cap_skips_the_remaining_paid_rungs(env):
    fakes = {"kimi": FakeExecutor([costly_failure(0.6)]), "openrouter": FakeExecutor([good]),
             "claude": FakeExecutor([good])}
    job_id = make_job(env, [MECH], max_spend=0.5)
    assert run(env, job_id, fakes)[0] is JobStatus.DONE
    assert rungs_tried(env, job_id) == [("kimi", AttemptOutcome.FAILED), ("claude", AttemptOutcome.PASSED)]
    assert fakes["openrouter"].calls == []
    second = [e.data for e in env["ledger"].events(job_id) if e.kind == "ladder_choice"][-1]
    assert any("spending cap" in why for _, why in second["skipped"])
    report = build_report(env["ledger"], job_id, config=parse_config({}))
    assert "Ladder: 0 of 1 steps passed below Claude; spent $0.60: Kimi $0.60; Claude used on 1 step" in report
    assert "$0.60" in report.split("Attempt 1 (")[1].splitlines()[0]
    assert "billed $0.60 by their providers" in report


def test_a_claude_only_profile_never_leaves_claude(env):
    fakes = {"kimi": FakeExecutor([good]), "openrouter": FakeExecutor([good]), "claude": FakeExecutor([good])}
    job_id = make_job(env, [MECH], ladder="claude-only")
    assert run(env, job_id, fakes)[0] is JobStatus.DONE
    assert rungs_tried(env, job_id) == [("claude", AttemptOutcome.PASSED)]
    assert fakes["kimi"].calls == [] and fakes["openrouter"].calls == []


def test_a_parent_is_checked_by_the_gates_and_its_fixes_are_claude_only(env, monkeypatch):
    parent = NodeSpec(id="p", goal="maths helpers", kind="mechanical")
    child = NodeSpec(id="c", goal="add sub()", kind="mechanical")
    fakes = {"kimi": FakeExecutor([good]), "openrouter": FakeExecutor([]), "claude": FakeExecutor([])}
    job_id = make_job(env, [parent, (child, "p")])
    assert run(env, job_id, fakes)[0] is JobStatus.DONE
    tried = [(a.node_id, a.executor, (a.usage or {}).get("rung")) for a in env["ledger"].attempts(job_id)]
    assert tried == [("c", "opencode", "kimi"), ("p", "integration-check", None)]
    # The report counts steps (leaves) and shows all of Claude's work, planner and reviewer included.
    from mullet.jobs.runner import claude_usage
    for role, turns in (("planner", 12), ("reviewer", 4), ("reviewer", 3)):
        env["ledger"].add_event(job_id, "claude_usage", claude_usage(role, {
            "num_turns": turns, "total_cost_usd": 0.5, "usage": {"input_tokens": 1000, "output_tokens": 500}}))
    report = build_report(env["ledger"], job_id, config=parse_config({}))
    assert "Ladder: 1 of 1 steps passed below Claude" in report
    assert "planner 1 session (12 turns, 1,500 tokens); reviewer 2 sessions (7 turns, 3,000 tokens)" in report
    # Were a fix needed, a mechanical parent's rungs are Claude only.
    runner = JobRunner(env["ledger"], job_id, ladder=make_ladder(), rung_executors=lambda r: fakes[r.name])
    runner._job, runner._profile = env["ledger"].get_job(job_id), Profile.model_validate(env["ledger"].get_job(job_id).profile)
    choice, _ = runner._rung_choice(parent, integration=True)
    assert [r.name for r in choice.rungs] == ["claude", "kimi"]


def test_design_steps_go_straight_to_claude(env):
    fakes = {"kimi": FakeExecutor([good]), "openrouter": FakeExecutor([]), "claude": FakeExecutor([good])}
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()", kind="design")])
    assert run(env, job_id, fakes)[0] is JobStatus.DONE
    assert rungs_tried(env, job_id) == [("claude", AttemptOutcome.PASSED)]


@pytest.mark.parametrize("level, claude_tries", [("medium", 2), ("high", 1)])
def test_claude_is_the_last_resort_and_gets_fewer_tries_when_its_week_is_tight(env, level, claude_tries):
    fakes = {"kimi": FakeExecutor([bad]), "claude": FakeExecutor([bad] * 3)}
    job_id = make_job(env, [MECH])
    ladder = make_ladder({"kimi": KIMI, "openrouter": []}, level=level)
    status, notes = run(env, job_id, fakes, ladder=ladder)
    assert status is JobStatus.BLOCKED
    assert rungs_tried(env, job_id) == [("kimi", AttemptOutcome.FAILED)] + [("claude", AttemptOutcome.FAILED)] * claude_tries
    node = env["ledger"].get_node(job_id, "n1")
    assert node.status is NodeStatus.BLOCKED and "ran out of rungs" in node.block_reason
    assert "ran out of rungs" in env["ledger"].get_job(job_id).error
    assert [e for e, _ in notes.sent] == ["blocked"] and "ran out of rungs" in notes.sent[0][1]


def test_when_claudes_week_has_room_it_comes_before_the_paid_rungs(env):
    fakes = {"free": FakeExecutor([bad]), "local": FakeExecutor([bad]), "kimi": FakeExecutor([good]),
             "claude": FakeExecutor([good])}
    config = {"kimi": KIMI, "openrouter": [], "free": "openrouter/x/free-coder:free",
              "local": "lmstudio/qwen3.6-35b-a3b"}
    job_id = make_job(env, [MECH], data_class="public")
    assert run(env, job_id, fakes, ladder=make_ladder(config, level="low", machine_ok=True, public=True))[0] \
        is JobStatus.DONE
    assert rungs_tried(env, job_id) == [("free", AttemptOutcome.FAILED), ("local", AttemptOutcome.FAILED),
                                        ("claude", AttemptOutcome.PASSED)]
    assert fakes["kimi"].calls == []


def test_with_room_a_design_step_goes_to_claude_first(env):
    design = NodeSpec(id="n1", goal="add sub()", kind="design", scope=["app.py", "tests/**"],
                      acceptance=["python3 runtests.py"])
    fakes = {"kimi": FakeExecutor([good]), "claude": FakeExecutor([good])}
    for level, first in (("low", "claude"), ("medium", "kimi")):
        job_id = make_job(env, [design])
        assert run(env, job_id, fakes, ladder=make_ladder({"kimi": KIMI, "openrouter": []}, level=level))[0] \
            is JobStatus.DONE
        assert rungs_tried(env, job_id) == [(first, AttemptOutcome.PASSED)]


def test_free_then_local_when_both_are_allowed(env):
    fakes = {"free": FakeExecutor([bad]), "local": FakeExecutor([good]), "kimi": FakeExecutor([]),
             "claude": FakeExecutor([])}
    config = {"kimi": KIMI, "openrouter": [], "free": "openrouter/qwen/qwen3-coder:free",
              "local": "lmstudio/qwen3.6-35b-a3b"}
    job_id = make_job(env, [MECH], data_class="public")
    assert run(env, job_id, fakes, ladder=make_ladder(config, machine_ok=True, public=True))[0] is JobStatus.DONE
    assert rungs_tried(env, job_id) == [("free", AttemptOutcome.FAILED), ("local", AttemptOutcome.PASSED)]
    checks = [e.data for e in env["ledger"].events(job_id) if e.kind == "public_check"]
    assert len(checks) == 1 and checks[0]["public"] is True  # checked once, kept for the job


def test_a_github_failure_skips_the_free_rung_for_that_attempt_only(env):
    fakes = {"free": FakeExecutor([good]), "kimi": FakeExecutor([bad]), "claude": FakeExecutor([])}
    config = {"kimi": KIMI, "openrouter": [], "free": "openrouter/x/free-coder:free"}
    answers = [PublicCheck(False, "GitHub's API: unreachable: URLError", transient=True),
               PublicCheck(True, "public on GitHub", "o/r", "main", time.time())]
    calls = []

    def check(repo, sha, profile):
        calls.append(sha)
        return answers[min(len(calls) - 1, 1)]

    job_id = make_job(env, [MECH], data_class="public")
    assert run(env, job_id, fakes, ladder=make_ladder(config, public_check=check))[0] is JobStatus.DONE
    assert rungs_tried(env, job_id) == [("kimi", AttemptOutcome.FAILED), ("free", AttemptOutcome.PASSED)]
    checks = [e.data for e in env["ledger"].events(job_id) if e.kind == "public_check"]
    assert [c["transient"] for c in checks] == [True, False]
    assert len(calls) == 2  # the good answer is cached for the rest


def test_a_local_step_never_runs_while_another_process_holds_the_gpu_lease(env):
    fakes = {"local": FakeExecutor([good]), "kimi": FakeExecutor([good]), "claude": FakeExecutor([])}
    config = {"kimi": KIMI, "openrouter": [], "local": "lmstudio/qwen3.6-35b-a3b"}
    job_id = make_job(env, [MECH])
    # The machine check said yes a moment ago; a reindex took the lease since.
    with machine.gpu_lease("reindex of pool docs", wait=False):
        assert run(env, job_id, fakes, ladder=make_ladder(config, machine_ok=True))[0] is JobStatus.DONE
    assert fakes["local"].calls == []
    assert rungs_tried(env, job_id) == [("local", AttemptOutcome.ABORTED), ("kimi", AttemptOutcome.PASSED)]
    assert "GpuBusy" in env["ledger"].attempts(job_id, "n1")[0].reason


class Resumable:
    """A fake worker with sessions: each run fails the gates and reports a session id."""

    def __init__(self, prefix, steps):
        self.name = prefix
        self.prefix = prefix
        self.steps = list(steps)
        self.runs, self.resumed = 0, []

    def _go(self, work, task):
        step = self.steps.pop(0)
        step(work, task)
        return ExecResult(exit_code=0, claim={"status": "done"}, session_id=f"{self.prefix}-{self.runs + len(self.resumed)}")

    def run(self, task, work, budget):
        self.runs += 1
        return self._go(work, task)

    def resume(self, task, work, budget, session_id):
        self.resumed.append(session_id)
        return self._go(work, task)


def test_a_session_is_resumed_only_on_the_rung_that_started_it(env):
    kimi = Resumable("kimi", [bad])
    claude = Resumable("claude", [bad, bad, good])
    job_id = make_job(env, [MECH], max_attempts=4)
    status, _ = run(env, job_id, {"kimi": kimi, "claude": claude}, ladder=make_ladder({"kimi": KIMI, "openrouter": []}))
    assert status is JobStatus.DONE
    assert kimi.resumed == []
    # Claude's first attempt starts fresh (kimi's session is not Claude's);
    # its second resumes its own; its last try always starts fresh.
    assert claude.runs == 2 and claude.resumed == ["claude-1"]


def test_paused_for_a_usage_limit_notifies_and_carries_on(env):
    calls = []

    def limited(work, task):
        calls.append("limited")
        return ExecResult(exit_code=None, paused_until=time.time() - 1, error="usage limit")

    fakes = {"claude": FakeExecutor([limited, good])}
    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()", kind="design")])
    status, notes = run(env, job_id, fakes, options=RunnerOptions(pause_margin_s=0, pause_poll_s=0.01),
                        sleep=lambda s: None)
    assert status is JobStatus.DONE
    assert [e for e, _ in notes.sent] == ["paused", "finished"]


def test_a_plain_job_notifies_too(env):
    ledger = env["ledger"]
    profile = Profile(repo_id="fixture", checks=CHECKS, env={"PYTHONDONTWRITEBYTECODE": "1"})
    job_id = ledger.create_job(str(env["source"]), "improve the app", "main", resolve_commit(env["source"], "HEAD"),
                               profile.model_dump(), budget={"max_turns": 5, "wall_clock_s": 60})
    ledger.set_plan(job_id, [(NodeSpec(id="n1", goal="add sub()"), None)])
    ledger.set_job_status(job_id, JobStatus.AWAITING_APPROVAL)
    ledger.approve(job_id)
    notes = Notes()
    assert JobRunner(ledger, job_id, FakeExecutor([bad] * 3), notify=notes).run() is JobStatus.BLOCKED
    assert notes.sent == [("blocked", "node n1 ran out of attempts")]


def test_a_risky_step_is_reviewed_even_with_review_off_and_must_pass(env, monkeypatch):
    from mullet.jobs.types import GateResult, GateStatus

    reviews = []

    def reviewer(ctx):
        reviews.append(ctx.node.id)
        verdict = GateStatus.FAIL if len(reviews) == 1 else GateStatus.PASS
        return GateResult("G7", verdict, "reviewer blocked" if verdict is GateStatus.FAIL else "fine")

    monkeypatch.setattr(JobRunner, "_make_reviewer",
                        lambda self, job, profile, executor, force=False: reviewer if force else None)
    risky = NodeSpec(id="n1", goal="add sub() and kill stray workers", kind="mechanical")
    safe = NodeSpec(id="n2", goal="write notes", kind="mechanical")
    notes = lambda w, t: (w / "NOTES.md").write_text("notes\n") and None  # noqa: E731
    # n1: kimi's work fails the safety review, openrouter's passes it; n2: kimi.
    fakes = {"kimi": FakeExecutor([good, notes]), "openrouter": FakeExecutor([good]), "claude": FakeExecutor([])}
    job_id = make_job(env, [risky, safe])
    assert run(env, job_id, fakes, ladder=make_ladder({"kimi": KIMI, "openrouter": [QWEN]}))[0] is JobStatus.DONE
    first, second = env["ledger"].attempts(job_id, "n1")
    g7 = {r.gate: r for r in env["ledger"].gate_results(first.id)}["G7"]
    assert first.outcome is AttemptOutcome.FAILED and g7.status is GateStatus.FAIL and "safety review" in g7.summary
    assert second.outcome is AttemptOutcome.PASSED
    assert reviews == ["n1", "n1"]  # n2 isn't risky and review is off: not reviewed


def test_retry_gives_a_blocked_step_fresh_attempts_on_a_named_rung(env):
    kimi = Resumable("kimi", [bad])
    claude = Resumable("claude", [bad, bad, good])
    job_id = make_job(env, [MECH], max_attempts=2)
    ladder = make_ladder({"kimi": KIMI, "openrouter": []}, level="high")
    status, notes = run(env, job_id, {"kimi": kimi, "claude": claude}, ladder=ladder)
    assert status is JobStatus.BLOCKED and [e for e, _ in notes.sent] == ["blocked"]
    assert rungs_tried(env, job_id) == [("kimi", AttemptOutcome.FAILED), ("claude", AttemptOutcome.FAILED)]
    ledger = env["ledger"]
    last = ledger.attempts(job_id, "n1")[-1].id
    ledger.add_event(job_id, "node_retry", {"node_id": "n1", "rung": "claude", "after_attempt": last})
    status, notes = run(env, job_id, {"kimi": kimi, "claude": claude}, ladder=ladder)
    assert status is JobStatus.DONE and [e for e, _ in notes.sent] == ["finished"]
    # Two fresh tries on Claude (the job's max_attempts), the first a fresh run, never kimi's session.
    assert rungs_tried(env, job_id)[2:] == [("claude", AttemptOutcome.FAILED), ("claude", AttemptOutcome.PASSED)]
    assert claude.runs == 3 and claude.resumed == [] and kimi.runs == 1


def test_retry_on_the_ladder_starts_it_again(env):
    job_id = make_job(env, [MECH], max_attempts=1)
    ladder = make_ladder({"kimi": KIMI, "openrouter": []}, level="high")
    fakes = {"kimi": FakeExecutor([bad, good]), "claude": FakeExecutor([bad])}
    assert run(env, job_id, fakes, ladder=ladder)[0] is JobStatus.BLOCKED
    last = env["ledger"].attempts(job_id, "n1")[-1].id
    env["ledger"].add_event(job_id, "node_retry", {"node_id": "n1", "rung": None, "after_attempt": last})
    assert run(env, job_id, fakes, ladder=ladder)[0] is JobStatus.DONE
    assert rungs_tried(env, job_id)[-1] == ("kimi", AttemptOutcome.PASSED)


def test_over_the_cap_with_claudes_week_tight_the_job_pauses_instead_of_using_claude(env):
    fakes = {"kimi": FakeExecutor([costly_failure(0.6)]), "claude": FakeExecutor([good])}
    job_id = make_job(env, [MECH], max_spend=0.5)
    ladder = make_ladder({"kimi": KIMI, "openrouter": []}, level="high")
    status, notes = run(env, job_id, fakes, ladder=ladder)
    assert status is JobStatus.PAUSED and fakes["claude"].calls == []
    assert [e for e, _ in notes.sent] == ["spend"]
    job = env["ledger"].get_job(job_id)
    assert "spending cap reached ($0.60 of $0.50)" in job.error and "--max-spend" in job.error
    # The owner raises the cap (mullet job resume --max-spend): kimi's model failed, so Claude takes the step.
    env["ledger"].add_event(job_id, "max_spend", {"value": 5.0})
    status, notes = run(env, job_id, fakes, ladder=ladder)
    assert status is JobStatus.DONE and rungs_tried(env, job_id)[-1] == ("claude", AttemptOutcome.PASSED)


class Recorder:
    """A fake opencode-like worker: reports usage after each step, as opencode's stream does."""

    def __init__(self, step, progress=None, raises=None):
        self.name = "opencode"
        self.on_progress = None
        self.price = {}
        self.spend_limit_usd = None
        self.step, self.progress, self.raises = step, progress, raises
        self.budgets, self.limits = [], []

    def run(self, task, work, budget):
        self.budgets.append(budget)
        self.limits.append((self.spend_limit_usd, dict(self.price)))
        if self.progress:
            self.on_progress(self.progress)
        if self.raises:
            raise self.raises
        self.step(work, task)
        return ExecResult(exit_code=0, claim={"status": "done"}, usage={"model": KIMI})


def test_usage_is_recorded_as_it_comes_so_a_broken_attempt_keeps_its_cost(env):
    progress = {"total_cost_usd": 0.42, "cost_reported": True, "tokens": {"input": 9000}, "num_turns": 3}
    kimi = Recorder(good, progress=progress, raises=RuntimeError("the runner died mid-attempt"))
    job_id = make_job(env, [MECH], max_spend=2.0)
    status, _ = run(env, job_id, {"kimi": kimi, "claude": FakeExecutor([])})
    assert status is JobStatus.BLOCKED
    attempt = env["ledger"].attempts(job_id, "n1")[0]
    assert attempt.outcome is AttemptOutcome.ABORTED and attempt.usage["total_cost_usd"] == 0.42
    assert attempt.usage["rung"] == "kimi" and attempt.usage["in_progress"] is True
    from mullet.jobs.ladder import spend_state
    assert spend_state([attempt.usage], 2.0, parse_config({})).spent == pytest.approx(0.42)


def test_each_paid_attempt_gets_the_remaining_spend_and_its_price(env):
    kimi = Recorder(bad)
    qwen = Recorder(good)
    config = {"kimi": KIMI, "openrouter": [QWEN], "prices": {QWEN: {"input": 0.2, "output": 0.8}}}
    job_id = make_job(env, [MECH], max_spend=1.5)
    assert run(env, job_id, {"kimi": kimi, "openrouter": qwen, "claude": FakeExecutor([])},
               ladder=make_ladder(config))[0] is JobStatus.DONE
    assert kimi.limits[0][0] == pytest.approx(1.5)
    assert qwen.limits[0] == (pytest.approx(1.5), {"input": 0.2, "output": 0.8})


def test_free_and_local_attempts_get_the_shorter_time(env):
    local = Recorder(good)
    config = {"kimi": KIMI, "openrouter": [], "local": "lmstudio/qwen3.6-35b-a3b", "cheap_minutes": 0.25}
    job_id = make_job(env, [MECH])
    assert run(env, job_id, {"local": local, "claude": FakeExecutor([])},
               ladder=make_ladder(config, machine_ok=True))[0] is JobStatus.DONE
    assert local.budgets[0].wall_clock_s == 15 and local.limits[0][0] is None  # local isn't paid


def test_claudes_rate_limits_from_its_own_stream_keep_the_budget_fresh(env, monkeypatch, tmp_path):
    from mullet.hud import usage

    monkeypatch.setattr(usage, "_CACHE_FILE", tmp_path / "usage.json")
    info = {"status": "allowed", "unifiedWindows": {"seven_day": {"utilization": 0.3, "resetsAt": time.time() + 9e4}}}

    def claude_step(work, task):
        good_change(work)
        return ExecResult(exit_code=0, claim={"status": "done"}, usage={"rate_limit": info})

    job_id = make_job(env, [NodeSpec(id="n1", goal="add sub()", kind="analysis", expects_change=True)])
    assert run(env, job_id, {"claude": FakeExecutor([claude_step]), "kimi": FakeExecutor([])})[0] is JobStatus.DONE
    assert usage.fetch_usage().weekly_pct == pytest.approx(30.0)


def test_only_trusted_check_commands_may_run_as_written(env):
    from mullet.jobs.runner import profile_exact_commands

    profile = Profile(repo_id="r", checks=Checks(lint="ruff . ; mypy .", test_full="pytest -q"))
    job_id = make_job(env, [MECH])
    job = env["ledger"].get_job(job_id)
    assert profile_exact_commands(profile, job) == []  # nothing recorded as trusted
    trusted = job.__class__(**{**job.__dict__, "budget": {**job.budget, "trusted_commands": ["ruff . ; mypy ."]}})
    assert profile_exact_commands(profile, trusted) == ["ruff . ; mypy ."]


def test_a_failing_notifier_never_changes_the_outcome(env):
    def broken(*args, **kwargs):
        raise RuntimeError("no dbus")

    job_id = make_job(env, [MECH])
    status, _ = run(env, job_id, {"kimi": FakeExecutor([good]), "claude": FakeExecutor([])}, notify=broken)
    assert status is JobStatus.DONE


def test_night_mode_doubles_a_local_attempts_wall_clock(env):
    local = Recorder(good)
    config = {"kimi": KIMI, "openrouter": [], "local": "lmstudio/qwen3.6-35b-a3b", "cheap_minutes": 0.25}
    job_id = make_job(env, [MECH])
    status, _ = run(env, job_id, {"local": local, "claude": FakeExecutor([])},
                    ladder=make_ladder(config, machine_ok=True,
                                       night=lambda: NightState(on=True, until=time.time() + 3600)))
    assert status is JobStatus.DONE
    assert local.budgets[0].wall_clock_s == 30  # 0.5 min * 60, twice the day value
    night_events = [e.data for e in env["ledger"].events(job_id) if e.kind == "night"]
    assert len(night_events) == 1 and night_events[0]["on"] is True


def test_night_events_are_recorded_once_per_change(env):
    states = [
        NightState(on=True, until=time.time() + 3600),
        NightState(on=True, until=time.time() + 3600),
        NightState(on=False, reason="you're back"),
    ]
    calls = 0

    def night():
        nonlocal calls
        state = states[min(calls, len(states) - 1)]
        calls += 1
        return state

    nodes = [NodeSpec(id=f"n{i}", goal=f"task {i}", kind="mechanical") for i in range(3)]
    config = {"kimi": KIMI, "openrouter": [], "local": "lmstudio/qwen3.6-35b-a3b"}
    job_id = make_job(env, nodes)
    status, _ = run(env, job_id, {"local": Recorder(good), "claude": FakeExecutor([])},
                    ladder=make_ladder(config, machine_ok=True, night=night))
    assert status is JobStatus.DONE
    assert calls == 3
    night_events = [e.data for e in env["ledger"].events(job_id) if e.kind == "night"]
    assert [e["on"] for e in night_events] == [True, False]
    assert night_events[-1].get("reason") == "you're back"


def test_without_night_mode_there_is_no_night_event_and_the_day_order_holds(env):
    design = NodeSpec(id="n1", goal="add a helper", kind="design", scope=["app.py", "tests/**"],
                      acceptance=["python3 runtests.py"])
    job_id = make_job(env, [design])
    status, _ = run(env, job_id, {"kimi": FakeExecutor([good]), "claude": FakeExecutor([])},
                    ladder=make_ladder({"kimi": KIMI, "openrouter": []}, level="medium"))
    assert status is JobStatus.DONE
    assert rungs_tried(env, job_id) == [("kimi", AttemptOutcome.PASSED)]
    assert not any(e.kind == "night" for e in env["ledger"].events(job_id))
