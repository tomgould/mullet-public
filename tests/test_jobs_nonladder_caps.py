"""Non-ladder opencode jobs: the spend allowance, the shared limits and the free-model public check."""

import pytest

from mullet.jobs import sandbox
from mullet.jobs.clone import resolve_commit
from mullet.jobs.executors.base import ExecResult
from mullet.jobs.executors.fake import FakeExecutor
from mullet.jobs.ladder import Ladder, LadderConfig, PublicCheck, Rung, config_path, model_rung, parse_config
from mullet.jobs.ledger import Ledger
from mullet.jobs.profile import Profile
from mullet.jobs.runner import ExecutorNotAllowed, JobRunner, solo_rung
from mullet.jobs.types import JobStatus, NodeSpec

from tests.jobs_helpers import good_change, make_source_repo
from tests.test_jobs_budget import NOW, _job_with_attempt, usage
from tests.test_jobs_runner import CHECKS, break_add

needs_sandbox = pytest.mark.skipif(not sandbox.check().ok, reason=sandbox.check().reason)

PAID = "moonshotai/kimi-k2.7-code"
FREE = "openrouter/qwen/qwen3-coder:free"
BOTH = ["claude-code", "opencode"]


class ArmedFake(FakeExecutor):
    """A scripted opencode worker that records the allowance the runner gave each attempt."""

    name = "opencode"

    def __init__(self, steps):
        super().__init__(steps)
        self.on_progress = None
        self.spend_limit_usd = None
        self.price: dict = {}
        self.price_configured = False
        self.limits: list = []

    def run(self, task, work, budget):
        self.limits.append(self.spend_limit_usd)
        return super().run(task, work, budget)


def bare_worker():
    return type("Worker", (), {"on_progress": None})()


# model_rung and solo_rung


@pytest.mark.parametrize("model, rung, paid", [
    (FREE, "free", False),
    ("lmstudio/qwen3.6-35b-a3b", "local", False),
    ("ollama/qwen2.5-coder:7b", "local", False),
    ("openrouter/z-ai/glm-5.3", "openrouter", True),
    (PAID, "kimi", True),
])
def test_model_rung_tells_the_rung_and_whether_it_is_paid(model, rung, paid):
    found = model_rung(model)
    assert (found.name, found.model, found.paid) == (rung, model, paid)


# The allowance, unit level


@pytest.fixture
def solo_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path / "cfg"))
    (tmp_path / "cfg").mkdir()
    ledger = Ledger(tmp_path / "jobs" / "ledger.db")

    def make(model=PAID, max_spend=2.0, data_class="private"):
        profile = Profile(repo_id="fixture", allowed_executors=BOTH, data_class=data_class)
        job_id = ledger.create_job("/repo", "goal", "main", "base000", profile.model_dump(), max_attempts=3,
                                   budget={"ladder": False, "executor": "opencode", "executor_model": model,
                                           "max_spend": max_spend})
        ledger.set_plan(job_id, [(NodeSpec(id="n1", goal="do n1"), None)])
        ledger.set_job_status(job_id, JobStatus.AWAITING_APPROVAL)
        ledger.approve(job_id)
        ledger.set_job_status(job_id, JobStatus.RUNNING)
        return job_id

    def add_attempt(job_id, cost, rung="kimi"):
        attempt_id = ledger.start_attempt(job_id, "n1", "opencode", "base000")
        ledger.record_progress(attempt_id, usage(rung, cost))
        ledger._conn.execute("UPDATE attempts SET started_at = ? WHERE id = ?", (NOW - 3600, attempt_id))

    def write_config(text):
        path = config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def runner(job_id, public=None):
        ladder = Ladder(config=LadderConfig(), now=lambda: NOW)
        if public is not None:
            ladder.public_check = lambda repo, sha, profile: public
        built = JobRunner(ledger, job_id, ladder=ladder, now=lambda: NOW)
        job = ledger.get_job(job_id)
        built._job, built._profile = job, Profile.model_validate(job.profile)
        return built

    yield {"make": make, "add_attempt": add_attempt, "write_config": write_config, "runner": runner,
           "ledger": ledger}
    ledger.close()


def test_a_paid_model_gets_its_max_spend_minus_what_the_job_spent(solo_env):
    job_id = solo_env["make"](max_spend=2.0)
    solo_env["add_attempt"](job_id, 0.5)
    runner = solo_env["runner"](job_id)
    worker = bare_worker()
    rung = solo_rung(solo_env["ledger"].get_job(job_id))
    assert rung == Rung("kimi", PAID)
    runner._arm(worker, rung, 1)
    assert worker.spend_limit_usd == pytest.approx(1.5)


def test_the_envelope_lowers_a_paid_models_allowance(solo_env):
    solo_env["write_config"]("budget:\n  envelope:\n    daily: 1.0\n")
    job_id = solo_env["make"](max_spend=2.0)
    _job_with_attempt(solo_env["ledger"], "other", usage(cost=0.75), NOW - 60)
    runner = solo_env["runner"](job_id)
    worker = bare_worker()
    runner._arm(worker, solo_rung(solo_env["ledger"].get_job(job_id)), 1)
    assert worker.spend_limit_usd == pytest.approx(0.25)


def test_a_per_rung_limit_lowers_a_paid_models_allowance(solo_env):
    solo_env["write_config"]("budget:\n  per_rung:\n    openrouter:\n      daily: 0.3\n")
    job_id = solo_env["make"](model="openrouter/z-ai/glm-5.3", max_spend=2.0)
    runner = solo_env["runner"](job_id)
    worker = bare_worker()
    runner._arm(worker, solo_rung(solo_env["ledger"].get_job(job_id)), 1)
    assert worker.spend_limit_usd == pytest.approx(0.3)


def test_the_price_comes_from_the_ladder_config(solo_env):
    job_id = solo_env["make"](model=PAID)
    runner = solo_env["runner"](job_id)
    runner._ladder = Ladder(config=parse_config({"prices": {PAID: {"input": 0.6, "output": 2.5}}}),
                            now=lambda: NOW)
    worker = bare_worker()
    runner._arm(worker, solo_rung(solo_env["ledger"].get_job(job_id)), 1)
    assert worker.price == {"input": 0.6, "output": 2.5} and worker.price_configured is True


@pytest.mark.parametrize("model", ["lmstudio/qwen3.6-35b-a3b", "ollama/qwen2.5-coder:7b", FREE])
def test_a_local_or_free_model_has_no_allowance(solo_env, model):
    job_id = solo_env["make"](model=model)
    runner = solo_env["runner"](job_id)
    worker = bare_worker()
    runner._arm(worker, solo_rung(solo_env["ledger"].get_job(job_id)), 1)
    assert worker.spend_limit_usd is None


def test_a_free_model_on_a_private_repo_is_refused(solo_env):
    job_id = solo_env["make"](model=FREE)
    runner = solo_env["runner"](job_id)
    with pytest.raises(ExecutorNotAllowed, match="free model"):
        runner._check_free_model(solo_rung(solo_env["ledger"].get_job(job_id)))


def test_a_free_model_on_a_public_repo_is_allowed(solo_env):
    job_id = solo_env["make"](model=FREE, data_class="public")
    runner = solo_env["runner"](job_id, public=PublicCheck(True, "public on GitHub", checked_at=NOW))
    runner._check_free_model(solo_rung(solo_env["ledger"].get_job(job_id)))
    assert any(e.kind == "public_check" for e in solo_env["ledger"].events(job_id))


def test_a_ladder_or_claude_job_has_no_solo_rung(solo_env):
    ledger = solo_env["ledger"]
    for budget in ({"ladder": True, "executor": "opencode", "executor_model": PAID}, {"executor": "claude-code"}):
        job_id = ledger.create_job("/repo", "goal", "main", "base000", Profile(repo_id="x").model_dump(),
                                   budget=budget)
        assert solo_rung(ledger.get_job(job_id)) is None


# Whole jobs


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path / "cfg"))
    source = make_source_repo(tmp_path / "src")
    ledger = Ledger(tmp_path / "jobs" / "ledger.db")
    yield {"tmp": tmp_path, "source": source, "ledger": ledger}
    ledger.close()


def make_job(env, model, max_spend=2.0, data_class="private") -> str:
    source, ledger = env["source"], env["ledger"]
    profile = Profile(repo_id="fixture", checks=CHECKS, env={"PYTHONDONTWRITEBYTECODE": "1"},
                      allowed_executors=BOTH, data_class=data_class)
    job_id = ledger.create_job(str(source), "improve the app", "main", resolve_commit(source, "HEAD"),
                               profile.model_dump(), max_attempts=3,
                               budget={"max_turns": 5, "wall_clock_s": 60, "review": False, "ladder": False,
                                       "executor": "opencode", "executor_model": model, "max_spend": max_spend})
    ledger.set_plan(job_id, [(NodeSpec(id="n1", goal="add sub()"), None)])
    ledger.set_job_status(job_id, JobStatus.AWAITING_APPROVAL)
    ledger.approve(job_id)
    return job_id


def costly_break(cost):
    def step(work, task):
        break_add(work)
        return ExecResult(exit_code=0, claim={"status": "done"},
                          usage={"total_cost_usd": cost, "cost_reported": True, "model": PAID})
    return step


@needs_sandbox
def test_a_paid_job_counts_its_spend_and_the_next_attempt_gets_the_rest(env):
    job_id = make_job(env, PAID, max_spend=2.0)
    worker = ArmedFake([costly_break(0.5), lambda w, t: good_change(w)])
    ladder = Ladder(config=LadderConfig())
    assert JobRunner(env["ledger"], job_id, worker, ladder=ladder).run() is JobStatus.DONE
    assert worker.limits == [pytest.approx(2.0), pytest.approx(1.5)]
    first = env["ledger"].attempts(job_id, "n1")[0].usage
    assert (first["rung"], first["model"]) == ("kimi", PAID)


@needs_sandbox
def test_a_paid_job_at_its_cap_pauses_instead_of_running(env):
    job_id = make_job(env, PAID, max_spend=0.4)
    worker = ArmedFake([costly_break(0.5), lambda w, t: good_change(w)])
    assert JobRunner(env["ledger"], job_id, worker, ladder=Ladder(config=LadderConfig())).run() is JobStatus.PAUSED
    job = env["ledger"].get_job(job_id)
    assert "spending cap is reached" in job.error and "--max-spend" in job.error
    assert "Claude" not in job.error
    assert len(env["ledger"].attempts(job_id, "n1")) == 1


def test_a_free_model_job_on_a_private_repo_blocks_before_any_attempt(env):
    job_id = make_job(env, FREE)
    worker = ArmedFake([lambda w, t: good_change(w)])
    assert JobRunner(env["ledger"], job_id, worker, ladder=Ladder(config=LadderConfig())).run() is JobStatus.BLOCKED
    assert "free model" in env["ledger"].get_job(job_id).error
    assert env["ledger"].attempts(job_id) == [] and worker.limits == []


@needs_sandbox
def test_a_free_model_job_on_a_public_repo_runs_without_an_allowance(env):
    job_id = make_job(env, FREE, data_class="public")
    ladder = Ladder(config=LadderConfig(), public_check=lambda repo, sha, profile: PublicCheck(True, "public"))
    worker = ArmedFake([lambda w, t: good_change(w)])
    assert JobRunner(env["ledger"], job_id, worker, ladder=ladder).run() is JobStatus.DONE
    assert worker.limits == [None]
    assert env["ledger"].attempts(job_id, "n1")[0].usage["rung"] == "free"
