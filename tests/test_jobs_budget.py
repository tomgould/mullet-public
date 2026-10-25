"""Tests for mullet.jobs.budget and how the runner, ladder, ledger and config validation use it."""

import datetime as dt

import pytest

from mullet.config.validate import validate_config
from mullet.jobs.budget import (
    BudgetConfig, Limits, earliest_start, headroom, load_budget, parse_budget, window_start,
)
from mullet.jobs.ladder import (
    Ladder, LadderConfig, Rung, SpendState, attempt_cost, config_path, parse_config, plan_rungs,
)
from mullet.jobs.ledger import Ledger
from mullet.jobs.profile import Profile
from mullet.jobs.runner import JobRunner
from mullet.jobs.types import JobStatus, NodeSpec

from tests.test_jobs_ladder import BOTH, FULL, LOCAL_OK, MECH, MEDIUM, PUBLIC, names

KIMI_ID = "moonshotai/kimi-k2.7-code"


def utc(year, month, day, hour=0, minute=0):
    return dt.datetime(year, month, day, hour, minute, tzinfo=dt.timezone.utc).timestamp()


# 2026-10-21 is a Wednesday; its Monday is the 19th and the previous Sunday the 18th.
WEDNESDAY_NOON = utc(2026, 10, 21, 12)
DAY_START = utc(2026, 10, 21)
WEEK_START = utc(2026, 10, 19)


def usage(rung="kimi", cost=2.0, model=KIMI_ID):
    return {"rung": rung, "total_cost_usd": cost, "cost_reported": True, "model": model}


def config(envelope=None, per_rung=None):
    return BudgetConfig(envelope=Limits(**(envelope or {})),
                        per_rung={k: Limits(**v) for k, v in (per_rung or {}).items()})


# parse_budget


def test_parse_a_valid_envelope_and_per_rung_limits():
    parsed = parse_budget({"envelope": {"daily": 25, "weekly": 100.5},
                           "per_rung": {"kimi": {"daily": 15}, "openrouter": {"weekly": 40}}})
    assert parsed.problems == []
    assert parsed.envelope == Limits(daily=25.0, weekly=100.5)
    assert parsed.per_rung == {"kimi": Limits(daily=15.0), "openrouter": Limits(weekly=40.0)}
    assert parsed.active


def test_the_legacy_keys_are_ignored_without_problems():
    parsed = parse_budget({"per_task": 1.0, "per_session": 5.0, "daily": 10.0})
    assert parsed.problems == []
    assert parsed.envelope == Limits()
    assert not parsed.active


@pytest.mark.parametrize("bad", [-1, True, "5", [3]])
def test_a_bad_dollar_value_is_a_problem_and_ignored(bad):
    parsed = parse_budget({"envelope": {"daily": bad, "weekly": 10}})
    assert len(parsed.problems) == 1 and "envelope.daily" in parsed.problems[0]
    assert parsed.envelope == Limits(daily=None, weekly=10.0)


def test_an_unknown_window_is_a_problem_and_ignored():
    parsed = parse_budget({"envelope": {"monthly": 5, "daily": 1}})
    assert len(parsed.problems) == 1 and "monthly" in parsed.problems[0]
    assert parsed.envelope == Limits(daily=1.0)


def test_a_non_mapping_limits_block_is_a_problem_and_ignored():
    parsed = parse_budget({"envelope": 5})
    assert len(parsed.problems) == 1 and "envelope" in parsed.problems[0]
    assert not parsed.active
    parsed = parse_budget({"per_rung": ["kimi"]})
    assert len(parsed.problems) == 1 and "per_rung" in parsed.problems[0]
    assert not parsed.active


def test_a_per_rung_name_that_is_not_a_paid_rung_is_a_problem_and_ignored():
    parsed = parse_budget({"per_rung": {"claude": {"daily": 5}, "kimi": {"daily": 2}}})
    assert len(parsed.problems) == 1 and "claude" in parsed.problems[0]
    assert list(parsed.per_rung) == ["kimi"]


def test_problems_name_the_source():
    parsed = parse_budget({"envelope": {"daily": -1}}, source="config.yaml budget")
    assert parsed.problems[0].startswith("config.yaml budget.envelope.daily")


@pytest.mark.parametrize("empty", [None, {}, "text", 5])
def test_no_budget_means_no_limits(empty):
    parsed = parse_budget(empty)
    assert parsed.problems == [] and not parsed.active
    assert parsed.envelope == Limits() and parsed.per_rung == {}


def test_a_zero_limit_is_a_limit():
    parsed = parse_budget({"envelope": {"daily": 0}})
    assert parsed.envelope.daily == 0.0 and parsed.active


# load_budget


def test_a_missing_file_has_no_limits(tmp_path):
    loaded = load_budget(tmp_path / "missing.yaml")
    assert not loaded.active and loaded.problems == []


def test_invalid_yaml_has_no_limits_and_a_problem(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("budget: [unclosed\n  envelope: {")
    loaded = load_budget(path)
    assert not loaded.active
    assert len(loaded.problems) == 1 and "unreadable" in loaded.problems[0]


def test_a_budget_block_is_read_from_the_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("ladder:\n  max_spend: 3\nbudget:\n  envelope:\n    daily: 4\n  per_rung:\n    kimi:\n      weekly: 9\n")
    loaded = load_budget(path)
    assert loaded.problems == []
    assert loaded.envelope == Limits(daily=4.0)
    assert loaded.per_rung == {"kimi": Limits(weekly=9.0)}


def test_a_file_without_a_budget_block_has_no_limits(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("ladder:\n  max_spend: 3\n")
    assert not load_budget(path).active


def test_a_file_that_is_not_a_mapping_has_no_limits(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- just\n- a list\n")
    loaded = load_budget(path)
    assert not loaded.active and loaded.problems == []


def test_load_budget_defaults_to_the_users_config_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yaml").write_text("budget:\n  envelope:\n    weekly: 7\n")
    assert load_budget().envelope == Limits(weekly=7.0)


# window_start and earliest_start


def test_the_daily_window_starts_at_midnight_utc():
    assert window_start("daily", WEDNESDAY_NOON) == DAY_START
    assert window_start("daily", DAY_START) == DAY_START
    assert window_start("daily", DAY_START + 86399) == DAY_START


def test_the_weekly_window_starts_on_monday_midnight_utc():
    assert window_start("weekly", WEDNESDAY_NOON) == WEEK_START
    assert window_start("weekly", utc(2026, 10, 25, 23, 59)) == WEEK_START  # Sunday
    assert window_start("weekly", utc(2026, 10, 26, 0, 1)) == utc(2026, 10, 26)  # Monday
    assert window_start("weekly", WEEK_START) == WEEK_START


def test_earliest_start_is_none_when_nothing_is_set():
    assert earliest_start(BudgetConfig(), WEDNESDAY_NOON) is None


def test_earliest_start_is_the_daily_start_when_only_daily_is_set():
    assert earliest_start(config({"daily": 1}), WEDNESDAY_NOON) == DAY_START


def test_earliest_start_is_the_weekly_start_when_only_weekly_is_set():
    assert earliest_start(config({"weekly": 1}), WEDNESDAY_NOON) == WEEK_START
    assert earliest_start(config(per_rung={"kimi": {"weekly": 1}}), WEDNESDAY_NOON) == WEEK_START


def test_earliest_start_is_the_earlier_of_both():
    assert earliest_start(config({"daily": 1, "weekly": 5}), WEDNESDAY_NOON) == WEEK_START
    assert earliest_start(config({"daily": 1}, {"kimi": {"weekly": 5}}), WEDNESDAY_NOON) == WEEK_START


# headroom


LADDER = LadderConfig()


def room(cfg, records, now=WEDNESDAY_NOON):
    return headroom(cfg, records, LADDER, now)


def test_no_spend_leaves_the_whole_envelope():
    found = room(config({"daily": 5}), [])
    assert found.left == 5.0 and "daily" in found.reason


def test_unlimited_when_no_envelope_is_set():
    found = room(config(per_rung={"kimi": {"daily": 1}}), [(DAY_START + 60, usage())])
    assert found.left is None and found.reason == ""


def test_spend_inside_the_window_is_subtracted():
    found = room(config({"daily": 5}), [(DAY_START + 60, usage(cost=2.0)), (DAY_START + 120, usage("openrouter", 0.5))])
    assert found.left == pytest.approx(2.5)


def test_spend_before_the_daily_window_does_not_count_daily_but_counts_weekly():
    yesterday = DAY_START - 3600
    assert room(config({"daily": 5}), [(yesterday, usage())]).left == 5.0
    found = room(config({"weekly": 5}), [(yesterday, usage())])
    assert found.left == pytest.approx(3.0)


def test_spend_before_the_weekly_window_counts_nowhere():
    last_week = WEEK_START - 3600
    assert room(config({"daily": 5, "weekly": 5}), [(last_week, usage())]).left == 5.0


def test_an_attempt_starting_exactly_at_the_window_start_counts():
    assert room(config({"daily": 5}), [(DAY_START, usage())]).left == pytest.approx(3.0)


def test_an_attempt_that_started_before_the_window_is_excluded():
    assert room(config({"daily": 5}), [(DAY_START - 1, usage(cost=4.0))]).left == 5.0


def test_the_tightest_window_names_itself_in_the_reason():
    records = [(DAY_START + 60, usage(cost=1.0)), (WEEK_START + 60, usage(cost=3.0))]
    found = room(config({"daily": 2, "weekly": 4}), records)
    assert found.left == pytest.approx(0.0)  # weekly: 4 - 4; daily: 2 - 1 = 1
    assert "weekly" in found.reason and "budget.envelope.weekly" in found.reason
    assert "$4.00 of $4.00" in found.reason
    found = room(config({"daily": 1.5, "weekly": 40}), records)
    assert found.left == pytest.approx(0.5)
    assert "daily" in found.reason and "budget.envelope.daily" in found.reason


def test_an_exceeded_envelope_has_negative_room():
    found = room(config({"daily": 1}), [(DAY_START + 60, usage(cost=2.0))])
    assert found.left == pytest.approx(-1.0)
    assert "$2.00 of $1.00" in found.reason


def test_per_rung_room_counts_only_that_rungs_spend():
    cfg = config(per_rung={"kimi": {"daily": 3}, "openrouter": {"daily": 1}})
    records = [(DAY_START + 60, usage("kimi", 2.0)), (DAY_START + 90, usage("openrouter", 0.25, "openrouter/x/y"))]
    found = room(cfg, records)
    assert found.left is None
    assert found.rung_left["kimi"] == pytest.approx(1.0)
    assert found.rung_left["openrouter"] == pytest.approx(0.75)
    assert "kimi daily limit is reached" in found.rung_reasons["kimi"]
    assert "budget.per_rung.kimi.daily" in found.rung_reasons["kimi"]


def test_per_rung_keeps_the_tightest_window():
    cfg = config(per_rung={"kimi": {"daily": 10, "weekly": 4}})
    found = room(cfg, [(DAY_START + 60, usage(cost=3.0))])
    assert found.rung_left["kimi"] == pytest.approx(1.0)
    assert "weekly" in found.rung_reasons["kimi"]


def test_a_rung_with_no_spend_has_its_full_limit():
    found = room(config(per_rung={"kimi": {"daily": 3}}), [(DAY_START + 60, usage("openrouter", 1.0, "openrouter/x/y"))])
    assert found.rung_left == {"kimi": 3.0}


def test_per_rung_spend_still_counts_toward_the_envelope():
    cfg = config({"daily": 5}, {"kimi": {"daily": 1}})
    found = room(cfg, [(DAY_START + 60, usage("kimi", 2.0))])
    assert found.left == pytest.approx(3.0) and found.rung_left["kimi"] == pytest.approx(-1.0)


@pytest.mark.parametrize("rung", ["claude", "local", "free"])
def test_unpaid_rungs_cost_nothing(rung):
    found = room(config({"daily": 5}), [(DAY_START + 60, usage(rung, 9.0, "whatever"))])
    assert found.left == 5.0


def test_missing_usage_is_skipped():
    found = room(config({"daily": 5}), [(DAY_START + 60, None), (DAY_START + 61, {}), (DAY_START + 62, usage())])
    assert found.left == pytest.approx(3.0)


def test_an_unreported_cost_is_priced_from_tokens():
    tokens = {"rung": "kimi", "model": KIMI_ID, "input_tokens": 1_000_000, "output_tokens": 0}
    priced, _ = attempt_cost(tokens, LADDER)
    found = room(config({"daily": 100}), [(DAY_START + 60, tokens)])
    assert found.left == pytest.approx(100 - priced)


# plan_rungs with a SpendState carrying budget limits


def choose(spend, **kw):
    return plan_rungs(MECH, BOTH, MEDIUM, LOCAL_OK, spend, config=FULL, public=PUBLIC, has_key=lambda r: True, **kw)


def test_an_envelope_cap_puts_its_reason_in_the_skipped_paid_rungs():
    reason = "the daily spending envelope is reached ($1.00 of $1.00, budget.envelope.daily)"
    choice = choose(SpendState(cap=1.0, spent=1.0, reason=reason))
    assert choice.capped
    paid = [why for label, why in choice.skipped if label.split()[0] in ("kimi", "openrouter")]
    assert paid and all(why == reason for why in paid)
    assert [r.name for r in choice.rungs] == ["claude", "local", "free"][:len(choice.rungs)] or all(
        not r.paid for r in choice.rungs)


def test_a_job_cap_without_a_reason_keeps_the_job_cap_message():
    choice = choose(SpendState(cap=1.0, spent=1.0))
    assert choice.capped
    assert any("the job's spending cap is reached" in why for _, why in choice.skipped)


def test_a_rung_with_no_room_is_skipped_but_other_paid_rungs_stay():
    reason = "the kimi daily limit is reached ($3.00 of $3.00, budget.per_rung.kimi.daily)"
    choice = choose(SpendState(cap=5.0, spent=0.0, rung_left={"kimi": 0.0}, rung_reasons={"kimi": reason}))
    labels = names(choice.rungs)
    assert "kimi" not in labels
    assert any(label.startswith("openrouter") for label in labels)
    assert any(why == reason for label, why in choice.skipped if label.startswith("kimi"))
    assert choice.capped is False  # another paid rung can still take the step, so the job is not paused


def test_a_negative_rung_room_also_skips_the_rung():
    choice = choose(SpendState(cap=5.0, rung_left={"kimi": -0.5}))
    assert "kimi" not in names(choice.rungs)
    assert any("kimi" in label and "spending limit is reached" in why for label, why in choice.skipped)


def test_a_rung_with_room_left_is_kept():
    choice = choose(SpendState(cap=5.0, rung_left={"kimi": 0.01}))
    assert "kimi" in names(choice.rungs)


def test_capped_is_true_when_rung_limits_block_every_paid_rung():
    choice = choose(SpendState(cap=5.0, rung_left={"kimi": 0.0, "openrouter": 0.0}))
    assert choice.capped
    assert all(not r.paid for r in choice.rungs)


# Ledger.attempt_usages_since


def _job_with_attempt(ledger, node, run_usage, started_at):
    job_id = ledger.create_job("/repo", "goal", "main", "base000", {}, max_attempts=3)
    ledger.set_plan(job_id, [(NodeSpec(id=node, goal=f"do {node}"), None)])
    ledger.set_job_status(job_id, JobStatus.AWAITING_APPROVAL)
    ledger.approve(job_id)
    ledger.set_job_status(job_id, JobStatus.RUNNING)
    attempt_id = ledger.start_attempt(job_id, node, "opencode", "base000")
    if run_usage is not None:
        ledger.record_progress(attempt_id, run_usage)
    ledger._conn.execute("UPDATE attempts SET started_at = ? WHERE id = ?", (started_at, attempt_id))
    return job_id, attempt_id


def test_attempt_usages_since_returns_recent_attempts_with_usage_across_jobs(tmp_path):
    with Ledger(tmp_path / "ledger.db") as ledger:
        _job_with_attempt(ledger, "a", usage(cost=1.0), 1000.0)
        _job_with_attempt(ledger, "b", usage(cost=2.0), 2000.0)
        _job_with_attempt(ledger, "c", usage(cost=3.0), 3000.0)
        _job_with_attempt(ledger, "d", None, 4000.0)
        found = ledger.attempt_usages_since(2000.0)
        assert sorted((started, u["total_cost_usd"]) for started, u in found) == [(2000.0, 2.0), (3000.0, 3.0)]
        assert ledger.attempt_usages_since(5000.0) == []
        assert len(ledger.attempt_usages_since(0.0)) == 3


# JobRunner._spend with the budget block


NOW = WEDNESDAY_NOON


@pytest.fixture
def spend_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path / "cfg"))
    (tmp_path / "cfg").mkdir()
    ledger = Ledger(tmp_path / "jobs" / "ledger.db")
    profile = Profile(repo_id="fixture")
    job_id = ledger.create_job("/repo", "goal", "main", "base000", profile.model_dump(), max_attempts=3,
                               budget={"ladder": True, "max_spend": 2.0})
    ledger.set_plan(job_id, [(NodeSpec(id="n1", goal="do n1", kind="mechanical"), None)])
    ledger.set_job_status(job_id, JobStatus.AWAITING_APPROVAL)
    ledger.approve(job_id)
    ledger.set_job_status(job_id, JobStatus.RUNNING)

    def add_attempt(cost, started_at=NOW - 3600, rung="kimi"):
        attempt_id = ledger.start_attempt(job_id, "n1", "opencode", "base000")
        ledger.record_progress(attempt_id, usage(rung, cost))
        ledger._conn.execute("UPDATE attempts SET started_at = ? WHERE id = ?", (started_at, attempt_id))

    def write_config(text):
        path = config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def runner():
        built = JobRunner(ledger, job_id, ladder=Ladder(config=parse_config({"kimi": KIMI_ID}), now=lambda: NOW),
                          now=lambda: NOW)
        built._job = ledger.get_job(job_id)
        return built

    yield {"add_attempt": add_attempt, "write_config": write_config, "runner": runner, "ledger": ledger}
    ledger.close()


def test_spend_is_unchanged_without_a_budget_block(spend_env):
    spend_env["add_attempt"](0.9)
    state = spend_env["runner"]()._spend()
    assert state.cap == 2.0 and state.reason == ""
    assert state.spent == pytest.approx(0.9)
    assert state.rung_left == {} and state.rung_reasons == {}


def test_the_envelope_lowers_the_cap_to_the_room_left(spend_env):
    spend_env["write_config"]("budget:\n  envelope:\n    daily: 1.0\n")
    spend_env["add_attempt"](0.9)
    state = spend_env["runner"]()._spend()
    assert state.spent == pytest.approx(0.9)
    assert state.cap == pytest.approx(1.0)
    assert state.cap - state.spent == pytest.approx(0.10)
    assert "daily spending envelope" in state.reason
    assert not state.capped


def test_an_exceeded_envelope_caps_the_job_at_what_it_spent(spend_env):
    spend_env["write_config"]("budget:\n  envelope:\n    daily: 1.0\n")
    spend_env["add_attempt"](2.0)
    spend_env["ledger"]._conn.execute("UPDATE jobs SET budget_json = ?", ('{"ladder": true, "max_spend": 5.0}',))
    state = spend_env["runner"]()._spend()
    assert state.cap == pytest.approx(state.spent) == pytest.approx(2.0)
    assert state.capped
    assert "daily spending envelope" in state.reason


def test_spend_by_other_jobs_counts_against_the_envelope(spend_env):
    spend_env["write_config"]("budget:\n  envelope:\n    daily: 1.0\n")
    _job_with_attempt(spend_env["ledger"], "x", usage(cost=0.75), NOW - 60)
    state = spend_env["runner"]()._spend()
    assert state.spent == 0.0
    assert state.cap == pytest.approx(0.25)
    assert "daily spending envelope" in state.reason


def test_an_envelope_larger_than_the_job_cap_leaves_the_cap_alone(spend_env):
    spend_env["write_config"]("budget:\n  envelope:\n    daily: 50\n")
    spend_env["add_attempt"](0.5)
    state = spend_env["runner"]()._spend()
    assert state.cap == 2.0 and state.reason == ""


def test_spend_before_the_window_does_not_count(spend_env):
    spend_env["write_config"]("budget:\n  envelope:\n    daily: 1.0\n")
    spend_env["add_attempt"](5.0, started_at=DAY_START - 10)
    state = spend_env["runner"]()._spend()
    assert state.cap == 2.0 and state.reason == ""


def test_per_rung_limits_are_carried_on_the_spend_state(spend_env):
    spend_env["write_config"]("budget:\n  per_rung:\n    kimi:\n      daily: 1.0\n")
    spend_env["add_attempt"](0.6)
    state = spend_env["runner"]()._spend()
    assert state.cap == 2.0
    assert state.rung_left["kimi"] == pytest.approx(0.4)
    assert "kimi daily limit" in state.rung_reasons["kimi"]


def test_an_invalid_budget_block_is_ignored_by_spend(spend_env):
    spend_env["write_config"]("budget:\n  envelope:\n    daily: -5\n")
    state = spend_env["runner"]()._spend()
    assert state.cap == 2.0 and state.reason == ""


def test_arm_gives_the_worker_the_smaller_of_job_and_rung_room(spend_env):
    spend_env["write_config"]("budget:\n  per_rung:\n    kimi:\n      daily: 0.5\n")
    runner = spend_env["runner"]()
    worker = type("Worker", (), {"on_progress": None})()
    runner._arm(worker, Rung("kimi", KIMI_ID), 1)
    assert worker.spend_limit_usd == pytest.approx(0.5)


def test_arm_uses_the_envelope_room_for_a_paid_rung(spend_env):
    spend_env["write_config"]("budget:\n  envelope:\n    daily: 1.0\n")
    spend_env["add_attempt"](0.9)
    runner = spend_env["runner"]()
    worker = type("Worker", (), {"on_progress": None})()
    runner._arm(worker, Rung("kimi", KIMI_ID), 1)
    assert worker.spend_limit_usd == pytest.approx(0.10)


# Config validation


def test_validate_rejects_a_negative_envelope():
    result = validate_config({"budget": {"envelope": {"daily": -1}}})
    assert not result.valid
    assert any("budget" in issue.path and "envelope.daily" in issue.message for issue in result.issues)


def test_validate_rejects_a_per_rung_name_that_is_not_paid():
    result = validate_config({"budget": {"per_rung": {"claude": {"daily": 1}}}})
    assert not result.valid and any("claude" in issue.message for issue in result.issues)


def test_validate_accepts_a_valid_envelope():
    result = validate_config({"budget": {"envelope": {"daily": 25, "weekly": 100},
                                         "per_rung": {"kimi": {"daily": 5}}}})
    assert result.valid, result.issues


def test_validate_still_accepts_the_legacy_keys():
    result = validate_config({"budget": {"per_task": 1.0, "per_session": 5.0, "daily": 10.0}})
    assert result.valid, result.issues


# Provider snapshots recorded by the runner


def test_the_runner_records_a_provider_snapshot_when_given_a_probe(spend_env):
    built = JobRunner(spend_env["ledger"], spend_env["runner"]().job_id, now=lambda: NOW,
                      snapshot=lambda: {"openrouter": {"used_usd": 20.0, "remaining_usd": None}})
    built._take_snapshot("start")
    events = [e for e in spend_env["ledger"].events(built.job_id) if e.kind == "provider_snapshot"]
    assert len(events) == 1
    assert events[0].data["when"] == "start" and events[0].data["providers"]["openrouter"]["used_usd"] == 20.0


def test_the_runner_probes_nothing_without_a_probe(spend_env):
    runner = spend_env["runner"]()
    runner._take_snapshot("start")
    assert not [e for e in spend_env["ledger"].events(runner.job_id) if e.kind == "provider_snapshot"]


def test_a_failing_probe_never_changes_the_job(spend_env):
    def broken():
        raise RuntimeError("offline")

    built = JobRunner(spend_env["ledger"], spend_env["runner"]().job_id, now=lambda: NOW, snapshot=broken)
    built._take_snapshot("end")
    assert not [e for e in spend_env["ledger"].events(built.job_id) if e.kind == "provider_snapshot"]


def test_the_report_compares_the_ledger_with_the_providers(spend_env):
    from mullet.jobs.report import build_report

    spend_env["add_attempt"](1.0)
    built = JobRunner(spend_env["ledger"], spend_env["runner"]().job_id, now=lambda: NOW, snapshot=None)
    ledger = spend_env["ledger"]
    ledger.add_event(built.job_id, "provider_snapshot",
                     {"when": "start", "at": NOW, "providers": {"kimi": {"used_usd": None, "remaining_usd": 100.0}}})
    ledger.add_event(built.job_id, "provider_snapshot",
                     {"when": "end", "at": NOW, "providers": {"kimi": {"used_usd": None, "remaining_usd": 99.0}}})
    report = build_report(ledger, built.job_id, config=parse_config({"kimi": KIMI_ID}))
    assert "Provider check" in report and "kimi: recorded $1.00, the provider says $1.00 (agrees)" in report


# Notifications carry credit and Claude's week


def test_notifications_carry_credit_and_claude_week_when_a_probe_is_given(spend_env):
    calls = []
    built = JobRunner(spend_env["ledger"], spend_env["runner"]().job_id, now=lambda: NOW,
                      notify=lambda *args, **kw: calls.append((args, kw)),
                      snapshot=lambda: {"kimi": {"used_usd": None, "remaining_usd": 100.5}})
    built._rate_limit = {"unifiedWindows": {"five_hour": {"utilization": 0.04, "resetsAt": NOW + 3600},
                                            "seven_day": {"utilization": 0.38, "resetsAt": NOW + 5 * 86400}}}
    built._notify("spend", "")
    (args, kw), = calls
    assert kw["status"] == "Credit: kimi $100.50 left; Claude week 38% used"


def test_notifications_are_unchanged_without_a_probe(spend_env):
    calls = []
    built = JobRunner(spend_env["ledger"], spend_env["runner"]().job_id, now=lambda: NOW,
                      notify=lambda *args, **kw: calls.append((args, kw)))
    built._notify("spend", "")
    (args, kw), = calls
    assert kw == {} and args[0] == "spend"
