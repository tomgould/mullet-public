"""Tests for mullet.jobs.ladder: config, the balanced policy, spend, the public check and the machine check.

Everything outside is faked: GitHub answers come from a dict, git runs on a
real throwaway repo, LM Studio is a loopback server or a function, and the
machine guard's snapshot is built by hand.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from mullet.hud.usage import RateLimits
from mullet.jobs import ladder as lad
from mullet.jobs import machine
from mullet.jobs import night as night_mod
from mullet.jobs.ladder import (
    BudgetState, GitHubError, LadderConfig, MachineState, PublicCheck, Rung, SpendState, Tried, apply_night,
    attempt_cost, budget_state, cached_public_check, check_public, choose_rungs, claude_tries, github_slug,
    load_config, machine_state, next_rung, parse_config, plan_rungs, spend_state,
)
from mullet.jobs.machine import GpuHolder, GpuProcess, Headroom
from mullet.jobs.night import NightState
from mullet.jobs.profile import Profile
from mullet.jobs.types import NodeSpec

from tests.jobs_helpers import git, make_source_repo

MECH = NodeSpec(id="m", goal="rename a helper", kind="mechanical")
KIMI_ID = "moonshotai/kimi-k2.7-code"
DESIGN = NodeSpec(id="d", goal="design the cache", kind="design")
ANALYSIS = NodeSpec(id="a", goal="find the slow query", kind="analysis")

RAW = {
    "kimi": "moonshotai/kimi-k2.7-code",
    "openrouter": ["openrouter/qwen/qwen3-coder", "openrouter/moonshotai/kimi-k2.7-code"],
    "free": "openrouter/qwen/qwen3-coder:free",
    "local": {"model": "lmstudio/qwen3.6-35b-a3b"},
}
FULL = parse_config(RAW)
BOTH = Profile(repo_id="r", allowed_executors=["claude-code", "opencode"], data_class="public")
PRIVATE = Profile(repo_id="r", allowed_executors=["claude-code", "opencode"])
MEDIUM = BudgetState("medium", 0.6)
LOCAL_OK = MachineState(True, "LM Studio has it", "loaded")
LOCAL_NO = MachineState(False, "LM Studio doesn't answer")
PUBLIC = PublicCheck(True, "public", "o/r", "main")
UNSPENT = SpendState(cap=2.0)


def names(rungs):
    return [r.name if r.name != "openrouter" else f"openrouter:{r.model.split('/', 2)[1]}" for r in rungs]


def choose(node=MECH, profile=BOTH, budget=MEDIUM, machine=LOCAL_OK, spend=UNSPENT, *, config=FULL,
           public=PUBLIC, has_key=lambda r: True, **kw):
    return plan_rungs(node, profile, budget, machine, spend, config=config, public=public, has_key=has_key, **kw)


# Config


def test_defaults_without_a_ladder_section(tmp_path):
    config = load_config(tmp_path / "missing.yaml")
    assert not config.configured
    assert config.kimi_model == "moonshotai/kimi-k2.7-code"
    assert config.openrouter_models[-1] == "openrouter/moonshotai/kimi-k2.7-code"
    assert config.free_model is None and config.local_model is None  # unset means skipped
    assert (config.claude_low, config.claude_high, config.max_spend) == (0.5, 0.8, 2.0)
    assert [r.name for r in config.rungs()] == ["kimi", "openrouter", "openrouter", "claude"]


def test_the_config_comes_from_the_users_mullet_config(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "ladder:\n  kimi: off\n  openrouter: [openrouter/x/y]\n  max_spend: 0.5\n  claude_low: 0.3\n"
        "  prices:\n    openrouter/x/y: {input: 1, output: 4}\n")
    config = load_config()
    assert config.configured and config.kimi_model is None and config.openrouter_models == ["openrouter/x/y"]
    assert config.max_spend == 0.5 and config.claude_low == 0.3
    assert config.price("openrouter/x/y") == {"input": 1.0, "output": 4.0}
    assert config.source == str(tmp_path / "config.yaml")


@pytest.mark.parametrize("data, problem", [
    ({"free": "openrouter/qwen/qwen3-coder"}, "not an OpenRouter :free model"),
    ({"free": "moonshotai/kimi:free"}, "not an OpenRouter :free model"),
    ({"local": "openrouter/x/y"}, "not an LM Studio model"),
    ({"local": {"model": "lmstudio/x", "url": "http://evil.example:1234"}}, "must be http://127.0.0.1"),
    ({"kimi": "no-slash"}, "not an opencode model id"),
    ({"max_spend": -1}, "not a valid number"),
    ({"claude_low": 0.9, "claude_high": 0.2}, "claude_high is below claude_low"),
])
def test_bad_values_are_dropped_with_a_note_never_guessed(data, problem):
    config = parse_config(data)
    assert any(problem in p for p in config.problems), config.problems
    assert config.free_model is None or config.free_model.endswith(":free")
    assert config.local_url == lad.DEFAULT_LOCAL_URL


def test_a_repo_file_in_the_working_directory_is_never_read(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mullet.yaml").write_text("ladder:\n  free: openrouter/evil/model:free\n")
    assert load_config().free_model is None


# The policy


LOW, HIGH = BudgetState("low", 0.2), BudgetState("high", 0.95)
CHECKABLE = NodeSpec(id="d", goal="add a retry helper", kind="design", scope=["app/retry.py", "tests/test_retry.py"],
                     acceptance=["pytest tests/test_retry.py"])
ALL_OR = ["openrouter:qwen", "openrouter:moonshotai"]


@pytest.mark.parametrize("budget, node, integration, expected", [
    (LOW, MECH, False, ["free", "local", "claude", "kimi", *ALL_OR]),  # Claude is paid for already
    (LOW, CHECKABLE, False, ["claude", "kimi"]),
    (LOW, ANALYSIS, False, ["claude", "kimi"]),
    (LOW, MECH, True, ["claude", "kimi"]),
    (MEDIUM, MECH, False, ["free", "local", "kimi", *ALL_OR, "claude"]),
    (MEDIUM, CHECKABLE, False, ["kimi", *ALL_OR, "claude"]),
    (MEDIUM, DESIGN, False, ["claude", "kimi"]),  # no acceptance commands
    (MEDIUM, ANALYSIS, False, ["claude", "kimi"]),
    (MEDIUM, MECH, True, ["claude", "kimi"]),
    (HIGH, MECH, False, ["free", "local", "kimi", *ALL_OR, "claude"]),
    (HIGH, CHECKABLE, False, ["free", "local", "kimi", *ALL_OR, "claude"]),
    (HIGH, ANALYSIS, False, ["kimi", *ALL_OR, "claude"]),  # analysis never goes to free or local
    (HIGH, DESIGN, False, ["kimi", *ALL_OR, "claude"]),
    (HIGH, MECH, True, ["kimi", *ALL_OR, "claude"]),
])
def test_the_order_follows_claudes_budget_and_the_step(budget, node, integration, expected):
    assert names(choose(node, budget=budget, integration=integration).rungs) == expected


@pytest.mark.parametrize("budget", [LOW, MEDIUM, HIGH])
def test_a_claude_only_profile_keeps_every_step_on_claude(budget):
    profile = BOTH.model_copy(update={"ladder": "claude-only"})
    for node in (MECH, CHECKABLE, ANALYSIS):
        assert names(choose(node, profile=profile, budget=budget).rungs) == ["claude"]


@pytest.mark.parametrize("update, group", [
    ({}, lad.CHECKABLE),
    ({"acceptance": []}, lad.CLAUDE_FIRST),
    ({"scope": ["**"]}, lad.CLAUDE_FIRST),
    ({"scope": ["a/*", "b/*", "c/*", "d/*", "e/*"]}, lad.CLAUDE_FIRST),
    ({"goal": "kill stray workers after the job"}, lad.CLAUDE_FIRST),
    ({"goal": "send the report to the team"}, lad.CLAUDE_FIRST),
])
def test_only_a_design_step_its_checks_can_judge_and_that_isnt_risky_is_checkable(update, group):
    assert lad.step_group(CHECKABLE.model_copy(update=update), FULL)[0] == group


def test_risk_is_read_from_goal_acceptance_and_scope():
    assert not lad.is_risky(NodeSpec(id="x", goal="parse the token stream over http"), FULL)
    assert not lad.is_risky(NodeSpec(id="x", goal="post a network socket message"), FULL)
    assert lad.is_risky(NodeSpec(id="x", goal="tidy", scope=["mullet/rm/**"]), FULL)
    assert lad.is_risky(NodeSpec(id="x", goal="tidy", acceptance=["python3 -m pytest tests/test_kill.py"]), FULL)
    for goal in ("Delete old clones", "remove the cache", "truncate the log", "drop the table", "push the branch",
                 "upload the report", "terminate idle workers", "wipe temp dirs", "unlink stale sockets"):
        assert lad.is_risky(NodeSpec(id="x", goal=goal), FULL), goal
    assert lad.is_risky(NodeSpec(id="x", goal="rename it"), parse_config({"risky_words": ["rename"]}))


def test_the_cheapest_rung_per_kind_comes_from_config():
    config = parse_config({"kimi": KIMI_ID, "openrouter": [], "free": "openrouter/x/y:free",
                           "start": {"mechanical": "kimi", "design": "claude"}})
    assert names(choose(config=config).rungs) == ["kimi", "claude"]
    assert names(choose(CHECKABLE, config=config).rungs) == ["claude"]
    assert names(choose(MECH, budget=LOW, config=config).rungs) == ["claude", "kimi"]
    assert parse_config({"start": {"mechanical": "cheapest"}}).start["mechanical"] == "free"
    bad = parse_config({"start": {"design": "gemini", "chores": "kimi"}})
    assert bad.start == lad.DEFAULT_START and len(bad.problems) == 2


def test_new_config_keys():
    config = parse_config({"max_below_claude": 2, "cheap_minutes": 5})
    assert (config.max_below_claude, config.cheap_minutes) == (2, 5.0)
    assert parse_config({}).max_below_claude == 3 and parse_config({}).cheap_minutes == 10.0
    assert parse_config({"cheap_minutes": 0}).problems


def test_a_private_repo_never_gets_the_free_rung():
    choice = choose(profile=PRIVATE, public=PublicCheck(False, "the repo's data class is private"))
    assert names(choice.rungs)[0] == "local"
    assert any("free" in label and "data class is private" in why for label, why in choice.skipped)
    assert names(choose(profile=PRIVATE, public=None).rungs)[0] == "local"  # not checked is not public


def test_local_is_skipped_without_headroom_with_the_reason():
    choice = choose(machine=LOCAL_NO)
    assert "local" not in names(choice.rungs)
    assert ("local (lmstudio/qwen3.6-35b-a3b)", "LM Studio doesn't answer") in choice.skipped


def test_allowed_executors_still_apply():
    choice = choose(profile=Profile(repo_id="r", data_class="public"))  # claude-code only
    assert names(choice.rungs) == ["claude"]
    assert any("doesn't allow the opencode executor" in why for _, why in choice.skipped)
    no_claude = Profile(repo_id="r", allowed_executors=["opencode"], data_class="public")
    assert names(choose(profile=no_claude).rungs)[-1] == "openrouter:moonshotai"


def test_the_spending_cap_skips_only_the_paid_rungs_and_says_so():
    choice = choose(spend=SpendState(cap=2.0, spent=2.0))
    assert names(choice.rungs) == ["free", "local", "claude"] and choice.capped
    assert any("spending cap" in why for _, why in choice.skipped)
    assert not choose().capped


def test_a_rung_without_a_key_is_skipped():
    choice = choose(has_key=lambda r: r.name != "kimi")
    assert "kimi" not in names(choice.rungs)
    assert ("kimi (moonshotai/kimi-k2.7-code)", "no API key for its provider") in choice.skipped


def test_choose_rungs_returns_just_the_rungs():
    rungs = choose_rungs(MECH, BOTH, MEDIUM, LOCAL_OK, UNSPENT, config=FULL, public=PUBLIC, has_key=lambda r: True)
    assert rungs == choose().rungs and all(isinstance(r, Rung) for r in rungs)


def test_rung_identity_and_executor():
    kimi = Rung("kimi", "moonshotai/kimi-k2.7-code")
    assert kimi.executor == "opencode" and kimi.paid and kimi.key == "kimi:moonshotai/kimi-k2.7-code"
    assert Rung("claude").executor == "claude-code" and not Rung("claude").paid
    assert not Rung("free", "openrouter/x:free").paid and "/" not in kimi.slug
    local = FULL.rungs()[1]
    assert local.upstream == "http://127.0.0.1:1234/v1"


def test_keys_come_from_mullets_own_env(monkeypatch):
    monkeypatch.setattr("mullet.core.env.dotenv_value", lambda name: None)
    kimi = Rung("kimi", "moonshotai/kimi-k2.7-code")
    assert not lad.rung_has_key(kimi)
    monkeypatch.setenv("KIMI_API_KEY", "sk-kimi-alias")  # the alias Mullet's .env may use
    assert lad.rung_has_key(kimi)
    assert lad.rung_has_key(Rung("claude")) and lad.rung_has_key(FULL.rungs()[1])  # local needs none


# Walking the ladder


def tried(rung, model=None, *, failed=True, moved=False):
    key = f"{rung}:{model or ''}"
    return Tried(key, rung, model, counted=not moved, failed=failed and not moved, moved_up=moved)


def walk(choice, history, budget=MEDIUM, max_attempts=3, config=FULL, **kw):
    found = next_rung(choice, history, budget, max_attempts=max_attempts, config=config, **kw)
    return (found.rung.name if found.rung else None), found


def test_one_attempt_per_rung_then_claude_gets_the_rest():
    choice = choose(MECH, public=None, machine=LOCAL_NO)  # kimi, or qwen, or kimi, claude
    assert walk(choice, [])[0] == "kimi"
    assert walk(choice, [tried("kimi", KIMI_ID, moved=True)])[1].rung.model == "openrouter/qwen/qwen3-coder"
    history = [tried("openrouter", "openrouter/qwen/qwen3-coder")]
    rung, found = walk(choice, history, max_attempts=4)
    assert rung == "kimi" and not found.final


def test_a_model_that_failed_the_gates_is_never_tried_again_elsewhere():
    choice = choose(MECH, public=None, machine=LOCAL_NO)
    rung, found = walk(choice, [tried("kimi", KIMI_ID), tried("openrouter", "openrouter/qwen/qwen3-coder")])
    assert rung == "claude"  # OpenRouter's Kimi is the model that just failed on Moonshot
    assert ("openrouter (openrouter/moonshotai/kimi-k2.7-code)", "its model already failed this step's checks") \
        in found.skipped
    # A provider error on Moonshot is no verdict on the model: OpenRouter's Kimi may still run.
    rung, found = walk(choice, [tried("kimi", KIMI_ID, moved=True), tried("openrouter", "openrouter/qwen/qwen3-coder")])
    assert found.rung.model == "openrouter/moonshotai/kimi-k2.7-code"


def test_claude_gets_its_turn_after_max_below_claude_attempts():
    config = parse_config({**RAW, "openrouter": ["openrouter/deepseek/deepseek-coder"]})
    choice = choose(MECH, config=config)  # free, local, kimi, deepseek, claude
    history = [tried("free", "openrouter/qwen/qwen3-coder:free"), tried("local", "lmstudio/qwen3.6-35b-a3b"),
               tried("kimi", KIMI_ID)]
    rung, found = walk(choice, history, max_attempts=6, config=config)
    assert rung == "claude" and any("Claude's turn" in why for _, why in found.skipped)
    assert walk(choice, history[:2], max_attempts=6, config=config)[0] == "kimi"
    roomy = parse_config({**RAW, "openrouter": ["openrouter/deepseek/deepseek-coder"], "max_below_claude": 5})
    assert walk(choice, history, max_attempts=6, config=roomy)[1].rung.model == "openrouter/deepseek/deepseek-coder"


def test_claude_in_the_middle_when_its_week_has_room():
    choice = choose(MECH, budget=LOW, public=None, machine=LOCAL_NO)  # claude, kimi, or, or
    assert walk(choice, [], budget=LOW)[0] == "claude"
    rung, found = walk(choice, [tried("claude")] * 3, budget=LOW)
    assert rung == "kimi" and not found.final
    assert walk(choice, [tried("claude")], budget=LOW, max_attempts=3)[0] == "claude"


def test_claude_gets_one_try_when_its_week_is_tight():
    choice = choose(ANALYSIS, budget=HIGH)  # kimi, or, or, claude
    history = [tried("kimi", KIMI_ID), tried("openrouter", "openrouter/qwen/qwen3-coder")]
    rung, found = walk(choice, history, budget=HIGH)
    assert rung == "claude" and found.final
    rung, found = walk(choice, [*history, tried("claude")], budget=HIGH)
    assert rung is None and "ran out of rungs" in found.reason
    assert claude_tries(HIGH, 3, 0) == 1 and claude_tries(MEDIUM, 3, 1) == 2 and claude_tries(MEDIUM, 3, 5) == 1


def test_model_family():
    assert lad.model_family("openrouter/moonshotai/kimi-k2.7-code") == lad.model_family(KIMI_ID)
    assert lad.model_family("openrouter/qwen/qwen3-coder:free") == lad.model_family("openrouter/qwen/qwen3-coder")
    assert lad.model_family("lmstudio/qwen3.6") != lad.model_family("openrouter/qwen/qwen3-coder")


# Claude's budget level


def test_no_usage_figures_count_as_medium():
    assert budget_state(FULL, None) == BudgetState("medium", None)


def test_budget_levels_from_the_statusline_figures(monkeypatch):
    monkeypatch.setattr("mullet.hud.guard.pressure_components", lambda limits: (0.0, limits.weekly_pct / 100))
    assert budget_state(FULL, RateLimits(weekly_pct=20)).level == "low"
    assert budget_state(FULL, RateLimits(weekly_pct=60)).level == "medium"
    assert budget_state(FULL, RateLimits(weekly_pct=90)).level == "high"


def test_the_last_known_level_stands_in_for_six_hours(monkeypatch):
    monkeypatch.setattr("mullet.hud.guard.pressure_components", lambda limits: (0.0, limits.weekly_pct / 100))
    clock = [1_000_000.0]
    figures = [RateLimits(weekly_pct=90)]
    ladder = lad.Ladder(config=FULL, usage=lambda: figures[0], now=lambda: clock[0])
    assert ladder.budget().level == "high"
    figures[0] = None
    clock[0] += 5 * 3600
    known = ladder.budget()
    assert known.level == "high" and known.stale_s == pytest.approx(5 * 3600) and "last known" in known.describe()
    clock[0] += 2 * 3600
    assert ladder.budget() == BudgetState("medium", None)


def test_claude_stream_limits_keep_the_figures_fresh(monkeypatch, tmp_path):
    from mullet.hud import usage

    monkeypatch.setattr(usage, "_CACHE_FILE", tmp_path / "usage.json")
    assert not usage.record_stream_limits({"status": "allowed"})
    info = {"unifiedWindows": {"seven_day": {"utilization": 0.42, "resetsAt": 4_102_444_800},
                               "five_hour": {"utilization": 0.1}}}
    assert usage.record_stream_limits(info)
    limits = usage.fetch_usage()
    assert limits.weekly_pct == pytest.approx(42.0) and limits.five_hour_pct == pytest.approx(10.0)
    assert limits.weekly_resets.startswith("2100-01-01")
    # Without the weekly window nothing is stored, so an old weekly figure is never restamped as fresh.
    assert not usage.record_stream_limits({"unifiedWindows": {"five_hour": {"utilization": 0.5}}})
    assert usage.fetch_usage().five_hour_pct == pytest.approx(10.0)


def test_a_five_hour_reading_alone_never_makes_a_low_budget(monkeypatch, tmp_path):
    from mullet.hud import usage

    monkeypatch.setattr(usage, "_CACHE_FILE", tmp_path / "usage.json")
    assert not usage.record_stream_limits({"unifiedWindows": {"five_hour": {"utilization": 0.1}}})
    assert usage.fetch_usage() is None


# Spend


def test_reported_cost_counts_and_tokens_are_priced_when_not():
    assert attempt_cost({"rung": "kimi", "cost_reported": True, "total_cost_usd": 0.27}, FULL) == (0.27, False)
    config = parse_config({"prices": {"moonshotai/kimi-k2.7-code": {"input": 1.0, "output": 4.0}}})
    usage = {"rung": "kimi", "model": "moonshotai/kimi-k2.7-code", "cost_reported": False,
             "tokens": {"input": 1_000_000, "output": 500_000, "cache_read": 0}}
    assert attempt_cost(usage, config) == (pytest.approx(3.0), True)
    unpriced = {**usage, "model": "openrouter/x/y", "rung": "openrouter"}
    assert attempt_cost(unpriced, config)[0] == pytest.approx(2.0 + 4.0)  # the conservative fallback


def test_a_reported_zero_with_real_tokens_is_priced_from_the_tokens():
    usage = {"rung": "kimi", "model": "openrouter/x/y", "cost_reported": True, "total_cost_usd": 0.0,
             "tokens": {"input": 1_000_000, "output": 0}}
    assert attempt_cost(usage, FULL) == (pytest.approx(2.0), True)
    assert attempt_cost({**usage, "tokens": {}}, FULL) == (0.0, False)


def test_claude_free_and_local_attempts_cost_nothing_here():
    for rung in ("claude", "free", "local", None):
        assert attempt_cost({"rung": rung, "cost_reported": True, "total_cost_usd": 9.0}, FULL) == (0.0, False)


def test_spend_adds_up_by_rung():
    state = spend_state([
        {"rung": "kimi", "cost_reported": True, "total_cost_usd": 0.27},
        {"rung": "openrouter", "cost_reported": True, "total_cost_usd": 0.04},
        {"rung": "claude", "total_cost_usd": 1.5},
        None,
    ], cap=0.3, config=FULL)
    assert state.spent == pytest.approx(0.31) and state.capped
    assert state.by_rung == {"kimi": pytest.approx(0.27), "openrouter": pytest.approx(0.04)}


# The public check


@pytest.fixture
def public_repo(tmp_path):
    repo = make_source_repo(tmp_path / "src")
    git(repo, "remote", "add", "origin", "git@github.com:octo/widgets.git")
    return repo


def github(repo_info=None, compare_status="behind", fail=None):
    calls = []

    def fetch(path):
        calls.append(path)
        if fail:
            raise GitHubError(fail)
        if "/compare/" in path:
            return {"status": compare_status}
        return {"private": False, "fork": False, "visibility": "public", "default_branch": "main",
                **(repo_info or {})}
    fetch.calls = calls
    return fetch


def head(repo):
    return git(repo, "rev-parse", "HEAD").strip()


PUBLIC_PROFILE = Profile(repo_id="r", data_class="public")


def test_a_public_unforked_pushed_repo_passes(public_repo):
    fetch = github()
    found = check_public(public_repo, head(public_repo), PUBLIC_PROFILE, fetch=fetch)
    assert found.public and found.slug == "octo/widgets" and found.default_branch == "main"
    assert fetch.calls == ["/repos/octo/widgets", f"/repos/octo/widgets/compare/main...{head(public_repo)}"]


@pytest.mark.parametrize("profile, fetch, reason", [
    (Profile(repo_id="r"), github(), "data class is private"),
    (Profile(repo_id="r", data_class="public", copy_in=[".env"]), github(), "copy_in"),
    (PUBLIC_PROFILE, github({"private": True}), "private"),
    (PUBLIC_PROFILE, github({"visibility": "internal"}), "private"),
    (PUBLIC_PROFILE, github({"fork": True}), "fork"),
    (PUBLIC_PROFILE, github(compare_status="ahead"), "isn't on origin/main"),
    (PUBLIC_PROFILE, github(compare_status="diverged"), "isn't on origin/main"),
    (PUBLIC_PROFILE, github(fail="unreachable: URLError"), "GitHub's API: unreachable"),
])
def test_the_public_check_fails_closed(public_repo, profile, fetch, reason):
    found = check_public(public_repo, head(public_repo), profile, fetch=fetch)
    assert not found.public and reason in found.reason


def test_submodules_or_a_gitmodules_file_fail_the_check(public_repo):
    (public_repo / ".gitmodules").write_text('[submodule "x"]\n\tpath = x\n\turl = https://example.com/x\n')
    git(public_repo, "add", ".gitmodules")
    git(public_repo, "commit", "-qm", "submodules")
    found = check_public(public_repo, head(public_repo), PUBLIC_PROFILE, fetch=github())
    assert not found.public and ".gitmodules" in found.reason
    gitlink = make_source_repo(public_repo.parent / "linked")
    git(gitlink, "update-index", "--add", "--cacheinfo", f"160000,{head(gitlink)},vendor/lib")
    git(gitlink, "commit", "-qm", "gitlink")
    git(gitlink, "remote", "add", "origin", "https://github.com/octo/linked")
    found = check_public(gitlink, head(gitlink), PUBLIC_PROFILE, fetch=github())
    assert not found.public and "submodules" in found.reason


def test_a_remote_that_isnt_github_fails(public_repo):
    git(public_repo, "remote", "set-url", "origin", "https://gitlab.com/octo/widgets.git")
    found = check_public(public_repo, head(public_repo), PUBLIC_PROFILE, fetch=github())
    assert not found.public and "not a GitHub repository" in found.reason


def test_unreachable_github_really_fails_closed(public_repo):
    # The real fetcher, with the test suite's network guard in the way.
    found = check_public(public_repo, head(public_repo), PUBLIC_PROFILE)
    assert not found.public and "GitHub" in found.reason


@pytest.mark.parametrize("url, slug", [
    ("git@github.com:octo/widgets.git", "octo/widgets"),
    ("https://github.com/octo/widgets", "octo/widgets"),
    ("ssh://git@github.com/octo/widgets.git", "octo/widgets"),
    ("https://github.com.evil.example/octo/widgets", None),
    ("https://gitlab.com/octo/widgets", None),
])
def test_github_remotes(url, slug):
    assert github_slug(url) == slug


@pytest.mark.parametrize("url", ["https://github.com/../widgets", "git@github.com:octo/..", "https://github.com/./x",
                                 "https://github.com/octo/..git", "https://github.com/octo/."])
def test_github_remotes_never_climb_the_api_path(url):
    assert github_slug(url) is None


def test_a_github_outage_is_transient_and_a_404_is_not(public_repo):
    down = check_public(public_repo, head(public_repo), PUBLIC_PROFILE, fetch=github(fail="unreachable: URLError"))
    assert not down.public and down.transient

    def gone(path):
        raise GitHubError("not found (private, or not on GitHub)", transient=False)

    found = check_public(public_repo, head(public_repo), PUBLIC_PROFILE, fetch=gone)
    assert not found.public and not found.transient


def test_the_public_check_is_cached_per_repo_and_base_for_an_hour(public_repo):
    clock = [5_000.0]
    calls = []

    def check(repo, sha, profile):
        calls.append(sha)
        return PublicCheck(True, "public", "octo/widgets", "main", clock[0])

    sha = head(public_repo)
    for _ in range(3):
        assert cached_public_check(public_repo, sha, PUBLIC_PROFILE, check=check, now=lambda: clock[0]).public
    assert calls == [sha]
    clock[0] += 3601
    cached_public_check(public_repo, sha, PUBLIC_PROFILE, check=check, now=lambda: clock[0])
    assert calls == [sha, sha]
    cached_public_check(public_repo, "0" * 40, PUBLIC_PROFILE, check=check, now=lambda: clock[0])
    assert len(calls) == 3  # another base is another check


def test_a_transient_failure_is_never_cached(public_repo):
    answers = [PublicCheck(False, "GitHub's API: unreachable", transient=True), PublicCheck(True, "public")]
    check = lambda repo, sha, profile: answers.pop(0)  # noqa: E731
    sha = head(public_repo)
    assert not cached_public_check(public_repo, sha, PUBLIC_PROFILE, check=check).public
    assert cached_public_check(public_repo, sha, PUBLIC_PROFILE, check=check).public  # asked again


def test_a_public_check_round_trips_through_the_event_log():
    found = PublicCheck(True, "public", "o/r", "main", 12.0)
    assert PublicCheck.from_dict(json.loads(json.dumps(found.to_dict()))) == found


# The machine check, for the local rung


LOCAL = parse_config({"local": "lmstudio/qwen3.6-35b-a3b"})
ROOMY = Headroom(gpu_name="RTX", vram_free_mb=9000, vram_total_mb=12000, ram_free_mb=32000, load_1m=0.5)


def test_local_needs_a_configured_model():
    assert not machine_state(LadderConfig()).local_ok


def test_local_is_skipped_while_another_process_holds_the_gpu_lease():
    reindex = GpuHolder(pid=999999, purpose="reindex of pool docs", started=0.0)
    state = machine_state(LOCAL, holder=lambda: reindex, models=lambda url: {"qwen3.6-35b-a3b": "loaded"},
                          sample=lambda: ROOMY)
    assert not state.local_ok and "reindex of pool docs" in state.reason


def test_local_is_skipped_when_lm_studio_does_not_answer_or_lacks_the_model():
    def down(url):
        raise OSError("connection refused")
    state = machine_state(LOCAL, holder=lambda: None, models=down, sample=lambda: ROOMY)
    assert not state.local_ok and "doesn't answer" in state.reason
    state = machine_state(LOCAL, holder=lambda: None, models=lambda url: {"other": "loaded"}, sample=lambda: ROOMY)
    assert not state.local_ok and "no model" in state.reason


def test_local_is_skipped_without_headroom():
    hot = Headroom(gpu_name="RTX", vram_free_mb=9000, vram_total_mb=12000, gpu_temp_c=95.0, ram_free_mb=32000)
    state = machine_state(LOCAL, holder=lambda: None, models=lambda url: {"qwen3.6-35b-a3b": "not-loaded"},
                          sample=lambda: hot)
    assert not state.local_ok and "MULLET_GPU_MAX_TEMP_C" in state.reason


def test_vram_lm_studio_holds_counts_as_free_when_the_model_is_loaded():
    full = Headroom(gpu_name="RTX", vram_free_mb=300, vram_total_mb=12000, ram_free_mb=32000, load_1m=0.5,
                    gpu_processes=(GpuProcess(4242, "/opt/lmstudio/llama-server", 11000),))
    loaded = machine_state(LOCAL, holder=lambda: None, models=lambda url: {"qwen3.6-35b-a3b": "loaded"},
                           sample=lambda: full)
    assert loaded.local_ok and loaded.model_state == "loaded"
    cold = machine_state(LOCAL, holder=lambda: None, models=lambda url: {"qwen3.6-35b-a3b": "not-loaded"},
                         sample=lambda: full)
    assert not cold.local_ok and "MULLET_GPU_MIN_FREE_MB" in cold.reason


class _FakeTime:
    """A clock that moves only when the wait sleeps."""

    def __init__(self):
        self.t = 0.0
        self.sleeps = 0

    def clock(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps += 1
        self.t += max(seconds, 1.0)


def test_waiting_for_headroom_gives_up_with_the_same_limits():
    hot = Headroom(gpu_name="RTX", vram_free_mb=9000, vram_total_mb=12000, gpu_temp_c=95.0, ram_free_mb=32000)
    before = machine.limits()
    fake = _FakeTime()
    state = machine_state(LOCAL, holder=lambda: None, models=lambda url: {"qwen3.6-35b-a3b": "not-loaded"},
                          sample=lambda: hot, wait_s=1800.0, clock=fake.clock, sleep=fake.sleep)
    assert not state.local_ok and "MULLET_GPU_MAX_TEMP_C" in state.reason
    assert fake.t >= 1800.0 and fake.sleeps > 1  # it really waited
    assert machine.limits() == before


def test_waiting_still_counts_lm_studios_vram_as_free_when_the_model_is_loaded():
    full = Headroom(gpu_name="RTX", vram_free_mb=300, vram_total_mb=12000, ram_free_mb=32000, load_1m=0.5,
                    gpu_processes=(GpuProcess(4242, "/opt/lmstudio/llama-server", 11000),))
    fake = _FakeTime()
    loaded = machine_state(LOCAL, holder=lambda: None, models=lambda url: {"qwen3.6-35b-a3b": "loaded"},
                           sample=lambda: full, wait_s=1800.0, clock=fake.clock, sleep=fake.sleep)
    assert loaded.local_ok and loaded.model_state == "loaded"
    assert fake.sleeps == 0  # room at once, as a single sample finds
    cold = machine_state(LOCAL, holder=lambda: None, models=lambda url: {"qwen3.6-35b-a3b": "not-loaded"},
                         sample=lambda: full, wait_s=1800.0, clock=fake.clock, sleep=fake.sleep)
    assert not cold.local_ok and "MULLET_GPU_MIN_FREE_MB" in cold.reason


def test_waiting_follows_lm_studio_when_its_process_changes():
    first = Headroom(gpu_name="RTX", vram_free_mb=300, vram_total_mb=12000, ram_free_mb=32000, load_1m=0.5,
                     gpu_processes=(GpuProcess(1, "python3", 11000),))
    later = Headroom(gpu_name="RTX", vram_free_mb=300, vram_total_mb=12000, ram_free_mb=32000, load_1m=0.5,
                     gpu_processes=(GpuProcess(4243, "/opt/lmstudio/llama-server", 11000),))
    samples = iter([first, later])
    fake = _FakeTime()
    state = machine_state(LOCAL, holder=lambda: None, models=lambda url: {"qwen3.6-35b-a3b": "loaded"},
                          sample=lambda: next(samples), wait_s=1800.0, clock=fake.clock, sleep=fake.sleep)
    assert state.local_ok and fake.sleeps == 1


def test_without_a_gpu_the_wait_never_lets_the_local_rung_past_a_limit():
    busy = Headroom(ram_free_mb=10, load_1m=0.5)  # no GPU: the guard's wait goes ahead anyway
    fake = _FakeTime()
    state = machine_state(LOCAL, holder=lambda: None, models=lambda url: {"qwen3.6-35b-a3b": "loaded"},
                          sample=lambda: busy, wait_s=1800.0, clock=fake.clock, sleep=fake.sleep)
    assert not state.local_ok and "MULLET_RAM_MIN_FREE_MB" in state.reason


# Night mode


NIGHT_ON = NightState(on=True, until=1e10)


def test_apply_night_gives_cheap_attempts_longer_and_leaves_money_and_models_alone():
    config = parse_config({**RAW, "prices": {KIMI_ID: {"input": 0.6, "output": 2.5}}, "max_spend": 1.5})
    config.problems.append("a note")
    at_night = apply_night(config, NIGHT_ON)
    assert at_night is not config
    assert at_night.cheap_minutes == config.cheap_minutes * 2 == 20.0
    assert at_night.max_below_claude == config.max_below_claude + 2 == 5
    for name in ("prices", "fallback_price", "max_spend", "claude_low", "claude_high", "start", "kimi_model",
                 "openrouter_models", "free_model", "local_model", "local_url", "problems"):
        assert getattr(at_night, name) == getattr(config, name), name
    assert at_night.problems is not config.problems
    assert (config.cheap_minutes, config.max_below_claude) == (10.0, 3)  # the original is untouched


def test_apply_night_off_returns_the_config_unchanged():
    assert apply_night(FULL, NightState(on=False, reason="you're back")) is FULL
    assert apply_night(FULL, NightState()) is FULL


def test_a_checkable_step_at_medium_tries_local_first_only_at_night():
    day = choose(CHECKABLE)
    assert names(day.rungs) == ["kimi", *ALL_OR, "claude"]
    assert lad.step_order("medium", lad.CHECKABLE) == ("kimi", "openrouter", "claude")
    assert lad.step_order("medium", lad.CHECKABLE) == lad.ORDERS["medium"][lad.CHECKABLE]
    at_night = choose(CHECKABLE, night=True)
    assert names(at_night.rungs) == ["local", "kimi", *ALL_OR, "claude"]
    assert not any(label.startswith("local") for label, _ in at_night.skipped)
    assert names(choose(CHECKABLE, machine=LOCAL_NO, night=True).rungs) == ["kimi", *ALL_OR, "claude"]


@pytest.mark.parametrize("level", ["low", "medium", "high"])
@pytest.mark.parametrize("group", [lad.MECHANICAL, lad.CHECKABLE, lad.CLAUDE_FIRST])
def test_every_other_order_is_the_same_at_night(level, group):
    if (level, group) == ("medium", lad.CHECKABLE):
        return
    assert lad.step_order(level, group, night=True) == lad.step_order(level, group) == lad.ORDERS[level][group]


@pytest.mark.parametrize("budget", [LOW, MEDIUM, HIGH])
def test_other_steps_choose_the_same_rungs_at_night(budget):
    for node, integration in ((MECH, False), (ANALYSIS, False), (DESIGN, False), (MECH, True)):
        assert choose(node, budget=budget, integration=integration, night=True) == \
            choose(node, budget=budget, integration=integration)
    if budget is not MEDIUM:
        assert choose(CHECKABLE, budget=budget, night=True) == choose(CHECKABLE, budget=budget)


def test_the_ladder_asks_night_mode_through_a_swappable_hook():
    assert lad.Ladder(config=FULL).night is night_mod.check
    assert lad.Ladder(config=FULL, night=lambda: NIGHT_ON).night() == NIGHT_ON


class _LMStudio(BaseHTTPRequestHandler):
    seen: list[tuple[str, str]] = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        type(self).seen.append(("GET", self.path))
        body = json.dumps({"data": [{"id": "qwen3.6-35b-a3b", "state": "loaded"},
                                    {"id": "ornith-1.5-9b", "state": "not-loaded"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        type(self).seen.append(("POST", self.path))
        self.send_response(500)
        self.end_headers()


def test_the_lm_studio_health_check_reads_model_states_and_never_sends_a_completion():
    _LMStudio.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _LMStudio)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        assert lad.lmstudio_models(url) == {"qwen3.6-35b-a3b": "loaded", "ornith-1.5-9b": "not-loaded"}
        config = parse_config({"local": {"model": "lmstudio/qwen3.6-35b-a3b", "url": url}})
        assert machine_state(config, holder=lambda: None, sample=lambda: ROOMY).local_ok
    finally:
        server.shutdown()
    assert _LMStudio.seen and all(seen == ("GET", "/api/v0/models") for seen in _LMStudio.seen)


def test_the_ladder_reads_no_completion_endpoint_anywhere():
    source = Path(lad.__file__).read_text()
    assert "chat/completions" not in source and "/v1/completions" not in source


def test_paid_openrouter_requests_never_go_to_hosts_that_keep_prompts():
    config = parse_config({"routing": {
        "openrouter/z-ai/glm-5.3": {"only": ["z-ai"], "data_collection": "allow", "allow_fallbacks": False},
        "openrouter/x/y": {"sort": "cheapest", "quantizations": "fp4"},
        "openrouter/bad": ["z-ai"],
    }})
    glm = config.request_fields(Rung("openrouter", "openrouter/z-ai/glm-5.3"))
    assert glm == {"provider": {"data_collection": "deny", "quantizations": ["fp8", "fp16", "bf16", "fp32"],
                                "only": ["z-ai"], "allow_fallbacks": False},
                   "usage": {"include": True}}  # OpenRouter then states each request's real cost
    # Settings Mullet doesn't pass on are dropped with a note, never guessed.
    assert config.request_fields(Rung("openrouter", "openrouter/x/y")) == \
        LadderConfig().request_fields(Rung("openrouter", "openrouter/x/y"))
    assert len([p for p in config.problems if "ladder.routing" in p]) == 4
    # Only the paid OpenRouter rungs are routed.
    for rung in (Rung("kimi", KIMI_ID), Rung("free", "openrouter/x/y:free"), Rung("claude")):
        assert config.request_fields(rung) is None


def test_local_by_day_false_keeps_the_local_rung_for_night_mode():
    config = parse_config({**RAW, "local_by_day": False})
    day = choose(config=config)
    assert "local" not in names(day.rungs)
    assert ("local (lmstudio/qwen3.6-35b-a3b)", "local runs only in night mode (ladder.local_by_day: false)") \
        in day.skipped
    assert "local" in names(choose(config=config, night=True).rungs)
    assert "local" in names(choose().rungs)  # the default keeps local by day
    assert parse_config({**RAW, "local_by_day": "no"}).problems == ["ladder.local_by_day: expected true or false; using true"]
