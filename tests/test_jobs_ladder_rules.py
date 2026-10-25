"""Tests for three ladder rules: the cap on attempts below Claude, OpenRouter privacy by where requests go,
and no undeclared quantisation by default."""

import pytest

from mullet.jobs.ladder import (
    DEFAULT_OPENROUTER_ROUTING, BudgetState, LadderConfig, MachineState, NightSettings, PublicCheck, Rung,
    SpendState, Tried, apply_night, next_rung, parse_config, plan_rungs,
)
from mullet.jobs.night import NightState
from mullet.jobs.profile import Profile
from mullet.jobs.types import NodeSpec

MECH = NodeSpec(id="m", goal="rename a helper", kind="mechanical")
KIMI_ID = "moonshotai/kimi-k2.7-code"
FREE_ID = "openrouter/mistralai/devstral-small:free"  # not a family any other rung uses
LOCAL_ID = "lmstudio/qwen3.6-35b-a3b"
QWEN = "openrouter/qwen/qwen3-coder"
DEEPSEEK = "openrouter/deepseek/deepseek-coder"
RAW = {"kimi": KIMI_ID, "openrouter": [QWEN, DEEPSEEK], "free": FREE_ID, "local": LOCAL_ID}
CONFIG = parse_config(RAW)
BOTH = Profile(repo_id="r", allowed_executors=["claude-code", "opencode"], data_class="public")
OPENCODE_ONLY = Profile(repo_id="r", allowed_executors=["opencode"], data_class="public")
LOW, MEDIUM = BudgetState("low", 0.2), BudgetState("medium", 0.6)
LOCAL_OK = MachineState(True, "LM Studio has it", "loaded")
PUBLIC = PublicCheck(True, "public", "o/r", "main")


def choose(budget=MEDIUM, profile=BOTH, config=CONFIG):
    return plan_rungs(MECH, profile, budget, LOCAL_OK, SpendState(cap=2.0), config=config, public=PUBLIC,
                      has_key=lambda r: True)


def failed(rung, model=None):
    return Tried(f"{rung}:{model or ''}", rung, model, counted=True, failed=True, moved_up=False)


def moved(rung, model=None):
    return Tried(f"{rung}:{model or ''}", rung, model, counted=False, failed=False, moved_up=True)


def below(history):
    return sum(1 for t in history if t.counted and t.rung != "claude")


def run_out(choice, budget, config, *, max_attempts=3, extra_claude_tries=0, history=None, outcome=failed):
    """Walk the ladder as the runner does, every attempt ending with `outcome`, until no rung is left."""
    history = list(history or [])
    for _ in range(20):
        found = next_rung(choice, history, budget, max_attempts=max_attempts, config=config,
                          extra_claude_tries=extra_claude_tries)
        if found.rung is None:
            return history, found
        history.append(outcome(found.rung.name, found.rung.model))
    raise AssertionError("the walk never ran out of rungs")


# 1. The cap on attempts below Claude


def test_after_claudes_turn_no_fourth_attempt_runs_below_claude():
    # Medium, mechanical: free, local, kimi, qwen, deepseek, claude. After three cheap failures Claude
    # takes its turn and fails; before the fix qwen then ran as a fourth attempt below Claude.
    choice = choose()
    history = [failed("free", FREE_ID), failed("local", LOCAL_ID), failed("kimi", KIMI_ID), failed("claude")]
    found = next_rung(choice, history, MEDIUM, max_attempts=3, config=CONFIG)
    assert found.rung is None and "ran out of rungs" in found.reason
    assert any("3 attempts below Claude already" in why for _, why in found.skipped)


def test_an_unblocked_step_gives_claude_the_extra_try_not_a_cheaper_rung():
    choice = choose()
    history = [failed("free", FREE_ID), failed("local", LOCAL_ID), failed("kimi", KIMI_ID), failed("claude")]
    found = next_rung(choice, history, MEDIUM, max_attempts=3, config=CONFIG, extra_claude_tries=1)
    assert found.rung is not None and found.rung.name == "claude"


def test_with_claude_in_the_middle_the_paid_rungs_after_it_share_the_same_cap():
    # Low, mechanical: free, local, claude, kimi, qwen, deepseek.
    history, _ = run_out(choose(LOW), LOW, CONFIG)
    assert [t.rung for t in history] == ["free", "local", "claude", "kimi"]
    assert below(history) == 3


@pytest.mark.parametrize("budget", [LOW, MEDIUM, BudgetState("high", 0.9)])
@pytest.mark.parametrize("max_attempts", [1, 3, 6])
@pytest.mark.parametrize("extra", [0, 2])
def test_no_walk_ever_counts_more_attempts_below_claude_than_the_cap(budget, max_attempts, extra):
    for cap in (0, 1, 3):
        config = parse_config({**RAW, "max_below_claude": cap})
        history, _ = run_out(choose(budget, config=config), budget, config, max_attempts=max_attempts,
                             extra_claude_tries=extra)
        assert below(history) <= cap


def test_without_claude_on_the_ladder_the_cap_still_holds():
    choice = choose(profile=OPENCODE_ONLY)
    assert "claude" not in [r.name for r in choice.rungs]
    history, found = run_out(choice, MEDIUM, CONFIG)
    assert below(history) == 3 and found.rung is None
    assert any("no more below Claude" in why for _, why in found.skipped)


def test_attempts_that_did_not_count_leave_the_allowance_alone():
    # Provider errors on free and local move the step up without using any of the three attempts.
    choice = choose()
    history, _ = run_out(choice, MEDIUM, CONFIG, history=[moved("free", FREE_ID), moved("local", LOCAL_ID)])
    assert [t.rung for t in history if t.counted] == ["kimi", "openrouter", "openrouter", "claude"]
    assert below(history) == 3


def test_the_pick_that_reaches_the_cap_is_final_when_claude_has_no_tries_left():
    config = parse_config({**RAW, "max_below_claude": 1})
    choice = choose(LOW, config=config)  # free, local, claude, kimi, ...
    found = next_rung(choice, [failed("claude")], LOW, max_attempts=1, config=config)
    assert found.rung is not None and found.rung.name == "free" and found.final


def test_night_mode_still_raises_the_cap():
    settings = NightSettings()
    night = apply_night(CONFIG, NightState(on=True, until=1e10), settings)
    assert night.max_below_claude == CONFIG.max_below_claude + settings.extra_below_claude
    history, _ = run_out(choose(), MEDIUM, night)
    assert below(history) == min(night.max_below_claude, 5)  # five rungs below Claude are configured


# 2. OpenRouter privacy follows where the requests go


@pytest.mark.parametrize("rung", [
    Rung("openrouter", QWEN),
    Rung("kimi", "openrouter/moonshotai/kimi-k2.7-code"),  # a kimi rung pointed at OpenRouter
    Rung("helper", "openrouter/z-ai/glm-5.3"),  # any future rung name
    Rung("openrouter", "openrouter/x/y:free"),  # a :free model on a rung without the public check
])
def test_every_rung_that_sends_to_openrouter_denies_data_collection(rung):
    fields = LadderConfig().request_fields(rung)
    assert fields is not None and fields["provider"]["data_collection"] == "deny"
    assert fields["provider"]["quantizations"] == ["fp8", "fp16", "bf16", "fp32"]


@pytest.mark.parametrize("rung", [
    Rung("kimi", KIMI_ID),  # Moonshot direct
    Rung("local", LOCAL_ID, upstream="http://127.0.0.1:1234/v1"),
    Rung("claude"),
    Rung("free", FREE_ID),  # public code only: no filter that could leave a free model without a host
])
def test_rungs_that_do_not_send_to_openrouter_or_only_send_public_code_carry_no_routing(rung):
    assert LadderConfig().request_fields(rung) is None


def test_the_user_cannot_turn_data_collection_back_on_for_any_openrouter_rung():
    config = parse_config({"routing": {"openrouter/moonshotai/kimi-k2.7-code": {"data_collection": "allow"}}})
    fields = config.request_fields(Rung("kimi", "openrouter/moonshotai/kimi-k2.7-code"))
    assert fields["provider"]["data_collection"] == "deny"


# 3. No undeclared quantisation by default, but configurable


def test_unknown_quantisation_is_refused_by_default():
    assert DEFAULT_OPENROUTER_ROUTING["quantizations"] == ["fp8", "fp16", "bf16", "fp32"]
    for refused in ("unknown", "fp4", "fp6", "int4", "int8"):
        assert refused not in DEFAULT_OPENROUTER_ROUTING["quantizations"]


def test_a_user_may_accept_unknown_quantisation_per_model():
    wanted = ["fp8", "fp16", "bf16", "fp32", "unknown"]
    config = parse_config({"routing": {QWEN: {"quantizations": wanted}}})
    assert not config.problems
    assert config.request_fields(Rung("openrouter", QWEN))["provider"]["quantizations"] == wanted
    # Other models keep the default.
    assert "unknown" not in config.request_fields(Rung("openrouter", DEEPSEEK))["provider"]["quantizations"]
