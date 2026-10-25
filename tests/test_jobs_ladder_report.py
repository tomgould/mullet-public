"""Tests for mullet.jobs.ladder_report.

These tests build a ledger by hand using the same API the runner uses, with no
models, providers or network calls.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from mullet.jobs.ladder import LadderConfig
from mullet.jobs.ladder_report import DEFAULT_CLAUDE_MODEL, build
from mullet.jobs.ledger import Ledger
from mullet.jobs.types import GateResult, GateStatus, JobStatus, NodeSpec


class _Clock:
    """A test clock that only moves when asked, so wall times are exact."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, delta: float) -> None:
        self.t += delta


def _make_config() -> LadderConfig:
    """A config that prices the Kimi model used in the tests."""
    config = LadderConfig()
    config.prices["moonshotai/kimi-k2.7-code"] = {"input": 0.6, "output": 2.5}
    return config


def _pass_all(ledger: Ledger, attempt_id: int) -> None:
    for gate in ("G1", "G2", "G3", "G4", "G5", "G6"):
        ledger.record_gate(attempt_id, GateResult(gate=gate, status=GateStatus.PASS, summary="ok"))


def _start_and_finish(
    ledger: Ledger,
    job_id: str,
    node_id: str,
    executor: str,
    usage: dict[str, Any] | None,
    *,
    pass_attempt: bool = True,
    resumed: bool = False,
    clock: _Clock | None = None,
    wall_seconds: float = 10.0,
) -> int:
    """Start an attempt, record usage, pass or fail it, and finish it."""
    if clock is None:
        clock = _Clock()
    attempt_id = ledger.start_attempt(job_id, node_id, executor, "base")
    ledger.record_executor_result(attempt_id, 0, usage, None, None)
    if pass_attempt:
        _pass_all(ledger, attempt_id)
    else:
        ledger.record_gate(attempt_id, GateResult(gate="G1", status=GateStatus.FAIL, summary="nope"))
    if clock:
        clock.advance(wall_seconds)
    end_sha = f"end-{attempt_id}"
    ledger.finish_attempt(attempt_id, end_sha)
    return attempt_id


def _abort_provider_error(
    ledger: Ledger,
    job_id: str,
    node_id: str,
    executor: str,
    usage: dict[str, Any] | None,
    clock: _Clock | None = None,
) -> int:
    attempt_id = ledger.start_attempt(job_id, node_id, executor, "base")
    ledger.record_executor_result(attempt_id, None, usage, None, None)
    if clock:
        clock.advance(5.0)
    ledger.abort_attempt(attempt_id, "provider error from OpenRouter")
    return attempt_id


def _setup_ladder_job(ledger: Ledger, clock: _Clock, budget: dict[str, Any]) -> str:
    job_id = ledger.create_job("/tmp/repo", "test ladder job", "main", "abc123", {}, budget=budget)
    nodes = [
        (NodeSpec(id="s1", goal="mechanical step", kind="mechanical"), None),
        (NodeSpec(id="s2", goal="design step", kind="design"), None),
        (NodeSpec(id="s3", goal="analysis step", kind="analysis"), None),
        (NodeSpec(id="s4", goal="mechanical step with provider error", kind="mechanical"), None),
        (NodeSpec(id="s5", goal="design step with token pricing", kind="design"), None),
        (NodeSpec(id="s6", goal="analysis escalation", kind="analysis"), None),
        (NodeSpec(id="p1", goal="parent integration"), None),
        (NodeSpec(id="c1", goal="child mechanical", kind="mechanical"), "p1"),
        (NodeSpec(id="s7", goal="pre-ladder analysis", kind="analysis"), None),
    ]
    ledger.set_plan(job_id, nodes)
    ledger.set_job_status(job_id, JobStatus.AWAITING_APPROVAL)
    ledger.approve(job_id)
    ledger.set_job_status(job_id, JobStatus.RUNNING)
    return job_id


def _fill_ladder_job(ledger: Ledger, job_id: str, clock: _Clock) -> None:
    free_model = "openrouter/qwen/qwen3-coder:free"
    kimi_model = "moonshotai/kimi-k2.7-code"

    # s1: mechanical, free fails, Kimi passes (escalation free -> kimi).
    _start_and_finish(
        ledger, job_id, "s1", "opencode",
        {"rung": "free", "model": free_model},
        pass_attempt=False, clock=clock, wall_seconds=5.0,
    )
    _start_and_finish(
        ledger, job_id, "s1", "opencode",
        {"rung": "kimi", "model": kimi_model, "tokens": {"input": 500_000, "output": 100_000}},
        pass_attempt=True, clock=clock, wall_seconds=10.0,
    )

    # s2: design, Kimi passes on a resumed session.
    _start_and_finish(
        ledger, job_id, "s2", "opencode",
        {"rung": "kimi", "model": kimi_model, "resumed": True,
         "tokens": {"input": 400_000, "output": 80_000}},
        pass_attempt=True, clock=clock, wall_seconds=8.0,
    )

    # s3: analysis, Claude passes first try.
    _start_and_finish(
        ledger, job_id, "s3", "claude-code",
        {"rung": "claude"},
        pass_attempt=True, clock=clock, wall_seconds=15.0,
    )

    # s4: mechanical, free aborts with a provider error.
    _abort_provider_error(
        ledger, job_id, "s4", "opencode",
        {"rung": "free", "model": free_model},
        clock=clock,
    )

    # s5: design, Kimi with cost_reported true and total_cost_usd 0; dollars come from tokens.
    _start_and_finish(
        ledger, job_id, "s5", "opencode",
        {"rung": "kimi", "model": kimi_model, "cost_reported": True, "total_cost_usd": 0,
         "tokens": {"input": 1_000_000, "output": 100_000}},
        pass_attempt=True, clock=clock, wall_seconds=12.0,
    )

    # s6: analysis, free fails, Kimi fails, Claude passes (escalations free -> kimi and kimi -> claude).
    _start_and_finish(
        ledger, job_id, "s6", "opencode",
        {"rung": "free", "model": free_model},
        pass_attempt=False, clock=clock, wall_seconds=4.0,
    )
    _start_and_finish(
        ledger, job_id, "s6", "opencode",
        {"rung": "kimi", "model": kimi_model, "tokens": {"input": 200_000, "output": 50_000}},
        pass_attempt=False, clock=clock, wall_seconds=9.0,
    )
    _start_and_finish(
        ledger, job_id, "s6", "claude-code",
        {"rung": "claude"},
        pass_attempt=True, clock=clock, wall_seconds=20.0,
    )

    # p1: parent integration attempt on Claude.
    _start_and_finish(
        ledger, job_id, "p1", "integration-check",
        {"rung": "claude"},
        pass_attempt=True, clock=clock, wall_seconds=6.0,
    )

    # c1: child mechanical, free fails, Kimi passes.
    _start_and_finish(
        ledger, job_id, "c1", "opencode",
        {"rung": "free", "model": free_model},
        pass_attempt=False, clock=clock, wall_seconds=3.0,
    )
    _start_and_finish(
        ledger, job_id, "c1", "opencode",
        {"rung": "kimi", "model": kimi_model, "tokens": {"input": 300_000, "output": 60_000}},
        pass_attempt=True, clock=clock, wall_seconds=7.0,
    )

    # s7: pre-ladder analysis attempt (no rung), executor claude-code passes.
    _start_and_finish(
        ledger, job_id, "s7", "claude-code",
        {},
        pass_attempt=True, clock=clock, wall_seconds=11.0,
    )

    ledger.add_event(job_id, "claude_usage", {"role": "planner", "num_turns": 3})
    ledger.add_event(job_id, "claude_usage", {"role": "reviewer", "num_turns": 2})


def _setup_non_ladder_job(ledger: Ledger, clock: _Clock) -> str:
    job_id = ledger.create_job("/tmp/repo", "non-ladder job", "main", "abc123", {}, budget=None)
    nodes = [(NodeSpec(id="n1", goal="plain step", kind="mechanical"), None)]
    ledger.set_plan(job_id, nodes)
    ledger.set_job_status(job_id, JobStatus.AWAITING_APPROVAL)
    ledger.approve(job_id)
    ledger.set_job_status(job_id, JobStatus.RUNNING)
    _start_and_finish(
        ledger, job_id, "n1", "opencode",
        {"rung": "kimi", "model": "moonshotai/kimi-k2.7-code"},
        pass_attempt=True, clock=clock,
    )
    return job_id


@pytest.fixture
def report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    clock = _Clock(start=1_000.0)
    monkeypatch.setattr("mullet.jobs.ledger._now", clock)

    ledger = Ledger(tmp_path / "ledger.db")
    _setup_non_ladder_job(ledger, clock)
    clock.advance(100.0)
    job_id = _setup_ladder_job(ledger, clock, budget={"ladder": True})
    _fill_ladder_job(ledger, job_id, clock)

    config = _make_config()
    return build(ledger, config=config), ledger, clock


def test_totals(report: Any) -> None:
    rep, *_ = report
    assert rep.jobs == 1
    assert rep.steps == 8
    assert rep.steps_passed_below_claude == 5
    assert rep.claude_worker_attempts == 4
    assert rep.claude_planner_sessions == 1
    assert rep.claude_reviewer_sessions == 1


def test_dollars_by_provider(report: Any) -> None:
    rep, *_ = report
    # Only paid rungs count; only the moonshotai Kimi model costs anything here.
    assert rep.dollars_by_provider == {"moonshotai": pytest.approx(2.415)}


def test_per_rung(report: Any) -> None:
    rep, *_ = report
    free = rep.per_rung["free"]
    assert free.attempts == 4
    assert free.steps_first_tried == 4
    assert free.steps_passed == 0
    assert free.first_attempt_passes == 0
    assert free.resumed_passes == 0
    assert free.escalations_up == 3
    assert free.provider_errors == 1
    assert free.dollars == 0.0
    assert free.tokens_in == 0
    assert free.tokens_out == 0
    assert free.wall_seconds == pytest.approx(5.0 + 5.0 + 4.0 + 3.0)

    kimi = rep.per_rung["kimi"]
    assert kimi.attempts == 5
    assert kimi.steps_first_tried == 2
    assert kimi.steps_passed == 4
    assert kimi.first_attempt_passes == 2
    assert kimi.resumed_passes == 1
    assert kimi.escalations_up == 1
    assert kimi.provider_errors == 0
    assert kimi.dollars == pytest.approx(2.415)
    assert kimi.tokens_in == 2_400_000
    assert kimi.tokens_out == 390_000
    assert kimi.wall_seconds == pytest.approx(10.0 + 8.0 + 12.0 + 9.0 + 7.0)

    claude = rep.per_rung["claude"]
    assert claude.attempts == 3
    assert claude.steps_first_tried == 1
    assert claude.steps_passed == 2
    assert claude.first_attempt_passes == 1
    assert claude.resumed_passes == 0
    assert claude.escalations_up == 0
    assert claude.provider_errors == 0
    assert claude.dollars == 0.0
    assert claude.wall_seconds == pytest.approx(15.0 + 20.0 + 6.0)

    pre = rep.per_rung["claude-code"]
    assert pre.attempts == 1
    assert pre.steps_first_tried == 1
    assert pre.steps_passed == 1
    assert pre.first_attempt_passes == 1
    assert pre.wall_seconds == pytest.approx(11.0)


def test_per_kind(report: Any) -> None:
    rep, *_ = report
    mechanical = rep.per_kind["mechanical"]
    assert mechanical.attempts == 5
    assert mechanical.steps_first_tried == 3
    assert mechanical.steps_passed == 2
    assert mechanical.first_attempt_passes == 0
    assert mechanical.resumed_passes == 0
    assert mechanical.escalations_up == 2
    assert mechanical.provider_errors == 1
    assert mechanical.dollars == pytest.approx(0.88)
    assert mechanical.tokens_in == 800_000
    assert mechanical.tokens_out == 160_000
    assert mechanical.wall_seconds == pytest.approx(5.0 + 10.0 + 5.0 + 3.0 + 7.0)

    design = rep.per_kind["design"]
    assert design.attempts == 2
    assert design.steps_first_tried == 2
    assert design.steps_passed == 2
    assert design.first_attempt_passes == 2
    assert design.resumed_passes == 1
    assert design.escalations_up == 0
    assert design.provider_errors == 0
    assert design.dollars == pytest.approx(1.29)
    assert design.tokens_in == 1_400_000
    assert design.tokens_out == 180_000
    assert design.wall_seconds == pytest.approx(8.0 + 12.0)

    analysis = rep.per_kind["analysis"]
    assert analysis.attempts == 5
    assert analysis.steps_first_tried == 3
    assert analysis.steps_passed == 3
    assert analysis.first_attempt_passes == 2
    assert analysis.resumed_passes == 0
    assert analysis.escalations_up == 2
    assert analysis.provider_errors == 0
    assert analysis.dollars == pytest.approx(0.245)
    assert analysis.tokens_in == 200_000
    assert analysis.tokens_out == 50_000
    assert analysis.wall_seconds == pytest.approx(15.0 + 4.0 + 9.0 + 20.0 + 11.0)

    integration = rep.per_kind["integration"]
    assert integration.attempts == 1
    assert integration.steps_first_tried == 0
    assert integration.steps_passed == 0
    assert integration.first_attempt_passes == 0
    assert integration.wall_seconds == pytest.approx(6.0)


def test_per_model(report: Any) -> None:
    rep, *_ = report
    free_model = "openrouter/qwen/qwen3-coder:free"
    kimi_model = "moonshotai/kimi-k2.7-code"

    assert rep.per_model[free_model].attempts == 4
    assert rep.per_model[free_model].passes == 0
    assert rep.per_model[free_model].dollars == 0.0

    assert rep.per_model[kimi_model].attempts == 5
    assert rep.per_model[kimi_model].passes == 4
    assert rep.per_model[kimi_model].dollars == pytest.approx(2.415)


def test_since_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock(start=1_000.0)
    monkeypatch.setattr("mullet.jobs.ledger._now", clock)
    ledger = Ledger(tmp_path / "ledger.db")

    # First (non-ladder) job is created early.
    _setup_non_ladder_job(ledger, clock)
    clock.advance(100.0)

    # Second (ladder) job is created later.
    ladder_job_id = _setup_ladder_job(ledger, clock, budget={"ladder": True})
    ladder_created = ledger.get_job(ladder_job_id).created_at
    _fill_ladder_job(ledger, ladder_job_id, clock)

    config = _make_config()

    # since below the ladder job includes it.
    rep = build(ledger, since=ladder_created - 1.0, config=config)
    assert rep.jobs == 1
    assert rep.steps == 8

    # since at the ladder job includes it.
    rep = build(ledger, since=ladder_created, config=config)
    assert rep.jobs == 1

    # since above the ladder job excludes it.
    rep = build(ledger, since=ladder_created + 1.0, config=config)
    assert rep.jobs == 0
    assert rep.steps == 0


def test_claude_avoided(report: Any) -> None:
    rep, *_ = report
    avoided = rep.claude_avoided
    # Winners below Claude: s1, s2, s5 and c1 on Kimi, s7 pre-ladder with no tokens.
    assert avoided.tokens_in == 500_000 + 400_000 + 1_000_000 + 300_000
    assert avoided.tokens_out == 100_000 + 80_000 + 100_000 + 60_000
    # claude-sonnet-5 in pricing.yaml: $0.002 in and $0.01 out per 1K tokens.
    assert avoided.model == DEFAULT_CLAUDE_MODEL == "claude-sonnet-5"
    assert avoided.input_per_1k == pytest.approx(0.002)
    assert avoided.output_per_1k == pytest.approx(0.01)
    assert avoided.estimated_dollars == pytest.approx(2_200 * 0.002 + 340 * 0.01)
    assert avoided.estimated_dollars == pytest.approx(7.8)
    assert avoided.is_estimate is True
    assert "estimate" in avoided.note
    assert "allowance not used" in avoided.note


def test_claude_avoided_falls_back_without_pricing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mullet.core.providers import pricing

    monkeypatch.setattr(pricing, "load_pricing", lambda: {})
    ledger = _rules_ledger(tmp_path, monkeypatch)
    _rules_job(ledger, [("design", [_try("kimi", KIMI, True, tokens={"input": 1_000, "output": 1_000})])])
    avoided = build(ledger, config=_make_config()).claude_avoided
    assert (avoided.input_per_1k, avoided.output_per_1k) == (0.002, 0.01)
    assert avoided.estimated_dollars == pytest.approx(0.012)


def test_fixture_has_no_changes(report: Any) -> None:
    rep, *_ = report
    assert rep.changes == []


# "What to change" rules. Each gets a small ledger where it fires and one just
# under its data threshold where it stays quiet.

KIMI = "moonshotai/kimi-k2.7-code"
FREE = "openrouter/qwen/qwen3-coder:free"
FREE_OTHER = "openrouter/other/coder:free"
EXPENSIVE = "openrouter/expensive-model"  # unpriced: the $2/$8 per million fallback


def _try(rung: str, model: str | None, passed: bool, *, tokens: dict[str, int] | None = None,
         gates: dict[str, GateStatus] | None = None) -> dict[str, Any]:
    return {"rung": rung, "model": model, "passed": passed, "tokens": tokens, "gates": gates}


def _rules_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Ledger:
    monkeypatch.setattr("mullet.jobs.ledger._now", _Clock(start=1_000.0))
    return Ledger(tmp_path / "ledger.db")


def _rules_job(ledger: Ledger, steps: list[tuple[str, list[dict[str, Any]]]]) -> str:
    """A ladder job with one leaf per (kind, attempts) entry."""
    job_id = ledger.create_job("/tmp/repo", "rules job", "main", "abc123", {}, budget={"ladder": True})
    ledger.set_plan(job_id, [(NodeSpec(id=f"s{i}", goal=f"step {i}", kind=kind), None)
                             for i, (kind, _) in enumerate(steps)])
    ledger.set_job_status(job_id, JobStatus.AWAITING_APPROVAL)
    ledger.approve(job_id)
    ledger.set_job_status(job_id, JobStatus.RUNNING)
    for i, (_, tries) in enumerate(steps):
        for t in tries:
            usage: dict[str, Any] = {"rung": t["rung"]}
            if t["model"]:
                usage["model"] = t["model"]
            if t["tokens"]:
                usage["tokens"] = t["tokens"]
            executor = "claude-code" if t["rung"] == "claude" else "opencode"
            attempt_id = ledger.start_attempt(job_id, f"s{i}", executor, "base")
            ledger.record_executor_result(attempt_id, 0, usage, None, None)
            if t["gates"]:
                for gate, status in t["gates"].items():
                    ledger.record_gate(attempt_id, GateResult(gate=gate, status=status, summary="x"))
            elif t["passed"]:
                _pass_all(ledger, attempt_id)
            else:
                ledger.record_gate(attempt_id, GateResult(gate="G1", status=GateStatus.FAIL, summary="nope"))
            ledger.finish_attempt(attempt_id, f"end-{attempt_id}")
    return job_id


def _changes(ledger: Ledger) -> list[str]:
    return build(ledger, config=_make_config()).changes


def test_drop_rule_fires_per_rung_and_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = _rules_ledger(tmp_path, monkeypatch)
    # FREE fails 5 times; FREE_OTHER on the same rung passes, which must not hide FREE.
    steps = [("mechanical", [_try("free", FREE, False), _try("free", FREE_OTHER, True)]) for _ in range(5)]
    _rules_job(ledger, steps)
    assert _changes(ledger) == [f"consider dropping free/{FREE} (5 attempts, 0 passes)"]


def test_drop_rule_quiet_under_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = _rules_ledger(tmp_path, monkeypatch)
    steps = [("mechanical", [_try("free", FREE, False), _try("free", FREE_OTHER, True)]) for _ in range(4)]
    _rules_job(ledger, steps)
    assert _changes(ledger) == []


def test_start_higher_fires(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = _rules_ledger(tmp_path, monkeypatch)
    steps = [("design", [_try("claude", None, True)]) for _ in range(4)]
    steps.append(("design", [_try("kimi", KIMI, True)]))
    _rules_job(ledger, steps)
    assert _changes(ledger) == ["start design steps higher (4 of 5 steps reached Claude)"]


def test_start_higher_quiet_under_step_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = _rules_ledger(tmp_path, monkeypatch)
    # Four steps, every one reaching Claude: too few steps to say anything.
    _rules_job(ledger, [("design", [_try("claude", None, True)]) for _ in range(4)])
    assert _changes(ledger) == []


def test_start_higher_quiet_at_ratio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = _rules_ledger(tmp_path, monkeypatch)
    # 3 of 5 is exactly 60%, which is not more than 60%.
    steps = [("design", [_try("claude", None, True)]) for _ in range(3)]
    steps += [("design", [_try("kimi", KIMI, True)]) for _ in range(2)]
    _rules_job(ledger, steps)
    assert _changes(ledger) == []


def test_prefer_cheaper_fires(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = _rules_ledger(tmp_path, monkeypatch)
    _rules_job(ledger, [
        # Kimi: 200K in at $0.6/M + 40K out at $2.5/M = $0.22 per pass.
        ("design", [_try("kimi", KIMI, True, tokens={"input": 200_000, "output": 40_000})]),
        # Fallback price: 100K in at $2/M + 50K out at $8/M = $0.60 per pass.
        ("design", [_try("openrouter", EXPENSIVE, True, tokens={"input": 100_000, "output": 50_000})]),
        # A free model passing at $0 is not a paid model and must not be the cheapest.
        ("design", [_try("free", FREE, True, tokens={"input": 100_000, "output": 50_000})]),
    ])
    assert _changes(ledger) == [f"prefer {KIMI} over {EXPENSIVE} ($0.22 vs $0.60 per pass)"]


def test_prefer_cheaper_quiet_under_ratio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = _rules_ledger(tmp_path, monkeypatch)
    _rules_job(ledger, [
        ("design", [_try("kimi", KIMI, True, tokens={"input": 200_000, "output": 40_000})]),
        # 100K in at $2/M + 25K out at $8/M = $0.40 per pass, under twice $0.22.
        ("design", [_try("openrouter", EXPENSIVE, True, tokens={"input": 100_000, "output": 25_000})]),
        ("design", [_try("free", FREE, True)]),
    ])
    assert _changes(ledger) == []


def test_prefer_cheaper_quiet_with_one_passing_paid_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = _rules_ledger(tmp_path, monkeypatch)
    _rules_job(ledger, [
        ("design", [_try("openrouter", EXPENSIVE, False, tokens={"input": 1_000, "output": 1_000}),
                    _try("kimi", KIMI, True, tokens={"input": 200_000, "output": 40_000})]),
        ("design", [_try("free", FREE, True)]),
    ])
    assert _changes(ledger) == []


def _g3_fail() -> dict[str, GateStatus]:
    return {"G1": GateStatus.PASS, "G2": GateStatus.PASS, "G3": GateStatus.FAIL}


def test_gate_rule_fires_across_cheap_rungs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = _rules_ledger(tmp_path, monkeypatch)
    # 3 on free and 3 on Kimi: neither reaches 5 alone, together they do.
    steps = [("mechanical", [_try("free", FREE, False, gates=_g3_fail()), _try("kimi", KIMI, True)])
             for _ in range(3)]
    steps += [("mechanical", [_try("kimi", KIMI, False, gates=_g3_fail()), _try("claude", None, True)])
              for _ in range(3)]
    _rules_job(ledger, steps)
    assert _changes(ledger) == ["look at why cheap attempts fail G3 (6 of 9)"]


def test_gate_rule_quiet_under_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = _rules_ledger(tmp_path, monkeypatch)
    # 4 cheap attempts fail G3; Claude's G3 failures don't count; G3 not run doesn't count.
    steps = [("mechanical", [_try("free", FREE, False, gates=_g3_fail()), _try("claude", None, True)])
             for _ in range(4)]
    steps += [("analysis", [_try("claude", None, False, gates=_g3_fail()), _try("claude", None, True)])
              for _ in range(3)]
    steps.append(("design", [_try("free", FREE_OTHER, False,
                                      gates={"G1": GateStatus.FAIL, "G3": GateStatus.SKIPPED})]))
    _rules_job(ledger, steps)
    assert _changes(ledger) == []


# Renderers.

import json  # noqa: E402
import re  # noqa: E402

from mullet.cli.ladder_cli import cmd_ladder  # noqa: E402
from mullet.jobs.ladder_report import render_json, render_text  # noqa: E402


def _table_lines(text: str, title: str) -> list[str]:
    """The header and rows of the table under `title`, up to the next blank line."""
    lines = text.splitlines()
    start = lines.index(title) + 1
    end = lines.index("", start)
    return lines[start:end]


def _cell_spans(line: str) -> list[tuple[int, int]]:
    # Columns are at least two spaces apart; a cell may hold single spaces.
    return [m.span() for m in re.finditer(r"\S+(?: \S+)*", line)]


def _assert_aligned(rows: list[str]) -> None:
    header = _cell_spans(rows[0])
    for row in rows[1:]:
        spans = _cell_spans(row)
        assert len(spans) == len(header), row
        assert spans[0][0] == header[0][0] == 0, row  # names left-aligned
        assert [s[1] for s in spans[1:]] == [h[1] for h in header[1:]], row  # numbers right-aligned


def test_render_json_shape_and_values(report: Any) -> None:
    rep, *_ = report
    data = json.loads(json.dumps(render_json(rep)))
    assert set(data) >= {"since", "totals", "rungs", "kinds", "models", "claude_avoided", "changes"}
    assert data["since"] is None
    assert data["totals"]["jobs"] == 1
    assert data["totals"]["steps"] == 8
    assert data["totals"]["steps_passed_below_claude"] == 5
    assert data["totals"]["claude_worker_attempts"] == 4
    assert data["totals"]["claude_planner_sessions"] == 1
    assert data["totals"]["claude_reviewer_sessions"] == 1
    assert data["totals"]["dollars_by_provider"] == {"moonshotai": pytest.approx(2.415)}

    assert list(data["rungs"]) == ["free", "kimi", "claude", "claude-code"]
    kimi = data["rungs"]["kimi"]
    assert kimi == {
        "attempts": 5, "steps_first_tried": 2, "steps_passed": 4, "first_attempt_passes": 2,
        "resumed_passes": 1, "escalations_up": 1, "provider_errors": 0,
        "dollars": pytest.approx(2.415), "tokens_in": 2_400_000, "tokens_out": 390_000,
        "wall_seconds": pytest.approx(46.0),
    }
    assert data["rungs"]["free"]["provider_errors"] == 1
    assert data["rungs"]["free"]["escalations_up"] == 3

    assert list(data["kinds"]) == ["mechanical", "design", "analysis", "integration"]
    assert data["kinds"]["design"]["resumed_passes"] == 1
    assert data["kinds"]["mechanical"]["dollars"] == pytest.approx(0.88)

    assert data["models"] == {
        FREE: {"attempts": 4, "passes": 0, "dollars": 0.0},
        KIMI: {"attempts": 5, "passes": 4, "dollars": pytest.approx(2.415)},
    }

    avoided = data["claude_avoided"]
    assert avoided["estimate"] is True
    assert avoided["model"] == DEFAULT_CLAUDE_MODEL
    assert avoided["tokens_in"] == 2_200_000
    assert avoided["tokens_out"] == 340_000
    assert avoided["estimated_dollars"] == pytest.approx(7.8)
    assert "allowance not used" in avoided["note"]
    assert data["changes"] == []


def test_render_json_carries_since_and_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = _rules_ledger(tmp_path, monkeypatch)
    _rules_job(ledger, [("mechanical", [_try("free", FREE, False), _try("free", FREE_OTHER, True)])
                        for _ in range(5)])
    data = json.loads(json.dumps(render_json(build(ledger, since=500.0, config=_make_config()))))
    assert data["since"] == 500.0
    assert data["totals"]["jobs"] == 1
    assert data["changes"] == [f"consider dropping free/{FREE} (5 attempts, 0 passes)"]


def test_render_text_tables(report: Any) -> None:
    rep, *_ = report
    text = render_text(rep)
    assert "—" not in text
    assert text.splitlines()[0] == "Ladder report across all jobs: 1 job, 8 steps, 5 passed below Claude"
    for title in ("Per rung", "Per step kind", "Per model", "Dollars by provider", "Claude sessions"):
        rows = _table_lines(text, title)
        assert len(rows) >= 2, title
        _assert_aligned(rows)

    rungs = _table_lines(text, "Per rung")
    assert rungs[0].split()[:2] == ["rung", "attempts"]
    assert [r.split()[0] for r in rungs[1:]] == ["free", "kimi", "claude", "claude-code"]
    assert rungs[2].split()[:8] == ["kimi", "5", "2", "4", "2", "1", "1", "0"]
    assert "$2.42" in rungs[2]
    assert "46s" in rungs[2]

    kinds = _table_lines(text, "Per step kind")
    assert [r.split()[0] for r in kinds[1:]] == ["mechanical", "design", "analysis", "integration"]

    models = _table_lines(text, "Per model")
    assert models[1].split() == [KIMI, "5", "4", "$2.42"]
    assert models[2].split() == [FREE, "4", "0", "$0.00"]

    providers = _table_lines(text, "Dollars by provider")
    assert providers[1].split() == ["moonshotai", "$2.42"]

    sessions = _table_lines(text, "Claude sessions")
    assert [s.split() for s in sessions[1:]] == [
        ["worker", "attempts", "4"], ["planner", "sessions", "1"], ["reviewer", "sessions", "1"],
    ]


def test_render_text_avoided_and_changes(report: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rep, *_ = report
    text = render_text(rep)
    assert "Claude work avoided (an estimate at Claude API prices" in text
    assert "on a subscription it isn't billed, it is allowance not used" in text
    assert "2,200,000 in, 340,000 out" in text
    assert "estimated Claude cost: $7.80" in text
    assert "What to change" in text
    assert "Not enough evidence yet" in text

    rules = tmp_path / "rules"
    rules.mkdir()
    ledger = _rules_ledger(rules, monkeypatch)
    _rules_job(ledger, [("mechanical", [_try("free", FREE, False), _try("free", FREE_OTHER, True)])
                        for _ in range(5)])
    text = render_text(build(ledger, since=500.0, config=_make_config()))
    assert "—" not in text
    assert text.splitlines()[0].startswith("Ladder report since ")
    lines = text.splitlines()
    assert lines[lines.index("What to change") + 1] == f"  - consider dropping free/{FREE} (5 attempts, 0 passes)"
    assert "Not enough evidence yet" not in text


def test_render_empty_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = _rules_ledger(tmp_path, monkeypatch)
    rep = build(ledger, config=_make_config())
    text = render_text(rep)
    assert "\n" not in text
    assert "No delegation ladder jobs" in text
    assert "—" not in text
    data = json.loads(json.dumps(render_json(rep)))
    assert data["totals"]["jobs"] == 0
    assert data["rungs"] == {} and data["kinds"] == {} and data["models"] == {}
    assert data["claude_avoided"]["estimate"] is True
    assert data["changes"] == []


# CLI.


def _cli_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, clock_start: float = 1_000.0) -> tuple[_Clock, float]:
    """Build a ladder job in a temporary jobs dir and return the clock and job creation time."""
    clock = _Clock(start=clock_start)
    monkeypatch.setattr("mullet.jobs.ledger._now", clock)
    monkeypatch.setenv("MULLET_JOBS_DIR", str(tmp_path))
    ledger = Ledger()
    job_id = _setup_ladder_job(ledger, clock, budget={"ladder": True})
    created_at = ledger.get_job(job_id).created_at
    _fill_ladder_job(ledger, job_id, clock)
    ledger.close()
    return clock, created_at


def test_cmd_ladder_report_text(capsys, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _cli_ledger(tmp_path, monkeypatch)
    monkeypatch.setattr("mullet.cli.ladder_cli._now", lambda: 2_000.0)
    assert cmd_ladder(["report"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.startswith("Ladder report across all jobs")
    assert "Per rung" in captured.out
    assert "Per step kind" in captured.out
    assert "Per model" in captured.out
    assert "Claude work avoided" in captured.out
    assert "What to change" in captured.out


def test_cmd_ladder_report_json(capsys, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _cli_ledger(tmp_path, monkeypatch)
    monkeypatch.setattr("mullet.cli.ladder_cli._now", lambda: 2_000.0)
    assert cmd_ladder(["report", "--json"]) == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["totals"]["jobs"] == 1
    assert data["totals"]["steps"] == 8
    assert "rungs" in data
    assert "kinds" in data
    assert "models" in data
    assert "claude_avoided" in data
    assert "changes" in data


def test_cmd_ladder_days_filter(capsys, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _cli_ledger(tmp_path, monkeypatch)
    monkeypatch.setattr("mullet.cli.ladder_cli._now", lambda: 2_000.0)
    assert cmd_ladder(["report", "--days", "1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["totals"]["jobs"] == 1

    monkeypatch.setattr("mullet.cli.ladder_cli._now", lambda: 100_000.0)
    assert cmd_ladder(["report", "--days", "1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["totals"]["jobs"] == 0


def test_cmd_ladder_since_filter(capsys, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, created_at = _cli_ledger(tmp_path, monkeypatch)
    monkeypatch.setattr("mullet.cli.ladder_cli._now", lambda: 2_000.0)
    created_date = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d")
    assert cmd_ladder(["report", "--since", created_date, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["totals"]["jobs"] == 1

    next_day = (datetime.fromtimestamp(created_at) + timedelta(days=1)).strftime("%Y-%m-%d")
    assert cmd_ladder(["report", "--since", next_day, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["totals"]["jobs"] == 0


def test_cmd_ladder_bad_since_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MULLET_JOBS_DIR", str(tmp_path))
    assert cmd_ladder(["report", "--since", "not-a-date"]) == 2


def test_cmd_ladder_since_and_days_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MULLET_JOBS_DIR", str(tmp_path))
    assert cmd_ladder(["report", "--since", "2024-01-01", "--days", "1"]) == 2
