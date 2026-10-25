"""Cross-job report of how the delegation ladder has done, from the ledger only.

The report reads the jobs ledger and never calls a model, a provider or the
network. It is a pure function of the ledger plus the ladder's pricing config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from mullet.jobs import ladder
from mullet.jobs.ladder import LadderConfig, attempt_cost
from mullet.jobs.ledger import Ledger
from mullet.jobs.text import one_line
from mullet.jobs.types import AttemptOutcome, GateStatus, NodeStatus

# The Claude model whose API prices value the Claude work avoided, and the
# price used if pricing.yaml lacks it (USD per 1K tokens, as in pricing.yaml).
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5"
CLAUDE_FALLBACK_PER_1K = {"input": 0.002, "output": 0.01}
AVOIDED_NOTE = ("estimate at Claude API prices; on a subscription it is "
                "allowance not used, not money saved")

# Rungs below Claude whose failures the gate rule looks at.
CHEAP_RUNGS = ("free", "local", "kimi", "openrouter")

# "What to change" thresholds. Below the minimum data a rule says nothing.
DROP_MIN_ATTEMPTS = 5  # a rung/model with this many attempts and no passes
START_HIGHER_MIN_STEPS = 5  # a kind needs this many steps
START_HIGHER_SHARE = 0.6  # ... and more than this share reaching Claude
PREFER_RATIO = 2.0  # a paid model's cost per pass over the cheapest's
GATE_RULE_GATES = ("G3", "G5", "G6", "G7")
GATE_MIN_ATTEMPTS = 5  # cheap-rung attempts where the gate ran
GATE_FAIL_SHARE = 0.5  # more than this share failing it


@dataclass
class RungStats:
    """Counts for one ladder rung (or a pre-ladder executor name)."""

    attempts: int = 0
    steps_first_tried: int = 0
    steps_passed: int = 0
    steps_unverified: int = 0
    first_attempt_passes: int = 0
    resumed_passes: int = 0
    escalations_up: int = 0
    provider_errors: int = 0
    dollars: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    wall_seconds: float = 0.0


@dataclass
class KindStats:
    """Counts for one step kind (mechanical, design, analysis, integration)."""

    attempts: int = 0
    steps_first_tried: int = 0
    steps_passed: int = 0
    steps_unverified: int = 0
    first_attempt_passes: int = 0
    resumed_passes: int = 0
    escalations_up: int = 0
    provider_errors: int = 0
    dollars: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    wall_seconds: float = 0.0


@dataclass
class ModelStats:
    """Counts for one model id."""

    attempts: int = 0
    passes: int = 0
    unverified: int = 0
    dollars: float = 0.0


@dataclass
class ClaudeAvoided:
    """Claude work avoided: what the steps passed below Claude would have cost on Claude.

    `tokens_in` and `tokens_out` are what the winning attempts of the steps
    passed below Claude used. `estimated_dollars` is an estimate only: those
    tokens priced at Claude API prices for `model` (DEFAULT_CLAUDE_MODEL), per
    1K tokens. On a subscription Claude isn't billed per token, so this is
    allowance not used, not money saved. Renders must show `note` beside it.
    """

    tokens_in: int = 0
    tokens_out: int = 0
    estimated_dollars: float = 0.0
    model: str = ""
    input_per_1k: float = 0.0
    output_per_1k: float = 0.0
    is_estimate: bool = True
    note: str = ""


@dataclass
class LadderReport:
    """The full cross-job ladder report."""

    jobs: int = 0
    steps: int = 0
    steps_passed_below_claude: int = 0
    steps_unverified_below_claude: int = 0
    dollars_by_provider: dict[str, float] = field(default_factory=dict)
    claude_worker_attempts: int = 0
    claude_planner_sessions: int = 0
    claude_reviewer_sessions: int = 0
    per_rung: dict[str, RungStats] = field(default_factory=dict)
    per_kind: dict[str, KindStats] = field(default_factory=dict)
    per_model: dict[str, ModelStats] = field(default_factory=dict)
    claude_avoided: ClaudeAvoided = field(default_factory=ClaudeAvoided)
    changes: list[str] = field(default_factory=list)  # "what to change", each naming its evidence
    failed_attempts: int = 0
    failures_by_gate: dict[str, int] = field(default_factory=dict)  # first failing gate -> attempts
    failure_reasons: dict[str, str] = field(default_factory=dict)  # gate -> its most common summary
    since: float | None = None  # the earliest job creation time counted, if any


REASON_LIMIT = 120


def _short(text: str) -> str:
    """A gate's own summary on one line, sanitised (it can quote a model) and cut to REASON_LIMIT characters."""
    return one_line(text, REASON_LIMIT)


def _rung_of(attempt) -> str:
    """The rung an attempt ran on, or the executor name for pre-ladder jobs."""
    usage = attempt.usage or {}
    return usage.get("rung") or attempt.executor


def _kind_of(node, is_leaf: bool) -> str:
    """The step kind for a node: leaf kind from its spec, parent as integration."""
    if is_leaf:
        return node.spec.kind
    return "integration"


def _tokens(usage: dict[str, Any] | None) -> tuple[int, int]:
    """(tokens_in, tokens_out) for an attempt's usage record."""
    if not isinstance(usage, dict):
        return 0, 0
    tokens = usage.get("tokens")
    if not isinstance(tokens, dict) or not tokens:
        return 0, 0

    def total(*names: str) -> int:
        return sum(int(tokens.get(n) or 0) for n in names if isinstance(tokens.get(n), (int, float)))

    return total("input", "cache_read", "cache_write"), total("output", "reasoning")


def _provider_of(model: str | None) -> str:
    """The provider part of a model id."""
    return str(model or "?").split("/", 1)[0]


def _claude_price(model: str = DEFAULT_CLAUDE_MODEL) -> dict[str, float]:
    """Claude API prices per 1K tokens from the local pricing.yaml, else the fallback.

    pricing.load_pricing only reads the yaml shipped beside it; nothing is fetched.
    """
    try:
        from mullet.core.providers import pricing

        entry = pricing.load_pricing().get(model) or {}
        return {"input": float(entry["input_per_1k"]), "output": float(entry["output_per_1k"])}
    except Exception:  # missing file, bad yaml or no entry: use the fixed price
        return dict(CLAUDE_FALLBACK_PER_1K)


@dataclass
class _Tally:
    """Attempts, passes and dollars, for the rules."""

    attempts: int = 0
    passes: int = 0
    dollars: float = 0.0


def _changes(by_rung_model: dict[tuple[str, str | None], _Tally],
             kind_steps: dict[str, tuple[int, int]],
             gate_runs: dict[str, tuple[int, int]]) -> list[str]:
    """The "what to change" lines from the fixed rules, each naming its evidence.

    `kind_steps` is kind -> (steps, steps that reached Claude); `gate_runs` is
    gate -> (cheap-rung attempts where it ran, of those that failed it).
    """
    out: list[str] = []

    for (rung, model), tally in sorted(by_rung_model.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        if tally.attempts >= DROP_MIN_ATTEMPTS and tally.passes == 0:
            label = f"{rung}/{model}" if model else rung
            out.append(f"consider dropping {label} ({tally.attempts} attempts, 0 passes)")

    for kind, (steps, reached) in sorted(kind_steps.items()):
        if steps >= START_HIGHER_MIN_STEPS and reached > START_HIGHER_SHARE * steps:
            out.append(f"start {kind} steps higher ({reached} of {steps} steps reached Claude)")

    paid: dict[str, _Tally] = {}
    for (rung, model), tally in by_rung_model.items():
        if rung in ladder.PAID_RUNGS and model:
            merged = paid.setdefault(model, _Tally())
            merged.attempts += tally.attempts
            merged.passes += tally.passes
            merged.dollars += tally.dollars
    per_pass = {model: t.dollars / t.passes for model, t in paid.items() if t.passes}
    if per_pass:
        cheapest = min(per_pass, key=lambda m: (per_pass[m], m))
        low = per_pass[cheapest]
        for model, cost in sorted(per_pass.items()):
            if model != cheapest and cost > PREFER_RATIO * low:
                out.append(f"prefer {cheapest} over {model} (${low:.2f} vs ${cost:.2f} per pass)")

    for gate in GATE_RULE_GATES:
        ran, failed = gate_runs.get(gate, (0, 0))
        if ran >= GATE_MIN_ATTEMPTS and failed > GATE_FAIL_SHARE * ran:
            out.append(f"look at why cheap attempts fail {gate} ({failed} of {ran})")

    return out


def build(ledger: Ledger, *, since: float | None = None, config: LadderConfig | None = None) -> LadderReport:
    """Build a cross-job ladder report from the ledger.

    Only ladder jobs are included. `since` filters by `job.created_at`.
    `config` defaults to the user's ladder config and supplies token prices.
    """
    config = config or ladder.load_config()
    report = LadderReport()
    report.since = since
    by_rung_model: dict[tuple[str, str | None], _Tally] = {}
    kind_steps: dict[str, list[int]] = {}  # kind -> [steps tried, steps that reached Claude]
    gate_runs: dict[str, list[int]] = {}  # gate -> [cheap attempts where it ran, of those failed]
    reasons: dict[str, dict[str, int]] = {}  # first failing gate -> summary -> attempts

    for job in ledger.list_jobs(limit=1_000_000):
        if not (job.budget or {}).get("ladder"):
            continue
        if since is not None and job.created_at < since:
            continue

        report.jobs += 1
        nodes = ledger.nodes(job.id)
        attempts = ledger.attempts(job.id)

        parents = {n.parent_id for n in nodes if n.parent_id}
        leaves = [n for n in nodes if n.node_id not in parents]

        report.steps += len(leaves)

        attempts_by_node: dict[str, list] = {}
        for attempt in attempts:
            attempts_by_node.setdefault(attempt.node_id, []).append(attempt)

        node_facts: dict[str, dict[str, Any]] = {}
        for node in nodes:
            node_attempts = attempts_by_node.get(node.node_id, [])
            is_leaf = node.node_id not in parents
            kind = _kind_of(node, is_leaf)

            winner = None
            if node.status in (NodeStatus.PASSED, NodeStatus.UNVERIFIED):
                winner = next(
                    (a for a in reversed(node_attempts)
                     if a.outcome in (AttemptOutcome.PASSED, AttemptOutcome.UNVERIFIED)),
                    None,
                )

            first_attempt = node_attempts[0] if node_attempts else None
            first_rung = _rung_of(first_attempt) if first_attempt else None

            node_facts[node.node_id] = {
                "kind": kind,
                "is_leaf": is_leaf,
                "winner": winner,
                "first_attempt": first_attempt,
                "first_rung": first_rung,
            }

        for node in nodes:
            facts = node_facts[node.node_id]
            kind = facts["kind"]
            node_attempts = attempts_by_node.get(node.node_id, [])

            for i, attempt in enumerate(node_attempts):
                rung = _rung_of(attempt)
                usage = attempt.usage or {}
                dollars, _ = attempt_cost(usage, config)
                tokens_in, tokens_out = _tokens(usage)
                wall = 0.0
                if attempt.started_at is not None and attempt.finished_at is not None:
                    wall = attempt.finished_at - attempt.started_at
                is_provider_error = (
                    attempt.outcome is AttemptOutcome.ABORTED
                    and attempt.reason is not None
                    and "provider error" in attempt.reason
                )
                escalated = i + 1 < len(node_attempts) and _rung_of(node_attempts[i + 1]) != rung

                rung_stats = report.per_rung.setdefault(rung, RungStats())
                rung_stats.attempts += 1
                rung_stats.dollars += dollars
                rung_stats.tokens_in += tokens_in
                rung_stats.tokens_out += tokens_out
                rung_stats.wall_seconds += wall
                if is_provider_error:
                    rung_stats.provider_errors += 1
                if escalated:
                    rung_stats.escalations_up += 1

                kind_stats = report.per_kind.setdefault(kind, KindStats())
                kind_stats.attempts += 1
                kind_stats.dollars += dollars
                kind_stats.tokens_in += tokens_in
                kind_stats.tokens_out += tokens_out
                kind_stats.wall_seconds += wall
                if is_provider_error:
                    kind_stats.provider_errors += 1
                if escalated:
                    kind_stats.escalations_up += 1

                model = usage.get("model")
                tally = by_rung_model.setdefault((rung, model), _Tally())
                tally.attempts += 1
                tally.dollars += dollars
                if attempt.outcome is AttemptOutcome.PASSED:
                    tally.passes += 1

                if attempt.outcome is AttemptOutcome.FAILED:
                    first = next((r for r in ledger.gate_results(attempt.id) if r.status is GateStatus.FAIL), None)
                    if first is not None:
                        report.failed_attempts += 1
                        report.failures_by_gate[first.gate] = report.failures_by_gate.get(first.gate, 0) + 1
                        seen_reasons = reasons.setdefault(first.gate, {})
                        key = _short(first.summary)
                        seen_reasons[key] = seen_reasons.get(key, 0) + 1

                if rung in CHEAP_RUNGS:
                    statuses: dict[str, set[GateStatus]] = {}
                    for result in ledger.gate_results(attempt.id):
                        statuses.setdefault(result.gate, set()).add(result.status)
                    for gate in GATE_RULE_GATES:
                        seen = statuses.get(gate, set())
                        if GateStatus.FAIL in seen or GateStatus.PASS in seen:
                            runs = gate_runs.setdefault(gate, [0, 0])
                            runs[0] += 1
                            if GateStatus.FAIL in seen:
                                runs[1] += 1

                if model:
                    model_stats = report.per_model.setdefault(model, ModelStats())
                    model_stats.attempts += 1
                    model_stats.dollars += dollars
                    if attempt.outcome is AttemptOutcome.PASSED:
                        model_stats.passes += 1
                    elif attempt.outcome is AttemptOutcome.UNVERIFIED:
                        model_stats.unverified += 1

                if usage.get("rung") in ladder.PAID_RUNGS:
                    provider = _provider_of(model)
                    report.dollars_by_provider[provider] = (
                        report.dollars_by_provider.get(provider, 0.0) + dollars
                    )

                if rung == "claude" or attempt.executor not in ("opencode", "integration-check"):
                    report.claude_worker_attempts += 1

        for node in leaves:
            facts = node_facts[node.node_id]
            kind = facts["kind"]
            winner = facts["winner"]
            first_attempt = facts["first_attempt"]
            first_rung = facts["first_rung"]

            if first_rung:
                report.per_rung.setdefault(first_rung, RungStats()).steps_first_tried += 1
                report.per_kind.setdefault(kind, KindStats()).steps_first_tried += 1
                counts = kind_steps.setdefault(kind, [0, 0])
                counts[0] += 1
                if any(_rung_of(a) == "claude" for a in attempts_by_node.get(node.node_id, [])):
                    counts[1] += 1

            if winner:
                winner_rung = _rung_of(winner)
                winner_usage = winner.usage or {}
                verified = winner.outcome is AttemptOutcome.PASSED
                if verified:
                    report.per_rung.setdefault(winner_rung, RungStats()).steps_passed += 1
                    report.per_kind.setdefault(kind, KindStats()).steps_passed += 1
                else:
                    report.per_rung.setdefault(winner_rung, RungStats()).steps_unverified += 1
                    report.per_kind.setdefault(kind, KindStats()).steps_unverified += 1

                if verified and winner == first_attempt:
                    report.per_rung.setdefault(winner_rung, RungStats()).first_attempt_passes += 1
                    report.per_kind.setdefault(kind, KindStats()).first_attempt_passes += 1

                if verified and winner_usage.get("resumed"):
                    report.per_rung.setdefault(winner_rung, RungStats()).resumed_passes += 1
                    report.per_kind.setdefault(kind, KindStats()).resumed_passes += 1

                if winner_rung != "claude" and not verified:
                    report.steps_unverified_below_claude += 1
                elif winner_rung != "claude":
                    report.steps_passed_below_claude += 1
                    won_in, won_out = _tokens(winner_usage)
                    report.claude_avoided.tokens_in += won_in
                    report.claude_avoided.tokens_out += won_out

        for event in ledger.events(job.id):
            if event.kind == "claude_usage":
                role = event.data.get("role")
                if role == "planner":
                    report.claude_planner_sessions += 1
                elif role == "reviewer":
                    report.claude_reviewer_sessions += 1

    avoided = report.claude_avoided
    price = _claude_price()
    avoided.model = DEFAULT_CLAUDE_MODEL
    avoided.input_per_1k = price["input"]
    avoided.output_per_1k = price["output"]
    avoided.estimated_dollars = (avoided.tokens_in * price["input"]
                                 + avoided.tokens_out * price["output"]) / 1_000
    avoided.note = AVOIDED_NOTE

    report.failures_by_gate = dict(sorted(report.failures_by_gate.items(), key=lambda kv: (-kv[1], kv[0])))
    report.failure_reasons = {gate: min(seen.items(), key=lambda kv: (-kv[1], kv[0]))[0]
                              for gate, seen in reasons.items()}
    report.changes = _changes(
        by_rung_model,
        {kind: (c[0], c[1]) for kind, c in kind_steps.items()},
        {gate: (r[0], r[1]) for gate, r in gate_runs.items()},
    )
    return report


# Rendering. Known rungs and kinds come first in ladder order, anything else
# (such as a pre-ladder executor name) after them by name.
RUNG_ORDER = ("free", "local", "kimi", "openrouter", "claude")
KIND_ORDER = ("mechanical", "design", "analysis", "integration")

_STEP_HEADERS = ("attempts", "first tried", "passed", "unverified", "first-try passes", "resumed passes",
                 "escalated up", "provider errors", "dollars", "tokens in", "tokens out", "wall time")


def _ordered(names, order: tuple[str, ...]) -> list[str]:
    known = [n for n in order if n in names]
    return known + sorted(n for n in names if n not in order)


def _date(ts: float | None) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts is not None else ""


def _money(dollars: float) -> str:
    return f"${dollars:,.2f}"


def _count(n: int) -> str:
    return f"{n:,}"


def _duration(seconds: float) -> str:
    whole = int(round(seconds))
    if whole < 60:
        return f"{whole}s"
    minutes, secs = divmod(whole, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _table(title: str, headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    """A title line and a plain table: first column left-aligned, the rest right-aligned."""
    cells = [headers, *rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    lines = [title]
    for row in cells:
        parts = [row[0].ljust(widths[0])] + [cell.rjust(w) for cell, w in zip(row[1:], widths[1:])]
        lines.append("  ".join(parts).rstrip())
    return lines


def _step_row(name: str, s: RungStats | KindStats) -> tuple[str, ...]:
    return (name, _count(s.attempts), _count(s.steps_first_tried), _count(s.steps_passed), _count(s.steps_unverified),
            _count(s.first_attempt_passes), _count(s.resumed_passes), _count(s.escalations_up),
            _count(s.provider_errors), _money(s.dollars), _count(s.tokens_in), _count(s.tokens_out),
            _duration(s.wall_seconds))


def _failure_lines(report: LadderReport) -> list[str]:
    if not report.failed_attempts:
        return ["Why attempts failed", "  no failed attempts"]
    lines = [f"Why attempts failed (first failing gate, {report.failed_attempts} failed attempts)"]
    for gate, count in report.failures_by_gate.items():
        lines.append(f"  {gate}: {count} of {report.failed_attempts}, most often: {report.failure_reasons.get(gate, '')}")
    return lines


def render_text(report: LadderReport) -> str:
    """The report as plain aligned tables and the "what to change" list."""
    period = f"since {_date(report.since)}" if report.since is not None else "across all jobs"
    if report.jobs == 0:
        return (f"No delegation ladder jobs {period}, so there is nothing to report yet. "
                "Jobs show up here once they run with the ladder on.")

    jobs = "job" if report.jobs == 1 else "jobs"
    steps = "step" if report.steps == 1 else "steps"
    lines = [f"Ladder report {period}: {report.jobs} {jobs}, {report.steps} {steps}, "
             f"{report.steps_passed_below_claude} passed below Claude"
             + (f", {report.steps_unverified_below_claude} unverified"
                if report.steps_unverified_below_claude else ""), ""]

    lines += _table("Per rung", ("rung", *_STEP_HEADERS),
                    [_step_row(name, report.per_rung[name])
                     for name in _ordered(report.per_rung, RUNG_ORDER)])
    lines.append("")
    lines += _table("Per step kind", ("kind", *_STEP_HEADERS),
                    [_step_row(name, report.per_kind[name])
                     for name in _ordered(report.per_kind, KIND_ORDER)])
    lines.append("")
    lines += _table("Per model", ("model", "attempts", "passes", "unverified", "dollars"),
                    [(name, _count(m.attempts), _count(m.passes), _count(m.unverified), _money(m.dollars))
                     for name, m in sorted(report.per_model.items())] or [("(none)", "0", "0", "0", _money(0.0))])
    lines.append("")

    providers = [(name, _money(d)) for name, d in sorted(report.dollars_by_provider.items())]
    providers.append(("all providers", _money(sum(report.dollars_by_provider.values()))))
    lines += _table("Dollars by provider", ("provider", "dollars"), providers)
    lines.append("")
    lines += _failure_lines(report)
    lines.append("")
    lines += _table("Claude sessions", ("session", "count"), [
        ("worker attempts", _count(report.claude_worker_attempts)),
        ("planner sessions", _count(report.claude_planner_sessions)),
        ("reviewer sessions", _count(report.claude_reviewer_sessions)),
    ])
    lines.append("")

    avoided = report.claude_avoided
    lines += [
        "Claude work avoided (an estimate at Claude API prices; on a subscription it isn't "
        "billed, it is allowance not used)",
        f"  steps passed below Claude: {_count(report.steps_passed_below_claude)}",
        f"  steps unverified below Claude: {_count(report.steps_unverified_below_claude)} "
        "(not counted above or in the tokens below)",
        f"  tokens those attempts used: {_count(avoided.tokens_in)} in, {_count(avoided.tokens_out)} out",
        f"  estimated Claude cost: {_money(avoided.estimated_dollars)} "
        f"({avoided.model} at ${avoided.input_per_1k:g} in and ${avoided.output_per_1k:g} out "
        "per 1K tokens)",
        "",
        "What to change",
    ]
    if report.changes:
        lines += [f"  - {change}" for change in report.changes]
    else:
        lines.append("  Not enough evidence yet to suggest a change.")
    return "\n".join(lines)


def _step_dict(s: RungStats | KindStats) -> dict[str, Any]:
    return {"attempts": s.attempts, "steps_first_tried": s.steps_first_tried,
            "steps_passed": s.steps_passed, "steps_unverified": s.steps_unverified, "first_attempt_passes": s.first_attempt_passes,
            "resumed_passes": s.resumed_passes, "escalations_up": s.escalations_up,
            "provider_errors": s.provider_errors, "dollars": s.dollars, "tokens_in": s.tokens_in,
            "tokens_out": s.tokens_out, "wall_seconds": s.wall_seconds}


def render_json(report: LadderReport) -> dict[str, Any]:
    """The report as a dict that json.dumps can write."""
    avoided = report.claude_avoided
    return {
        "since": report.since,
        "since_date": _date(report.since) or None,
        "totals": {
            "jobs": report.jobs,
            "steps": report.steps,
            "steps_passed_below_claude": report.steps_passed_below_claude,
            "steps_unverified_below_claude": report.steps_unverified_below_claude,
            "dollars_by_provider": dict(sorted(report.dollars_by_provider.items())),
            "claude_worker_attempts": report.claude_worker_attempts,
            "claude_planner_sessions": report.claude_planner_sessions,
            "claude_reviewer_sessions": report.claude_reviewer_sessions,
        },
        "rungs": {name: _step_dict(report.per_rung[name]) for name in _ordered(report.per_rung, RUNG_ORDER)},
        "kinds": {name: _step_dict(report.per_kind[name]) for name in _ordered(report.per_kind, KIND_ORDER)},
        "models": {name: {"attempts": m.attempts, "passes": m.passes, "unverified": m.unverified, "dollars": m.dollars}
                   for name, m in sorted(report.per_model.items())},
        "claude_avoided": {
            "estimate": True,
            "tokens_in": avoided.tokens_in,
            "tokens_out": avoided.tokens_out,
            "estimated_dollars": avoided.estimated_dollars,
            "model": avoided.model,
            "input_per_1k": avoided.input_per_1k,
            "output_per_1k": avoided.output_per_1k,
            "note": avoided.note,
        },
        "why_attempts_failed": {
            "failed_attempts": report.failed_attempts,
            "gates": [{"gate": gate, "attempts": count, "reason": report.failure_reasons.get(gate, "")}
                      for gate, count in report.failures_by_gate.items()],
        },
        "changes": list(report.changes),
    }
