"""A job report built from the ledger and nothing else.

Every status comes from recorded gate results. What an executor said about
its own work appears only as a labelled claim. Nodes whose checks could not
all run are reported as unverified, never as passed.
"""

from __future__ import annotations

import time

from mullet.jobs.ladder import LadderConfig, attempt_cost, load_config, model_rung, spend_state
from mullet.jobs.ledger import Ledger
from mullet.jobs.reconcile import reconcile
from mullet.jobs.types import AttemptOutcome, GateStatus, NodeStatus

_GATE_WORD = {
    GateStatus.PASS: "pass", GateStatus.FAIL: "FAIL", GateStatus.NOT_RUN: "not run", GateStatus.SKIPPED: "skipped",
}


def _when(ts: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "-"


def _cost(usage: dict | None) -> float:
    try:
        return float((usage or {}).get("total_cost_usd") or 0)
    except (TypeError, ValueError):
        return 0.0


_RUNG_NAMES = {"kimi": "Kimi", "openrouter": "OpenRouter", "free": "free", "local": "local", "claude": "Claude"}


def _tokens(usage: dict) -> str:
    tokens = usage.get("tokens")
    if not isinstance(tokens, dict) or not tokens:
        return ""
    def total(*names: str) -> int:
        return sum(int(tokens.get(n) or 0) for n in names if isinstance(tokens.get(n), (int, float)))
    inp, out = total("input", "cache_read", "cache_write"), total("output", "reasoning")
    return f", {inp:,} tokens in, {out:,} out" if inp or out else ""


def _attempt_line(i: int, attempt, config: LadderConfig) -> str:
    """`Attempt 2 (opencode, rung kimi, moonshotai/kimi-k2.7-code): passed, 11 turns, $0.05, ...`."""
    usage = attempt.usage or {}
    outcome = attempt.outcome.value if attempt.outcome else "in progress"
    turns = usage.get("num_turns")
    who = attempt.executor
    if usage.get("rung"):
        who += f", rung {usage['rung']}"
    if usage.get("model"):
        who += f", {usage['model']}"
    cost = ""
    if usage.get("rung") in ("kimi", "openrouter"):
        dollars, estimated = attempt_cost(usage, config)
        cost = f", ${dollars:.2f}" + (" estimated from tokens" if estimated else "")
    elif usage.get("cost_reported") and usage.get("total_cost_usd") is not None:
        cost = f", ${_cost(usage):.2f}"
    reason = (f", reason: {attempt.reason}" if attempt.reason and attempt.outcome
              in (AttemptOutcome.ABORTED, AttemptOutcome.PAUSED) else "")
    return (f"Attempt {i} ({who}): {outcome}" + (f", {turns} turns" if turns else "") + cost + _tokens(usage)
            + reason)


def ladder_summary(nodes, attempts, config: LadderConfig) -> str:
    """Like `4 of 6 steps passed below Claude; spent $0.31: Kimi $0.27, OpenRouter $0.04; Claude used on 2 steps`.

    Steps are the leaves: a parent's integration check isn't a step of its own.
    """
    parents = {n.parent_id for n in nodes if n.parent_id}
    nodes = [n for n in nodes if n.node_id not in parents]
    below = 0
    unverified = 0
    claude_steps = set()
    for node in nodes:
        mine = [a for a in attempts if a.node_id == node.node_id]
        if any((a.usage or {}).get("rung") == "claude" for a in mine):
            claude_steps.add(node.node_id)
        if node.status in (NodeStatus.PASSED, NodeStatus.UNVERIFIED):
            winner = next((a for a in reversed(mine) if a.outcome in (AttemptOutcome.PASSED, AttemptOutcome.UNVERIFIED)),
                          None)
            rung = (winner.usage or {}).get("rung") if winner else None
            if rung and rung != "claude":
                if node.status is NodeStatus.PASSED:
                    below += 1
                else:
                    unverified += 1
    spend = spend_state([a.usage for a in attempts], config.max_spend, config)
    passed = f"{below} of {len(nodes)} steps passed below Claude"
    parts = [passed + (f", {unverified} unverified" if unverified else "")]
    spent = f"spent ${spend.spent:.2f}"
    if spend.by_rung:
        spent += ": " + ", ".join(f"{_RUNG_NAMES.get(r, r)} ${v:.2f}" for r, v in sorted(spend.by_rung.items(),
                                                                                   key=lambda kv: -kv[1]))
    if spend.estimated:
        spent += " (partly estimated from tokens)"
    parts.append(spent)
    parts.append(f"Claude used on {len(claude_steps)} step{'s' if len(claude_steps) != 1 else ''}")
    return "; ".join(parts)


def _claude_session_tokens(data: dict) -> int:
    tokens = data.get("tokens") or {}
    return sum(int(v) for v in tokens.values() if isinstance(v, (int, float)))


def usage_lines(job, attempts, config: LadderConfig, events=()) -> list[str]:
    """What the job cost, by who did the work.

    opencode attempts are billed by their provider (the cost it reported, or
    tokens times the configured price); free and local ones cost nothing.
    Claude's figure is Claude Code's own estimate, billed only with
    `--billing api`; on a subscription it counts against the plan's limits.
    """
    opencode = [a for a in attempts if a.executor == "opencode"]
    claude = [a for a in attempts if a.executor not in ("opencode", "integration-check")]
    lines = [f"Attempts: {len(attempts)}."]
    if opencode:
        billed = 0.0
        estimated = False
        by_provider: dict[str, float] = {}
        for a in opencode:
            usage = dict(a.usage or {})
            # A single-executor opencode attempt recorded before rungs were kept for it: tell from the model.
            usage.setdefault("rung", model_rung(str(usage.get("model") or "")).name)
            if usage["rung"] not in ("kimi", "openrouter"):
                continue
            dollars, guess = attempt_cost(usage, config)
            billed += dollars
            estimated = estimated or guess
            provider = str(usage.get("model") or "?").split("/", 1)[0]
            by_provider[provider] = by_provider.get(provider, 0.0) + dollars
        detail = ", ".join(f"{p} ${v:.2f}" for p, v in sorted(by_provider.items(), key=lambda kv: -kv[1]))
        lines.append(f"opencode attempts ({len(opencode)}): billed ${billed:.2f} by their providers"
                     + (f" ({detail})" if detail else "")
                     + ("; part of it is estimated from tokens times the configured price" if estimated else "")
                     + ". Free and local attempts cost nothing.")
    sessions = [e.data for e in events if e.kind == "claude_usage"]
    if claude or sessions:
        billing = (job.budget or {}).get("billing", "subscription")
        total = sum(_cost(a.usage) for a in claude) + sum(_cost(d) for d in sessions)
        parts = []
        if claude:
            turns = sum(int((a.usage or {}).get("num_turns") or 0) for a in claude)
            parts.append(f"{len(claude)} worker attempt{'s' if len(claude) != 1 else ''} ({turns} turns)")
        for role in ("planner", "reviewer"):
            mine = [d for d in sessions if d.get("role") == role]
            if mine:
                turns = sum(int(d.get("num_turns") or 0) for d in mine)
                tokens = sum(_claude_session_tokens(d) for d in mine)
                parts.append(f"{role} {len(mine)} session{'s' if len(mine) != 1 else ''} "
                             f"({turns} turns, {tokens:,} tokens)")
        work = "Claude's work: " + "; ".join(parts)
        if billing == "api":
            lines.append(f"{work}. ${total:.2f}, billed to your API key.")
        else:
            lines.append(f"{work}. About ${total:.2f} at API prices (Claude Code's own estimate); on a "
                         "subscription this isn't billed, it counts against the plan's limits.")
    return lines


def provider_check_lines(events, attempts, config: LadderConfig) -> list[str]:
    """The ledger's spend against each paid provider's own figure, from the snapshots taken around the job."""
    snapshots = [e.data for e in events if e.kind == "provider_snapshot"]
    starts = [s for s in snapshots if s.get("when") == "start"]
    ends = [s for s in snapshots if s.get("when") == "end"]
    if not starts or not ends:
        return []
    recorded: dict[str, float] = {}
    for attempt in attempts:
        usage = attempt.usage or {}
        if usage.get("rung") in ("kimi", "openrouter"):
            recorded[usage["rung"]] = recorded.get(usage["rung"], 0.0) + attempt_cost(usage, config)[0]
    found = reconcile(starts[0].get("providers") or {}, ends[-1].get("providers") or {}, recorded)
    return ["Provider check (the ledger against each provider's own figure):"] + [f"- {r.line()}" for r in found]


def build_report(ledger: Ledger, job_id: str, config: LadderConfig | None = None) -> str:
    """The job's report as Markdown. `config` (the ladder's prices) defaults to the user's."""
    job = ledger.get_job(job_id)
    nodes = ledger.nodes(job_id)
    events = ledger.events(job_id)
    attempts = ledger.attempts(job_id)
    config = config or load_config()
    if (job.budget or {}).get("max_spend") is not None:
        config.max_spend = float(job.budget["max_spend"])
    ladder = bool((job.budget or {}).get("ladder"))
    lines = [
        f"# Job {job.id}",
        "",
        f"Goal: {job.goal}",
        f"Status: {job.status.value}" + (f" ({job.error})" if job.error else ""),
        f"Repository: {job.repo_path} at {job.base_ref} ({job.base_sha[:12]})",
        f"Branch: {job.branch or '-'}",
        f"Created {_when(job.created_at)}, approved {_when(job.approved_at)}, updated {_when(job.updated_at)}",
        "",
    ]

    counts: dict[NodeStatus, int] = {}
    for node in nodes:
        counts[node.status] = counts.get(node.status, 0) + 1
    lines.append("Nodes: " + ", ".join(f"{n} {s.value}" for s, n in counts.items()))
    if ladder:
        lines.append("Ladder: " + ladder_summary(nodes, attempts, config))
        lines.extend(provider_check_lines(events, attempts, config))
    unverified = [n.node_id for n in nodes if n.status is NodeStatus.UNVERIFIED]
    if unverified:
        lines.append(f"Not verified (some checks could not run): {', '.join(unverified)}")
    lines.append("")

    for node in nodes:
        lines.append(f"## {node.node_id}: {node.status.value}")
        lines.append("")
        lines.append(node.spec.goal)
        if node.end_sha:
            lines.append(f"Commit: {node.end_sha[:12]}")
        if node.block_reason:
            lines.append(f"Blocked: {node.block_reason}")
        node_attempts = [a for a in attempts if a.node_id == node.node_id]
        for i, attempt in enumerate(node_attempts, 1):
            lines.append("")
            lines.append(_attempt_line(i, attempt, config))
            results = ledger.gate_results(attempt.id)
            if results:
                lines.append("")
                lines.append("| Gate | Result | Detail |")
                lines.append("|---|---|---|")
                for r in results:
                    lines.append(f"| {r.gate} | {_GATE_WORD[r.status]} | {r.summary.replace('|', '/')} |")
            if attempt.claim:
                # One line, quoted: a claim must not be able to fake report structure.
                summary = " ".join(str(attempt.claim.get("summary", "")).split())
                if summary:
                    lines.append("")
                    lines.append(f"Executor's own summary (a claim, not checked): \"{summary[:600]}\"")
        lines.append("")

    final = [e.data for e in events if e.kind == "final_check"]
    if final:
        lines.append("## Final check on the whole job")
        lines.append("")
        last = final[-1]
        if "skipped" in last:
            lines.append(f"Skipped: {last['skipped']}")
        for r in last.get("results", []):
            lines.append(f"- {r['gate']}: {r['status']}, {r['summary']}")
        if last.get("lost_tests_checked") is False:
            lines.append("- A node was allowed to remove tests, so tests missing from this run were not checked.")
        lines.append("")

    lines.append("## Usage")
    lines.append("")
    lines += usage_lines(job, attempts, config, events)
    lines.append("")

    problems = ledger.audit(job_id)
    lines.append("## Ledger audit")
    lines.append("")
    lines += [f"- {p}" for p in problems] if problems else ["Clean: every passed node is backed by its gate results."]
    return "\n".join(lines) + "\n"
