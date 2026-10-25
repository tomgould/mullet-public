"""CLI subcommand: ``mullet routing <action>``.

Actions
-------
show                          Print per-provider summary table.
recent [--limit N]            List last N outcomes, newest first.
signature <task>              Print computed 16-char signature for *task*.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict

from mullet.core.routing_history import (
    _percentile,
    iter_recent,
    make_signature,
    summary_by_provider,
)
from mullet.core.router.resolver import build_default_router
from mullet.core.presets import PresetManager

def cyan(t: str) -> str: return t
def dim(t: str) -> str: return t
def bold(t: str) -> str: return t
def yellow(t: str) -> str: return t
def green(t: str) -> str: return t
RESET = ""
_HAS_COLOR = False

try:
    from mullet.hud.colors import cyan, dim, bold, yellow, green
    _HAS_COLOR = True
except Exception:  # pragma: no cover — colors always available in test env
    pass  # pragma: no cover


try:
    from mullet.core.router.decision import (
        QUALITY_THRESHOLD as _QUALITY_THRESHOLD,
        QUALITY_MIN_SAMPLES as _QUALITY_MIN_SAMPLES,
        RATE_LIMIT_THRESHOLD as _RL_THRESHOLD,
        RATE_LIMIT_WATCHLIST_THRESHOLD as _RL_WATCHLIST_THRESHOLD,
    )
except ImportError:  # pragma: no cover — router.decision always importable in test env
    _QUALITY_THRESHOLD = 0.60  # pragma: no cover
    _QUALITY_MIN_SAMPLES = 3  # pragma: no cover
    _RL_THRESHOLD = 0.40  # pragma: no cover
    _RL_WATCHLIST_THRESHOLD = 0.20  # pragma: no cover


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rate_str(rate: float) -> str:
    pct = rate * 100
    s = f"{pct:5.1f}%"
    if pct >= 90:
        return green(s)
    if pct >= 70:
        return yellow(s)
    return s


def _provider_col(provider: str) -> str:
    return cyan(f"{provider:<12}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _cmd_show(args: list[str]) -> int:
    """Print summary_by_provider as an aligned table."""
    p = argparse.ArgumentParser(prog="mullet routing show", add_help=False)
    p.add_argument("--limit", type=int, default=2000, help="Max records to aggregate")
    ns, _ = p.parse_known_args(args)

    summary = summary_by_provider(limit=ns.limit)
    if not summary:
        print("No routing history yet.  Records accumulate automatically as you use mullet.")
        return 0

    header = (
        f"{'Provider':<12}  {'Calls':>6}  {'Succ':>5}  {'Rate':>7}  "
        f"{'Avg cost':>10}  {'Total $':>10}  {'Avg ms':>8}  {'Flag':<6}"
    )
    print(bold(header))
    print(dim("-" * len(header)))

    any_flagged = False
    for provider, s in sorted(summary.items()):
        rate_s = _rate_str(s["success_rate"])
        cost_s = f"${s['avg_cost']:.5f}"
        total_s = f"${s.get('total_cost', 0.0):.4f}"
        wall_s = f"{s['avg_wall_ms']:,}"
        calls = s["calls"]
        rate = s["success_rate"]
        rl_rate = s.get("rate_limit_rate", 0.0)
        flags = []
        if calls >= _QUALITY_MIN_SAMPLES and rate < _QUALITY_THRESHOLD:
            flags.append(yellow("⚠ skip"))
            any_flagged = True
        if calls >= 5 and rl_rate > _RL_THRESHOLD:
            flags.append(yellow(f"⚠ 429 {rl_rate:.0%}"))
            any_flagged = True
        print(
            f"{_provider_col(provider)}  "
            f"{calls:>6}  "
            f"{s['successes']:>5}  "
            f"{rate_s}  "
            f"{cost_s:>10}  "
            f"{total_s:>10}  "
            f"{wall_s:>8}  "
            f"{' '.join(flags)}"
        )

    if any_flagged:
        print()
        print(dim(f"  ⚠ skip = router skips this provider (≥{_QUALITY_MIN_SAMPLES} samples, rate < {_QUALITY_THRESHOLD:.0%})"))
        print(dim(f"  ⚠ 429  = high rate-limit rate (≥5 samples, rl_rate > {_RL_THRESHOLD:.0%})"))

    return 0


def _cmd_recent(args: list[str]) -> int:
    """List last N outcomes, newest first."""
    p = argparse.ArgumentParser(prog="mullet routing recent", add_help=False)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--provider", default=None, help="Filter by provider name (e.g. kimi, openrouter)")
    p.add_argument("--failed", action="store_true", help="Show only failed outcomes")
    ns, _ = p.parse_known_args(args)

    outcomes = list(iter_recent(limit=ns.limit if not (ns.provider or ns.failed) else ns.limit * 5))
    if ns.provider:
        outcomes = [o for o in outcomes if o.provider == ns.provider]
    if ns.failed:
        outcomes = [o for o in outcomes if not o.success]
    outcomes = outcomes[:ns.limit]

    if not outcomes:
        filter_desc = ""
        if ns.provider:
            filter_desc += f" for provider={ns.provider!r}"
        if ns.failed:
            filter_desc += " (failed only)"
        print(f"No routing history yet{filter_desc}.")
        return 0

    for o in outcomes:
        status = green("OK") if o.success else yellow(f"FAIL({o.error_class or '?'})")
        cost_s = f"${o.cost_usd:.5f}"
        print(
            f"  {dim(o.timestamp[:19])}  {_provider_col(o.provider)}"
            f"  {o.tier:<10}  {status:<20}  "
            f"in={o.tokens_in:>6}  out={o.tokens_out:>5}  "
            f"cost={cost_s}  {o.wall_time_ms:>5}ms"
        )
        print(f"    {dim(o.task_preview[:100])}")

    return 0


def _cmd_signature(args: list[str]) -> int:
    """Print the computed signature for a task string (debugging aid)."""
    if not args:
        print("Usage: mullet routing signature <task> [--tier TIER] [--family FAMILY]",
              file=sys.stderr)
        return 1

    p = argparse.ArgumentParser(prog="mullet routing signature", add_help=False)
    p.add_argument("task", nargs=argparse.REMAINDER)
    p.add_argument("--tier", default="STANDARD")
    p.add_argument("--family", default="run")
    ns, _ = p.parse_known_args(args)

    task = " ".join(ns.task).strip()
    if not task:
        print("Error: task string is required.", file=sys.stderr)
        return 1

    sig = make_signature(task, tool_family=ns.family, tier=ns.tier)
    print(sig)
    return 0


# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------

def _cmd_insights(args: list[str]) -> int:
    """Print a human-readable routing insights report."""
    p = argparse.ArgumentParser(prog="mullet routing insights", add_help=False)
    p.add_argument("--limit", type=int, default=2000)
    ns, _ = p.parse_known_args(args)

    outcomes = list(iter_recent(limit=ns.limit))
    if not outcomes:
        print("no data yet — run some tasks first")
        return 0

    # --- Section 1: Top signatures by call volume ---
    sig_stats: dict[str, dict] = {}
    for o in outcomes:
        s = o.task_signature
        if s not in sig_stats:
            sig_stats[s] = {"count": 0, "preview": o.task_preview, "providers": defaultdict(lambda: {"ok": 0, "total": 0, "cost": 0.0})}
        sig_stats[s]["count"] += 1
        sig_stats[s]["providers"][o.provider]["total"] += 1
        if o.success:
            sig_stats[s]["providers"][o.provider]["ok"] += 1
        sig_stats[s]["providers"][o.provider]["cost"] += o.cost_usd

    top5 = sorted(sig_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:5]

    print(bold("=== 1. Top Signatures by Call Volume ==="))
    for sig, st in top5:
        # find cheapest provider with >75% success
        cheap = None
        cheap_cost = float("inf")
        for prov, ps in st["providers"].items():
            rate = ps["ok"] / ps["total"] if ps["total"] else 0
            avg_c = ps["cost"] / ps["total"] if ps["total"] else 0
            if rate >= 0.75 and avg_c < cheap_cost:
                cheap = prov
                cheap_cost = avg_c
        cheap_str = cheap or dim("none")
        print(f"  {cyan(sig)}  calls={st['count']:>4}  best={cheap_str}")
        print(f"    {dim(st['preview'][:80])}")
    print()

    # --- Section 2: Per-provider reliability ---
    prov_stats: dict[str, dict] = {}
    for o in outcomes:
        pr = o.provider
        if pr not in prov_stats:
            prov_stats[pr] = {"ok": 0, "total": 0, "cost": 0.0, "wall": 0}
        prov_stats[pr]["total"] += 1
        if o.success:
            prov_stats[pr]["ok"] += 1
        prov_stats[pr]["cost"] += o.cost_usd
        prov_stats[pr]["wall"] += o.wall_time_ms

    print(bold("=== 2. Per-Provider Reliability ==="))
    hdr = f"  {'Provider':<12}  {'Rate':>7}  {'AvgCost':>10}  {'TotalCost':>11}  {'AvgMs':>7}  {'Samples':>7}"
    print(dim(hdr))
    for pr, ps in sorted(prov_stats.items(), key=lambda x: x[1]["total"], reverse=True):
        rate = ps["ok"] / ps["total"] if ps["total"] else 0.0
        avg_c = ps["cost"] / ps["total"] if ps["total"] else 0.0
        avg_w = ps["wall"] // ps["total"] if ps["total"] else 0
        print(
            f"  {_provider_col(pr)}  {_rate_str(rate)}  ${avg_c:.5f}"
            f"  ${ps['cost']:.5f}    {avg_w:>7}ms  {ps['total']:>7}"
        )
    print()

    # --- Section 3: Best lane per tier ---
    tier_stats: dict[str, dict[str, dict]] = {}
    for o in outcomes:
        t = o.tier
        pr = o.provider
        if t not in tier_stats:
            tier_stats[t] = {}
        if pr not in tier_stats[t]:
            tier_stats[t][pr] = {"ok": 0, "total": 0, "cost": 0.0}
        tier_stats[t][pr]["total"] += 1
        if o.success:
            tier_stats[t][pr]["ok"] += 1
        tier_stats[t][pr]["cost"] += o.cost_usd

    print(bold("=== 3. Best Lane per Tier (success/cost score, top 3) ==="))
    for tier in sorted(tier_stats):
        scored = []
        for prov, ps in tier_stats[tier].items():
            rate = ps["ok"] / ps["total"] if ps["total"] else 0.0
            avg_c = ps["cost"] / ps["total"] if ps["total"] else 0.0
            score = rate / max(avg_c, 0.0001)
            scored.append((prov, rate, avg_c, score))
        scored.sort(key=lambda x: x[3], reverse=True)
        top3 = scored[:3]
        leaders = "  ".join(f"{cyan(p)} score={s:.1f}" for p, _r, _c, s in top3)
        print(f"  {yellow(tier)}: {leaders}")
    print()

    # --- Section 4: Anti-recommendations ---
    print(bold(f"=== 4. Anti-Recommendations (≥{_QUALITY_MIN_SAMPLES} samples, <{_QUALITY_THRESHOLD:.0%} success — router skips these) ==="))
    found_any = False
    for pr, ps in prov_stats.items():
        if ps["total"] >= _QUALITY_MIN_SAMPLES:
            rate = ps["ok"] / ps["total"]
            if rate < _QUALITY_THRESHOLD:
                found_any = True
                # Collect error_class breakdown
                err_counts: dict[str, int] = {}
                for o in outcomes:
                    if o.provider == pr and not o.success:
                        ec = o.error_class or "unknown"
                        err_counts[ec] = err_counts.get(ec, 0) + 1
                err_str = ", ".join(f"{ec}={n}" for ec, n in sorted(err_counts.items(), key=lambda x: -x[1]))
                print(f"  {_provider_col(pr)}  rate={_rate_str(rate)}  errors: {err_str}")
    if not found_any:
        print(dim(f"  (none — all providers with ≥{_QUALITY_MIN_SAMPLES} samples are above {_QUALITY_THRESHOLD:.0%})"))

    # Rate-limit watchlist — providers above quality threshold but with high 429 rates
    _RL_MIN_SAMPLES = 5
    rl_flagged = []
    for pr, ps in prov_stats.items():
        if ps["total"] < _RL_MIN_SAMPLES:
            continue
        rate = ps["ok"] / ps["total"]
        if rate >= _QUALITY_THRESHOLD:  # above quality threshold (not already in anti-recs)
            rl_count = sum(1 for o in outcomes if o.provider == pr and o.error_class == "rate_limit")
            rl_rate = rl_count / ps["total"]
            if rl_rate >= _RL_WATCHLIST_THRESHOLD:
                rl_flagged.append((pr, rl_rate, ps["total"]))
    if rl_flagged:
        print()
        print(dim(f"  Rate-limit watchlist (≥{_RL_WATCHLIST_THRESHOLD:.0%} 429 rate despite ≥{_QUALITY_THRESHOLD:.0%} success):"))
        for pr, rl_rate, total in sorted(rl_flagged, key=lambda x: -x[1]):
            print(f"    {_provider_col(pr)}  429 rate={rl_rate:.0%}  ({total} calls) — consider downweighting")
    print()

    # --- Section 5: Cost + latency tail (p50 / p95 per provider) ---
    # Built directly from outcomes (same dataset as sections 1-4; no extra file read).
    print(bold("=== 5. Cost & Latency Tail per Provider (p50 / p95) ==="))
    prov_costs: dict[str, list[float]] = {}
    prov_walls: dict[str, list[int]] = {}
    for o in outcomes:
        prov_costs.setdefault(o.provider, []).append(o.cost_usd)
        prov_walls.setdefault(o.provider, []).append(o.wall_time_ms)
    if prov_costs:
        hdr5 = (
            f"  {'Provider':<12}  {'p50 cost':>10}  {'p95 cost':>10}  "
            f"{'p50 ms':>8}  {'p95 ms':>8}  {'Calls':>6}"
        )
        print(dim(hdr5))
        for pr in sorted(prov_costs):
            sc = sorted(prov_costs[pr])
            sw = sorted(prov_walls.get(pr, []))
            p50c = _percentile(sc, 50)
            p95c = _percentile(sc, 95)
            p50w = _percentile(sw, 50)
            p95w = _percentile(sw, 95)
            cost_spike = p95c > 0 and p50c > 0 and p95c / p50c >= 5
            wall_spike = p95w > 0 and p50w > 0 and p95w / p50w >= 5
            spike = yellow(" ⚠ spike") if (cost_spike or wall_spike) else ""
            print(
                f"  {_provider_col(pr)}  ${p50c:.5f}    ${p95c:.5f}  "
                f"{int(p50w):>7}ms  {int(p95w):>7}ms  {len(sc):>6}{spike}"
            )
        print(dim("  ⚠ spike = p95 ≥ 5× p50 on cost or latency — check for runaway tasks"))
    else:  # pragma: no cover — prov_costs is never empty when outcomes is non-empty
        print(dim("  (no data)"))  # pragma: no cover

    return 0


# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------

def _cmd_simulate(args: list[str]) -> int:
    """Replay historical routing outcomes through the current router policy + active preset.

    Note: history stores only ``task_preview`` (<=2000 chars), so very long prompts
    still replay on a truncated prefix — complexity scoring may differ slightly from
    the original full-prompt decision in that case. ``none`` in the NOW column means
    the current policy has no eligible provider for that task.
    """
    p = argparse.ArgumentParser(prog="mullet routing simulate", add_help=False)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--provider", type=str, default=None)
    p.add_argument("--changed-only", action="store_true")
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.add_argument("--no-preset", action="store_true")
    ns, _ = p.parse_known_args(args)

    # Resolve allowlist from active preset (unless suppressed)
    if ns.no_preset:
        allowlist = None
    else:
        allowlist = PresetManager().active().router_allowlist()

    # Over-fetch when filtering by provider so we can trim to --limit after filter
    fetch_limit = ns.limit * 5 if ns.provider else ns.limit
    outcomes = list(iter_recent(limit=fetch_limit))

    # Filter by provider if requested
    if ns.provider:
        outcomes = [o for o in outcomes if o.provider == ns.provider][: ns.limit]

    if not outcomes:
        if ns.format == "json":
            print(json.dumps({
                "summary": {"replayed": 0, "changed": 0, "unchanged": 0, "transitions": {}},
                "rows": [],
            }))
        elif ns.provider:
            print(f"No matching routing history for provider={ns.provider!r}.")
        else:
            print("No routing history to simulate.")
        return 0

    router, _ = build_default_router()

    rows = []
    replayed = 0
    changed_count = 0
    transitions: Counter = Counter()

    for o in outcomes:
        decision = router.route(o.task_preview, allowed_providers=allowlist)
        new_provider = decision.provider if decision.provider is not None else "none"
        new_model = decision.model or ""
        new_tier = getattr(decision.tier, "name", str(decision.tier)).upper()

        was_provider = o.provider
        was_model = o.model or ""
        was_tier = (o.tier or "").upper()

        changed = (was_provider, was_model, was_tier) != (new_provider, new_model, new_tier)
        replayed += 1
        if changed:
            changed_count += 1
            transitions[f"{was_provider}→{new_provider}"] += 1

        rows.append({
            "signature": o.task_signature,
            "task_preview": o.task_preview.replace("\n", " "),
            "was": {"provider": was_provider, "model": was_model, "tier": was_tier},
            "now": {
                "provider": new_provider,
                "model": new_model,
                "tier": new_tier,
                "reasons": list(decision.reasons),
            },
            "changed": changed,
        })

    unchanged_count = replayed - changed_count
    pct = (changed_count / replayed * 100) if replayed else 0.0

    if ns.format == "json":
        filtered_rows = [r for r in rows if not ns.changed_only or r["changed"]]
        data = {
            "summary": {
                "replayed": replayed,
                "changed": changed_count,
                "unchanged": unchanged_count,
                "transitions": dict(transitions),
            },
            "rows": filtered_rows,
        }
        print(json.dumps(data, indent=2))
        return 0

    # Text output
    for row in rows:
        if ns.changed_only and not row["changed"]:
            continue
        marker = "~" if row["changed"] else "="
        was = row["was"]
        now = row["now"]
        print(
            f"  {marker} {was['provider']}/{was['model']} {was['tier']} "
            f"→ {now['provider']}/{now['model']} {now['tier']}"
        )

    # Summary footer
    print(
        f"\n{replayed} replayed, {changed_count} changed ({pct:.0f}%), {unchanged_count} unchanged"
    )
    for transition, count in sorted(transitions.items()):
        print(f"  {transition}: {count}")

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _cmd_log(args: list[str]) -> int:
    """Tail routing.log in the state directory — human-readable routing decision log."""
    p = argparse.ArgumentParser(prog="mullet routing log", add_help=False)
    p.add_argument("--tail", type=int, default=20, help="Last N lines to show")
    ns, _ = p.parse_known_args(args)

    from mullet.core.paths import state_dir
    log_path = state_dir() / "routing.log"
    if not log_path.exists():
        print("No routing.log yet — routing decisions are logged here as you run tasks.")
        return 0

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = lines[-ns.tail:] if len(lines) > ns.tail else lines
    print(f"Last {len(tail)} routing decisions (newest last):\n")
    for line in tail:
        parts = line.split("  ", maxsplit=4)
        if len(parts) >= 4:
            ts = dim(parts[0]) if parts else ""
            provider_model = cyan(parts[1]) if len(parts) > 1 else ""
            tier = yellow(parts[2]) if len(parts) > 2 else ""
            rest = "  ".join(parts[3:])
            print(f"  {ts}  {provider_model}  {tier}  {rest}")
        else:
            print(f"  {line}")
    return 0


def _cmd_compare(args: list[str]) -> int:
    """Side-by-side comparison of two providers from routing history."""
    p = argparse.ArgumentParser(prog="mullet routing compare", add_help=False)
    p.add_argument("providers", nargs=2, metavar="PROVIDER",
                   help="Two provider names to compare (e.g. kimi openrouter)")
    p.add_argument("--limit", type=int, default=2000)
    ns, _ = p.parse_known_args(args)

    p1, p2 = ns.providers
    summary = summary_by_provider(limit=ns.limit)

    if p1 not in summary and p2 not in summary:
        print(f"No data for either '{p1}' or '{p2}'. Run some tasks first.")
        return 1

    s1 = summary.get(p1)
    s2 = summary.get(p2)

    def _row(label: str, v1, v2, fmt=lambda x: str(x), better="higher") -> None:
        fv1 = fmt(v1) if v1 is not None else dim("no data")
        fv2 = fmt(v2) if v2 is not None else dim("no data")
        if v1 is not None and v2 is not None:
            if better == "higher":
                mark1 = green("◀ ") if v1 > v2 else ("  " if v1 == v2 else "  ")
                mark2 = green("◀ ") if v2 > v1 else ("  " if v1 == v2 else "  ")
            else:  # lower is better
                mark1 = green("◀ ") if v1 < v2 else ("  " if v1 == v2 else "  ")
                mark2 = green("◀ ") if v2 < v1 else ("  " if v1 == v2 else "  ")
        else:
            mark1 = mark2 = "  "
        print(f"  {label:<18}  {mark1}{fv1:<18}  {mark2}{fv2}")

    print(bold(f"\n=== Provider Comparison: {cyan(p1)} vs {cyan(p2)} ===\n"))
    print(f"  {'Metric':<18}  {bold(p1):<20}  {bold(p2)}")
    print(dim("  " + "-" * 58))
    _row("Calls",
         s1["calls"] if s1 else None, s2["calls"] if s2 else None,
         fmt=lambda x: str(x), better="higher")
    _row("Success rate",
         s1["success_rate"] if s1 else None, s2["success_rate"] if s2 else None,
         fmt=lambda x: f"{x:.1%}", better="higher")
    _row("Avg cost",
         s1["avg_cost"] if s1 else None, s2["avg_cost"] if s2 else None,
         fmt=lambda x: f"${x:.5f}", better="lower")
    _row("Total $",
         s1.get("total_cost") if s1 else None, s2.get("total_cost") if s2 else None,
         fmt=lambda x: f"${x:.4f}", better="lower")
    _row("Avg latency",
         s1["avg_wall_ms"] if s1 else None, s2["avg_wall_ms"] if s2 else None,
         fmt=lambda x: f"{x:,}ms", better="lower")
    _row("p50 cost",
         s1.get("p50_cost") if s1 else None, s2.get("p50_cost") if s2 else None,
         fmt=lambda x: f"${x:.5f}", better="lower")
    _row("p95 cost",
         s1.get("p95_cost") if s1 else None, s2.get("p95_cost") if s2 else None,
         fmt=lambda x: f"${x:.5f}", better="lower")
    _row("p50 latency",
         s1.get("p50_wall_ms") if s1 else None, s2.get("p50_wall_ms") if s2 else None,
         fmt=lambda x: f"{int(x):,}ms", better="lower")
    _row("p95 latency",
         s1.get("p95_wall_ms") if s1 else None, s2.get("p95_wall_ms") if s2 else None,
         fmt=lambda x: f"{int(x):,}ms", better="lower")
    print()
    print(dim("  ◀ = better value for this metric"))
    return 0


def _cmd_export(args: list[str]) -> int:
    """Export routing history to JSON or CSV."""
    p = argparse.ArgumentParser(prog="mullet routing export", add_help=False)
    p.add_argument("--format", choices=["json", "csv"], default="json",
                   help="Output format (default: json)")
    p.add_argument("--output", "-o", default=None,
                   help="Output file path (default: stdout)")
    p.add_argument("--limit", type=int, default=None,
                   help="Max records to export (default: all)")
    ns, _ = p.parse_known_args(args)

    outcomes = list(iter_recent(limit=ns.limit or 1_000_000))
    if not outcomes:
        print("No routing history to export.", file=sys.stderr)
        return 0

    import json as _json
    from dataclasses import asdict

    if ns.format == "json":
        data = [asdict(o) for o in outcomes]
        text = _json.dumps(data, indent=2)
    else:
        import io
        import csv
        if not outcomes:  # pragma: no cover — already returned above if empty
            return 0  # pragma: no cover
        buf = io.StringIO()
        fields = list(asdict(outcomes[0]).keys())
        writer = csv.DictWriter(buf, fieldnames=fields)
        writer.writeheader()
        for o in outcomes:
            writer.writerow(asdict(o))
        text = buf.getvalue()

    if ns.output:
        from pathlib import Path
        Path(ns.output).write_text(text, encoding="utf-8")
        print(f"Exported {len(outcomes)} records to {ns.output}")
    else:
        print(text)
    return 0


def _cmd_reset(args: list[str]) -> int:
    """Clear routing_history.jsonl — removes all recorded outcomes."""
    p = argparse.ArgumentParser(prog="mullet routing reset", add_help=False)
    p.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    ns, _ = p.parse_known_args(args)

    from mullet.core.routing_history import history_path
    hist = history_path()
    if not hist.exists():
        print("No routing history to clear.")
        return 0

    lines = [ln for ln in hist.read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(f"This will delete {len(lines)} routing record(s) from {hist}")
    if not ns.yes:
        try:
            ans = input("Confirm? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        if ans not in ("y", "yes"):
            print("Aborted.")
            return 1
    hist.write_text("", encoding="utf-8")
    print(f"Cleared {len(lines)} record(s).")
    return 0


def _cmd_purge_provider(args: list[str]) -> int:
    """Remove all routing history entries for a specific provider."""
    p = argparse.ArgumentParser(prog="mullet routing purge-provider", add_help=False)
    p.add_argument("provider", help="Provider name to remove (e.g. kimi, openrouter, local)")
    p.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    ns, _ = p.parse_known_args(args)

    from mullet.core.routing_history import history_path
    import json as _json
    hist = history_path()
    if not hist.exists():
        print("No routing history found.")
        return 0

    kept: list[str] = []
    removed = 0
    for line in hist.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = _json.loads(line)
        except _json.JSONDecodeError:
            kept.append(line)
            continue
        if d.get("provider", "") == ns.provider:
            removed += 1
        else:
            kept.append(line)

    if removed == 0:
        print(f"No entries found for provider '{ns.provider}'.")
        return 0

    print(f"This will remove {removed} entry/entries for '{ns.provider}' (keeping {len(kept)}).")
    if not ns.yes:
        try:
            ans = input("Confirm? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        if ans not in ("y", "yes"):
            print("Aborted.")
            return 1

    hist.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    print(f"Removed {removed} entry/entries for '{ns.provider}'.")
    return 0


def run(args: list[str]) -> int:
    """Entry point for ``mullet routing``."""
    if not args or args[0] in ("-h", "--help", "help"):
        print(
            "Usage: mullet routing <action>\n"
            "  show [--limit N]               Per-provider summary table with total cost\n"
            "  recent [--limit N] [--provider P] [--failed]  Last N outcomes (filterable)\n"
            "  log [--tail N]                 Tail human-readable routing decision log\n"
            "  signature <task>               Print 16-char signature for task\n"
            "  insights [--limit N]           Human-readable routing insights report (5 sections)\n"
            "  compare <p1> <p2> [--limit N]  Side-by-side provider comparison\n"
            "  export [--format json|csv] [-o FILE] [--limit N]  Export history\n"
            "  reset [--yes]                  Clear all routing history\n"
            "  purge-provider <name> [--yes]  Remove history for one provider\n"
            "  simulate [--limit N] [--provider P] [--changed-only] [--format text|json] [--no-preset]"
            "   Replay history through current policy ('none' = no eligible provider)\n"
            "\n"
            "All actions read the project's .mullet/routing_history.jsonl, written by every\n"
            "routed delegation. See docs/MODEL_ROUTING.md and\n"
            "mullet/core/router/routing_matrix.py for routing rules."
        )
        return 0

    action = args[0]
    rest = args[1:]

    if action == "show":
        return _cmd_show(rest)
    elif action == "recent":
        return _cmd_recent(rest)
    elif action == "log":
        return _cmd_log(rest)
    elif action == "signature":
        return _cmd_signature(rest)
    elif action == "insights":
        return _cmd_insights(rest)
    elif action == "compare":
        return _cmd_compare(rest)
    elif action == "export":
        return _cmd_export(rest)
    elif action == "reset":
        return _cmd_reset(rest)
    elif action == "purge-provider":
        return _cmd_purge_provider(rest)
    elif action == "simulate":
        return _cmd_simulate(rest)
    else:
        print(
            f"Unknown action: {action!r}.  "
            "Use show, recent, log, signature, insights, compare, export, reset, purge-provider, or simulate.",
            file=sys.stderr,
        )
        return 1
