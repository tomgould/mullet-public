"""Tests for mullet.core.routing_history."""
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path

import pytest


from mullet.core.routing_history import (
    RoutingOutcome,
    estimate_cost,
    history_path,
    iter_recent,
    make_outcome,
    make_signature,
    record,
    success_rate,
    summary_by_provider,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_outcome(
    tmp_path: Path,
    provider: str = "kimi",
    model: str = "kimi-k2",
    success: bool = True,
    task: str = "Summarise the codebase",
    tokens_in: int = 1000,
    tokens_out: int = 500,
    wall_time_ms: int = 1200,
    error_class: str | None = None,
) -> RoutingOutcome:
    o = make_outcome(
        task=task,
        provider=provider,
        model=model,
        tier="STANDARD",
        success=success,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        wall_time_ms=wall_time_ms,
        error_class=error_class,
    )
    record(o, base=tmp_path)
    return o


# ---------------------------------------------------------------------------
# record + iter_recent round-trip
# ---------------------------------------------------------------------------

def test_record_and_iter_recent_roundtrip(tmp_path: Path) -> None:
    o = _make_outcome(tmp_path, provider="kimi")
    results = list(iter_recent(limit=10, base=tmp_path))
    assert len(results) == 1
    r = results[0]
    assert r.provider == "kimi"
    assert r.success is True
    assert r.task_preview == "Summarise the codebase"


def test_iter_recent_newest_first(tmp_path: Path) -> None:
    _make_outcome(tmp_path, task="task A", provider="kimi")
    _make_outcome(tmp_path, task="task B", provider="gemini")
    _make_outcome(tmp_path, task="task C", provider="local")

    results = list(iter_recent(limit=3, base=tmp_path))
    # Newest first — recorded in order A, B, C so C should be first
    assert results[0].provider == "local"
    assert results[1].provider == "gemini"
    assert results[2].provider == "kimi"


def test_iter_recent_respects_limit(tmp_path: Path) -> None:
    for i in range(10):
        _make_outcome(tmp_path, task=f"task {i}")
    results = list(iter_recent(limit=5, base=tmp_path))
    assert len(results) == 5


def test_iter_recent_empty_file(tmp_path: Path) -> None:
    results = list(iter_recent(limit=10, base=tmp_path))
    assert results == []


# ---------------------------------------------------------------------------
# make_signature
# ---------------------------------------------------------------------------

def test_signature_stable_for_same_input() -> None:
    s1 = make_signature("Fix the bug", "run", "STANDARD")
    s2 = make_signature("Fix the bug", "run", "STANDARD")
    assert s1 == s2
    assert len(s1) == 16


def test_signature_different_for_different_task() -> None:
    s1 = make_signature("Fix the bug", "run", "STANDARD")
    s2 = make_signature("Review the code", "run", "STANDARD")
    assert s1 != s2


def test_signature_case_insensitive_on_prefix() -> None:
    s1 = make_signature("Fix the BUG", "run", "STANDARD")
    s2 = make_signature("fix the bug", "run", "STANDARD")
    assert s1 == s2


def test_signature_whitespace_normalised() -> None:
    s1 = make_signature("fix   the   bug", "run", "STANDARD")
    s2 = make_signature("fix the bug", "run", "STANDARD")
    assert s1 == s2


def test_signature_different_tier() -> None:
    s1 = make_signature("Fix the bug", "run", "STANDARD")
    s2 = make_signature("Fix the bug", "run", "ADVANCED")
    assert s1 != s2


def test_signature_different_family() -> None:
    s1 = make_signature("Fix the bug", "run", "STANDARD")
    s2 = make_signature("Fix the bug", "delegate", "STANDARD")
    assert s1 != s2


# ---------------------------------------------------------------------------
# success_rate
# ---------------------------------------------------------------------------

def test_success_rate_none_when_below_min_samples(tmp_path: Path) -> None:
    _make_outcome(tmp_path, provider="kimi", success=True)
    _make_outcome(tmp_path, provider="kimi", success=True)
    sig = make_signature("Summarise the codebase")
    rate = success_rate(sig, "kimi", base=tmp_path, min_samples=3)
    assert rate is None


def test_success_rate_computes_correctly(tmp_path: Path) -> None:
    _make_outcome(tmp_path, provider="kimi", success=True)
    _make_outcome(tmp_path, provider="kimi", success=True)
    _make_outcome(tmp_path, provider="kimi", success=False)
    sig = make_signature("Summarise the codebase")
    rate = success_rate(sig, "kimi", base=tmp_path, min_samples=3)
    assert rate is not None
    assert abs(rate - 2 / 3) < 1e-9


def test_success_rate_none_no_file(tmp_path: Path) -> None:
    sig = make_signature("anything")
    rate = success_rate(sig, "kimi", base=tmp_path, min_samples=1)
    assert rate is None


def test_success_rate_ignores_other_providers(tmp_path: Path) -> None:
    _make_outcome(tmp_path, provider="kimi", success=True)
    _make_outcome(tmp_path, provider="kimi", success=True)
    _make_outcome(tmp_path, provider="kimi", success=True)
    _make_outcome(tmp_path, provider="gemini", success=False)
    _make_outcome(tmp_path, provider="gemini", success=False)
    _make_outcome(tmp_path, provider="gemini", success=False)
    sig = make_signature("Summarise the codebase")
    rate = success_rate(sig, "kimi", base=tmp_path, min_samples=3)
    assert rate == 1.0


# ---------------------------------------------------------------------------
# summary_by_provider
# ---------------------------------------------------------------------------

def test_summary_by_provider_empty(tmp_path: Path) -> None:
    result = summary_by_provider(base=tmp_path)
    assert result == {}


def test_summary_by_provider_aggregates(tmp_path: Path) -> None:
    _make_outcome(tmp_path, provider="kimi", success=True, tokens_in=1000, tokens_out=500)
    _make_outcome(tmp_path, provider="kimi", success=True, tokens_in=2000, tokens_out=1000)
    _make_outcome(tmp_path, provider="kimi", success=False, tokens_in=500, tokens_out=0)
    _make_outcome(tmp_path, provider="gemini", success=True, tokens_in=5000, tokens_out=2000)

    summary = summary_by_provider(base=tmp_path)
    assert "kimi" in summary
    assert "gemini" in summary

    kimi = summary["kimi"]
    assert kimi["calls"] == 3
    assert kimi["successes"] == 2
    assert abs(kimi["success_rate"] - 2 / 3) < 1e-4

    gemini = summary["gemini"]
    assert gemini["calls"] == 1
    assert gemini["success_rate"] == 1.0


def test_summary_by_provider_respects_limit(tmp_path: Path) -> None:
    for _ in range(10):
        _make_outcome(tmp_path, provider="kimi")
    summary = summary_by_provider(base=tmp_path, limit=5)
    assert summary["kimi"]["calls"] == 5


def test_summary_by_provider_includes_cost_percentiles(tmp_path: Path) -> None:
    """summary_by_provider includes p50_cost and p95_cost per provider."""
    # 10 kimi calls with known costs: wall_time varies, tokens fixed
    # cost_usd is computed from tokens — use zero-cost provider by patching
    # Instead, inject raw JSONL with explicit cost_usd values.
    import json as _json
    from mullet.core.routing_history import history_path

    hp = history_path(tmp_path)
    hp.parent.mkdir(parents=True, exist_ok=True)
    costs = [0.001 * i for i in range(1, 11)]  # 0.001 … 0.010
    for c in costs:
        line = {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "task_signature": "aaaa",
            "task_preview": "t",
            "provider": "kimi",
            "model": "k2",
            "tier": "STANDARD",
            "success": True,
            "error_class": None,
            "tokens_in": 1000,
            "tokens_out": 500,
            "wall_time_ms": 1000,
            "cost_usd": c,
            "tags": [],
        }
        with open(hp, "a") as f:
            f.write(_json.dumps(line) + "\n")

    summary = summary_by_provider(base=tmp_path)
    kimi = summary["kimi"]
    assert "p50_cost" in kimi, "p50_cost missing from summary"
    assert "p95_cost" in kimi, "p95_cost missing from summary"
    # 10 values [0.001..0.010]: p50 = 0.0055, p95 = 0.00955 (linear interpolation)
    assert abs(kimi["p50_cost"] - 0.0055) < 1e-6
    assert abs(kimi["p95_cost"] - 0.00955) < 1e-6


def test_summary_by_provider_single_call_percentiles(tmp_path: Path) -> None:
    """Single call: p50 == p95 == the single cost value."""
    import json as _json
    from mullet.core.routing_history import history_path

    hp = history_path(tmp_path)
    hp.parent.mkdir(parents=True, exist_ok=True)
    line = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "task_signature": "bbbb",
        "task_preview": "t",
        "provider": "openrouter",
        "model": "deepseek-v4",
        "tier": "STANDARD",
        "success": True,
        "error_class": None,
        "tokens_in": 1000,
        "tokens_out": 500,
        "wall_time_ms": 1000,
        "cost_usd": 0.005,
        "tags": [],
    }
    with open(hp, "a") as f:
        f.write(_json.dumps(line) + "\n")

    summary = summary_by_provider(base=tmp_path)
    prov = summary["openrouter"]
    assert prov["p50_cost"] == 0.005
    assert prov["p95_cost"] == 0.005


def test_summary_by_provider_includes_wall_time_percentiles(tmp_path: Path) -> None:
    """summary_by_provider includes p50_wall_ms and p95_wall_ms per provider."""
    import json as _json
    from mullet.core.routing_history import history_path

    hp = history_path(tmp_path)
    hp.parent.mkdir(parents=True, exist_ok=True)
    wall_times = [100 * i for i in range(1, 11)]  # 100, 200, …, 1000 ms
    for wt in wall_times:
        line = {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "task_signature": "cccc",
            "task_preview": "t",
            "provider": "local",
            "model": "llama3",
            "tier": "LOW",
            "success": True,
            "error_class": None,
            "tokens_in": 500,
            "tokens_out": 200,
            "wall_time_ms": wt,
            "cost_usd": 0.0,
            "tags": [],
        }
        with open(hp, "a") as f:
            f.write(_json.dumps(line) + "\n")

    summary = summary_by_provider(base=tmp_path)
    local = summary["local"]
    assert "p50_wall_ms" in local, "p50_wall_ms missing from summary"
    assert "p95_wall_ms" in local, "p95_wall_ms missing from summary"
    # 10 values [100..1000]: p50 = 550, p95 = 955
    assert abs(local["p50_wall_ms"] - 550) < 1
    assert abs(local["p95_wall_ms"] - 955) < 1


# ---------------------------------------------------------------------------
# Cost math
# ---------------------------------------------------------------------------

def test_cost_opus_1m_in_zero_out() -> None:
    cost = estimate_cost("opus", tokens_in=1_000_000, tokens_out=0)
    assert abs(cost - 5.0) < 1e-9


def test_cost_kimi_100k_in_50k_out() -> None:
    # First KNOWN_PRICING match for "kimi" is k2.6: 0.00095/1K in, 0.004/1K out
    # 100K * 0.95/M + 50K * 4.0/M = 0.095 + 0.20 = 0.295
    cost = estimate_cost("kimi", tokens_in=100_000, tokens_out=50_000)
    assert abs(cost - 0.295) < 1e-9


def test_cost_local_is_zero() -> None:
    cost = estimate_cost("local", tokens_in=999_999, tokens_out=999_999)
    assert cost == 0.0


def test_cost_unknown_provider_is_zero() -> None:
    cost = estimate_cost("totally_unknown_xyz", tokens_in=1_000_000, tokens_out=1_000_000)
    assert cost == 0.0


def test_cost_model_name_wins_over_provider() -> None:
    # Provider "kimi" would match kimi-k2.7-code (0.00095/1K), but
    # explicit model="kimi-k2.5" (0.00060/1K in, 0.003/1K out) takes priority.
    cost = estimate_cost("kimi", tokens_in=100_000, tokens_out=50_000, model="kimi-k2.5")
    # 100K * 0.60/M + 50K * 3.0/M = 0.06 + 0.15 = 0.21
    assert abs(cost - 0.21) < 1e-9


def test_cost_model_exact_match() -> None:
    # Exact model key lookup — opus 4.8 at $0.005/1K input
    cost = estimate_cost("anthropic", tokens_in=1_000_000, tokens_out=0, model="claude-opus-4-8")
    assert abs(cost - 5.0) < 1e-9


def test_cost_stored_in_outcome(tmp_path: Path) -> None:
    o = _make_outcome(tmp_path, provider="opus", model="claude-opus-4-8", tokens_in=1_000_000, tokens_out=0)
    assert abs(o.cost_usd - 5.0) < 1e-9


# ---------------------------------------------------------------------------
# Atomic concurrency — 5 parallel writers, no lost lines
# ---------------------------------------------------------------------------

def _write_worker(args: tuple[Path, int]) -> None:
    base, idx = args
    o = make_outcome(
        task=f"concurrent task {idx}",
        provider="kimi",
        model="kimi-k2",
        tier="STANDARD",
        success=True,
        tokens_in=100,
        tokens_out=50,
        wall_time_ms=100,
    )
    record(o, base=base)


def test_concurrent_writes_no_lost_lines(tmp_path: Path) -> None:
    n = 5
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(_write_worker, [(tmp_path, i) for i in range(n)]))

    path = history_path(tmp_path)
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == n
    # Every line must be valid JSON
    for line in lines:
        obj = json.loads(line)
        assert "provider" in obj


# ---------------------------------------------------------------------------
# Serialisation fidelity
# ---------------------------------------------------------------------------

def test_from_dict_roundtrip(tmp_path: Path) -> None:
    o = make_outcome(
        task="test task",
        provider="haiku",
        model="claude-haiku-4-5",
        tier="LOW",
        success=False,
        tokens_in=200,
        tokens_out=0,
        wall_time_ms=300,
        error_class="TimeoutError",
        tags=["test", "haiku"],
    )
    record(o, base=tmp_path)
    results = list(iter_recent(limit=1, base=tmp_path))
    r = results[0]
    assert r.provider == "haiku"
    assert r.success is False
    assert r.error_class == "TimeoutError"
    assert r.tags == ["test", "haiku"]
    assert r.tier == "LOW"


# ---------------------------------------------------------------------------
# append_decision_log
# ---------------------------------------------------------------------------

def test_append_decision_log_creates_file(tmp_path):
    """append_decision_log writes a human-readable line to routing.log."""
    from mullet.core.routing_history import append_decision_log
    append_decision_log(
        provider="kimi",
        model="kimi-k2.6",
        tier="STANDARD",
        reasons=["complexity: moderate", "Kimi is cheapest cloud"],
        task_preview="write a unit test for the pool health checker",
        base=tmp_path,
    )
    log_file = tmp_path / ".mullet" / "routing.log"
    assert log_file.exists()
    content = log_file.read_text()
    assert "kimi" in content
    assert "kimi-k2.6" in content
    assert "STANDARD" in content
    assert "complexity: moderate" in content
    assert "write a unit test" in content


def test_append_decision_log_appends(tmp_path):
    """Multiple calls append lines, not overwrite."""
    from mullet.core.routing_history import append_decision_log
    for i in range(3):
        append_decision_log(
            provider="gemini",
            model="gemini-flash",
            tier="LOW",
            reasons=[f"reason-{i}"],
            task_preview=f"task {i}",
            base=tmp_path,
        )
    lines = (tmp_path / ".mullet" / "routing.log").read_text().strip().splitlines()
    assert len(lines) == 3
    assert "reason-0" in lines[0]
    assert "reason-2" in lines[2]


# --- provider_spend_5h ---

def test_provider_spend_5h_no_file(tmp_path):
    """Returns {} when routing_history.jsonl doesn't exist."""
    from mullet.core.routing_history import provider_spend_5h
    result = provider_spend_5h(base=tmp_path)
    assert result == {}


def test_provider_spend_5h_recent_entries(tmp_path):
    """Sums costs for entries within the last 5 hours."""
    from mullet.core.routing_history import make_outcome, record, provider_spend_5h
    outcome = make_outcome("test task", "kimi", "kimi-k2", "STANDARD", True, 100_000, 50_000, 500)
    record(outcome, base=tmp_path)
    result = provider_spend_5h(base=tmp_path)
    assert "kimi" in result
    assert result["kimi"] > 0.0


def test_provider_spend_5h_excludes_old_entries(tmp_path):
    """Excludes entries older than 5 hours from spend totals."""
    import json
    from datetime import datetime, timezone, timedelta
    from mullet.core.routing_history import history_path, provider_spend_5h
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
    entry = {
        "timestamp": old_ts,
        "task_signature": "abc123",
        "task_preview": "old task",
        "provider": "kimi",
        "model": "kimi-k2",
        "tier": "STANDARD",
        "success": True,
        "error_class": None,
        "tokens_in": 100000,
        "tokens_out": 50000,
        "wall_time_ms": 500,
        "cost_usd": 99.99,
        "tags": [],
    }
    history_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    history_path(tmp_path).write_text(json.dumps(entry) + "\n")
    result = provider_spend_5h(base=tmp_path)
    # Old entry should not be counted
    assert result.get("kimi", 0.0) == 0.0


# --- make_signature ---

def test_make_signature_deterministic():
    """Same inputs → same signature always."""
    from mullet.core.routing_history import make_signature
    s1 = make_signature("analyse this code", "run", "STANDARD")
    s2 = make_signature("analyse this code", "run", "STANDARD")
    assert s1 == s2


def test_make_signature_16_chars():
    """Output is exactly 16 hex characters."""
    from mullet.core.routing_history import make_signature
    sig = make_signature("any task string")
    assert len(sig) == 16
    assert all(c in "0123456789abcdef" for c in sig)


def test_make_signature_normalises_whitespace():
    """Extra whitespace is collapsed before hashing."""
    from mullet.core.routing_history import make_signature
    assert make_signature("hello   world") == make_signature("hello world")


def test_make_signature_different_tiers_differ():
    """Different tiers produce different signatures."""
    from mullet.core.routing_history import make_signature
    s_std = make_signature("task", "run", "STANDARD")
    s_crit = make_signature("task", "run", "CRITICAL")
    assert s_std != s_crit


# --- make_outcome ---

def test_make_outcome_sets_cost():
    """make_outcome auto-computes cost_usd from tokens."""
    from mullet.core.routing_history import make_outcome
    outcome = make_outcome("test task", "kimi", "kimi-k2", "STANDARD", True, 1_000_000, 0, 1000)
    # First KNOWN_PRICING match for "kimi" is k2.6 at $0.95/M in
    assert abs(outcome.cost_usd - 0.95) < 0.01


def test_make_outcome_sets_signature():
    """make_outcome computes task_signature and task_preview."""
    from mullet.core.routing_history import make_outcome, make_signature
    outcome = make_outcome("analyse this code", "local", "llama3", "LOW", True, 500, 100, 300)
    expected_sig = make_signature("analyse this code", "run", "LOW")
    assert outcome.task_signature == expected_sig
    assert "analyse" in outcome.task_preview


def test_make_outcome_truncates_and_stores_error_message():
    """make_outcome stores a truncated error_message, distinct from error_class."""
    from mullet.core.routing_history import make_outcome
    long_msg = "opencode session/prompt error: " + ("x" * 600)
    outcome = make_outcome(
        "test task", "kimi", "kimi-k2", "STANDARD", False, 0, 0, 1000,
        error_class="RuntimeError", error_message=long_msg,
    )
    assert outcome.error_class == "RuntimeError"
    assert len(outcome.error_message) == 500
    assert outcome.error_message.startswith("opencode session/prompt error:")


def test_make_outcome_error_message_defaults_to_none():
    """make_outcome with no error_message (the common success-path call) stores None."""
    from mullet.core.routing_history import make_outcome
    outcome = make_outcome("test task", "kimi", "kimi-k2", "STANDARD", True, 100, 50, 1000)
    assert outcome.error_message is None


def test_routing_outcome_from_dict_reads_error_message():
    """from_dict round-trips error_message when present."""
    from mullet.core.routing_history import RoutingOutcome
    d = {
        "timestamp": "2026-07-12T00:00:00+00:00", "task_signature": "abc",
        "task_preview": "x", "provider": "kimi", "model": "kimi-k2",
        "tier": "STANDARD", "success": False, "error_class": "RuntimeError",
        "error_message": "n_keep: 10842 >= n_ctx: 4096",
        "tokens_in": 0, "tokens_out": 0, "wall_time_ms": 100, "cost_usd": 0.0,
    }
    outcome = RoutingOutcome.from_dict(d)
    assert outcome.error_message == "n_keep: 10842 >= n_ctx: 4096"


def test_routing_outcome_from_dict_missing_error_message_is_none():
    """from_dict on an OLD jsonl line (no error_message field) defaults to None."""
    from mullet.core.routing_history import RoutingOutcome
    d = {
        "timestamp": "2026-07-12T00:00:00+00:00", "task_signature": "abc",
        "task_preview": "x", "provider": "kimi", "model": "kimi-k2",
        "tier": "STANDARD", "success": True, "error_class": None,
        "tokens_in": 0, "tokens_out": 0, "wall_time_ms": 100, "cost_usd": 0.0,
    }
    outcome = RoutingOutcome.from_dict(d)
    assert outcome.error_message is None


def test_summary_by_provider_tracks_rate_limit_rate(tmp_path: Path) -> None:
    """summary_by_provider includes rate_limit_rate computed from error_class field."""
    import json as _json
    from mullet.core.routing_history import history_path

    hp = history_path(tmp_path)
    hp.parent.mkdir(parents=True, exist_ok=True)

    _base_entry = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "task_signature": "dddd",
        "task_preview": "t",
        "model": "deepseek",
        "tier": "STANDARD",
        "tokens_in": 100,
        "tokens_out": 50,
        "wall_time_ms": 1000,
        "cost_usd": 0.0,
        "tags": [],
    }

    entries = [
        # 4 rate-limit errors
        {**_base_entry, "provider": "openrouter", "success": False, "error_class": "rate_limit"},
        {**_base_entry, "provider": "openrouter", "success": False, "error_class": "rate_limit"},
        {**_base_entry, "provider": "openrouter", "success": False, "error_class": "rate_limit"},
        {**_base_entry, "provider": "openrouter", "success": False, "error_class": "rate_limit"},
        # 6 successes
        {**_base_entry, "provider": "openrouter", "success": True, "error_class": None},
        {**_base_entry, "provider": "openrouter", "success": True, "error_class": None},
        {**_base_entry, "provider": "openrouter", "success": True, "error_class": None},
        {**_base_entry, "provider": "openrouter", "success": True, "error_class": None},
        {**_base_entry, "provider": "openrouter", "success": True, "error_class": None},
        {**_base_entry, "provider": "openrouter", "success": True, "error_class": None},
        # kimi: no rate limits
        {**_base_entry, "provider": "kimi", "success": True, "error_class": None},
        {**_base_entry, "provider": "kimi", "success": True, "error_class": None},
        {**_base_entry, "provider": "kimi", "success": True, "error_class": None},
    ]
    with open(hp, "w") as f:
        for e in entries:
            f.write(_json.dumps(e) + "\n")

    summary = summary_by_provider(base=tmp_path)

    # openrouter: 4/10 = 40% rate-limit rate
    assert "rate_limit_rate" in summary["openrouter"]
    assert abs(summary["openrouter"]["rate_limit_rate"] - 0.4) < 0.001

    # kimi: 0% rate-limit rate
    assert summary["kimi"]["rate_limit_rate"] == 0.0

    # success rates also correct
    assert abs(summary["openrouter"]["success_rate"] - 0.6) < 0.001
    assert summary["kimi"]["success_rate"] == 1.0


def test_daily_cost_summary_empty(tmp_path):
    from mullet.core.routing_history import daily_cost_summary
    result = daily_cost_summary(base=tmp_path)
    assert result == {}


def test_daily_cost_summary_groups_by_day(tmp_path):
    from mullet.core.routing_history import daily_cost_summary, RoutingOutcome, record
    # Write 3 records: 2 on one date, 1 on another
    for i, (date, cost) in enumerate([("2026-06-14", 1.5), ("2026-06-14", 0.5), ("2026-06-15", 2.0)]):
        o = RoutingOutcome(
            timestamp=f"{date}T12:0{i}:00+00:00",
            task_signature="sig",
            task_preview="test",
            provider="openrouter",
            model="test",
            tier="STANDARD",
            success=True,
            error_class=None,
            tokens_in=100,
            tokens_out=50,
            wall_time_ms=100,
            cost_usd=cost,
        )
        record(o, base=tmp_path)
    result = daily_cost_summary(base=tmp_path, limit_days=365)
    assert "2026-06-14" in result["days"]
    assert abs(result["days"]["2026-06-14"] - 2.0) < 1e-6
    assert abs(result["days"]["2026-06-15"] - 2.0) < 1e-6
    assert result["count"] == 2
    assert result["max"] == pytest.approx(2.0)


def test_history_path_default_uses_state_dir_and_does_not_touch_cwd(tmp_path, monkeypatch):
    """history_path() with no base must not create .mullet/ in the working directory.

    Reading the path and writing through record() should create the file in
    Mullet's state directory instead. Tests run with a throwaway HOME, so the
    state directory is safe to write to.
    """
    import mullet.core.routing_history as rh
    from mullet.core.paths import state_dir

    monkeypatch.chdir(tmp_path)
    path = rh.history_path()
    assert path == state_dir() / "routing_history.jsonl"
    assert not (tmp_path / ".mullet").exists()
    assert list(tmp_path.iterdir()) == []

    # Reading the path must not create anything in cwd either.
    assert list(rh.iter_recent(limit=10)) == []
    assert not (tmp_path / ".mullet").exists()

    # Only writing should create the file, and it should be under the state dir.
    outcome = rh.make_outcome("test task", "kimi", "kimi-k2", "STANDARD", True, 100, 50, 100)
    rh.record(outcome)
    assert path.exists()
    assert not (tmp_path / ".mullet").exists()
    assert list(rh.iter_recent(limit=10)) != []


def test_history_path_base_argument_still_uses_dot_mullet(tmp_path):
    """Passing an explicit base keeps the old .mullet/ layout."""
    from mullet.core.routing_history import history_path

    assert history_path(tmp_path) == tmp_path / ".mullet" / "routing_history.jsonl"
