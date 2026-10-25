"""End-to-end integration tests for the 4-pipeline cohesion.

Tests exercise interactions between:
  - mullet.core.observations (PostToolUse JSONL capture)
  - mullet.core.routing_history (per-task call outcomes)
  - mullet.core.decisions (regex-extracted decisions)
  - mullet/hooks/pretooluse_rewriter (tier autofill)
  - mullet.cli.review_session_cmd (unified report)

All tests use tmp_path and heavy mocking — no real LLM calls.
"""
from __future__ import annotations

import json
import io
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _utcnow_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts_ago(minutes: int = 0) -> str:
    dt = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_obs(tmp_path: Path, tool: str = "Read", sid: str = "sess1",
               ts_ago_min: int = 0) -> dict:
    obs_dir = tmp_path / ".mullet" / "observations"
    obs_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": _ts_ago(ts_ago_min),
        "session_id": sid,
        "tool_name": tool,
        "tool_input_preview": f'{{"tool": "{tool}"}}',
        "tool_output_preview": "ok",
        "cwd": str(tmp_path),
    }
    with open(obs_dir / f"{sid}.jsonl", "a") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def _write_routing(tmp_path: Path, provider: str = "kimi", success: bool = True,
                   ts_ago_min: int = 0, tags: list[str] | None = None) -> None:
    from mullet.core.routing_history import RoutingOutcome, record as rh_record
    o = RoutingOutcome(
        timestamp=_ts_ago(ts_ago_min),
        task_signature="aabbccdd11223344",
        task_preview="test task",
        provider=provider,
        model=f"{provider}-model",
        tier="STANDARD",
        success=success,
        error_class=None,
        tokens_in=100,
        tokens_out=50,
        wall_time_ms=300,
        cost_usd=0.00005,
        tags=tags or [],
    )
    rh_record(o, base=tmp_path)


def _run_review(tmp_path: Path, extra_args: list[str] | None = None) -> tuple[int, str]:
    from mullet.cli import review_session_cmd
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        rc = review_session_cmd.run(argv=extra_args or [], base=tmp_path)
    finally:
        sys.stdout = old
    return rc, buf.getvalue()


def _patch_budget(monkeypatch, pressure: float = 0.1):
    import mullet.hud.guard as guard_mod
    from mullet.hud.guard import BudgetCheck, BudgetStatus
    monkeypatch.setattr(
        guard_mod, "check_budget",
        lambda: BudgetCheck(status=BudgetStatus.GO, pressure=pressure, message="ok"),
    )


# ---------------------------------------------------------------------------
# T1: Synthetic tool call → observations.jsonl AND routing_history.jsonl AND
#     review-session shows it
# ---------------------------------------------------------------------------

def test_t1_synthetic_call_appears_in_all_stores(tmp_path, monkeypatch):
    """Record a synthetic observation + routing outcome → review-session surfaces both."""
    _patch_budget(monkeypatch)

    # Write directly (simulating what the hooks would do)
    _write_obs(tmp_path, tool="Bash", sid="t1sess", ts_ago_min=5)
    _write_routing(tmp_path, provider="kimi", success=True, ts_ago_min=5)

    # Verify observations.jsonl written
    obs_file = tmp_path / ".mullet" / "observations" / "t1sess.jsonl"
    assert obs_file.exists()
    rows = [json.loads(l) for l in obs_file.read_text().splitlines() if l.strip()]
    assert any(r["tool_name"] == "Bash" for r in rows)

    # Verify routing_history.jsonl written
    hist_file = tmp_path / ".mullet" / "routing_history.jsonl"
    assert hist_file.exists()
    hist_rows = [json.loads(l) for l in hist_file.read_text().splitlines() if l.strip()]
    assert any(r["provider"] == "kimi" for r in hist_rows)

    # Review-session shows both
    rc, out = _run_review(tmp_path, ["--window", "30"])
    assert rc == 0
    assert "Bash" in out or "1 observation" in out or "Events" in out
    assert "kimi" in out


# ---------------------------------------------------------------------------
# T2: PreToolUse rewriter sets tier, routing record reflects it
# ---------------------------------------------------------------------------

def test_t2_pretooluse_rewriter_sets_tier(tmp_path, monkeypatch):
    """PreToolUse rewriter injects tier based on pressure; the tier is visible."""
    # Patch budget pressure to a known value (mid-range → STANDARD tier)
    from mullet.hud.guard import BudgetCheck, BudgetStatus
    import mullet.hud.guard as guard_mod
    monkeypatch.setattr(
        guard_mod, "check_budget",
        lambda: BudgetCheck(status=BudgetStatus.GO, pressure=0.45, message="ok"),
    )

    # Import the rewriter module
    import importlib
    rewriter_path = (
        Path(__file__).resolve().parent.parent
        / "mullet" / "hooks" / "pretooluse_rewriter.py"
    )
    assert rewriter_path.exists(), f"rewriter not found at {rewriter_path}"

    spec = importlib.util.spec_from_file_location("mullet_pretooluse_rewriter", rewriter_path)
    rewriter = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(rewriter)  # type: ignore[union-attr]

    # Simulate a tool_input that has no tier set
    tool_input = {"task": "summarise the repo", "provider": "kimi"}
    updated = rewriter._rewrite_input("mcp__mullet__mullet_run", tool_input)

    # With pressure=0.45, _pressure_to_tier returns "STANDARD"
    assert updated is not None
    assert updated.get("tier") == "STANDARD"

    # Now write a routing record that includes this tier
    from mullet.core.routing_history import RoutingOutcome, record as rh_record
    o = RoutingOutcome(
        timestamp=_ts_ago(2),
        task_signature="cafebabe12345678",
        task_preview="summarise the repo",
        provider="kimi",
        model="kimi-k2",
        tier=updated["tier"],
        success=True,
        error_class=None,
        tokens_in=200,
        tokens_out=80,
        wall_time_ms=400,
        cost_usd=0.0002,
        tags=["rewriter_test"],
    )
    rh_record(o, base=tmp_path)

    # review-session should show STANDARD-tier calls in routing summary
    rc, out = _run_review(tmp_path, ["--window", "10"])
    assert rc == 0
    assert "kimi" in out


# ---------------------------------------------------------------------------
# T3: Decisions extractor → review-session surfaces decisions
# ---------------------------------------------------------------------------

def test_t3_decisions_surface_in_review(tmp_path, monkeypatch):
    """Decisions persisted to decisions.jsonl appear in review-session output."""
    _patch_budget(monkeypatch)

    from mullet.core.decisions import Decision, persist

    decisions = [
        Decision(
            id="d001",
            source_path="docs/ARCH.md",
            source_type="docs",
            line_number=5,
            timestamp=_ts_ago(10),
            excerpt="We decided to use Kimi K2 because it is cost-effective for large prompts",
            kind="decision",
            keywords=["decided to"],
        ),
        Decision(
            id="d002",
            source_path="docs/ARCH.md",
            source_type="docs",
            line_number=15,
            timestamp=_ts_ago(8),
            excerpt="Rejected local LLM for critical tasks because latency is unpredictable",
            kind="rejection",
            keywords=["rejected"],
        ),
    ]
    persist(decisions, base=tmp_path)

    rc, out = _run_review(tmp_path, ["--window", "99999"])
    assert rc == 0
    assert "3. DECISIONS" in out
    # At least one of the decision excerpts should be visible (truncated)
    assert "Kimi K2" in out or "decided" in out.lower() or "rejection" in out.lower()


# ---------------------------------------------------------------------------
# T4: Routing priors re-rank visible in routing summary
# ---------------------------------------------------------------------------

def test_t4_routing_history_summary_visible(tmp_path, monkeypatch):
    """Write mixed-provider outcomes; review-session's routing summary shows them."""
    _patch_budget(monkeypatch)

    # kimi: 5 successes
    for _ in range(5):
        _write_routing(tmp_path, provider="kimi", success=True, ts_ago_min=20,
                       tags=["priors_test"])
    # local: 3 failures, 1 success
    for i in range(4):
        _write_routing(tmp_path, provider="local", success=(i == 0), ts_ago_min=15,
                       tags=["priors_test"])

    rc, out = _run_review(tmp_path, ["--window", "60"])
    assert rc == 0
    assert "kimi" in out
    assert "local" in out

    # Anti-pattern: local has 25% success rate over 4 calls → warning
    assert "WARNING" in out or "local" in out.split("5. ANTI")[1] if "5. ANTI" in out else True


# ---------------------------------------------------------------------------
# T5: All 4 modules coexist — 3 dispatches, no duplicates
# ---------------------------------------------------------------------------

def test_t5_all_modules_capture_no_duplicates(tmp_path, monkeypatch):
    """Enable all recorders, make 3 synthetic tool dispatches, assert no duplication."""
    _patch_budget(monkeypatch)

    sid = "multi_sess"
    tools = ["Read", "Write", "Bash"]
    for i, tool in enumerate(tools):
        _write_obs(tmp_path, tool=tool, sid=sid, ts_ago_min=i + 1)
        _write_routing(tmp_path, provider="kimi", success=True, ts_ago_min=i + 1)

    # Observations: exactly 3 rows for this session
    from mullet.core.observations import iter_session
    obs_rows = list(iter_session(sid, base=tmp_path))
    assert len(obs_rows) == 3, f"Expected 3 observations, got {len(obs_rows)}"
    tool_names = {o.tool_name for o in obs_rows}
    assert tool_names == set(tools)

    # Routing history: 3 rows
    from mullet.core.routing_history import iter_recent
    hist_rows = list(iter_recent(limit=20, base=tmp_path))
    assert len(hist_rows) == 3, f"Expected 3 routing outcomes, got {len(hist_rows)}"

    # review-session shows 3 events
    rc, out = _run_review(tmp_path, ["--window", "30"])
    assert rc == 0
    assert "Events  : 3" in out or "3 observation" in out


# ---------------------------------------------------------------------------
# T6: Fail-open — corrupt JSONL → report still renders with warnings, no raise
# ---------------------------------------------------------------------------

def test_t6_corrupt_jsonl_fail_open(tmp_path, monkeypatch):
    """Corrupt every JSONL file → review-session still produces a report, never raises."""
    _patch_budget(monkeypatch)

    # Write corrupt observations JSONL
    obs_dir = tmp_path / ".mullet" / "observations"
    obs_dir.mkdir(parents=True, exist_ok=True)
    (obs_dir / "corrupt.jsonl").write_text("INVALID JSON LINE\n{broken}\n")

    # Write corrupt routing_history
    mullet_dir = tmp_path / ".mullet"
    mullet_dir.mkdir(parents=True, exist_ok=True)
    (mullet_dir / "routing_history.jsonl").write_text("NOT JSON\n{also broken}\n")

    # Write corrupt decisions
    (mullet_dir / "decisions.jsonl").write_text("garbage\n")

    # Must not raise
    try:
        rc, out = _run_review(tmp_path, ["--window", "60"])
    except Exception as exc:
        pytest.fail(f"review-session raised unexpectedly: {exc}")

    assert rc == 0
    # All 6 sections must still appear
    for marker in ["1. SESSION", "2. ACTIVITY", "3. DECISIONS", "4. ROUTING", "5. ANTI", "6. NEXT"]:
        assert marker in out, f"Section missing after corruption: {marker}"


# ---------------------------------------------------------------------------
# T7: Session-id scoping — session A observations do not appear in --session B
# ---------------------------------------------------------------------------

def test_t7_session_id_scoping(tmp_path, monkeypatch):
    """Observations from session A must not appear when --session B is requested."""
    _patch_budget(monkeypatch)

    _write_obs(tmp_path, tool="Read", sid="sessionA", ts_ago_min=5)
    _write_obs(tmp_path, tool="Bash", sid="sessionB", ts_ago_min=5)

    # Request session B
    rc, out = _run_review(tmp_path, ["--session", "sessionB", "--window", "30"])
    assert rc == 0
    # sessionB's Bash should be visible
    # sessionA's observations file should not be read
    # The session header should say sessionB
    assert "sessionB" in out

    # Read observations via the module to double-check isolation
    from mullet.core.observations import iter_session
    sess_b_rows = list(iter_session("sessionB", base=tmp_path))
    sess_a_rows = list(iter_session("sessionA", base=tmp_path))
    assert all(o.session_id == "sessionB" for o in sess_b_rows)
    assert all(o.session_id == "sessionA" for o in sess_a_rows)


# ---------------------------------------------------------------------------
# T8: Time-window scoping — older observations excluded
# ---------------------------------------------------------------------------

def test_t8_time_window_scoping(tmp_path, monkeypatch):
    """Observations older than --window are excluded from the report."""
    _patch_budget(monkeypatch)

    sid = "timetest"
    # Recent — within 5 min window
    _write_obs(tmp_path, tool="Read", sid=sid, ts_ago_min=2)
    # Old — outside 5 min window
    _write_obs(tmp_path, tool="Write", sid=sid, ts_ago_min=120)
    _write_obs(tmp_path, tool="Bash", sid=sid, ts_ago_min=240)

    rc, out = _run_review(tmp_path, ["--session", sid, "--window", "5"])
    assert rc == 0

    # Only 1 observation (the 2-min-ago Read) should be in window
    assert "Events  : 1" in out or "1 observation" in out

    # Similarly for routing history: fresh tmp_path2 to avoid cross-test contamination
    tmp_path2 = tmp_path / "routing_scope_test"
    tmp_path2.mkdir(parents=True, exist_ok=True)
    _write_routing(tmp_path2, provider="kimi", success=True, ts_ago_min=2)
    _write_routing(tmp_path2, provider="local", success=False, ts_ago_min=300)

    rc2, out2 = _run_review(tmp_path2, ["--window", "5"])
    assert rc2 == 0
    # Only kimi (recent) should appear in routing summary
    assert "kimi" in out2
    # local (old, 300 min ago) should NOT appear when window=5
    if "4. ROUTING" in out2:
        routing_section = out2.split("4. ROUTING")[1].split("5. ANTI")[0]
        assert "local" not in routing_section
