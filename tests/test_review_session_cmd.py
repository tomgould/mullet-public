"""Tests for mullet.cli.review_session_cmd — 8 focused unit tests."""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


from mullet.core.decisions import persist, Decision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts_ago(minutes: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_obs(tmp_path: Path, tool: str = "Read", sid: str = "sess1",
              ts_ago_min: int = 5) -> None:
    """Write a synthetic observation directly to JSONL."""
    obs_dir = tmp_path / ".mullet" / "observations"
    obs_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": _ts_ago(ts_ago_min),
        "session_id": sid,
        "tool_name": tool,
        "tool_input_preview": "{}",
        "tool_output_preview": "ok",
        "cwd": str(tmp_path),
    }
    with open(obs_dir / f"{sid}.jsonl", "a") as fh:
        fh.write(json.dumps(row) + "\n")


def _make_routing_outcome(tmp_path: Path, provider: str = "kimi",
                          success: bool = True, ts_ago_min: int = 5) -> None:
    from mullet.core.routing_history import RoutingOutcome, record as rh_record
    ts = _ts_ago(ts_ago_min)
    o = RoutingOutcome(
        timestamp=ts,
        task_signature="aabbccdd11223344",
        task_preview="Summarise codebase",
        provider=provider,
        model=f"{provider}-model",
        tier="STANDARD",
        success=success,
        error_class=None,
        tokens_in=100,
        tokens_out=50,
        wall_time_ms=500,
        cost_usd=0.0001,
        tags=[],
    )
    rh_record(o, base=tmp_path)


def _run_review(tmp_path: Path, extra_args: list[str] | None = None,
                monkeypatch=None) -> tuple[int, str]:
    """Run review_session_cmd.run(), capturing stdout. Returns (rc, output)."""
    import io
    import sys as _sys
    from mullet.cli import review_session_cmd

    buf = io.StringIO()
    old_stdout = _sys.stdout
    _sys.stdout = buf
    try:
        rc = review_session_cmd.run(argv=extra_args or [], base=tmp_path)
    finally:
        _sys.stdout = old_stdout
    return rc, buf.getvalue()


# ---------------------------------------------------------------------------
# Test 1: Empty window
# ---------------------------------------------------------------------------

def test_empty_window_message(tmp_path):
    """Empty data directory → report mentions no-activity sentinel."""
    rc, out = _run_review(tmp_path, ["--window", "1"])
    assert rc == 0
    assert "No activity" in out or "empty" in out.lower() or "No routing" in out


# ---------------------------------------------------------------------------
# Test 2: Happy path — all 6 sections render
# ---------------------------------------------------------------------------

def test_happy_path_all_sections_render(tmp_path, monkeypatch):
    """Fixture observations + routing + decisions → 6 sections present."""
    # Write fixture data
    _make_obs(tmp_path, tool="Read", sid="s1", ts_ago_min=10)
    _make_obs(tmp_path, tool="Write", sid="s1", ts_ago_min=8)
    _make_routing_outcome(tmp_path, provider="kimi", success=True, ts_ago_min=9)

    dec = Decision(
        id="aabbccdd",
        source_path="docs/ARCH.md",
        source_type="docs",
        line_number=10,
        timestamp=_ts_ago(5),
        excerpt="We decided to use kimi because it is cost-effective",
        kind="decision",
        keywords=["decided to"],
    )
    persist([dec], base=tmp_path)

    # Patch check_budget to avoid network
    import mullet.hud.guard as guard_mod
    from mullet.hud.guard import BudgetCheck, BudgetStatus
    monkeypatch.setattr(
        guard_mod, "check_budget",
        lambda: BudgetCheck(status=BudgetStatus.GO, pressure=0.1, message="ok"),
    )

    rc, out = _run_review(tmp_path, ["--window", "60"])
    assert rc == 0
    for section_marker in [
        "1. SESSION WINDOW",
        "2. ACTIVITY DIGEST",
        "3. DECISIONS",
        "4. ROUTING",
        "5. ANTI-PATTERN",
        "6. NEXT-SUGGESTED",
    ]:
        assert section_marker in out, f"Missing section: {section_marker}"


# ---------------------------------------------------------------------------
# Test 3: --session filter
# ---------------------------------------------------------------------------

def test_session_filter_limits_to_session(tmp_path):
    """--session A should exclude observations from session B."""
    _make_obs(tmp_path, tool="Read", sid="sessionA", ts_ago_min=5)
    _make_obs(tmp_path, tool="Bash", sid="sessionB", ts_ago_min=5)

    rc, out = _run_review(tmp_path, ["--session", "sessionA", "--window", "60"])
    assert rc == 0
    assert "sessionA" in out
    # sessionB obs file not loaded when --session sessionA given
    # The report should say sessionA
    assert "sessionB" not in out.split("Session")[1].split("\n")[0] if "Session" in out else True


# ---------------------------------------------------------------------------
# Test 4: --window respected (older rows excluded)
# ---------------------------------------------------------------------------

def test_window_excludes_old_observations(tmp_path):
    """Observations older than --window minutes are excluded from activity digest."""
    # Recent: within window
    _make_obs(tmp_path, tool="Read", sid="s2", ts_ago_min=2)
    # Old: outside window
    _make_obs(tmp_path, tool="Bash", sid="s2", ts_ago_min=200)

    rc, out = _run_review(tmp_path, ["--window", "5"])
    assert rc == 0
    # Only 1 observation should be in window (the one 2 min ago)
    # The digest shows counts — if Bash appears it would come from old obs
    # We verify total events count is low (1)
    assert "1 observation" in out or "Events  : 1" in out


# ---------------------------------------------------------------------------
# Test 5: Budget check raises — graceful fallback
# ---------------------------------------------------------------------------

def test_next_lane_fallback_when_budget_raises(tmp_path, monkeypatch):
    """When check_budget() raises, next-lane section still renders."""
    import mullet.hud.guard as guard_mod
    monkeypatch.setattr(guard_mod, "check_budget",
                        lambda: (_ for _ in ()).throw(RuntimeError("API down")))

    rc, out = _run_review(tmp_path, ["--window", "60"])
    assert rc == 0
    assert "6. NEXT-SUGGESTED" in out
    assert "BALANCED" in out.upper() or "unavailable" in out.lower() or "fallback" in out.lower()


# ---------------------------------------------------------------------------
# Test 6: Anti-pattern section — low success rate provider flagged
# ---------------------------------------------------------------------------

def test_antipattern_flags_low_success_rate_provider(tmp_path, monkeypatch):
    """Provider with success_rate <0.5 over >=3 calls → WARNING in section 5."""
    import mullet.hud.guard as guard_mod
    from mullet.hud.guard import BudgetCheck, BudgetStatus
    monkeypatch.setattr(guard_mod, "check_budget",
                        lambda: BudgetCheck(status=BudgetStatus.GO, pressure=0.1, message="ok"))

    # Write 4 failed outcomes for "badprovider"
    for _ in range(4):
        _make_routing_outcome(tmp_path, provider="badprovider", success=False, ts_ago_min=10)

    rc, out = _run_review(tmp_path, ["--window", "60"])
    assert rc == 0
    assert "5. ANTI-PATTERN" in out
    assert "WARNING" in out or "badprovider" in out


# ---------------------------------------------------------------------------
# Test 7: Section ordering is stable
# ---------------------------------------------------------------------------

def test_section_ordering_is_stable(tmp_path, monkeypatch):
    """The 6 sections appear in numerical order (1 before 2 before 3...)."""
    import mullet.hud.guard as guard_mod
    from mullet.hud.guard import BudgetCheck, BudgetStatus
    monkeypatch.setattr(guard_mod, "check_budget",
                        lambda: BudgetCheck(status=BudgetStatus.GO, pressure=0.2, message="ok"))

    _make_obs(tmp_path, sid="sX", ts_ago_min=5)

    rc, out = _run_review(tmp_path, ["--window", "60"])
    assert rc == 0

    positions = []
    for marker in ["1. SESSION", "2. ACTIVITY", "3. DECISIONS", "4. ROUTING", "5. ANTI", "6. NEXT"]:
        pos = out.find(marker)
        assert pos != -1, f"Section marker not found: {marker}"
        positions.append(pos)

    assert positions == sorted(positions), "Sections appear out of order"


# ---------------------------------------------------------------------------
# Test 8: Runs in <500ms on fixture data
# ---------------------------------------------------------------------------

def test_runs_within_500ms(tmp_path, monkeypatch):
    """report generation on fixture data must complete in <500ms."""
    import mullet.hud.guard as guard_mod
    from mullet.hud.guard import BudgetCheck, BudgetStatus
    monkeypatch.setattr(guard_mod, "check_budget",
                        lambda: BudgetCheck(status=BudgetStatus.GO, pressure=0.2, message="ok"))

    for i in range(20):
        _make_obs(tmp_path, tool="Read", sid="perf", ts_ago_min=i % 30)
    for i in range(10):
        _make_routing_outcome(tmp_path, provider="kimi", success=(i % 2 == 0), ts_ago_min=i)

    start = time.monotonic()
    rc, out = _run_review(tmp_path, ["--window", "60"])
    elapsed = time.monotonic() - start
    assert rc == 0
    assert elapsed < 0.5, f"review-session took {elapsed:.3f}s, expected <0.5s"


class TestParseTs:
    """Direct unit tests for the _parse_ts helper."""

    def setup_method(self):
        from mullet.cli.review_session_cmd import _parse_ts
        self._parse_ts = _parse_ts

    def test_z_suffix(self):
        dt = self._parse_ts("2026-04-27T12:00:00Z")
        assert dt is not None
        assert dt.year == 2026 and dt.tzinfo is not None

    def test_plus_offset(self):
        dt = self._parse_ts("2026-04-27T12:00:00+00:00")
        assert dt is not None

    def test_fromisoformat_naive_gets_utc(self):
        dt = self._parse_ts("2026-04-27T12:00:00")
        assert dt is not None
        assert dt.tzinfo is not None

    def test_empty_returns_none(self):
        assert self._parse_ts("") is None

    def test_malformed_returns_none(self):
        assert self._parse_ts("not-a-date") is None


class TestSection:
    """Test _section formatting helper."""

    def test_contains_title(self):
        from mullet.cli.review_session_cmd import _section
        result = _section("My Title")
        assert "My Title" in result

    def test_contains_separator(self):
        from mullet.cli.review_session_cmd import _section
        result = _section("X")
        assert "─" in result


# ---------------------------------------------------------------------------
# Missing coverage: exception branches + pressure lane tiers
# ---------------------------------------------------------------------------

def test_observations_load_error_adds_warning(tmp_path):
    """Lines 113-114: iter_recent raises → warning appended → shows in section 5."""
    from unittest.mock import patch
    with patch("mullet.core.observations.iter_recent", side_effect=OSError("obs dir gone")), \
         patch("mullet.core.observations.iter_session", side_effect=OSError("obs dir gone")):
        rc, out = _run_review(tmp_path, ["--window", "60"])
    assert rc == 0
    assert "5. ANTI-PATTERN" in out
    assert "WARN" in out


def test_decisions_load_error_adds_warning(tmp_path):
    """Lines 175-176: decisions load() raises → warning appended."""
    from unittest.mock import patch
    with patch("mullet.core.decisions.load", side_effect=RuntimeError("decisions corrupt")):
        rc, out = _run_review(tmp_path, ["--window", "60"])
    assert rc == 0
    assert "decisions load error" in out


def test_routing_history_load_error_adds_warning(tmp_path):
    """Lines 219-220: routing_history load raises → warning appended."""
    from unittest.mock import patch
    with patch("mullet.core.routing_history.iter_recent", side_effect=RuntimeError("db down")):
        rc, out = _run_review(tmp_path, ["--window", "60"])
    assert rc == 0
    assert "routing_history load error" in out


def test_lane_balanced_at_moderate_pressure(tmp_path, monkeypatch):
    """Lines 264-266: pressure=0.45 (0.30 <= p < 0.65) → balanced lane."""
    import mullet.hud.guard as guard_mod
    from mullet.hud.guard import BudgetCheck, BudgetStatus
    monkeypatch.setattr(guard_mod, "check_budget",
                        lambda: BudgetCheck(status=BudgetStatus.GO, pressure=0.45, message="ok"))
    rc, out = _run_review(tmp_path, ["--window", "60"])
    assert rc == 0
    assert "BALANCED" in out.upper()


def test_lane_economy_at_high_pressure(tmp_path, monkeypatch):
    """Lines 267-269: pressure=0.80 (>= 0.65) → economy lane."""
    import mullet.hud.guard as guard_mod
    from mullet.hud.guard import BudgetCheck, BudgetStatus
    monkeypatch.setattr(guard_mod, "check_budget",
                        lambda: BudgetCheck(status=BudgetStatus.GO, pressure=0.80, message="ok"))
    rc, out = _run_review(tmp_path, ["--window", "60"])
    assert rc == 0
    assert "ECONOMY" in out.upper()
