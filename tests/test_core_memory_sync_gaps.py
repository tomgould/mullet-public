"""Gap tests for mullet.core.memory_sync — targets uncovered lines."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import mullet.core.memory_sync as ms_mod
from mullet.core.memory_sync import (
    _get_store,
    _write_fallback,
    _load_fallback_ids,
    sync_observation_summary,
    sync_decisions,
    backfill_all,
)
from mullet.core.observation_summary import SessionSummary
from mullet.core.decisions import Decision


# ---------------------------------------------------------------------------
# _get_store  (lines 42-51)
# ---------------------------------------------------------------------------

def test_get_store_import_failure_returns_none():
    """Lines 42-51: ImportError / Exception → None and warning logged."""
    with patch("mullet.knowledge.engine.memory.MemoryStore", side_effect=ImportError("no module")):
        result = _get_store()
    assert result is None


# ---------------------------------------------------------------------------
# _write_fallback  (lines 68-69)
# ---------------------------------------------------------------------------

def test_write_fallback_oserror_logged(tmp_path, caplog):
    """Lines 68-69: OSError during fallback write is logged as debug."""
    with caplog.at_level("DEBUG", logger="mullet.core.memory_sync"):
        with patch.object(Path, "open", side_effect=OSError("disk full")):
            _write_fallback({"id": "x"}, base=tmp_path)
    assert "fallback write failed" in caplog.text


# ---------------------------------------------------------------------------
# _load_fallback_ids  (lines 82, 86-87)
# ---------------------------------------------------------------------------

def test_load_fallback_ids_skips_blank_lines(tmp_path):
    """Line 82: blank lines in fallback JSONL are skipped."""
    fb = tmp_path / ".mullet" / "memory_sync.jsonl"
    fb.parent.mkdir(parents=True, exist_ok=True)
    fb.write_text(
        json.dumps({"id": "a"}) + "\n\n" + json.dumps({"id": "b"}) + "\n"
    )
    seen = _load_fallback_ids(base=tmp_path)
    assert seen == {"a", "b"}


def test_load_fallback_ids_exception_returns_empty(tmp_path, caplog):
    """Lines 86-87: corrupt fallback file → empty set, no raise."""
    fb = tmp_path / ".mullet" / "memory_sync.jsonl"
    fb.parent.mkdir(parents=True, exist_ok=True)
    fb.write_text("not json")
    seen = _load_fallback_ids(base=tmp_path)
    assert seen == set()


# ---------------------------------------------------------------------------
# sync_observation_summary — asdict exception  (lines 139-140)
# ---------------------------------------------------------------------------

def test_sync_observation_summary_asdict_exception_falls_back(tmp_path):
    """Lines 139-140: asdict raises → fallback getattr dict is used."""
    summary = _make_summary("sess-asdict")
    store = MagicMock()
    store.store.return_value = "mem-id"

    with patch("mullet.core.memory_sync._get_store", return_value=store):
        with patch("mullet.core.memory_sync.asdict", side_effect=RuntimeError("fail")):
            mem_id = sync_observation_summary(summary, base=tmp_path)

    assert mem_id == "mem-id"
    # Ensure store.store was called with metadata containing data
    call_kwargs = store.store.call_args.kwargs
    assert "data" in call_kwargs["metadata"]
    assert call_kwargs["metadata"]["data"]["session_id"] == "sess-asdict"


# ---------------------------------------------------------------------------
# sync_decisions — fallback dedup continue  (line 228)
# ---------------------------------------------------------------------------

def test_sync_decisions_fallback_skips_existing_ids(tmp_path):
    """Line 228: decisions already in fallback JSONL are skipped."""
    fb = tmp_path / ".mullet" / "memory_sync.jsonl"
    fb.parent.mkdir(parents=True, exist_ok=True)
    fb.write_text(json.dumps({"id": "id001"}) + "\n")

    _write_decisions(tmp_path, [Decision(
        id="id001",
        source_path="a.md",
        source_type="docs",
        line_number=1,
        timestamp="2026-04-24T08:00:00Z",
        excerpt="old",
        kind="decision",
        keywords=["decided to"],
    )])

    with patch("mullet.core.memory_sync._get_store", return_value=None):
        count = sync_decisions(base=tmp_path)

    assert count == 0


# ---------------------------------------------------------------------------
# sync_decisions — store.store exception  (lines 260-261)
# ---------------------------------------------------------------------------

def test_sync_decisions_store_exception_continues(tmp_path, caplog):
    """Lines 260-261: single decision store failure is logged, others proceed."""
    decisions = [
        _make_decision("idOK"),
        _make_decision("idFAIL"),
    ]
    _write_decisions(tmp_path, decisions)

    store = MagicMock()
    def _side_effect(*, key, **kwargs):
        if "idFAIL" in key:
            raise RuntimeError("disk full")
        return "mem-id"
    store.store.side_effect = _side_effect

    with patch("mullet.core.memory_sync._get_store", return_value=store):
        with caplog.at_level("WARNING", logger="mullet.core.memory_sync"):
            count = sync_decisions(base=tmp_path)

    assert count == 1
    assert "failed to store decision" in caplog.text


# ---------------------------------------------------------------------------
# sync_decisions — outer exception  (lines 270-272)
# ---------------------------------------------------------------------------

def test_sync_decisions_outer_exception_returns_zero(tmp_path, caplog):
    """Lines 270-272: unexpected exception → 0, warning logged."""
    _write_decisions(tmp_path, [_make_decision()])
    with patch("mullet.core.memory_sync._get_store", side_effect=RuntimeError("boom")):
        with caplog.at_level("WARNING", logger="mullet.core.memory_sync"):
            count = sync_decisions(base=tmp_path)
    assert count == 0
    assert "sync_decisions failed" in caplog.text


# ---------------------------------------------------------------------------
# backfill_all — per-summary exception  (lines 315-320)
# ---------------------------------------------------------------------------

def test_backfill_all_skips_corrupt_summary_file(tmp_path, caplog):
    """Lines 315-320: corrupt JSON summary is skipped, others processed."""
    summaries_dir = tmp_path / ".mullet" / "observations" / "summaries"
    summaries_dir.mkdir(parents=True)
    (summaries_dir / "good.json").write_text(json.dumps({
        "session_id": "s1",
        "started_at": "",
        "ended_at": "",
        "tool_counts": {},
        "files_touched": [],
        "summary": "",
        "bullet_points": [],
        "open_questions": [],
    }))
    (summaries_dir / "bad.json").write_text("not json")

    store = MagicMock()
    store.store.return_value = "mem-id"

    with patch("mullet.core.memory_sync._get_store", return_value=store):
        with caplog.at_level("WARNING", logger="mullet.core.memory_sync"):
            result = backfill_all(base=tmp_path)

    assert result["observations"] == 1
    assert "backfill skipped" in caplog.text


# ---------------------------------------------------------------------------
# backfill_all — decisions exception  (lines 325-326)
# ---------------------------------------------------------------------------

def test_backfill_all_decisions_exception_returns_zero_obs(tmp_path, caplog):
    """Lines 325-326: sync_decisions exception inside backfill is caught."""
    with patch("mullet.core.memory_sync.sync_decisions", side_effect=RuntimeError("boom")):
        with caplog.at_level("WARNING", logger="mullet.core.memory_sync"):
            result = backfill_all(base=tmp_path)
    assert result["decisions"] == 0
    assert "backfill decisions failed" in caplog.text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_summary(session_id: str = "sess-gap") -> SessionSummary:
    return SessionSummary(
        session_id=session_id,
        started_at="2026-04-24T08:00:00Z",
        ended_at="2026-04-24T09:00:00Z",
        tool_counts={},
        files_touched=[],
        summary="gap test",
        bullet_points=[],
        open_questions=[],
    )


def _make_decision(did: str = "id000") -> Decision:
    return Decision(
        id=did,
        source_path="docs/README.md",
        source_type="docs",
        line_number=1,
        timestamp="2026-04-24T08:00:00Z",
        excerpt="gap",
        kind="decision",
        keywords=["decided to"],
    )


def _write_decisions(base: Path, decisions: list) -> None:
    path = base / ".mullet" / "decisions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for d in decisions:
            fh.write(json.dumps(d.to_dict()) + "\n")
