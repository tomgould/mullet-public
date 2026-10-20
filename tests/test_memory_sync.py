"""Tests for mullet.core.memory_sync."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch


from mullet.core.observation_summary import SessionSummary
from mullet.core.decisions import Decision


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_summary(session_id: str = "sess-abc") -> SessionSummary:
    return SessionSummary(
        session_id=session_id,
        started_at="2026-04-24T08:00:00Z",
        ended_at="2026-04-24T09:00:00Z",
        tool_counts={"Read": 5, "Edit": 3, "Bash": 2},
        files_touched=["/tmp/project/foo.py", "/tmp/project/bar.py"],
        summary="The developer refactored the routing module and added unit tests.",
        bullet_points=["Refactored router", "Added tests", "Fixed bug in decisions.py"],
        open_questions=["Should we add caching?", "Is the fallback path tested?"],
    )


def _make_decision(
    did: str = "abc1234567890abc",
    kind: str = "decision",
    excerpt: str = "decided to use SQLite for the memory store because it is lightweight",
) -> Decision:
    return Decision(
        id=did,
        source_path="docs/ARCHITECTURE.md",
        source_type="docs",
        line_number=42,
        timestamp="2026-04-24T08:30:00Z",
        excerpt=excerpt,
        kind=kind,
        keywords=["decided to"],
    )


def _real_store(tmp_path: Path):
    """Return a real MemoryStore backed by a temp SQLite DB."""
    from mullet.knowledge.engine.memory import MemoryStore

    return MemoryStore(tmp_path / "memory.db")


# ---------------------------------------------------------------------------
# sync_observation_summary
# ---------------------------------------------------------------------------


def test_sync_observation_summary_returns_mem_id(tmp_path):
    """Round-trip: summary stored, mem_id returned."""
    store = _real_store(tmp_path)

    with patch("mullet.core.memory_sync._get_store", return_value=store):
        from mullet.core.memory_sync import sync_observation_summary

        mem_id = sync_observation_summary(_make_summary(), base=tmp_path)

    assert mem_id is not None
    assert isinstance(mem_id, str)
    assert len(mem_id) > 0


def test_sync_observation_summary_content_and_tags(tmp_path):
    """Content contains summary text; metadata has correct tags."""
    store = _real_store(tmp_path)
    summary = _make_summary("sess-xyz")

    with patch("mullet.core.memory_sync._get_store", return_value=store):
        from mullet.core.memory_sync import sync_observation_summary

        sync_observation_summary(summary, base=tmp_path)

    results = store.list(namespace="mullet-pipeline")
    assert len(results) == 1
    entry = results[0]

    assert "sess-xyz" in entry["content"]
    assert "refactored the routing module" in entry["content"]

    tags = entry["metadata"]["tags"]
    assert "mullet" in tags
    assert "pipeline" in tags
    assert "observations" in tags
    assert "session-summary" in tags
    assert "session:sess-xyz" in tags


def test_sync_observation_summary_metadata_roundtrip(tmp_path):
    """metadata['data'] preserves original dataclass fields."""
    store = _real_store(tmp_path)
    summary = _make_summary("sess-rt")

    with patch("mullet.core.memory_sync._get_store", return_value=store):
        from mullet.core.memory_sync import sync_observation_summary

        sync_observation_summary(summary, base=tmp_path)

    results = store.list(namespace="mullet-pipeline")
    data = results[0]["metadata"]["data"]
    assert data["session_id"] == "sess-rt"
    assert data["tool_counts"] == {"Read": 5, "Edit": 3, "Bash": 2}
    assert len(data["files_touched"]) == 2


def test_sync_observation_summary_none_input(tmp_path):
    """None summary returns None gracefully."""
    with patch("mullet.core.memory_sync._get_store", return_value=_real_store(tmp_path)):
        from mullet.core.memory_sync import sync_observation_summary

        result = sync_observation_summary(None, base=tmp_path)
    assert result is None


def test_sync_observation_summary_fail_open_when_store_raises(tmp_path):
    """MemoryStore.store() raising does not propagate — returns None."""
    broken_store = MagicMock()
    broken_store.store.side_effect = RuntimeError("DB locked")

    with patch("mullet.core.memory_sync._get_store", return_value=broken_store):
        from mullet.core.memory_sync import sync_observation_summary

        result = sync_observation_summary(_make_summary(), base=tmp_path)

    assert result is None


def test_sync_observation_summary_fallback_jsonl_when_store_unavailable(tmp_path):
    """When _get_store returns None, falls back to local JSONL."""
    with patch("mullet.core.memory_sync._get_store", return_value=None):
        from mullet.core.memory_sync import sync_observation_summary

        result = sync_observation_summary(_make_summary("sess-fb"), base=tmp_path)

    assert result is None  # no store, no ID
    fallback = tmp_path / ".mullet" / "memory_sync.jsonl"
    assert fallback.exists()
    lines = [l for l in fallback.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["key"] == "obs-summary:sess-fb"


# ---------------------------------------------------------------------------
# sync_decisions
# ---------------------------------------------------------------------------


def test_sync_decisions_basic(tmp_path):
    """Decisions from jsonl are synced into store."""
    decisions = [
        _make_decision("id001", "decision"),
        _make_decision("id002", "rationale", "because it reduces coupling"),
    ]
    _write_decisions(tmp_path, decisions)

    store = _real_store(tmp_path)
    with patch("mullet.core.memory_sync._get_store", return_value=store):
        from mullet.core.memory_sync import sync_decisions

        count = sync_decisions(base=tmp_path)

    assert count == 2
    results = store.list(namespace="mullet-pipeline")
    assert len(results) == 2


def test_sync_decisions_dedup_across_repeated_runs(tmp_path):
    """Running sync twice does not duplicate entries (store upserts by key)."""
    _write_decisions(tmp_path, [_make_decision("idAAA")])
    store = _real_store(tmp_path)

    with patch("mullet.core.memory_sync._get_store", return_value=store):
        from mullet.core.memory_sync import sync_decisions

        sync_decisions(base=tmp_path)
        sync_decisions(base=tmp_path)

    results = store.list(namespace="mullet-pipeline")
    assert len(results) == 1  # upsert, not duplicate


def test_sync_decisions_tag_shape(tmp_path):
    """Tags include mullet, pipeline, decisions, and the kind."""
    _write_decisions(tmp_path, [_make_decision("idTAG", kind="rejection")])
    store = _real_store(tmp_path)

    with patch("mullet.core.memory_sync._get_store", return_value=store):
        from mullet.core.memory_sync import sync_decisions

        sync_decisions(base=tmp_path)

    entry = store.list(namespace="mullet-pipeline")[0]
    tags = entry["metadata"]["tags"]
    assert set(tags) == {"mullet", "pipeline", "decisions", "rejection"}


def test_sync_decisions_fail_open(tmp_path):
    """Store unavailable → returns 0, writes fallback JSONL."""
    _write_decisions(tmp_path, [_make_decision("idFO")])

    with patch("mullet.core.memory_sync._get_store", return_value=None):
        from mullet.core.memory_sync import sync_decisions

        count = sync_decisions(base=tmp_path)

    assert count == 1  # written to fallback
    fallback = tmp_path / ".mullet" / "memory_sync.jsonl"
    assert fallback.exists()


def test_sync_decisions_limit(tmp_path):
    """limit= parameter caps the number of decisions synced."""
    decisions = [_make_decision(f"id{i:03d}") for i in range(10)]
    _write_decisions(tmp_path, decisions)

    store = _real_store(tmp_path)
    with patch("mullet.core.memory_sync._get_store", return_value=store):
        from mullet.core.memory_sync import sync_decisions

        count = sync_decisions(base=tmp_path, limit=3)

    assert count == 3


# ---------------------------------------------------------------------------
# backfill_all
# ---------------------------------------------------------------------------


def test_backfill_all_counts(tmp_path):
    """backfill_all returns correct observation + decision counts."""
    # Create two summary JSON files
    summaries_dir = tmp_path / ".mullet" / "observations" / "summaries"
    summaries_dir.mkdir(parents=True)
    for i in range(2):
        import dataclasses

        s = _make_summary(f"sess-{i}")
        p = summaries_dir / f"sess-{i}-20260424T000000Z.json"
        p.write_text(json.dumps(dataclasses.asdict(s)))

    # Create decisions
    _write_decisions(tmp_path, [_make_decision("d1"), _make_decision("d2")])

    store = _real_store(tmp_path)
    with patch("mullet.core.memory_sync._get_store", return_value=store):
        from mullet.core.memory_sync import backfill_all

        result = backfill_all(base=tmp_path)

    assert result["observations"] == 2
    assert result["decisions"] == 2


def test_backfill_all_empty_dirs(tmp_path):
    """backfill_all with no data returns zeros without error."""
    store = _real_store(tmp_path)
    with patch("mullet.core.memory_sync._get_store", return_value=store):
        from mullet.core.memory_sync import backfill_all

        result = backfill_all(base=tmp_path)

    assert result == {"observations": 0, "decisions": 0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_decisions(base: Path, decisions: list) -> None:
    """Write a list of Decision objects to .mullet/decisions.jsonl."""
    path = base / ".mullet" / "decisions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for d in decisions:
            fh.write(json.dumps(d.to_dict()) + "\n")


# ---------------------------------------------------------------------------
# _get_store — direct call (lines 42-51)
# ---------------------------------------------------------------------------

def test_get_store_returns_store_or_none(tmp_path):
    """Lines 42-51: _get_store() either succeeds or logs warning and returns None."""
    from mullet.core.memory_sync import _get_store
    result = _get_store(tmp_path)
    # May return None if MemoryStore unavailable; no exception should propagate
    assert result is None or hasattr(result, "store")


# ---------------------------------------------------------------------------
# _write_fallback — except branch (lines 68-69)
# ---------------------------------------------------------------------------

def test_write_fallback_oserror_swallowed(tmp_path):
    """Lines 68-69: _write_fallback() when path is unwritable → swallowed."""
    from mullet.core.memory_sync import _write_fallback
    # Create the directory and make it unwritable
    mullet_dir = tmp_path / ".mullet"
    mullet_dir.mkdir(parents=True, exist_ok=True)
    mullet_dir.chmod(0o000)
    try:
        _write_fallback({"id": "x", "content": "test"}, base=tmp_path)
        # No exception raised
    finally:
        mullet_dir.chmod(0o755)


# ---------------------------------------------------------------------------
# _load_fallback_ids — file exists with valid IDs (lines 79-88)
# ---------------------------------------------------------------------------

def test_load_fallback_ids_with_existing_file(tmp_path):
    """Lines 77-88: _load_fallback_ids reads IDs from an existing JSONL file."""
    from mullet.core.memory_sync import _load_fallback_ids, _fallback_jsonl
    path = _fallback_jsonl(tmp_path)
    path.write_text(
        '{"id": "abc123"}\n'
        '{"id": "def456"}\n'
        '\n'  # blank line — skipped
        '{"content": "no id"}\n',  # no id key — not added
        encoding="utf-8",
    )
    ids = _load_fallback_ids(tmp_path)
    assert "abc123" in ids
    assert "def456" in ids
    assert len(ids) == 2


def test_load_fallback_ids_corrupt_file_swallowed(tmp_path):
    """Lines 86-87: _load_fallback_ids with bad JSON → except pass → return empty set."""
    from mullet.core.memory_sync import _load_fallback_ids, _fallback_jsonl
    path = _fallback_jsonl(tmp_path)
    path.write_text("{ bad json\n", encoding="utf-8")
    ids = _load_fallback_ids(tmp_path)
    assert isinstance(ids, set)


# ---------------------------------------------------------------------------
# sync_observation_summary — asdict fallback (lines 139-140)
# ---------------------------------------------------------------------------

def test_sync_observation_summary_non_dataclass_uses_getattr_fallback(tmp_path):
    """Lines 139-140: asdict(non_dataclass) raises → fallback getattr dict."""
    class FakeSummary:
        session_id = "sess-fake"
        started_at = "2026-01-01T00:00:00Z"
        ended_at = "2026-01-01T01:00:00Z"
        tool_counts = {}
        files_touched = []
        summary = "fake summary text"
        bullet_points = []
        open_questions = []

    store = _real_store(tmp_path)
    with patch("mullet.core.memory_sync._get_store", return_value=store):
        from mullet.core.memory_sync import sync_observation_summary
        result = sync_observation_summary(FakeSummary(), base=tmp_path)
    assert result is not None


# ---------------------------------------------------------------------------
# sync_decisions — fallback dedup (line 228)
# ---------------------------------------------------------------------------

def test_sync_decisions_fallback_dedup_skips_existing(tmp_path):
    """Line 228: fallback mode, same ID in existing_ids → continue (not re-written)."""
    _write_decisions(tmp_path, [_make_decision("idFB1")])

    with patch("mullet.core.memory_sync._get_store", return_value=None):
        from mullet.core.memory_sync import sync_decisions

        # First call: written to fallback
        count1 = sync_decisions(base=tmp_path)
        assert count1 == 1

        # Second call: already in fallback JSONL → skipped
        count2 = sync_decisions(base=tmp_path)
        assert count2 == 0


# ---------------------------------------------------------------------------
# sync_decisions — store.store() raises (lines 260-261)
# ---------------------------------------------------------------------------

def test_sync_decisions_store_raise_logs_and_continues(tmp_path):
    """Lines 260-261: store.store() raises → logged, decision not counted."""
    _write_decisions(tmp_path, [_make_decision("idERR"), _make_decision("idOK2")])

    broken_store = MagicMock()
    broken_store.store.side_effect = [RuntimeError("locked"), None]

    with patch("mullet.core.memory_sync._get_store", return_value=broken_store):
        from mullet.core.memory_sync import sync_decisions
        count = sync_decisions(base=tmp_path)

    # First fails (not counted), second succeeds
    assert count == 1


# ---------------------------------------------------------------------------
# sync_decisions — outer except (lines 270-272)
# ---------------------------------------------------------------------------

def test_sync_decisions_outer_exception_returns_zero(tmp_path):
    """Lines 270-272: unexpected exception in sync_decisions → fail-open, return 0."""
    with patch("mullet.core.memory_sync.sync_decisions.__wrapped__", create=True):
        pass
    # Patch 'load' inside decisions module to raise
    with patch("mullet.core.decisions.load", side_effect=RuntimeError("corrupt")):
        from mullet.core.memory_sync import sync_decisions
        count = sync_decisions(base=tmp_path)
    assert count == 0


# ---------------------------------------------------------------------------
# backfill_all — corrupt summary file skipped (lines 315-318)
# ---------------------------------------------------------------------------

def test_backfill_all_skips_corrupt_summary_file(tmp_path):
    """Lines 315-318: corrupt JSON summary file → logged, skipped, count not incremented."""
    import dataclasses
    summaries_dir = tmp_path / ".mullet" / "observations" / "summaries"
    summaries_dir.mkdir(parents=True)

    # Good summary
    good = _make_summary("sess-good")
    (summaries_dir / "good.json").write_text(json.dumps(dataclasses.asdict(good)))

    # Bad JSON file
    (summaries_dir / "bad.json").write_text("{ broken json [")

    store = _real_store(tmp_path)
    with patch("mullet.core.memory_sync._get_store", return_value=store):
        from mullet.core.memory_sync import backfill_all
        result = backfill_all(base=tmp_path)

    # good file counted; bad file skipped
    assert result["observations"] == 1


# ---------------------------------------------------------------------------
# backfill_all — else branch (line 314) and outer excepts (lines 319-320, 325-326)
# ---------------------------------------------------------------------------

def test_backfill_all_fallback_mode_hits_else_branch(tmp_path):
    """Line 314: sync_observation_summary returns None (fallback) → else branch."""
    import dataclasses
    summaries_dir = tmp_path / ".mullet" / "observations" / "summaries"
    summaries_dir.mkdir(parents=True)
    s = _make_summary("sess-fb2")
    (summaries_dir / "sess-fb2.json").write_text(json.dumps(dataclasses.asdict(s)))

    # store=None → sync_observation_summary returns None → else branch
    with patch("mullet.core.memory_sync._get_store", return_value=None):
        from mullet.core.memory_sync import backfill_all
        result = backfill_all(base=tmp_path)
    assert result["observations"] == 1  # still counted via else branch


def test_backfill_all_outer_except_observation(tmp_path):
    """Lines 319-320: unexpected error outside per-file loop → outer except swallowed."""
    from mullet.core.observation_summary import SessionSummary
    summaries_dir = tmp_path / ".mullet" / "observations" / "summaries"
    summaries_dir.mkdir(parents=True)

    store = _real_store(tmp_path)
    # Patch glob to raise inside the is_dir branch
    with patch("mullet.core.memory_sync._get_store", return_value=store), \
         patch("pathlib.Path.glob", side_effect=OSError("disk error")):
        from mullet.core.memory_sync import backfill_all
        result = backfill_all(base=tmp_path)
    # Outer except caught: obs_count stays 0
    assert result["observations"] == 0


def test_backfill_all_outer_except_decisions(tmp_path):
    """Lines 325-326: sync_decisions raises unexpectedly → outer except swallowed."""
    store = _real_store(tmp_path)
    with patch("mullet.core.memory_sync._get_store", return_value=store), \
         patch("mullet.core.memory_sync.sync_decisions", side_effect=RuntimeError("kaboom")):
        from mullet.core.memory_sync import backfill_all
        result = backfill_all(base=tmp_path)
    assert result["decisions"] == 0


def test_get_store_import_error_returns_none(tmp_path):
    """Lines 49-51: MemoryStore unavailable → except → return None."""
    from mullet.core.memory_sync import _get_store
    with patch("mullet.knowledge.engine.memory.MemoryStore", side_effect=ImportError("no store")):
        result = _get_store(tmp_path)
    assert result is None
