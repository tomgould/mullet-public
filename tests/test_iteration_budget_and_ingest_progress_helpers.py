"""Tests for IterationBudget pure properties and IngestionProgress/Checkpoint."""

import time

from mullet.core.agent.iteration_budget import IterationBudget
from mullet.knowledge.engine.ingest_progress import (
    IngestionProgress,
    IngestionCheckpoint,
    CHECKPOINT_FILENAME,
)


# ---------------------------------------------------------------------------
# IterationBudget
# ---------------------------------------------------------------------------

class TestIterationBudget:
    def test_initial_used_is_zero(self):
        b = IterationBudget(max_total=10)
        assert b.used == 0

    def test_remaining_starts_at_max(self):
        b = IterationBudget(max_total=10)
        assert b.remaining == 10

    def test_consume_increments_used(self):
        b = IterationBudget(max_total=10)
        b.consume()
        assert b.used == 1
        assert b.remaining == 9

    def test_consume_returns_true_when_available(self):
        b = IterationBudget(max_total=5)
        assert b.consume() is True

    def test_consume_returns_false_when_exhausted(self):
        b = IterationBudget(max_total=1)
        b.consume()
        assert b.consume() is False

    def test_exhausted_false_initially(self):
        b = IterationBudget(max_total=5)
        assert b.exhausted is False

    def test_exhausted_true_after_max_consumes(self):
        b = IterationBudget(max_total=2)
        b.consume()
        b.consume()
        assert b.exhausted is True

    def test_refund_decrements_used(self):
        b = IterationBudget(max_total=10)
        b.consume()
        b.consume()
        b.refund(1)
        assert b.used == 1

    def test_refund_does_not_go_below_zero(self):
        b = IterationBudget(max_total=10)
        b.refund(5)
        assert b.used == 0

    def test_no_timeout_never_times_out(self):
        b = IterationBudget(max_total=10, timeout_seconds=None)
        assert b.timed_out is False

    def test_expired_timeout(self):
        b = IterationBudget(max_total=10, timeout_seconds=0.0)
        time.sleep(0.01)
        assert b.timed_out is True

    def test_elapsed_seconds_positive(self):
        b = IterationBudget(max_total=10)
        assert b.elapsed_seconds >= 0

    def test_pressure_none_at_start(self):
        b = IterationBudget(max_total=100)
        assert b.pressure == "none"

    def test_pressure_caution_at_70_percent(self):
        b = IterationBudget(max_total=10)
        for _ in range(7):
            b.consume()
        assert b.pressure == "caution"

    def test_pressure_urgent_at_90_percent(self):
        # With 20 max, consuming 18 = 90%, remaining = 2 > 1 → urgent
        b = IterationBudget(max_total=20)
        for _ in range(18):
            b.consume()
        assert b.pressure == "urgent"

    def test_pressure_critical_at_last_iteration(self):
        b = IterationBudget(max_total=10)
        for _ in range(9):
            b.consume()
        # remaining = 1 → critical
        assert b.pressure == "critical"

    def test_get_pressure_message_none_when_no_pressure(self):
        b = IterationBudget(max_total=100)
        assert b.get_pressure_message() is None

    def test_get_pressure_message_string_when_caution(self):
        b = IterationBudget(max_total=10)
        for _ in range(7):
            b.consume()
        msg = b.get_pressure_message()
        assert isinstance(msg, str)
        assert len(msg) > 0


# ---------------------------------------------------------------------------
# IngestionProgress
# ---------------------------------------------------------------------------

class TestIngestionProgress:
    def test_percent_zero_files_is_100(self):
        p = IngestionProgress(total_files=0)
        assert p.percent == 100.0

    def test_percent_partial(self):
        p = IngestionProgress(total_files=10, processed_files=5)
        assert p.percent == 50.0

    def test_percent_complete(self):
        p = IngestionProgress(total_files=4, processed_files=4)
        assert p.percent == 100.0

    def test_status_in_progress(self):
        p = IngestionProgress(total_files=10, processed_files=5)
        assert p.status == "in_progress"

    def test_status_completed(self):
        p = IngestionProgress(total_files=4, processed_files=4)
        assert p.status == "completed"

    def test_status_completed_with_errors(self):
        p = IngestionProgress(total_files=4, processed_files=4, failed_files=["bad.py"])
        assert p.status == "completed_with_errors"

    def test_eta_seconds_none_when_not_started(self):
        p = IngestionProgress(total_files=10, processed_files=0)
        assert p.eta_seconds is None

    def test_format_eta_calculating_when_none(self):
        p = IngestionProgress(total_files=10, processed_files=0)
        assert "calculating" in p.format_eta()

    def test_format_eta_seconds(self):
        p = IngestionProgress(total_files=2, processed_files=1, start_time=time.time() - 1.0)
        result = p.format_eta()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_defaults(self):
        p = IngestionProgress(total_files=5)
        assert p.processed_files == 0
        assert p.total_chunks == 0
        assert p.failed_files == []


# ---------------------------------------------------------------------------
# IngestionCheckpoint
# ---------------------------------------------------------------------------

class TestIngestionCheckpoint:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / CHECKPOINT_FILENAME
        ck = IngestionCheckpoint(
            client_name="mypool",
            processed_files=["a.py", "b.py"],
            total_files=["a.py", "b.py", "c.py"],
            chunks_indexed=42,
        )
        ck.save(path)
        loaded = IngestionCheckpoint.load(path)
        assert loaded is not None
        assert loaded.client_name == "mypool"
        assert loaded.chunks_indexed == 42
        assert "a.py" in loaded.processed_files

    def test_load_nonexistent_returns_none(self, tmp_path):
        path = tmp_path / "missing.json"
        assert IngestionCheckpoint.load(path) is None

    def test_checkpoint_path_uses_constant(self, tmp_path):
        result = IngestionCheckpoint.checkpoint_path(tmp_path)
        assert result.name == CHECKPOINT_FILENAME

    def test_save_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b" / CHECKPOINT_FILENAME
        ck = IngestionCheckpoint(
            client_name="p",
            processed_files=[],
            total_files=["x.py"],
        )
        ck.save(nested)
        assert nested.exists()

    def test_load_invalid_json_returns_none(self, tmp_path):
        path = tmp_path / CHECKPOINT_FILENAME
        path.write_text("not json {")
        assert IngestionCheckpoint.load(path) is None
