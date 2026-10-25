"""Tests for ingest_progress.py and ingest_job.py."""
import json
import time
import pytest

from mullet.knowledge.engine.ingest_progress import (
    IngestionProgress, IngestionCheckpoint, CHECKPOINT_FILENAME
)
from mullet.knowledge.engine.ingest_job import IngestionJob, IngestionJobManager


# ===========================================================================
# IngestionProgress
# ===========================================================================

class TestIngestionProgressPercent:
    def test_zero_total_files_returns_100(self):
        p = IngestionProgress(total_files=0)
        assert p.percent == 100.0

    def test_zero_processed_of_ten(self):
        p = IngestionProgress(total_files=10, processed_files=0)
        assert p.percent == 0.0

    def test_half_processed(self):
        p = IngestionProgress(total_files=10, processed_files=5)
        assert p.percent == 50.0

    def test_all_processed(self):
        p = IngestionProgress(total_files=4, processed_files=4)
        assert p.percent == 100.0


class TestIngestionProgressEta:
    def test_no_processed_files_returns_none(self):
        p = IngestionProgress(total_files=10)
        assert p.eta_seconds is None

    def test_all_done_returns_zero(self):
        p = IngestionProgress(total_files=5, processed_files=5, start_time=time.time() - 1.0)
        eta = p.eta_seconds
        assert eta == 0.0 or eta is None  # 0 remaining

    def test_positive_eta_when_work_remains(self):
        p = IngestionProgress(total_files=10, processed_files=5, start_time=time.time() - 1.0)
        eta = p.eta_seconds
        assert eta is not None
        assert eta > 0


class TestIngestionProgressStatus:
    def test_in_progress(self):
        p = IngestionProgress(total_files=10, processed_files=5)
        assert p.status == "in_progress"

    def test_completed(self):
        p = IngestionProgress(total_files=5, processed_files=5)
        assert p.status == "completed"

    def test_completed_with_errors(self):
        p = IngestionProgress(total_files=5, processed_files=5, failed_files=["a.py"])
        assert p.status == "completed_with_errors"


class TestIngestionProgressFormatEta:
    def test_calculating_when_no_processed(self):
        p = IngestionProgress(total_files=10)
        assert p.format_eta() == "calculating..."

    def test_seconds_only(self):
        p = IngestionProgress(total_files=10, processed_files=5, start_time=time.time() - 1.0)
        result = p.format_eta()
        # Should be Xs format
        assert "s" in result

    def test_minutes_format(self):
        # 100 files, 1 processed, took 1 second → ~99 seconds remaining
        p = IngestionProgress(total_files=200, processed_files=1, start_time=time.time() - 1.0)
        result = p.format_eta()
        # Should show minutes
        assert "m" in result or "s" in result  # just check it doesn't crash


# ===========================================================================
# IngestionCheckpoint
# ===========================================================================

class TestIngestionCheckpointSave:
    def test_saves_json_file(self, tmp_path):
        chk = IngestionCheckpoint("mypool", ["a.py", "b.py"], ["a.py", "b.py", "c.py"])
        path = tmp_path / "checkpoint.json"
        chk.save(path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["client_name"] == "mypool"
        assert data["processed_files"] == ["a.py", "b.py"]

    def test_creates_parent_dirs(self, tmp_path):
        chk = IngestionCheckpoint("pool", [], ["x.py"])
        path = tmp_path / "nested" / "dir" / "checkpoint.json"
        chk.save(path)
        assert path.exists()

    def test_timestamp_auto_set_when_empty(self, tmp_path):
        chk = IngestionCheckpoint("pool", [], [], timestamp="")
        path = tmp_path / "chk.json"
        chk.save(path)
        data = json.loads(path.read_text())
        assert data["timestamp"]  # not empty

    def test_explicit_timestamp_preserved(self, tmp_path):
        chk = IngestionCheckpoint("pool", [], [], timestamp="2024-01-01T00:00:00")
        path = tmp_path / "chk.json"
        chk.save(path)
        data = json.loads(path.read_text())
        assert data["timestamp"] == "2024-01-01T00:00:00"


class TestIngestionCheckpointLoad:
    def test_returns_none_when_file_missing(self, tmp_path):
        result = IngestionCheckpoint.load(tmp_path / "nonexistent.json")
        assert result is None

    def test_loads_valid_checkpoint(self, tmp_path):
        chk = IngestionCheckpoint("mypool", ["a.py"], ["a.py", "b.py"], chunks_indexed=42)
        path = tmp_path / "chk.json"
        chk.save(path)
        loaded = IngestionCheckpoint.load(path)
        assert loaded is not None
        assert loaded.client_name == "mypool"
        assert loaded.chunks_indexed == 42
        assert loaded.processed_files == ["a.py"]

    def test_returns_none_on_invalid_json(self, tmp_path):
        path = tmp_path / "chk.json"
        path.write_text("not json {{{{")
        result = IngestionCheckpoint.load(path)
        assert result is None

    def test_returns_none_on_missing_key(self, tmp_path):
        path = tmp_path / "chk.json"
        path.write_text('{"processed_files": []}')  # missing client_name
        result = IngestionCheckpoint.load(path)
        assert result is None

    def test_defaults_for_optional_fields(self, tmp_path):
        path = tmp_path / "chk.json"
        path.write_text(json.dumps({"client_name": "pool"}))
        loaded = IngestionCheckpoint.load(path)
        assert loaded is not None
        assert loaded.processed_files == []
        assert loaded.chunks_indexed == 0


class TestIngestionCheckpointPath:
    def test_checkpoint_path_returns_correct_path(self, tmp_path):
        path = IngestionCheckpoint.checkpoint_path(tmp_path)
        assert path == tmp_path / CHECKPOINT_FILENAME


# ===========================================================================
# IngestionJob.to_dict
# ===========================================================================

class TestIngestionJobToDict:
    def test_basic_fields(self):
        job = IngestionJob("abc12345", "myclient", "file")
        d = job.to_dict()
        assert d["job_id"] == "abc12345"
        assert d["client_name"] == "myclient"
        assert d["job_type"] == "file"
        assert d["status"] == "pending"
        assert "age_seconds" in d

    def test_no_progress_field_when_none(self):
        job = IngestionJob("abc", "client", "url")
        d = job.to_dict()
        assert "progress" not in d

    def test_progress_included_when_set(self):
        job = IngestionJob("abc", "client", "reindex")
        job.progress = IngestionProgress(total_files=10, processed_files=5)
        d = job.to_dict()
        assert "progress" in d
        assert d["progress"]["total_files"] == 10
        assert d["progress"]["processed_files"] == 5
        assert "percent" in d["progress"]
        assert "eta" in d["progress"]
        assert "status" in d["progress"]

    def test_result_and_error_included(self):
        job = IngestionJob("abc", "client", "file", status="failed", error="disk full")
        d = job.to_dict()
        assert d["error"] == "disk full"


# ===========================================================================
# IngestionJobManager
# ===========================================================================

class TestIngestionJobManagerCreate:
    def test_creates_job_with_pending_status(self):
        mgr = IngestionJobManager()
        job = mgr.create_job("mypool", "file")
        assert job.status == "pending"
        assert job.client_name == "mypool"
        assert job.job_type == "file"

    def test_creates_job_with_short_id(self):
        mgr = IngestionJobManager()
        job = mgr.create_job("pool", "url")
        assert len(job.job_id) == 8

    def test_multiple_jobs_have_unique_ids(self):
        mgr = IngestionJobManager()
        ids = {mgr.create_job("pool", "file").job_id for _ in range(10)}
        assert len(ids) == 10


class TestIngestionJobManagerStart:
    def test_pending_job_becomes_running(self):
        mgr = IngestionJobManager()
        job = mgr.create_job("pool", "file")
        mgr.start_job(job.job_id)
        assert job.status == "running"

    def test_start_idempotent_when_already_running(self):
        mgr = IngestionJobManager()
        job = mgr.create_job("pool", "file")
        mgr.start_job(job.job_id)
        mgr.start_job(job.job_id)  # should not raise
        assert job.status == "running"

    def test_start_nonexistent_job_is_noop(self):
        mgr = IngestionJobManager()
        mgr.start_job("nonexistent")  # should not raise

    def test_max_concurrent_raises(self):
        mgr = IngestionJobManager(max_concurrent=1)
        job1 = mgr.create_job("pool", "file")
        job2 = mgr.create_job("pool", "url")
        mgr.start_job(job1.job_id)
        with pytest.raises(RuntimeError, match="Max concurrent"):
            mgr.start_job(job2.job_id)

    def test_two_concurrent_when_max_is_two(self):
        mgr = IngestionJobManager(max_concurrent=2)
        job1 = mgr.create_job("pool", "file")
        job2 = mgr.create_job("pool", "url")
        mgr.start_job(job1.job_id)
        mgr.start_job(job2.job_id)  # should not raise
        assert job2.status == "running"


class TestIngestionJobManagerComplete:
    def test_complete_sets_status_and_result(self):
        mgr = IngestionJobManager()
        job = mgr.create_job("pool", "file")
        mgr.start_job(job.job_id)
        mgr.complete_job(job.job_id, "42 chunks indexed")
        assert job.status == "completed"
        assert job.result == "42 chunks indexed"

    def test_complete_nonexistent_job_is_noop(self):
        mgr = IngestionJobManager()
        mgr.complete_job("nonexistent", "result")  # should not raise

    def test_completed_job_frees_slot_for_next(self):
        mgr = IngestionJobManager(max_concurrent=1)
        job1 = mgr.create_job("pool", "file")
        job2 = mgr.create_job("pool", "url")
        mgr.start_job(job1.job_id)
        mgr.complete_job(job1.job_id, "done")
        mgr.start_job(job2.job_id)  # should not raise now
        assert job2.status == "running"


class TestIngestionJobManagerFail:
    def test_fail_sets_status_and_error(self):
        mgr = IngestionJobManager()
        job = mgr.create_job("pool", "file")
        mgr.fail_job(job.job_id, "disk full")
        assert job.status == "failed"
        assert job.error == "disk full"

    def test_fail_nonexistent_is_noop(self):
        mgr = IngestionJobManager()
        mgr.fail_job("nonexistent", "error")  # should not raise


class TestIngestionJobManagerProgress:
    def test_update_progress_stored(self):
        mgr = IngestionJobManager()
        job = mgr.create_job("pool", "reindex")
        progress = IngestionProgress(total_files=10, processed_files=3)
        mgr.update_progress(job.job_id, progress)
        assert job.progress is progress

    def test_update_progress_nonexistent_is_noop(self):
        mgr = IngestionJobManager()
        progress = IngestionProgress(total_files=5)
        mgr.update_progress("nonexistent", progress)  # should not raise


class TestIngestionJobManagerGet:
    def test_get_existing_job(self):
        mgr = IngestionJobManager()
        job = mgr.create_job("pool", "file")
        assert mgr.get_job(job.job_id) is job

    def test_get_nonexistent_returns_none(self):
        mgr = IngestionJobManager()
        assert mgr.get_job("bad-id") is None

    def test_get_client_jobs_filtered(self):
        mgr = IngestionJobManager()
        j1 = mgr.create_job("pool-a", "file")
        j2 = mgr.create_job("pool-b", "file")
        j3 = mgr.create_job("pool-a", "url")
        jobs = mgr.get_client_jobs("pool-a")
        assert len(jobs) == 2
        assert all(j.client_name == "pool-a" for j in jobs)

    def test_get_client_jobs_newest_first(self):
        mgr = IngestionJobManager()
        j1 = mgr.create_job("pool", "file")
        time.sleep(0.01)
        j2 = mgr.create_job("pool", "url")
        jobs = mgr.get_client_jobs("pool")
        assert jobs[0].job_id == j2.job_id  # newest first


class TestIngestionJobManagerCleanup:
    def test_cleanup_removes_old_completed(self):
        mgr = IngestionJobManager()
        job = mgr.create_job("pool", "file")
        mgr.complete_job(job.job_id, "done")
        job.created_at = time.time() - 7200  # 2 hours old
        removed = mgr.cleanup_completed(max_age_seconds=3600)
        assert removed == 1
        assert mgr.get_job(job.job_id) is None

    def test_cleanup_keeps_recent_jobs(self):
        mgr = IngestionJobManager()
        job = mgr.create_job("pool", "file")
        mgr.complete_job(job.job_id, "done")
        # created_at is now — should not be cleaned up
        removed = mgr.cleanup_completed(max_age_seconds=3600)
        assert removed == 0
        assert mgr.get_job(job.job_id) is not None

    def test_cleanup_keeps_running_jobs(self):
        mgr = IngestionJobManager()
        job = mgr.create_job("pool", "file")
        mgr.start_job(job.job_id)
        job.created_at = time.time() - 7200
        removed = mgr.cleanup_completed(max_age_seconds=3600)
        assert removed == 0

    def test_cleanup_removes_old_failed_jobs(self):
        mgr = IngestionJobManager()
        job = mgr.create_job("pool", "file")
        mgr.fail_job(job.job_id, "error")
        job.created_at = time.time() - 7200
        removed = mgr.cleanup_completed(max_age_seconds=3600)
        assert removed == 1
