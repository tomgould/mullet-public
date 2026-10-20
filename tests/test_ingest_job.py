"""Tests for mullet.knowledge.engine.ingest_job."""
from __future__ import annotations

import time

import pytest

from mullet.knowledge.engine.ingest_job import IngestionJob, IngestionJobManager
from mullet.knowledge.engine.ingest_progress import IngestionProgress


# ---------------------------------------------------------------------------
# IngestionJob.to_dict
# ---------------------------------------------------------------------------

class TestIngestionJobToDict:
    def test_basic_fields_present(self) -> None:
        job = IngestionJob(job_id="ab12cd34", client_name="acme", job_type="file")
        d = job.to_dict()
        assert d["job_id"] == "ab12cd34"
        assert d["client_name"] == "acme"
        assert d["job_type"] == "file"
        assert d["status"] == "pending"
        assert d["result"] is None
        assert d["error"] is None

    def test_age_seconds_is_non_negative(self) -> None:
        job = IngestionJob(job_id="x", client_name="c", job_type="url")
        d = job.to_dict()
        assert d["age_seconds"] >= 0

    def test_no_progress_key_when_none(self) -> None:
        job = IngestionJob(job_id="x", client_name="c", job_type="file")
        d = job.to_dict()
        assert "progress" not in d

    def test_progress_included_when_set(self) -> None:
        job = IngestionJob(job_id="x", client_name="c", job_type="file")
        job.progress = IngestionProgress(total_files=10, processed_files=5, total_chunks=50)
        d = job.to_dict()
        assert "progress" in d
        assert d["progress"]["total_files"] == 10
        assert d["progress"]["processed_files"] == 5
        assert d["progress"]["total_chunks"] == 50

    def test_completed_job_has_result(self) -> None:
        job = IngestionJob(
            job_id="done",
            client_name="c",
            job_type="reindex",
            status="completed",
            result="42 files indexed",
        )
        d = job.to_dict()
        assert d["status"] == "completed"
        assert d["result"] == "42 files indexed"

    def test_failed_job_has_error(self) -> None:
        job = IngestionJob(
            job_id="fail",
            client_name="c",
            job_type="file",
            status="failed",
            error="Connection refused",
        )
        d = job.to_dict()
        assert d["status"] == "failed"
        assert d["error"] == "Connection refused"


# ---------------------------------------------------------------------------
# IngestionJobManager
# ---------------------------------------------------------------------------

class TestIngestionJobManagerCreate:
    def test_create_job_returns_job(self) -> None:
        mgr = IngestionJobManager()
        job = mgr.create_job("client1", "file")
        assert job.client_name == "client1"
        assert job.job_type == "file"
        assert job.status == "pending"

    def test_job_id_is_short(self) -> None:
        mgr = IngestionJobManager()
        job = mgr.create_job("c", "url")
        assert len(job.job_id) <= 8

    def test_max_concurrent_raises(self) -> None:
        # Cap is enforced at start_job, not create_job — see IngestionJobManager
        # docstring. create_job is permissive (multiple pending jobs allowed).
        mgr = IngestionJobManager(max_concurrent=1)
        job = mgr.create_job("c", "file")
        mgr.start_job(job.job_id)
        second = mgr.create_job("c", "file")  # creating is fine
        with pytest.raises(RuntimeError, match="Max concurrent"):
            mgr.start_job(second.job_id)  # starting hits the cap

    def test_can_create_after_completing(self) -> None:
        mgr = IngestionJobManager(max_concurrent=1)
        job = mgr.create_job("c", "file")
        mgr.start_job(job.job_id)
        mgr.complete_job(job.job_id, "done")
        # Now should succeed
        job2 = mgr.create_job("c", "file")
        assert job2 is not None


class TestIngestionJobManagerLifecycle:
    def test_start_job_sets_running(self) -> None:
        mgr = IngestionJobManager()
        job = mgr.create_job("c", "file")
        mgr.start_job(job.job_id)
        assert mgr.get_job(job.job_id).status == "running"

    def test_complete_job_sets_completed(self) -> None:
        mgr = IngestionJobManager()
        job = mgr.create_job("c", "file")
        mgr.start_job(job.job_id)
        mgr.complete_job(job.job_id, "indexed 10 files")
        j = mgr.get_job(job.job_id)
        assert j.status == "completed"
        assert j.result == "indexed 10 files"

    def test_fail_job_sets_failed(self) -> None:
        mgr = IngestionJobManager()
        job = mgr.create_job("c", "file")
        mgr.start_job(job.job_id)
        mgr.fail_job(job.job_id, "disk full")
        j = mgr.get_job(job.job_id)
        assert j.status == "failed"
        assert j.error == "disk full"

    def test_update_progress(self) -> None:
        mgr = IngestionJobManager()
        job = mgr.create_job("c", "reindex")
        mgr.start_job(job.job_id)
        progress = IngestionProgress(total_files=100, processed_files=25)
        mgr.update_progress(job.job_id, progress)
        j = mgr.get_job(job.job_id)
        assert j.progress.processed_files == 25

    def test_get_unknown_job_returns_none(self) -> None:
        mgr = IngestionJobManager()
        assert mgr.get_job("nonexistent") is None

    def test_start_unknown_job_is_noop(self) -> None:
        mgr = IngestionJobManager()
        mgr.start_job("nonexistent")  # Should not raise

    def test_complete_unknown_job_is_noop(self) -> None:
        mgr = IngestionJobManager()
        mgr.complete_job("nonexistent", "result")  # Should not raise


class TestIngestionJobManagerQuery:
    def test_get_client_jobs_returns_jobs(self) -> None:
        mgr = IngestionJobManager(max_concurrent=10)
        mgr.create_job("acme", "file")
        mgr.create_job("acme", "url")
        mgr.create_job("other", "file")
        jobs = mgr.get_client_jobs("acme")
        assert len(jobs) == 2
        assert all(j.client_name == "acme" for j in jobs)

    def test_get_client_jobs_newest_first(self) -> None:
        mgr = IngestionJobManager(max_concurrent=10)
        j1 = mgr.create_job("c", "file")
        time.sleep(0.01)
        j2 = mgr.create_job("c", "url")
        jobs = mgr.get_client_jobs("c")
        assert jobs[0].job_id == j2.job_id

    def test_get_client_jobs_empty_when_none(self) -> None:
        mgr = IngestionJobManager()
        assert mgr.get_client_jobs("nobody") == []


class TestIngestionJobManagerCleanup:
    def test_cleanup_removes_old_completed(self) -> None:
        mgr = IngestionJobManager()
        job = mgr.create_job("c", "file")
        # Artificially age the job
        job.created_at = time.time() - 7200  # 2 hours ago
        job.status = "completed"
        removed = mgr.cleanup_completed(max_age_seconds=3600)
        assert removed == 1
        assert mgr.get_job(job.job_id) is None

    def test_cleanup_keeps_recent_completed(self) -> None:
        mgr = IngestionJobManager()
        job = mgr.create_job("c", "file")
        job.status = "completed"
        removed = mgr.cleanup_completed(max_age_seconds=3600)
        assert removed == 0
        assert mgr.get_job(job.job_id) is not None

    def test_cleanup_does_not_remove_running(self) -> None:
        mgr = IngestionJobManager()
        job = mgr.create_job("c", "file")
        job.created_at = time.time() - 7200
        mgr.start_job(job.job_id)
        removed = mgr.cleanup_completed(max_age_seconds=3600)
        assert removed == 0
