"""Tests for mullet.knowledge.engine.ingest_job — IngestionJob + IngestionJobManager."""

import threading
import time
from unittest.mock import MagicMock

import pytest

from mullet.knowledge.engine.ingest_job import IngestionJob, IngestionJobManager


# --- IngestionJob ----------------------------------------------------------


def test_ingestion_job_default_status_pending():
    job = IngestionJob(job_id="j1", client_name="c", job_type="file")
    assert job.status == "pending"
    assert job.progress is None
    assert job.result is None
    assert job.error is None


def test_ingestion_job_to_dict_basic():
    job = IngestionJob(job_id="j1", client_name="my-client", job_type="reindex")
    d = job.to_dict()
    assert d["job_id"] == "j1"
    assert d["client_name"] == "my-client"
    assert d["job_type"] == "reindex"
    assert d["status"] == "pending"
    assert d["result"] is None
    assert d["error"] is None
    assert "age_seconds" in d
    assert "progress" not in d  # No progress → key omitted


def test_ingestion_job_to_dict_with_progress():
    job = IngestionJob(job_id="j2", client_name="c", job_type="file")
    job.progress = MagicMock()
    job.progress.percent = 42.5
    job.progress.processed_files = 42
    job.progress.total_files = 100
    job.progress.total_chunks = 500
    job.progress.format_eta.return_value = "2m 30s"
    job.progress.failed_files = 1
    job.progress.status = "in-progress"

    d = job.to_dict()
    assert d["progress"]["percent"] == 42.5
    assert d["progress"]["processed_files"] == 42
    assert d["progress"]["total_files"] == 100
    assert d["progress"]["total_chunks"] == 500
    assert d["progress"]["eta"] == "2m 30s"
    assert d["progress"]["failed_files"] == 1


def test_ingestion_job_age_seconds_monotonic():
    """age_seconds should reflect elapsed time since creation."""
    job = IngestionJob(job_id="j3", client_name="c", job_type="file")
    job.created_at = time.time() - 5  # Force known age
    d = job.to_dict()
    assert d["age_seconds"] >= 5


# --- IngestionJobManager ---------------------------------------------------


def test_create_job_returns_new_job():
    mgr = IngestionJobManager()
    job = mgr.create_job("my-client", "file")
    assert job.job_id
    assert job.client_name == "my-client"
    assert job.job_type == "file"
    assert job.status == "pending"


def test_start_job_respects_max_concurrent():
    """Max 1 concurrent job by default; starting a second one raises.

    The cap lives at start_job (not create_job) — see the docstring on
    IngestionJobManager.start_job for the race-window rationale. create_job
    only registers intent; multiple pending jobs are allowed.
    """
    mgr = IngestionJobManager(max_concurrent=1)
    j1 = mgr.create_job("c1", "file")
    mgr.start_job(j1.job_id)

    j2 = mgr.create_job("c2", "file")  # create is permissive
    with pytest.raises(RuntimeError, match="Max concurrent"):
        mgr.start_job(j2.job_id)  # but start hits the cap


def test_create_job_after_completion_allowed():
    """Completed jobs don't count toward the running limit."""
    mgr = IngestionJobManager(max_concurrent=1)
    j1 = mgr.create_job("c1", "file")
    mgr.start_job(j1.job_id)
    mgr.complete_job(j1.job_id, "done")

    # Now can create another job
    j2 = mgr.create_job("c2", "file")
    assert j2.job_id != j1.job_id


def test_start_job_updates_status():
    mgr = IngestionJobManager()
    job = mgr.create_job("c", "file")
    mgr.start_job(job.job_id)
    assert mgr.get_job(job.job_id).status == "running"


def test_complete_job_sets_result():
    mgr = IngestionJobManager()
    job = mgr.create_job("c", "file")
    mgr.complete_job(job.job_id, "done successfully")
    retrieved = mgr.get_job(job.job_id)
    assert retrieved.status == "completed"
    assert retrieved.result == "done successfully"


def test_fail_job_sets_error():
    mgr = IngestionJobManager()
    job = mgr.create_job("c", "file")
    mgr.fail_job(job.job_id, "network down")
    retrieved = mgr.get_job(job.job_id)
    assert retrieved.status == "failed"
    assert retrieved.error == "network down"


def test_update_progress_attaches_progress_obj():
    mgr = IngestionJobManager()
    job = mgr.create_job("c", "file")
    progress = MagicMock()
    mgr.update_progress(job.job_id, progress)
    assert mgr.get_job(job.job_id).progress is progress


def test_get_job_missing_returns_none():
    mgr = IngestionJobManager()
    assert mgr.get_job("does-not-exist") is None


def test_get_client_jobs_sorted_newest_first():
    mgr = IngestionJobManager()
    j1 = mgr.create_job("c1", "file")
    mgr.start_job(j1.job_id)
    mgr.complete_job(j1.job_id, "")

    # Force different created_at
    j1.created_at = time.time() - 100

    j2 = mgr.create_job("c1", "url")
    j3 = mgr.create_job("c2", "file")  # different client

    c1_jobs = mgr.get_client_jobs("c1")
    assert len(c1_jobs) == 2
    assert c1_jobs[0].created_at >= c1_jobs[1].created_at  # newest first
    assert all(j.client_name == "c1" for j in c1_jobs)


def test_cleanup_completed_removes_old_jobs():
    mgr = IngestionJobManager()
    j1 = mgr.create_job("c", "file")
    mgr.start_job(j1.job_id)
    mgr.complete_job(j1.job_id, "")
    # Force age
    j1.created_at = time.time() - 7200  # 2 hours old

    j2 = mgr.create_job("c", "file")
    mgr.start_job(j2.job_id)
    mgr.fail_job(j2.job_id, "err")
    j2.created_at = time.time() - 7200

    # Pending/running shouldn't be cleaned
    j3 = mgr.create_job("c", "file")
    j3.created_at = time.time() - 7200

    removed = mgr.cleanup_completed(max_age_seconds=3600)
    assert removed == 2
    assert mgr.get_job(j1.job_id) is None
    assert mgr.get_job(j2.job_id) is None
    assert mgr.get_job(j3.job_id) is not None


def test_cleanup_completed_respects_max_age():
    """Recent completed jobs stay; only old ones removed."""
    mgr = IngestionJobManager()
    j1 = mgr.create_job("c", "file")
    mgr.start_job(j1.job_id)
    mgr.complete_job(j1.job_id, "")

    removed = mgr.cleanup_completed(max_age_seconds=3600)
    assert removed == 0
    assert mgr.get_job(j1.job_id) is not None


# --- thread safety ---------------------------------------------------------


def test_create_job_race_doesnt_exceed_limit():
    """Many threads racing to create jobs — only max_concurrent succeed."""
    mgr = IngestionJobManager(max_concurrent=3)
    created_ids = []
    failed_count = [0]
    lock = threading.Lock()

    def try_create():
        try:
            j = mgr.create_job("c", "file")
            mgr.start_job(j.job_id)
            with lock:
                created_ids.append(j.job_id)
        except RuntimeError:
            with lock:
                failed_count[0] += 1

    threads = [threading.Thread(target=try_create) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly max_concurrent should succeed
    assert len(created_ids) <= 3
    assert len(created_ids) + failed_count[0] == 20
