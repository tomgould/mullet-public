"""Tests for autoresearch pipeline and ingest_job manager helpers."""

import json
import pytest

from mullet.core.orchestration.autoresearch import (
    ResearchFinding,
    ResearchIteration,
    ResearchPipeline,
)
from mullet.knowledge.engine.ingest_job import (
    IngestionJob,
    IngestionJobManager,
)
from mullet.knowledge.engine.ingest_progress import IngestionProgress


# ---------------------------------------------------------------------------
# ResearchFinding
# ---------------------------------------------------------------------------

class TestResearchFinding:
    def test_fields_stored(self):
        f = ResearchFinding(content="Found X", source="web")
        assert f.content == "Found X"
        assert f.source == "web"

    def test_relevance_default(self):
        f = ResearchFinding(content="x", source="y")
        assert f.relevance == 1.0

    def test_timestamp_auto_set(self):
        f = ResearchFinding(content="x", source="y")
        assert bool(f.timestamp)

    def test_custom_relevance(self):
        f = ResearchFinding(content="x", source="y", relevance=0.7)
        assert f.relevance == 0.7


# ---------------------------------------------------------------------------
# ResearchIteration
# ---------------------------------------------------------------------------

class TestResearchIteration:
    def test_fields_stored(self):
        it = ResearchIteration(iteration=1, query="What is X?")
        assert it.iteration == 1
        assert it.query == "What is X?"

    def test_defaults(self):
        it = ResearchIteration(iteration=1, query="q")
        assert it.findings == []
        assert it.summary == ""
        assert it.score == 0.0

    def test_timestamp_auto_set(self):
        it = ResearchIteration(iteration=1, query="q")
        assert bool(it.timestamp)


# ---------------------------------------------------------------------------
# ResearchPipeline
# ---------------------------------------------------------------------------

def _make_finding(content="Found it", source="web"):
    return ResearchFinding(content=content, source=source)


class TestResearchPipelineInit:
    def test_question_stored(self):
        p = ResearchPipeline("What is Python?")
        assert p.question == "What is Python?"

    def test_defaults(self):
        p = ResearchPipeline("q")
        assert p.max_iterations == 5
        assert p.min_score == 0.8
        assert p.iterations == []
        assert p.status == "pending"


class TestShouldContinue:
    def test_no_iterations_returns_true(self):
        p = ResearchPipeline("q")
        assert p.should_continue() is True

    def test_max_iterations_reached_returns_false(self):
        p = ResearchPipeline("q", max_iterations=2)
        p.add_iteration("q1", [], "s", score=0.5)
        p.add_iteration("q2", [], "s", score=0.5)
        assert p.should_continue() is False

    def test_high_score_returns_false(self):
        p = ResearchPipeline("q", min_score=0.8)
        p.add_iteration("q1", [], "s", score=0.9)
        assert p.should_continue() is False

    def test_low_score_returns_true(self):
        p = ResearchPipeline("q", min_score=0.8)
        p.add_iteration("q1", [], "s", score=0.5)
        assert p.should_continue() is True


class TestAddIteration:
    def test_iteration_number_increments(self):
        p = ResearchPipeline("q")
        it1 = p.add_iteration("q1", [], "s1", 0.5)
        it2 = p.add_iteration("q2", [], "s2", 0.5)
        assert it1.iteration == 1
        assert it2.iteration == 2

    def test_status_running_while_below_threshold(self):
        p = ResearchPipeline("q", min_score=0.9)
        p.add_iteration("q1", [], "s", score=0.5)
        assert p.status == "running"

    def test_status_completed_when_threshold_met(self):
        p = ResearchPipeline("q", min_score=0.8)
        p.add_iteration("q1", [], "s", score=0.95)
        assert p.status == "completed"

    def test_findings_attached(self):
        p = ResearchPipeline("q")
        f = _make_finding()
        it = p.add_iteration("q1", [f], "s", 0.5)
        assert len(it.findings) == 1


class TestBestIteration:
    def test_empty_returns_none(self):
        p = ResearchPipeline("q")
        assert p.best_iteration() is None

    def test_returns_highest_score(self):
        p = ResearchPipeline("q")
        p.add_iteration("q1", [], "s", score=0.4)
        p.add_iteration("q2", [], "s", score=0.9)
        p.add_iteration("q3", [], "s", score=0.6)
        assert p.best_iteration().score == 0.9


class TestGetRefinedQuery:
    def test_empty_returns_original_question(self):
        p = ResearchPipeline("What is Python?")
        assert p.get_refined_query() == "What is Python?"

    def test_after_iteration_includes_score(self):
        p = ResearchPipeline("What is Python?")
        p.add_iteration("q", [], "summary text", score=0.4)
        refined = p.get_refined_query()
        assert "0.4" in refined
        assert "What is Python?" in refined


class TestResearchPipelineSerialization:
    def test_to_dict_question(self):
        p = ResearchPipeline("q?")
        d = p.to_dict()
        assert d["question"] == "q?"

    def test_to_dict_iterations(self):
        p = ResearchPipeline("q")
        p.add_iteration("q1", [_make_finding()], "s", 0.5)
        d = p.to_dict()
        assert len(d["iterations"]) == 1

    def test_from_dict_round_trip(self):
        p = ResearchPipeline("My question", max_iterations=3, min_score=0.7)
        p.add_iteration("refined", [_make_finding()], "summary", 0.6)
        p2 = ResearchPipeline.from_dict(p.to_dict())
        assert p2.question == "My question"
        assert p2.max_iterations == 3
        assert len(p2.iterations) == 1

    def test_to_dict_is_json_serializable(self):
        p = ResearchPipeline("q")
        assert json.dumps(p.to_dict())

    def test_save_and_load(self, tmp_path):
        p = ResearchPipeline("Saved question")
        p.add_iteration("q1", [], "s", 0.5)
        path = str(tmp_path / "research.json")
        p.save(path)
        p2 = ResearchPipeline.load(path)
        assert p2.question == "Saved question"
        assert len(p2.iterations) == 1


# ---------------------------------------------------------------------------
# IngestionJob
# ---------------------------------------------------------------------------

class TestIngestionJob:
    def test_fields_stored(self):
        job = IngestionJob(job_id="abc", client_name="pool_a", job_type="file")
        assert job.job_id == "abc"
        assert job.client_name == "pool_a"
        assert job.job_type == "file"

    def test_status_default_pending(self):
        job = IngestionJob(job_id="x", client_name="c", job_type="url")
        assert job.status == "pending"

    def test_to_dict_has_job_id(self):
        job = IngestionJob(job_id="abc", client_name="c", job_type="file")
        d = job.to_dict()
        assert d["job_id"] == "abc"

    def test_to_dict_no_progress_key_when_none(self):
        job = IngestionJob(job_id="abc", client_name="c", job_type="file")
        d = job.to_dict()
        assert "progress" not in d

    def test_to_dict_with_progress(self):
        prog = IngestionProgress(processed_files=5, total_files=10, total_chunks=50)
        job = IngestionJob(job_id="abc", client_name="c", job_type="file", progress=prog)
        d = job.to_dict()
        assert "progress" in d
        assert d["progress"]["percent"] == 50.0


# ---------------------------------------------------------------------------
# IngestionJobManager
# ---------------------------------------------------------------------------

class TestIngestionJobManager:
    def test_create_job_returns_job(self):
        mgr = IngestionJobManager()
        job = mgr.create_job("pool_a", "file")
        assert job.client_name == "pool_a"
        assert job.job_type == "file"

    def test_created_job_has_id(self):
        mgr = IngestionJobManager()
        job = mgr.create_job("c", "url")
        assert bool(job.job_id)

    def test_max_concurrent_enforced(self):
        # Cap is at start_job (not create_job) — see IngestionJobManager docstring.
        mgr = IngestionJobManager(max_concurrent=1)
        job = mgr.create_job("c", "file")
        mgr.start_job(job.job_id)
        second = mgr.create_job("c", "url")  # creating second is permissive
        with pytest.raises(RuntimeError):
            mgr.start_job(second.job_id)  # but starting it hits the cap

    def test_max_concurrent_ok_after_complete(self):
        mgr = IngestionJobManager(max_concurrent=1)
        job = mgr.create_job("c", "file")
        mgr.start_job(job.job_id)
        mgr.complete_job(job.job_id, "done")
        job2 = mgr.create_job("c", "url")  # should not raise
        assert job2 is not None

    def test_start_job_sets_running(self):
        mgr = IngestionJobManager()
        job = mgr.create_job("c", "file")
        mgr.start_job(job.job_id)
        assert mgr.get_job(job.job_id).status == "running"

    def test_complete_job_sets_completed(self):
        mgr = IngestionJobManager()
        job = mgr.create_job("c", "file")
        mgr.complete_job(job.job_id, "success")
        j = mgr.get_job(job.job_id)
        assert j.status == "completed"
        assert j.result == "success"

    def test_fail_job_sets_failed(self):
        mgr = IngestionJobManager()
        job = mgr.create_job("c", "file")
        mgr.fail_job(job.job_id, "error msg")
        j = mgr.get_job(job.job_id)
        assert j.status == "failed"
        assert j.error == "error msg"

    def test_get_job_unknown_returns_none(self):
        mgr = IngestionJobManager()
        assert mgr.get_job("nonexistent") is None

    def test_get_client_jobs_filters_by_client(self):
        mgr = IngestionJobManager(max_concurrent=5)
        mgr.create_job("pool_a", "file")
        mgr.create_job("pool_a", "url")
        mgr.create_job("pool_b", "file")
        jobs = mgr.get_client_jobs("pool_a")
        assert len(jobs) == 2
        assert all(j.client_name == "pool_a" for j in jobs)

    def test_get_client_jobs_sorted_newest_first(self):
        import time
        mgr = IngestionJobManager(max_concurrent=5)
        job1 = mgr.create_job("c", "file")
        time.sleep(0.01)
        job2 = mgr.create_job("c", "url")
        jobs = mgr.get_client_jobs("c")
        assert jobs[0].job_id == job2.job_id

    def test_cleanup_completed_removes_old_jobs(self):
        mgr = IngestionJobManager()
        job = mgr.create_job("c", "file")
        mgr.complete_job(job.job_id, "done")
        # Backdate the job
        mgr.get_job(job.job_id).created_at = 0.0
        removed = mgr.cleanup_completed(max_age_seconds=1)
        assert removed == 1
        assert mgr.get_job(job.job_id) is None

    def test_cleanup_does_not_remove_running(self):
        mgr = IngestionJobManager()
        job = mgr.create_job("c", "file")
        mgr.start_job(job.job_id)
        mgr.get_job(job.job_id).created_at = 0.0
        removed = mgr.cleanup_completed(max_age_seconds=1)
        assert removed == 0

    def test_update_progress_sets_progress(self):
        mgr = IngestionJobManager()
        job = mgr.create_job("c", "file")
        prog = IngestionProgress(processed_files=3, total_files=10, total_chunks=30)
        mgr.update_progress(job.job_id, prog)
        assert mgr.get_job(job.job_id).progress == prog
