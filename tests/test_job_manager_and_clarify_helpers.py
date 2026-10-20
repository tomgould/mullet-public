"""Tests for job_manager helpers and clarify tool pure functions."""

import json
import pytest

from mullet.core.knowledge.job_manager import Job, JobManager
from mullet.core.tools.clarify import (
    handle_clarify,
    CLARIFY_SCHEMA,
    MAX_CHOICES,
)


# ---------------------------------------------------------------------------
# Job dataclass
# ---------------------------------------------------------------------------

class TestJob:
    def test_fields_stored(self):
        j = Job(id="j1", description="Run tests")
        assert j.id == "j1"
        assert j.description == "Run tests"

    def test_status_default_pending(self):
        j = Job(id="j1", description="d")
        assert j.status == "pending"

    def test_progress_default_zero(self):
        j = Job(id="j1", description="d")
        assert j.progress == 0

    def test_created_at_auto_set(self):
        j = Job(id="j1", description="d")
        assert j.created_at is not None

    def test_started_at_none_by_default(self):
        j = Job(id="j1", description="d")
        assert j.started_at is None

    def test_completed_at_none_by_default(self):
        j = Job(id="j1", description="d")
        assert j.completed_at is None


# ---------------------------------------------------------------------------
# JobManager
# ---------------------------------------------------------------------------

class TestJobManagerStartJob:
    def test_start_job_returns_job(self):
        mgr = JobManager()
        job = mgr.start_job("j1", "Do something")
        assert job.id == "j1"

    def test_start_job_status_running(self):
        mgr = JobManager()
        job = mgr.start_job("j1", "d")
        assert job.status == "running"

    def test_start_job_started_at_set(self):
        mgr = JobManager()
        job = mgr.start_job("j1", "d")
        assert job.started_at is not None

    def test_duplicate_id_raises_value_error(self):
        mgr = JobManager()
        mgr.start_job("j1", "d")
        with pytest.raises(ValueError):
            mgr.start_job("j1", "d2")

    def test_max_concurrent_raises_runtime_error(self):
        mgr = JobManager(max_concurrent_jobs=1)
        mgr.start_job("j1", "d")
        with pytest.raises(RuntimeError):
            mgr.start_job("j2", "d2")

    def test_increments_running_count(self):
        mgr = JobManager()
        mgr.start_job("j1", "d")
        assert mgr.running_jobs_count == 1


class TestJobManagerGetJob:
    def test_get_existing_job(self):
        mgr = JobManager()
        mgr.start_job("j1", "d")
        job = mgr.get_job("j1")
        assert job is not None
        assert job.id == "j1"

    def test_get_nonexistent_returns_none(self):
        mgr = JobManager()
        assert mgr.get_job("ghost") is None


class TestJobManagerListJobs:
    def test_empty_list_initially(self):
        mgr = JobManager()
        assert mgr.list_jobs() == []

    def test_lists_added_jobs(self):
        mgr = JobManager(max_concurrent_jobs=5)
        mgr.start_job("j1", "d1")
        mgr.start_job("j2", "d2")
        jobs = mgr.list_jobs()
        assert len(jobs) == 2

    def test_returns_list(self):
        mgr = JobManager()
        assert isinstance(mgr.list_jobs(), list)


class TestJobManagerCancelJob:
    def test_cancel_sets_cancelled(self):
        mgr = JobManager()
        mgr.start_job("j1", "d")
        mgr.cancel_job("j1")
        assert mgr.get_job("j1").status == "cancelled"

    def test_cancel_sets_completed_at(self):
        mgr = JobManager()
        mgr.start_job("j1", "d")
        mgr.cancel_job("j1")
        assert mgr.get_job("j1").completed_at is not None

    def test_cancel_decrements_running_count(self):
        mgr = JobManager()
        mgr.start_job("j1", "d")
        mgr.cancel_job("j1")
        assert mgr.running_jobs_count == 0

    def test_cancel_nonexistent_raises(self):
        mgr = JobManager()
        with pytest.raises(ValueError):
            mgr.cancel_job("ghost")


class TestJobManagerCompleteJob:
    def test_complete_sets_completed(self):
        mgr = JobManager()
        mgr.start_job("j1", "d")
        mgr.complete_job("j1")
        assert mgr.get_job("j1").status == "completed"

    def test_complete_sets_progress_100(self):
        mgr = JobManager()
        mgr.start_job("j1", "d")
        mgr.complete_job("j1")
        assert mgr.get_job("j1").progress == 100

    def test_complete_custom_status(self):
        mgr = JobManager()
        mgr.start_job("j1", "d")
        mgr.complete_job("j1", status="failed")
        assert mgr.get_job("j1").status == "failed"

    def test_complete_decrements_running_count(self):
        mgr = JobManager()
        mgr.start_job("j1", "d")
        mgr.complete_job("j1")
        assert mgr.running_jobs_count == 0

    def test_complete_nonexistent_raises(self):
        mgr = JobManager()
        with pytest.raises(ValueError):
            mgr.complete_job("ghost")

    def test_complete_allows_new_job(self):
        mgr = JobManager(max_concurrent_jobs=1)
        mgr.start_job("j1", "d")
        mgr.complete_job("j1")
        job2 = mgr.start_job("j2", "d2")  # should not raise
        assert job2 is not None


class TestUpdateJobProgress:
    def test_update_progress_changes_value(self):
        mgr = JobManager()
        mgr.start_job("j1", "d")
        mgr.update_job_progress("j1", 50)
        assert mgr.get_job("j1").progress == 50

    def test_update_progress_nonexistent_raises(self):
        mgr = JobManager()
        with pytest.raises(ValueError):
            mgr.update_job_progress("ghost", 50)


# ---------------------------------------------------------------------------
# CLARIFY_SCHEMA
# ---------------------------------------------------------------------------

class TestClarifySchema:
    def test_has_description(self):
        assert "description" in CLARIFY_SCHEMA

    def test_has_parameters(self):
        assert "parameters" in CLARIFY_SCHEMA

    def test_question_required(self):
        assert "question" in CLARIFY_SCHEMA["parameters"]["required"]

    def test_question_param_exists(self):
        props = CLARIFY_SCHEMA["parameters"]["properties"]
        assert "question" in props

    def test_choices_param_exists(self):
        props = CLARIFY_SCHEMA["parameters"]["properties"]
        assert "choices" in props


# ---------------------------------------------------------------------------
# MAX_CHOICES constant
# ---------------------------------------------------------------------------

class TestMaxChoices:
    def test_max_choices_is_five(self):
        assert MAX_CHOICES == 5

    def test_max_choices_is_int(self):
        assert isinstance(MAX_CHOICES, int)


# ---------------------------------------------------------------------------
# handle_clarify
# ---------------------------------------------------------------------------

class TestHandleClarify:
    def _run(self, question, **kwargs):
        return json.loads(handle_clarify(question, **kwargs))

    def test_empty_question_returns_error(self):
        result = self._run("")
        assert "error" in result

    def test_none_question_returns_error(self):
        result = json.loads(handle_clarify(None))
        assert "error" in result

    def test_callback_response_returned(self):
        cb = lambda q, c: "my answer"
        result = self._run("What color?", _callback=cb)
        assert result["response"] == "my answer"
        assert result["question"] == "What color?"

    def test_choices_truncated_to_max(self):
        received = []
        def cb(q, choices):
            received.append(len(choices))
            return "ok"
        self._run("q?", choices=["a", "b", "c", "d", "e", "f"], _callback=cb)
        assert received[0] == MAX_CHOICES

    def test_callback_exception_returns_error(self):
        def bad_cb(q, c): raise RuntimeError("fail")
        result = self._run("q?", _callback=bad_cb)
        assert "error" in result

    def test_no_choices_callback_receives_none(self):
        received = []
        def cb(q, choices):
            received.append(choices)
            return "ok"
        self._run("Simple question?", _callback=cb)
        assert received[0] is None

    def test_returns_json_string(self):
        cb = lambda q, c: "answer"
        raw = handle_clarify("q?", _callback=cb)
        assert isinstance(raw, str)
        json.loads(raw)
