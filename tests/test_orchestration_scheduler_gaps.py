"""Coverage-gap tests for mullet.core.orchestration.scheduler."""

from mullet.core.orchestration.scheduler import TaskScheduler


class TestSchedulerGaps:
    def test_scheduler_resume_nonexistent_returns_false(self, tmp_path):
        """Cover line 97: resume returns False when task name not found."""
        scheduler = TaskScheduler(config_path=str(tmp_path / "scheduler.json"))
        assert scheduler.resume("nonexistent") is False

    def test_scheduler_mark_run_nonexistent_no_error(self, tmp_path):
        """Cover line 118: mark_run returns early when task name not found."""
        scheduler = TaskScheduler(config_path=str(tmp_path / "scheduler.json"))
        scheduler.mark_run("nonexistent")  # should not raise

    def test_scheduler_load_corrupt_json(self, tmp_path):
        """Cover lines 140-141: corrupt JSON in _load logs warning and continues."""
        path = tmp_path / "scheduler.json"
        path.write_text("not valid json")
        scheduler = TaskScheduler(config_path=str(path))
        assert scheduler.list_tasks() == []
