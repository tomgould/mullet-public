"""Tests for scheduler pure helpers and invoker pure functions."""

import pytest
from pathlib import Path

from mullet.core.orchestration.scheduler import (
    parse_interval,
    ScheduledTask,
    TaskScheduler,
)
from mullet.core.skills.types import Skill, SkillMetadata
from mullet.core.skills.invoker import (
    SkillInvoker,
    generate_runtime_guidance,
)


# ---------------------------------------------------------------------------
# parse_interval
# ---------------------------------------------------------------------------

class TestParseInterval:
    def test_seconds(self):
        assert parse_interval("30s") == 30

    def test_minutes(self):
        assert parse_interval("5m") == 300

    def test_hours(self):
        assert parse_interval("1h") == 3600

    def test_days(self):
        assert parse_interval("1d") == 86400

    def test_large_value(self):
        assert parse_interval("10m") == 600

    def test_whitespace_stripped(self):
        assert parse_interval(" 5m ") == 300

    def test_invalid_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_interval("5x")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            parse_interval("")

    def test_no_unit_raises(self):
        with pytest.raises(ValueError):
            parse_interval("300")

    def test_returns_int(self):
        assert isinstance(parse_interval("1h"), int)


# ---------------------------------------------------------------------------
# ScheduledTask
# ---------------------------------------------------------------------------

class TestScheduledTask:
    def test_fields_stored(self):
        t = ScheduledTask(name="sync", schedule="1h", command="mullet sync")
        assert t.name == "sync"
        assert t.schedule == "1h"
        assert t.command == "mullet sync"

    def test_defaults(self):
        t = ScheduledTask(name="t", schedule="5m", command="cmd")
        assert t.enabled is True
        assert t.run_count == 0
        assert t.last_run is None

    def test_created_at_auto_set(self):
        t = ScheduledTask(name="t", schedule="5m", command="cmd")
        assert bool(t.created_at)

    def test_next_run_computed_on_init(self):
        t = ScheduledTask(name="t", schedule="5m", command="cmd")
        assert t.next_run is not None

    def test_invalid_schedule_sets_next_run_none(self):
        t = ScheduledTask(name="t", schedule="INVALID", command="cmd")
        assert t.next_run is None


# ---------------------------------------------------------------------------
# TaskScheduler
# ---------------------------------------------------------------------------

class TestTaskScheduler:
    def test_empty_on_missing_file(self, tmp_path):
        sched = TaskScheduler(config_path=str(tmp_path / "sched.json"))
        assert sched.list_tasks() == []

    def test_add_task(self, tmp_path):
        sched = TaskScheduler(config_path=str(tmp_path / "sched.json"))
        task = sched.add("daily", "1h", "mullet sync")
        assert task.name == "daily"

    def test_add_duplicate_raises(self, tmp_path):
        sched = TaskScheduler(config_path=str(tmp_path / "sched.json"))
        sched.add("daily", "1h", "cmd")
        with pytest.raises(ValueError):
            sched.add("daily", "5m", "cmd2")

    def test_add_invalid_schedule_raises(self, tmp_path):
        sched = TaskScheduler(config_path=str(tmp_path / "sched.json"))
        with pytest.raises(ValueError):
            sched.add("daily", "invalid", "cmd")

    def test_list_tasks_returns_added(self, tmp_path):
        sched = TaskScheduler(config_path=str(tmp_path / "sched.json"))
        sched.add("t1", "1h", "cmd1")
        sched.add("t2", "5m", "cmd2")
        names = [t.name for t in sched.list_tasks()]
        assert "t1" in names
        assert "t2" in names

    def test_remove_existing(self, tmp_path):
        sched = TaskScheduler(config_path=str(tmp_path / "sched.json"))
        sched.add("t1", "1h", "cmd")
        assert sched.remove("t1") is True
        assert sched.list_tasks() == []

    def test_remove_nonexistent_returns_false(self, tmp_path):
        sched = TaskScheduler(config_path=str(tmp_path / "sched.json"))
        assert sched.remove("ghost") is False

    def test_pause_disables_task(self, tmp_path):
        sched = TaskScheduler(config_path=str(tmp_path / "sched.json"))
        sched.add("t1", "1h", "cmd")
        assert sched.pause("t1") is True
        task = sched.list_tasks()[0]
        assert task.enabled is False

    def test_pause_nonexistent_returns_false(self, tmp_path):
        sched = TaskScheduler(config_path=str(tmp_path / "sched.json"))
        assert sched.pause("ghost") is False

    def test_resume_enables_task(self, tmp_path):
        sched = TaskScheduler(config_path=str(tmp_path / "sched.json"))
        sched.add("t1", "1h", "cmd")
        sched.pause("t1")
        assert sched.resume("t1") is True
        task = sched.list_tasks()[0]
        assert task.enabled is True

    def test_mark_run_increments_run_count(self, tmp_path):
        sched = TaskScheduler(config_path=str(tmp_path / "sched.json"))
        sched.add("t1", "5m", "cmd")
        sched.mark_run("t1")
        task = sched.list_tasks()[0]
        assert task.run_count == 1

    def test_mark_run_sets_last_run(self, tmp_path):
        sched = TaskScheduler(config_path=str(tmp_path / "sched.json"))
        sched.add("t1", "5m", "cmd")
        sched.mark_run("t1")
        task = sched.list_tasks()[0]
        assert task.last_run is not None

    def test_persists_across_instances(self, tmp_path):
        path = str(tmp_path / "sched.json")
        s1 = TaskScheduler(config_path=path)
        s1.add("persisted", "1h", "cmd")
        s2 = TaskScheduler(config_path=path)
        assert any(t.name == "persisted" for t in s2.list_tasks())


# ---------------------------------------------------------------------------
# generate_runtime_guidance
# ---------------------------------------------------------------------------

def _make_skill(
    name="test_skill",
    tier_override=None,
    required_tools=None,
    required_env_vars=None,
    platforms=None,
):
    meta = SkillMetadata(
        name=name,
        tier_override=tier_override,
        required_tools=required_tools or [],
        required_env_vars=required_env_vars or [],
        platforms=platforms or [],
    )
    return Skill(metadata=meta, body="Body text", path=Path("/tmp/skill.md"))


class TestGenerateRuntimeGuidance:
    def test_no_metadata_returns_empty(self):
        skill = _make_skill()
        assert generate_runtime_guidance(skill) == ""

    def test_tier_override_in_output(self):
        skill = _make_skill(tier_override="premium")
        result = generate_runtime_guidance(skill)
        assert "premium" in result

    def test_required_tools_in_output(self):
        skill = _make_skill(required_tools=["Read", "Edit"])
        result = generate_runtime_guidance(skill)
        assert "Read" in result
        assert "Edit" in result

    def test_platforms_in_output(self):
        skill = _make_skill(platforms=["linux", "macos"])
        result = generate_runtime_guidance(skill)
        assert "linux" in result

    def test_has_header(self):
        skill = _make_skill(tier_override="fast")
        result = generate_runtime_guidance(skill)
        assert "Skill Runtime Guidance" in result

    def test_env_vars_dict_format_in_output(self):
        skill = _make_skill(required_env_vars=[{"name": "MY_TOKEN", "help": "API token"}])
        result = generate_runtime_guidance(skill)
        assert "MY_TOKEN" in result


# ---------------------------------------------------------------------------
# SkillInvoker.check_requirements
# ---------------------------------------------------------------------------

class TestSkillInvokerCheckRequirements:
    def setup_method(self):
        self.inv = SkillInvoker()

    def test_no_requirements_returns_empty(self):
        skill = _make_skill()
        assert self.inv.check_requirements(skill) == []

    def test_missing_env_var_returned(self):
        skill = _make_skill(required_env_vars=[{"name": "_MULLET_TEST_NEVER_SET", "help": "key"}])
        unmet = self.inv.check_requirements(skill)
        assert len(unmet) == 1
        assert "_MULLET_TEST_NEVER_SET" in unmet[0]

    def test_present_env_var_not_returned(self, monkeypatch):
        monkeypatch.setenv("_MULLET_TEST_KEY", "secret")
        skill = _make_skill(required_env_vars=[{"name": "_MULLET_TEST_KEY"}])
        assert self.inv.check_requirements(skill) == []

    def test_string_env_var_format(self):
        skill = _make_skill(required_env_vars=["_MULLET_TEST_MISSING_STRING_VAR"])
        unmet = self.inv.check_requirements(skill)
        assert len(unmet) == 1

    def test_returns_list(self):
        skill = _make_skill()
        assert isinstance(self.inv.check_requirements(skill), list)


# ---------------------------------------------------------------------------
# SkillInvoker.build_pipeline_context
# ---------------------------------------------------------------------------

class TestSkillInvokerBuildPipelineContext:
    def setup_method(self):
        self.inv = SkillInvoker()

    def test_no_pipeline_returns_none(self):
        skill = _make_skill()
        assert self.inv.build_pipeline_context(skill) is None

    def test_pipeline_context_has_pipeline_str(self):
        meta = SkillMetadata(
            name="step2",
            pipeline=["step1", "step2", "step3"],
            next_skill="step3",
        )
        skill = Skill(metadata=meta, body="body", path=Path("/tmp/skill.md"))
        ctx = self.inv.build_pipeline_context(skill)
        assert "step1" in ctx
        assert "step2" in ctx
        assert "step3" in ctx

    def test_pipeline_shows_current_stage(self):
        meta = SkillMetadata(
            name="step2",
            pipeline=["step1", "step2", "step3"],
            next_skill="step3",
        )
        skill = Skill(metadata=meta, body="body", path=Path("/tmp/skill.md"))
        ctx = self.inv.build_pipeline_context(skill)
        assert "Current stage" in ctx
        assert "2/3" in ctx

    def test_next_skill_only_returns_context(self):
        meta = SkillMetadata(name="step1", next_skill="step2")
        skill = Skill(metadata=meta, body="body", path=Path("/tmp/skill.md"))
        ctx = self.inv.build_pipeline_context(skill)
        assert ctx is not None
        assert "step2" in ctx
