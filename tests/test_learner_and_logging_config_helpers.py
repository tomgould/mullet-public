"""Tests for learner pure helpers and logging_config pure functions."""


from mullet.core.skills.learner import (
    LearnedSkill,
    extract_skill,
)
from mullet.knowledge.engine.logging_config import (
    AuditConfig,
    load_audit_config,
    get_default_audit_config,
)


# ---------------------------------------------------------------------------
# LearnedSkill.to_dict / from_dict
# ---------------------------------------------------------------------------

class TestLearnedSkillDict:
    def test_to_dict_fields(self):
        skill = LearnedSkill(name="my_skill", pattern="Fix bug", solution="Edit → test")
        d = skill.to_dict()
        assert d["name"] == "my_skill"
        assert d["pattern"] == "Fix bug"
        assert d["solution"] == "Edit → test"

    def test_to_dict_success_count(self):
        skill = LearnedSkill(name="s", pattern="p", solution="sol", success_count=3)
        assert skill.to_dict()["success_count"] == 3

    def test_to_dict_tags(self):
        skill = LearnedSkill(name="s", pattern="p", solution="sol", tags=["file-ops"])
        assert "file-ops" in skill.to_dict()["tags"]

    def test_from_dict_round_trip(self):
        skill = LearnedSkill(name="s", pattern="p", solution="x", tags=["t"])
        d = skill.to_dict()
        restored = LearnedSkill.from_dict(d)
        assert restored.name == "s"
        assert restored.pattern == "p"
        assert "t" in restored.tags

    def test_default_success_count_is_one(self):
        skill = LearnedSkill(name="s", pattern="p", solution="sol")
        assert skill.success_count == 1


# ---------------------------------------------------------------------------
# extract_skill
# ---------------------------------------------------------------------------

class TestExtractSkill:
    def test_fewer_than_two_calls_returns_none(self):
        result = extract_skill("fix bug", ["Read"], "done")
        assert result is None

    def test_returns_learned_skill(self):
        result = extract_skill("fix the auth bug", ["Read", "Edit", "Bash"], "Fixed")
        assert isinstance(result, LearnedSkill)

    def test_name_derived_from_task(self):
        result = extract_skill("fix the bug now", ["Read", "Edit"], "done")
        assert "fix" in result.name

    def test_pattern_from_task(self):
        result = extract_skill("fix the auth bug", ["Read", "Edit"], "done")
        assert "fix" in result.pattern.lower()

    def test_solution_joins_tools(self):
        result = extract_skill("task", ["Read", "Edit", "Bash"], "done")
        assert "Read" in result.solution
        assert "Edit" in result.solution

    def test_file_ops_tag_added(self):
        result = extract_skill("task", ["read_file", "write_file"], "done")
        assert "file-ops" in result.tags

    def test_no_tags_when_tools_generic(self):
        result = extract_skill("task", ["Tool1", "Tool2"], "done")
        assert isinstance(result.tags, list)

    def test_truncates_long_pattern(self):
        long_task = "a " * 200
        result = extract_skill(long_task, ["T1", "T2"], "done")
        assert len(result.pattern) <= 200


# ---------------------------------------------------------------------------
# AuditConfig
# ---------------------------------------------------------------------------

class TestAuditConfig:
    def test_defaults(self):
        cfg = AuditConfig()
        assert cfg.enabled is True
        assert cfg.log_level == "full"
        assert cfg.include_chunks is True

    def test_should_log_true_by_default(self):
        cfg = AuditConfig()
        assert cfg.should_log() is True

    def test_should_log_false_when_disabled(self):
        cfg = AuditConfig(enabled=False)
        assert cfg.should_log() is False

    def test_should_log_false_when_level_none(self):
        cfg = AuditConfig(log_level="none")
        assert cfg.should_log() is False

    def test_is_full_true_for_full(self):
        cfg = AuditConfig(log_level="full")
        assert cfg.is_full() is True

    def test_is_full_false_for_minimal(self):
        cfg = AuditConfig(log_level="minimal")
        assert cfg.is_full() is False

    def test_is_minimal_true_for_minimal(self):
        cfg = AuditConfig(log_level="minimal")
        assert cfg.is_minimal() is True

    def test_is_minimal_false_for_full(self):
        cfg = AuditConfig(log_level="full")
        assert cfg.is_minimal() is False

    def test_retention_days_defaults_none(self):
        cfg = AuditConfig()
        assert cfg.retention_days is None


# ---------------------------------------------------------------------------
# load_audit_config
# ---------------------------------------------------------------------------

class TestLoadAuditConfig:
    def test_empty_config_uses_defaults(self):
        cfg = load_audit_config({})
        assert cfg.enabled is True
        assert cfg.log_level == "full"

    def test_loads_enabled_false(self):
        cfg = load_audit_config({"audit": {"enabled": False}})
        assert cfg.enabled is False

    def test_loads_log_level(self):
        cfg = load_audit_config({"audit": {"log_level": "minimal"}})
        assert cfg.log_level == "minimal"

    def test_loads_retention_days(self):
        cfg = load_audit_config({"audit": {"retention_days": 30}})
        assert cfg.retention_days == 30

    def test_missing_audit_section_uses_defaults(self):
        cfg = load_audit_config({"other_key": "val"})
        assert cfg.enabled is True

    def test_returns_audit_config(self):
        assert isinstance(load_audit_config({}), AuditConfig)


# ---------------------------------------------------------------------------
# get_default_audit_config
# ---------------------------------------------------------------------------

class TestGetDefaultAuditConfig:
    def test_returns_audit_config(self):
        assert isinstance(get_default_audit_config(), AuditConfig)

    def test_enabled_by_default(self):
        cfg = get_default_audit_config()
        assert cfg.enabled is True

    def test_full_level_by_default(self):
        cfg = get_default_audit_config()
        assert cfg.log_level == "full"
