"""Tests for learned skills system."""

import pytest
from mullet.core.skills.learned import LearnedSkillManager, LearnedSkill


@pytest.fixture
def manager(tmp_path):
    m = LearnedSkillManager(skills_dir=tmp_path / "skills")
    return m


def test_record_and_list(manager):
    skill = manager.record("deploy", "Deploy workflow")
    assert isinstance(skill, LearnedSkill)
    assert skill.name == "deploy"
    assert skill.triggers == ["deploy"]

    skills = manager.list()
    assert len(skills) == 1
    assert skills[0].name == "deploy"


def test_persistence(tmp_path):
    dir_path = tmp_path / "skills"
    mgr1 = LearnedSkillManager(skills_dir=dir_path)
    mgr1.record("deploy", "Deploy workflow")

    # New manager instance should load existing skills
    mgr2 = LearnedSkillManager(skills_dir=dir_path)
    skills = mgr2.list()
    assert len(skills) == 1
    assert skills[0].name == "deploy"


def test_exact_match(manager):
    manager.record("deploy to production", "Deploy workflow")

    result = manager.match("please deploy to production now")
    assert result is not None
    assert result.name == "deploy_to_production"


def test_no_match(manager):
    manager.record("deploy", "Deploy workflow")

    result = manager.match("completely unrelated query")
    assert result is None


def test_match_increments_use_count(manager):
    manager.record("deploy", "Deploy workflow")

    result1 = manager.match("please deploy")
    assert result1 is not None
    assert result1.use_count == 1

    result2 = manager.match("deploy again")
    assert result2 is not None
    assert result2.use_count == 2


def test_match_is_case_insensitive(manager):
    manager.record("deploy", "Deploy workflow")

    result = manager.match("DEPLOY to production")
    assert result is not None


def test_list_returns_copy(manager):
    manager.record("deploy", "Deploy workflow")

    skills1 = manager.list()
    skills1.append(LearnedSkill(name="extra", description="extra"))

    skills2 = manager.list()
    assert len(skills2) == 1
