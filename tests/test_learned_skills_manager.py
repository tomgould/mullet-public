"""Tests for LearnedSkillManager scaffold."""

import pytest
from mullet.core.skills.learned import LearnedSkillManager, LearnedSkill


@pytest.fixture
def manager(tmp_path):
    return LearnedSkillManager(skills_dir=tmp_path)


def test_record_creates_skill(manager):
    skill = manager.record("deploy to staging", "Deployment workflow for staging env")
    assert isinstance(skill, LearnedSkill)
    assert skill.name == "deploy_to_staging"
    assert skill.pattern == "deploy to staging"
    assert skill.triggers == ["deploy to staging"]


def test_match_hits_trigger(manager):
    manager.record("deploy to staging", "Deployment workflow")
    result = manager.match("please deploy to staging now")
    assert result is not None
    assert result.name == "deploy_to_staging"
    assert result.use_count == 1


def test_match_misses(manager):
    manager.record("deploy to staging", "Deployment workflow")
    result = manager.match("completely unrelated query")
    assert result is None


def test_list_returns_all(manager):
    manager.record("deploy to staging", "Deployment workflow")
    manager.record("run tests", "Test runner workflow")
    skills = manager.list()
    assert len(skills) == 2
    names = {s.name for s in skills}
    assert names == {"deploy_to_staging", "run_tests"}
