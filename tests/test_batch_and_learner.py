"""Tests for learned skills."""

from mullet.core.skills.learner import extract_skill, save_skill, find_similar_skill, LearnedSkill


# === Learned Skills Tests ===

def test_extract_skill_returns_none_for_few_tools():
    result = extract_skill("simple task", ["one_tool"], "done")
    assert result is None


def test_extract_skill_returns_skill():
    skill = extract_skill("build a REST API", ["file_ops", "terminal", "web"], "API built")
    assert skill is not None
    assert isinstance(skill, LearnedSkill)
    assert "file-ops" in skill.tags
    assert "terminal" in skill.tags


def test_save_and_find_skill(tmp_path, monkeypatch):
    import mullet.core.skills.learner as learner
    monkeypatch.setattr(learner, "SKILLS_DIR", tmp_path)

    skill = LearnedSkill(name="test_skill", pattern="build something", solution="file → terminal → verify")
    path = save_skill(skill)
    assert path.exists()

    found = find_similar_skill("build something new")
    assert found is not None
    assert found.name == "test_skill"


def test_find_similar_no_match(tmp_path, monkeypatch):
    import mullet.core.skills.learner as learner
    monkeypatch.setattr(learner, "SKILLS_DIR", tmp_path)
    assert find_similar_skill("completely unrelated query") is None
