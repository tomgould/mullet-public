"""Tests covering exception-handler branches in skills modules."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch


from mullet.core.skills.learned import LearnedSkillManager
from mullet.core.skills.learner import LearnedSkill as LearnerSkill, save_skill, find_similar_skill
from mullet.core.skills import stats as stats_module
from mullet.core.skills.manager import SkillManager


# --- learned.py ---

def test_learned_manager_load_corrupt_json(tmp_path):
    bad_json = tmp_path / "learned_skills.json"
    bad_json.write_text("not json")
    manager = LearnedSkillManager(skills_dir=tmp_path)
    assert manager.list() == []


def test_learned_manager_save_oserror(tmp_path):
    manager = LearnedSkillManager(skills_dir=tmp_path)
    with patch.object(Path, "write_text", side_effect=OSError("disk full")):
        skill = manager.record("some pattern", "some context")
    assert skill.name == "some_pattern"


# --- learner.py ---

def test_learner_save_skill_corrupt_existing(tmp_path):
    with patch("mullet.core.skills.learner.SKILLS_DIR", tmp_path):
        skill_name = "test_skill"
        corrupt_file = tmp_path / f"{skill_name}.json"
        corrupt_file.write_text("not json")
        skill = LearnerSkill(name=skill_name, pattern="pat", solution="sol")
        path = save_skill(skill)
        assert path.exists()
        assert json.loads(path.read_text())["name"] == skill_name


def test_learner_find_similar_skill_bad_json(tmp_path):
    with patch("mullet.core.skills.learner.SKILLS_DIR", tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        bad_file = tmp_path / "bad_skill.json"
        bad_file.write_text("not json")
        result = find_similar_skill("some task description")
        assert result is None


# --- stats.py ---

def test_stats_record_invocation_save_error():
    with patch.object(stats_module, "_save", side_effect=Exception("boom")):
        stats_module.record_invocation("my_skill")


def test_stats_record_outcome_save_error():
    with patch.object(stats_module, "_save", side_effect=Exception("boom")):
        stats_module.record_outcome("my_skill", success=True)


# --- manager.py ---

def test_manager_delete_skill_oserror(tmp_path):
    with patch.object(Path, "home", return_value=tmp_path):
        manager = SkillManager()
        skill_dir = manager.skills_dir / "testskill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: testskill\n---\n")
        with patch.object(shutil, "rmtree", side_effect=OSError("access denied")):
            result = manager.delete("testskill")
    assert result["success"] is False
    assert "access denied" in result["message"]


def test_manager_list_user_bad_skill_md(tmp_path):
    with patch.object(Path, "home", return_value=tmp_path):
        manager = SkillManager()
        skill_dir = manager.skills_dir / "badyaml"
        skill_dir.mkdir(parents=True)
        # Write a SKILL.md with invalid YAML frontmatter
        (skill_dir / "SKILL.md").write_text("---\n{name: badyaml\n---\n")
        skills = manager.list_user()
    assert skills == []
