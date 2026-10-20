from pathlib import Path
from unittest.mock import patch


from mullet.core.skills.manager import SkillManager


class TestValidateName:
    def test_valid_alphanumeric(self):
        mgr = SkillManager()
        assert mgr._validate_name("my-skill-123") is True

    def test_valid_simple(self):
        mgr = SkillManager()
        assert mgr._validate_name("skill") is True

    def test_invalid_with_space(self):
        mgr = SkillManager()
        assert mgr._validate_name("my skill") is False

    def test_invalid_with_underscore(self):
        mgr = SkillManager()
        assert mgr._validate_name("my_skill") is False

    def test_invalid_empty(self):
        mgr = SkillManager()
        assert mgr._validate_name("") is False

    def test_invalid_special_chars(self):
        mgr = SkillManager()
        assert mgr._validate_name("skill@!") is False

    def test_invalid_uppercase_mixed(self):
        mgr = SkillManager()
        assert mgr._validate_name("My-Skill") is True


class TestCreate:
    def test_invalid_name(self):
        with patch.object(SkillManager, "get_skill_dir", return_value=Path("/tmp/skills")):
            mgr = SkillManager()
            result = mgr.create("bad name", "desc", "body")
        assert result["success"] is False

    def test_already_exists(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_path = skills_dir / "existing"
        skill_path.mkdir()
        (skill_path / "SKILL.md").write_text("content")
        with patch.object(SkillManager, "get_skill_dir", return_value=skills_dir):
            mgr = SkillManager()
            result = mgr.create("existing", "desc", "body")
        assert result["success"] is False
        assert "already exists" in result["message"]

    def test_success(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        with patch.object(SkillManager, "get_skill_dir", return_value=skills_dir):
            mgr = SkillManager()
            result = mgr.create("new-skill", "A description", "Skill body here", triggers=["foo"], tags=["bar"])
        assert result["success"] is True
        assert "new-skill" in result["message"]
        skill_file = skills_dir / "new-skill" / "SKILL.md"
        assert skill_file.exists()
        content = skill_file.read_text()
        assert "A description" in content
        assert "foo" in content
        assert "bar" in content
        assert "Skill body here" in content


class TestEdit:
    def test_not_found(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        with patch.object(SkillManager, "get_skill_dir", return_value=skills_dir):
            mgr = SkillManager()
            result = mgr.edit("missing", body="new")
        assert result["success"] is False
        assert "not found" in result["message"]

    def test_invalid_format(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_path = skills_dir / "bad"
        skill_path.mkdir()
        (skill_path / "SKILL.md").write_text("no frontmatter")
        with patch.object(SkillManager, "get_skill_dir", return_value=skills_dir):
            mgr = SkillManager()
            result = mgr.edit("bad", body="new")
        assert result["success"] is False
        assert "Invalid" in result["message"]

    def test_success(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_path = skills_dir / "editme"
        skill_path.mkdir()
        original = "---\nname: editme\ndescription: old\nversion: 1.0.0\ntriggers: []\ntags: []\nmode: null\n---\nold body"
        (skill_path / "SKILL.md").write_text(original)
        with patch.object(SkillManager, "get_skill_dir", return_value=skills_dir):
            mgr = SkillManager()
            result = mgr.edit("editme", description="new desc", body="new body", triggers=["t1"], tags=["tg1"])
        assert result["success"] is True
        content = (skill_path / "SKILL.md").read_text()
        assert "new desc" in content
        assert "new body" in content
        assert "t1" in content
        assert "tg1" in content


class TestDelete:
    def test_not_found(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        with patch.object(SkillManager, "get_skill_dir", return_value=skills_dir):
            mgr = SkillManager()
            result = mgr.delete("missing")
        assert result["success"] is False
        assert "not found" in result["message"]

    def test_success(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_path = skills_dir / "todelete"
        skill_path.mkdir()
        (skill_path / "SKILL.md").write_text("content")
        with patch.object(SkillManager, "get_skill_dir", return_value=skills_dir):
            mgr = SkillManager()
            result = mgr.delete("todelete")
        assert result["success"] is True
        assert not skill_path.exists()


class TestListUser:
    def test_empty(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        with patch.object(SkillManager, "get_skill_dir", return_value=skills_dir):
            mgr = SkillManager()
            result = mgr.list_user()
        assert result == []

    def test_lists_skills(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_path = skills_dir / "skill-a"
        skill_path.mkdir()
        frontmatter = "---\nname: skill-a\ndescription: desc-a\nversion: 1.0.0\ntriggers: []\ntags: [tag1]\n---\nbody"
        (skill_path / "SKILL.md").write_text(frontmatter)
        with patch.object(SkillManager, "get_skill_dir", return_value=skills_dir):
            mgr = SkillManager()
            result = mgr.list_user()
        assert len(result) == 1
        assert result[0]["name"] == "skill-a"
        assert result[0]["description"] == "desc-a"
        assert result[0]["tags"] == ["tag1"]

    def test_skips_bad_files(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_path = skills_dir / "bad"
        skill_path.mkdir()
        (skill_path / "SKILL.md").write_text("not yaml frontmatter")
        with patch.object(SkillManager, "get_skill_dir", return_value=skills_dir):
            mgr = SkillManager()
            result = mgr.list_user()
        assert result == []
