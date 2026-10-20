"""Tests for mullet.core.skills.learner — pure functions."""
from __future__ import annotations

import json
from unittest.mock import patch


from mullet.core.skills.learner import (
    LearnedSkill,
    extract_skill,
    find_similar_skill,
    list_skills,
    save_skill,
)


class TestLearnedSkill:
    def test_to_dict_roundtrip(self):
        s = LearnedSkill(name="fix_bug", pattern="fix the bug", solution="Read → Edit", tags=["file-ops"])
        d = s.to_dict()
        s2 = LearnedSkill.from_dict(d)
        assert s2.name == s.name
        assert s2.pattern == s.pattern
        assert s2.solution == s.solution
        assert s2.tags == s.tags

    def test_from_dict_defaults(self):
        s = LearnedSkill.from_dict({"name": "x", "pattern": "p", "solution": "s"})
        assert s.success_count == 1
        assert s.tags == []

    def test_to_dict_has_required_keys(self):
        s = LearnedSkill(name="n", pattern="p", solution="s")
        d = s.to_dict()
        for key in ("name", "pattern", "solution", "success_count", "tags"):
            assert key in d


class TestExtractSkill:
    def test_returns_none_for_single_tool(self):
        result = extract_skill("list files", ["Read"], "ok")
        assert result is None

    def test_returns_skill_for_multiple_tools(self):
        result = extract_skill("fix the bug in auth", ["Read", "Edit", "Bash"], "tests pass")
        assert result is not None
        assert isinstance(result, LearnedSkill)

    def test_name_derived_from_task(self):
        result = extract_skill("deploy the service", ["Read", "Bash", "Write"], "done")
        assert "deploy" in result.name

    def test_file_ops_tag(self):
        result = extract_skill("update config file", ["Read", "Edit_file", "Bash"], "done")
        assert "file-ops" in result.tags

    def test_solution_joins_tool_calls(self):
        tools = ["Read", "Edit", "Bash"]
        result = extract_skill("do task", tools, "done")
        assert "Read" in result.solution
        assert "Edit" in result.solution
        assert "Bash" in result.solution

    def test_terminal_tag(self):
        """Line 49: terminal tag added when tool name contains 'terminal'."""
        result = extract_skill("run shell command", ["Read", "terminal_run", "Edit"], "done")
        assert "terminal" in result.tags

    def test_web_tag(self):
        """Line 51: web tag added when tool name contains 'web'."""
        result = extract_skill("fetch page content", ["Read", "web_fetch", "Edit"], "done")
        assert "web" in result.tags

    def test_solution_truncated_at_10_tools(self):
        tools = [f"Tool{i}" for i in range(20)]
        result = extract_skill("big task", tools, "done")
        # Only first 10 tools in solution
        parts = result.solution.split(" → ")
        assert len(parts) == 10

    def test_pattern_truncated_at_200_chars(self):
        long_task = "x " * 200
        result = extract_skill(long_task, ["Read", "Edit"], "done")
        assert len(result.pattern) <= 200


class TestSaveSkill:
    def test_save_new_skill(self, tmp_path):
        skills_dir = tmp_path / "skills"
        with patch("mullet.core.skills.learner.SKILLS_DIR", skills_dir):
            skill = LearnedSkill(name="new_skill", pattern="p", solution="s")
            path = save_skill(skill)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["name"] == "new_skill"

    def test_save_increments_success_count_if_exists(self, tmp_path):
        """Lines 64-65: existing skill is loaded and success_count incremented."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        existing = LearnedSkill(name="my_skill", pattern="p", solution="s", success_count=3)
        (skills_dir / "my_skill.json").write_text(json.dumps(existing.to_dict()))
        with patch("mullet.core.skills.learner.SKILLS_DIR", skills_dir):
            skill = LearnedSkill(name="my_skill", pattern="p", solution="s")
            save_skill(skill)
        data = json.loads((skills_dir / "my_skill.json").read_text())
        assert data["success_count"] == 4

    def test_save_corrupt_existing_does_not_raise(self, tmp_path):
        """Line 66: corrupt existing file → exception swallowed, skill saved with count=1."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "bad.json").write_text("{corrupt")
        with patch("mullet.core.skills.learner.SKILLS_DIR", skills_dir):
            skill = LearnedSkill(name="bad", pattern="p", solution="s")
            save_skill(skill)
        data = json.loads((skills_dir / "bad.json").read_text())
        assert data["success_count"] == 1


class TestFindSimilarSkill:
    def test_returns_none_when_dir_missing(self, tmp_path):
        nonexistent = tmp_path / "no_skills_here"
        with patch("mullet.core.skills.learner.SKILLS_DIR", nonexistent):
            result = find_similar_skill("do something")
        assert result is None

    def test_returns_none_no_match(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill = LearnedSkill(name="deploy_service", pattern="deploy the service to production", solution="Read → Bash")
        (skills_dir / "deploy_service.json").write_text(json.dumps(skill.to_dict()))
        with patch("mullet.core.skills.learner.SKILLS_DIR", skills_dir):
            result = find_similar_skill("unrelated topic xyz")
        assert result is None

    def test_returns_match_above_threshold(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill = LearnedSkill(name="run_tests", pattern="run all pytest tests for auth module", solution="Bash")
        (skills_dir / "run_tests.json").write_text(json.dumps(skill.to_dict()))
        with patch("mullet.core.skills.learner.SKILLS_DIR", skills_dir):
            result = find_similar_skill("run all pytest tests for auth module")
        assert result is not None
        assert result.name == "run_tests"

    def test_skips_malformed_json(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "bad.json").write_text("{invalid json")
        with patch("mullet.core.skills.learner.SKILLS_DIR", skills_dir):
            result = find_similar_skill("anything")
        assert result is None

    def test_skips_skill_with_empty_pattern(self, tmp_path):
        """Line 88: skill with empty pattern is skipped (pattern_words is empty set)."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        empty_pattern = LearnedSkill(name="empty", pattern="", solution="s")
        (skills_dir / "empty.json").write_text(json.dumps(empty_pattern.to_dict()))
        with patch("mullet.core.skills.learner.SKILLS_DIR", skills_dir):
            result = find_similar_skill("anything at all")
        assert result is None


class TestListSkills:
    def test_returns_empty_when_dir_missing(self, tmp_path):
        nonexistent = tmp_path / "no_skills"
        with patch("mullet.core.skills.learner.SKILLS_DIR", nonexistent):
            result = list_skills()
        assert result == []

    def test_returns_all_skills(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        for name in ("alpha", "beta", "gamma"):
            s = LearnedSkill(name=name, pattern=f"pattern for {name}", solution="Read")
            (skills_dir / f"{name}.json").write_text(json.dumps(s.to_dict()))
        with patch("mullet.core.skills.learner.SKILLS_DIR", skills_dir):
            result = list_skills()
        assert len(result) == 3
        names = {s.name for s in result}
        assert names == {"alpha", "beta", "gamma"}

    def test_skips_malformed_files(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "good.json").write_text(json.dumps(
            LearnedSkill(name="good", pattern="p", solution="s").to_dict()
        ))
        (skills_dir / "bad.json").write_text("NOT JSON")
        with patch("mullet.core.skills.learner.SKILLS_DIR", skills_dir):
            result = list_skills()
        assert len(result) == 1
        assert result[0].name == "good"
