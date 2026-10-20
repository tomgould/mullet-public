from pathlib import Path
from unittest.mock import patch


from mullet.core.skills.registry import SkillRegistry
from mullet.core.skills.types import Skill, SkillMetadata


def make_skill(name, triggers, aliases=None, tags=None, description="test"):
    return Skill(
        metadata=SkillMetadata(
            name=name,
            description=description,
            triggers=list(triggers),
            aliases=list(aliases) if aliases else [],
            tags=list(tags) if tags else [],
        ),
        body="",
        path=Path("."),
    )


@patch("mullet.core.skills.learned.LearnedSkillManager", side_effect=ImportError)
class TestRegisterAndGet:
    def test_register_and_get_by_name(self, _mock):
        reg = SkillRegistry()
        skill = make_skill("test-skill", ["run"])
        reg.register(skill)
        assert reg.get("test-skill") is skill

    def test_get_by_alias(self, _mock):
        reg = SkillRegistry()
        skill = make_skill("test-skill", ["run"], aliases=["ts"])
        reg.register(skill)
        assert reg.get("ts") is skill

    def test_get_missing_returns_none(self, _mock):
        reg = SkillRegistry()
        assert reg.get("missing") is None


@patch("mullet.core.skills.learned.LearnedSkillManager", side_effect=ImportError)
class TestMatch:
    def test_slash_command_match(self, _mock):
        reg = SkillRegistry()
        skill = make_skill("deploy", ["deploy"])
        reg.register(skill)
        result = reg.match("/deploy")
        assert result is not None
        assert result.match_type == "slash_command"
        assert result.skill is skill
        assert result.confidence == 1.0
        assert result.trigger == "/deploy"

    def test_trigger_keyword_match(self, _mock):
        reg = SkillRegistry()
        skill = make_skill("deploy", ["deploy app"])
        reg.register(skill)
        result = reg.match("please deploy app now")
        assert result is not None
        assert result.match_type == "keyword"
        assert result.skill is skill
        assert result.confidence == 0.9
        assert result.trigger == "deploy app"

    def test_informational_query_does_not_trigger(self, _mock):
        reg = SkillRegistry()
        skill = make_skill("deploy", ["deploy"])
        reg.register(skill)
        assert reg.match("what is deploy?") is None

    def test_no_match_returns_none(self, _mock):
        reg = SkillRegistry()
        skill = make_skill("deploy", ["deploy"])
        reg.register(skill)
        assert reg.match("hello world") is None


@patch("mullet.core.skills.learned.LearnedSkillManager", side_effect=ImportError)
class TestListSkills:
    def test_returns_list_of_dicts_with_expected_keys(self, _mock):
        reg = SkillRegistry()
        skill = make_skill("s1", ["t1"], tags=["dev"], description="d1")
        reg.register(skill)
        result = reg.list_skills()
        assert len(result) == 1
        assert set(result[0].keys()) == {"name", "description", "scope", "tags", "mode"}
        assert result[0]["name"] == "s1"
        assert result[0]["description"] == "d1"
        assert result[0]["scope"] == "builtin"

    def test_tag_filter(self, _mock):
        reg = SkillRegistry()
        reg.register(make_skill("s1", ["t1"], tags=["dev"]))
        reg.register(make_skill("s2", ["t2"], tags=["ops"]))
        dev_skills = reg.list_skills(tags=["dev"])
        assert len(dev_skills) == 1
        assert dev_skills[0]["name"] == "s1"
        assert reg.list_skills(tags=["missing"]) == []

    def test_empty_registry_returns_empty_list(self, _mock):
        reg = SkillRegistry()
        assert reg.list_skills() == []


@patch("mullet.core.skills.learned.LearnedSkillManager", side_effect=ImportError)
class TestBuildIndexPrompt:
    def test_returns_string_with_name_and_description(self, _mock):
        reg = SkillRegistry()
        reg.register(make_skill("deploy", ["d"], tags=["ops"], description="Deploy things"))
        result = reg.build_index_prompt()
        assert "deploy" in result
        assert "Deploy things" in result

    def test_empty_registry_returns_empty_string(self, _mock):
        reg = SkillRegistry()
        assert reg.build_index_prompt() == ""
