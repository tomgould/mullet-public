"""Gap tests for mullet/core/skills/registry.py — target ≥95% coverage."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from mullet.core.skills.registry import SkillRegistry
from mullet.core.skills.types import Skill, SkillMatch, SkillMetadata


# ---------------------------------------------------------------------------
# Line 34 — load_all with no loader is a no-op
# ---------------------------------------------------------------------------

def test_load_all_without_loader():
    reg = SkillRegistry(loader=None)
    reg.load_all()
    assert reg._skills == {}


# ---------------------------------------------------------------------------
# Lines 106-112 — learned skill match succeeds
# ---------------------------------------------------------------------------

def test_match_learned_skill():
    reg = SkillRegistry()
    reg._learned_manager = MagicMock()

    learned_match = MagicMock()
    learned_match.confidence = 75
    learned_match.skill.skill = Skill(
        metadata=SkillMetadata(name="learned"),
        body="learned body",
        path=Path("."),
    )
    learned_match.matched_triggers = ["trigger1", "trigger2"]
    reg._learned_manager.match.return_value = [learned_match]

    result = reg.match("some text")
    assert result is not None
    assert result.match_type == "learned"
    assert result.confidence == 0.75
    assert "trigger1" in result.trigger


# ---------------------------------------------------------------------------
# Lines 113-114 — learned skill match raises
# ---------------------------------------------------------------------------

def test_match_learned_skill_exception():
    reg = SkillRegistry()
    reg._learned_manager = MagicMock()
    reg._learned_manager.match.side_effect = Exception("learned boom")

    with patch("mullet.core.skills.registry.logger") as mock_logger:
        result = reg.match("some text")
        assert result is None
        mock_logger.debug.assert_called_once()


# ---------------------------------------------------------------------------
# Lines 125-135 — invoke skill not found
# ---------------------------------------------------------------------------

def test_invoke_skill_not_found():
    reg = SkillRegistry()
    result = reg.invoke("nonexistent")
    assert "not found" in result.content
    assert result.unmet_requirements == ["Skill 'nonexistent' is not registered"]


# ---------------------------------------------------------------------------
# Lines 141-142 — record_invocation exception handled
# ---------------------------------------------------------------------------

def test_invoke_record_invocation_exception():
    reg = SkillRegistry()
    skill = Skill(metadata=SkillMetadata(name="test_skill"), body="body", path=Path("."))
    reg.register(skill)

    with patch("mullet.core.skills.stats.record_invocation", side_effect=Exception("record boom")):
        with patch("mullet.core.skills.registry.logger") as mock_logger:
            result = reg.invoke("test_skill")
            assert "Failed to record skill invocation" in str(mock_logger.warning.call_args)


# ---------------------------------------------------------------------------
# Lines 149-172 — SkillInvoker exception → fallback invocation
# ---------------------------------------------------------------------------

def test_invoke_skill_invoker_fallback():
    reg = SkillRegistry()
    skill = Skill(
        metadata=SkillMetadata(
            name="test_skill",
            required_env_vars=["MISSING_VAR"],
            mode="append",
            tier_override="LOW",
        ),
        body="body",
        path=Path("."),
    )
    reg.register(skill)

    with patch("mullet.core.skills.invoker.SkillInvoker", side_effect=Exception("invoker boom")):
        with patch("mullet.core.skills.registry.logger") as mock_logger:
            result = reg.invoke("test_skill", user_args="my args")

    assert result.skill == skill
    assert result.mode == "append"
    assert result.tier_override == "LOW"
    assert "my args" in result.content
    assert any("MISSING_VAR" in req for req in result.unmet_requirements)
    mock_logger.debug.assert_called_once()


# ---------------------------------------------------------------------------
# list_skills with tag filter
# ---------------------------------------------------------------------------

def test_list_skills_tag_filter():
    reg = SkillRegistry()
    reg.register(Skill(metadata=SkillMetadata(name="a", tags=["tag1"]), body="", path=Path(".")))
    reg.register(Skill(metadata=SkillMetadata(name="b", tags=["tag2"]), body="", path=Path(".")))

    all_skills = reg.list_skills()
    assert len(all_skills) == 2

    filtered = reg.list_skills(tags=["tag1"])
    assert len(filtered) == 1
    assert filtered[0]["name"] == "a"


# ---------------------------------------------------------------------------
# build_index_prompt empty vs populated
# ---------------------------------------------------------------------------

def test_build_index_prompt_empty():
    reg = SkillRegistry()
    assert reg.build_index_prompt() == ""


def test_build_index_prompt_populated():
    reg = SkillRegistry()
    reg.register(
        Skill(
            metadata=SkillMetadata(name="skill_a", description="does a", tags=["cat1"]),
            body="",
            path=Path("."),
        )
    )
    prompt = reg.build_index_prompt()
    assert "Available Skills" in prompt
    assert "/skill_a" in prompt


# ---------------------------------------------------------------------------
# Lines 28-29 — LearnedSkillManager init exception
# ---------------------------------------------------------------------------

def test_registry_init_learned_manager_exception():
    with patch("mullet.core.skills.learned.LearnedSkillManager", side_effect=Exception("no learned")):
        with patch("mullet.core.skills.registry.logger") as mock_logger:
            reg = SkillRegistry()
            assert reg._learned_manager is None
            mock_logger.debug.assert_called_once()


# ---------------------------------------------------------------------------
# Lines 35-43 — load_all loads learned skills
# ---------------------------------------------------------------------------

def test_load_all_learned_skills():
    mock_loader = MagicMock()
    skill = Skill(metadata=SkillMetadata(name="loaded"), body="", path=Path("."))
    mock_loader.scan.return_value = [skill]
    reg = SkillRegistry(loader=mock_loader)
    reg._learned_manager = MagicMock()
    reg.load_all()
    assert "loaded" in reg._skills
    reg._learned_manager.load.assert_called_once()


def test_load_all_learned_skills_load_exception():
    mock_loader = MagicMock()
    mock_loader.scan.return_value = []
    reg = SkillRegistry(loader=mock_loader)
    reg._learned_manager = MagicMock()
    reg._learned_manager.load.side_effect = Exception("load boom")
    with patch("mullet.core.skills.registry.logger") as mock_logger:
        reg.load_all()
        mock_logger.debug.assert_called_once()


# ---------------------------------------------------------------------------
# Line 50 — register with aliases
# ---------------------------------------------------------------------------

def test_register_with_alias():
    reg = SkillRegistry()
    skill = Skill(
        metadata=SkillMetadata(name="main_skill", aliases=["alias1", "alias2"]),
        body="",
        path=Path("."),
    )
    reg.register(skill)
    assert reg.get("alias1") == skill
    assert reg.get("alias2") == skill


# ---------------------------------------------------------------------------
# Lines 69-72 — slash command match
# ---------------------------------------------------------------------------

def test_match_slash_command():
    reg = SkillRegistry()
    skill = Skill(
        metadata=SkillMetadata(name="deploy", triggers=["deploy"]),
        body="",
        path=Path("."),
    )
    reg.register(skill)
    result = reg.match("/deploy")
    assert result is not None
    assert result.match_type == "slash_command"
    assert result.confidence == 1.0


# ---------------------------------------------------------------------------
# Lines 91-93 — informational intent filtering
# ---------------------------------------------------------------------------

def test_match_informational_intent():
    reg = SkillRegistry()
    skill = Skill(
        metadata=SkillMetadata(name="autopilot", triggers=["autopilot"]),
        body="",
        path=Path("."),
    )
    reg.register(skill)
    result = reg.match("what is autopilot?")
    assert result is None


# ---------------------------------------------------------------------------
# Lines 97-99 — keyword trigger match sorting (longest wins)
# ---------------------------------------------------------------------------

def test_match_longest_trigger_wins():
    reg = SkillRegistry()
    skill_a = Skill(
        metadata=SkillMetadata(name="a", triggers=["deploy"]),
        body="",
        path=Path("."),
    )
    skill_b = Skill(
        metadata=SkillMetadata(name="b", triggers=["deploy app"]),
        body="",
        path=Path("."),
    )
    reg.register(skill_a)
    reg.register(skill_b)
    result = reg.match("deploy app to prod")
    assert result is not None
    assert result.skill.metadata.name == "b"
    assert result.trigger == "deploy app"
