"""Tests for the SkillInvoker."""

from pathlib import Path
from mullet.core.skills.invoker import SkillInvoker, resolve_handoff_artifact
from mullet.core.skills.types import Skill, SkillMetadata, SkillInvocation


def _make_skill(**kwargs) -> Skill:
    meta = SkillMetadata(name="test", description="Test skill", **kwargs)
    return Skill(metadata=meta, body="# Test\nDo the thing.", path=Path("."))


def test_basic_invoke():
    invoker = SkillInvoker()
    skill = _make_skill()
    result = invoker.invoke(skill)
    assert isinstance(result, SkillInvocation)
    assert "Do the thing" in result.content
    assert result.mode is None


def test_invoke_with_mode():
    invoker = SkillInvoker()
    skill = _make_skill(mode="autopilot")
    result = invoker.invoke(skill)
    assert result.mode == "autopilot"


def test_invoke_with_user_args():
    invoker = SkillInvoker()
    skill = _make_skill()
    result = invoker.invoke(skill, user_args="--strict src/auth.py")
    assert "--strict" in result.content
    assert "src/auth.py" in result.content


def test_invoke_with_pipeline():
    invoker = SkillInvoker()
    skill = _make_skill(
        pipeline=["plan", "test", "verify"],
        next_skill="verify",
        handoff=".mullet/plans/*.md",
    )
    result = invoker.invoke(skill)
    assert "Pipeline" in result.content
    assert "verify" in result.content


def test_check_requirements_met():
    invoker = SkillInvoker()
    skill = _make_skill(required_env_vars=[{"name": "PATH", "help": "System path"}])
    unmet = invoker.check_requirements(skill)
    assert len(unmet) == 0  # PATH always exists


def test_check_requirements_missing():
    invoker = SkillInvoker()
    skill = _make_skill(required_env_vars=[{"name": "NONEXISTENT_VAR_12345"}])
    unmet = invoker.check_requirements(skill)
    assert len(unmet) == 1
    assert "NONEXISTENT_VAR_12345" in unmet[0]


def test_resolve_handoff_no_match(tmp_path):
    result = resolve_handoff_artifact(str(tmp_path / "*.nonexistent"))
    assert result is None


def test_resolve_handoff_finds_file(tmp_path):
    (tmp_path / "plan_001.md").write_text("plan content")
    result = resolve_handoff_artifact(str(tmp_path / "plan_*.md"))
    assert result is not None
    assert "plan_001.md" in result
