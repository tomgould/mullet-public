"""Tests for mullet.core.skills.invoker — SkillInvoker and helpers."""

from pathlib import Path

from mullet.core.skills.types import Skill, SkillMetadata, SkillInvocation
from mullet.core.skills.invoker import SkillInvoker, generate_runtime_guidance


def make_skill(
    body: str = "Do the thing.",
    name: str = "test_skill",
    tier_override: str | None = None,
    required_env_vars: list = None,
    required_tools: list = None,
    platforms: list = None,
    pipeline: list = None,
    next_skill: str | None = None,
    handoff: str | None = None,
) -> Skill:
    meta = SkillMetadata(
        name=name,
        tier_override=tier_override,
        required_env_vars=required_env_vars or [],
        required_tools=required_tools or [],
        platforms=platforms or [],
        pipeline=pipeline or [],
        next_skill=next_skill,
        handoff=handoff,
    )
    return Skill(metadata=meta, body=body, path=Path("/fake/skill.md"))


# ---------------------------------------------------------------------------
# SkillInvoker.check_requirements
# ---------------------------------------------------------------------------

class TestCheckRequirements:
    def setup_method(self):
        self.invoker = SkillInvoker()

    def test_no_requirements_returns_empty(self):
        skill = make_skill()
        assert self.invoker.check_requirements(skill) == []

    def test_missing_env_var_string_form(self, monkeypatch):
        monkeypatch.delenv("MY_TEST_VAR_XYZ", raising=False)
        skill = make_skill(required_env_vars=["MY_TEST_VAR_XYZ"])
        unmet = self.invoker.check_requirements(skill)
        assert len(unmet) == 1
        assert "MY_TEST_VAR_XYZ" in unmet[0]

    def test_present_env_var_not_unmet(self, monkeypatch):
        monkeypatch.setenv("MY_TEST_VAR_XYZ", "value")
        skill = make_skill(required_env_vars=["MY_TEST_VAR_XYZ"])
        assert self.invoker.check_requirements(skill) == []

    def test_missing_env_var_dict_form(self, monkeypatch):
        monkeypatch.delenv("DICT_VAR", raising=False)
        req = {"name": "DICT_VAR", "help": "Get it from the dashboard"}
        skill = make_skill(required_env_vars=[req])
        unmet = self.invoker.check_requirements(skill)
        assert any("DICT_VAR" in u for u in unmet)
        assert any("Get it from the dashboard" in u for u in unmet)

    def test_dict_env_var_without_help(self, monkeypatch):
        monkeypatch.delenv("NO_HELP_VAR", raising=False)
        req = {"name": "NO_HELP_VAR"}
        skill = make_skill(required_env_vars=[req])
        unmet = self.invoker.check_requirements(skill)
        assert len(unmet) == 1
        assert "NO_HELP_VAR" in unmet[0]

    def test_multiple_missing_all_reported(self, monkeypatch):
        monkeypatch.delenv("VAR_A", raising=False)
        monkeypatch.delenv("VAR_B", raising=False)
        skill = make_skill(required_env_vars=["VAR_A", "VAR_B"])
        unmet = self.invoker.check_requirements(skill)
        assert len(unmet) == 2

    def test_required_tools_not_checked(self):
        skill = make_skill(required_tools=["read_file", "terminal"])
        assert self.invoker.check_requirements(skill) == []


# ---------------------------------------------------------------------------
# SkillInvoker.build_pipeline_context
# ---------------------------------------------------------------------------

class TestBuildPipelineContext:
    def setup_method(self):
        self.invoker = SkillInvoker()

    def test_no_pipeline_returns_none(self):
        skill = make_skill()
        assert self.invoker.build_pipeline_context(skill) is None

    def test_next_skill_triggers_context(self):
        skill = make_skill(next_skill="review_skill")
        ctx = self.invoker.build_pipeline_context(skill)
        assert ctx is not None
        assert "review_skill" in ctx

    def test_pipeline_list_appears_in_context(self):
        skill = make_skill(
            name="step_one",
            pipeline=["step_one", "step_two", "step_three"],
            next_skill="step_two",
        )
        ctx = self.invoker.build_pipeline_context(skill)
        assert "step_one" in ctx
        assert "step_two" in ctx

    def test_current_stage_shows_position(self):
        skill = make_skill(
            name="step_two",
            pipeline=["step_one", "step_two", "step_three"],
            next_skill="step_three",
        )
        ctx = self.invoker.build_pipeline_context(skill)
        assert "2/3" in ctx

    def test_handoff_path_included(self):
        skill = make_skill(
            next_skill="next",
            handoff=".mullet/handoff.md",
        )
        ctx = self.invoker.build_pipeline_context(skill)
        assert ".mullet/handoff.md" in ctx

    def test_returns_string_when_pipeline(self):
        skill = make_skill(pipeline=["a", "b"], next_skill="b")
        ctx = self.invoker.build_pipeline_context(skill)
        assert isinstance(ctx, str)


# ---------------------------------------------------------------------------
# SkillInvoker.invoke
# ---------------------------------------------------------------------------

class TestSkillInvokerInvoke:
    def setup_method(self):
        self.invoker = SkillInvoker()

    def test_returns_skill_invocation(self):
        skill = make_skill()
        result = self.invoker.invoke(skill)
        assert isinstance(result, SkillInvocation)

    def test_body_in_content(self):
        skill = make_skill(body="Analyze this repository.")
        result = self.invoker.invoke(skill)
        assert "Analyze this repository." in result.content

    def test_user_args_in_content(self):
        skill = make_skill()
        result = self.invoker.invoke(skill, user_args="--verbose mode")
        assert "--verbose mode" in result.content

    def test_no_user_args_no_args_section(self):
        skill = make_skill()
        result = self.invoker.invoke(skill, user_args="")
        assert "User Arguments" not in result.content

    def test_unmet_requirements_in_content(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR_XYZ", raising=False)
        skill = make_skill(required_env_vars=["MISSING_VAR_XYZ"])
        result = self.invoker.invoke(skill)
        assert "MISSING_VAR_XYZ" in result.content

    def test_tier_override_propagated(self):
        skill = make_skill(tier_override="HIGH")
        result = self.invoker.invoke(skill)
        assert result.tier_override == "HIGH"

    def test_next_skill_propagated(self):
        skill = make_skill(next_skill="deploy_skill")
        result = self.invoker.invoke(skill)
        assert result.next_skill == "deploy_skill"

    def test_mode_propagated(self):
        skill = make_skill()
        skill.metadata.mode = "plan"
        result = self.invoker.invoke(skill)
        assert result.mode == "plan"


# ---------------------------------------------------------------------------
# generate_runtime_guidance
# ---------------------------------------------------------------------------

class TestGenerateRuntimeGuidance:
    def test_no_metadata_returns_empty(self):
        skill = make_skill()
        result = generate_runtime_guidance(skill)
        assert result == ""

    def test_tier_override_in_guidance(self):
        skill = make_skill(tier_override="HIGH")
        result = generate_runtime_guidance(skill)
        assert "HIGH" in result

    def test_required_tools_in_guidance(self):
        skill = make_skill(required_tools=["read_file", "write_file"])
        result = generate_runtime_guidance(skill)
        assert "read_file" in result
        assert "write_file" in result

    def test_platforms_in_guidance(self):
        skill = make_skill(platforms=["linux", "darwin"])
        result = generate_runtime_guidance(skill)
        assert "linux" in result

    def test_env_vars_in_guidance_string_form(self, monkeypatch):
        skill = make_skill(required_env_vars=["MY_ENV_TOKEN"])
        result = generate_runtime_guidance(skill)
        assert "MY_ENV_TOKEN" in result

    def test_env_vars_in_guidance_dict_form(self, monkeypatch):
        skill = make_skill(required_env_vars=[{"name": "DICT_TOKEN", "help": "..."}])
        result = generate_runtime_guidance(skill)
        assert "DICT_TOKEN" in result

    def test_header_only_returns_empty_string(self):
        skill = make_skill()
        result = generate_runtime_guidance(skill)
        assert result == ""
