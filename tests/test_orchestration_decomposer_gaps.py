"""Targeted tests for uncovered lines in mullet/core/orchestration/decomposer.py."""

from unittest.mock import MagicMock, patch

import pytest

from mullet.core.orchestration.decomposer import (
    _detect_dependencies,
    _estimate_complexity,
    _generate_criteria,
    generate_subtasks,
    TaskAnalysis,
)
from mullet.core.orchestration.llm_decomposer import LLMDecomposer, DecompositionError
from mullet.core.orchestration.swarm_types import SwarmContext


# Line 192: testing task type overrides tier to MEDIUM even for auth/database
class TestGenerateSubtasksTestingTier:
    def test_testing_task_type_sets_medium_tier_for_auth(self):
        analysis = TaskAnalysis(
            task="write auth tests",
            areas=["auth"],
            task_type="testing",
        )
        subtasks = generate_subtasks(analysis, "write auth tests")
        assert subtasks[0].model_tier == "MEDIUM"

    def test_testing_task_type_sets_medium_tier_for_database(self):
        analysis = TaskAnalysis(
            task="write database tests",
            areas=["database"],
            task_type="testing",
        )
        subtasks = generate_subtasks(analysis, "write database tests")
        assert subtasks[0].model_tier == "MEDIUM"


# Line 303: _estimate_complexity adds 0.2 when word count > 50
class TestEstimateComplexityLongText:
    def test_complexity_increases_with_long_text(self):
        text = " ".join(["word"] * 60)
        score = _estimate_complexity(text, [])
        # Base score for >50 words is 0.2, clamped to max 1.0
        assert score >= 0.2


# Line 329: database -> backend dependency
class TestDetectDependenciesDatabaseBackend:
    def test_database_before_backend(self):
        deps = _detect_dependencies(["database", "backend"], "feature")
        assert ("database", "backend") in deps


# Lines 341-343: config before everything else
class TestDetectDependenciesConfig:
    def test_config_before_all_other_areas(self):
        deps = _detect_dependencies(["config", "frontend", "backend"], "feature")
        assert ("config", "frontend") in deps
        assert ("config", "backend") in deps

    def test_config_not_dependent_on_itself(self):
        deps = _detect_dependencies(["config"], "feature")
        assert ("config", "config") not in deps


# Lines 374-375: testing task type acceptance criteria
class TestGenerateCriteriaTesting:
    def test_testing_criteria(self):
        criteria = _generate_criteria("frontend", "testing")
        assert "All new tests pass" in criteria
        assert "No existing tests broken" in criteria


# Lines 377-378: bugfix task type acceptance criteria
class TestGenerateCriteriaBugfix:
    def test_bugfix_criteria(self):
        criteria = _generate_criteria("frontend", "bugfix")
        assert "Bug is fixed and no regression introduced" in criteria
        assert "Root cause is addressed, not just symptoms" in criteria


# Lines 380-381: refactor task type acceptance criteria
class TestGenerateCriteriaRefactor:
    def test_refactor_criteria(self):
        criteria = _generate_criteria("frontend", "refactor")
        assert "Existing behavior is preserved" in criteria
        assert "Code is cleaner/simpler" in criteria


# Line 390: database area acceptance criteria
class TestGenerateCriteriaDatabaseArea:
    def test_database_area_criteria(self):
        criteria = _generate_criteria("database", "feature")
        assert "Migration is reversible" in criteria


# --- llm_decomposer.py gaps ---

class TestLLMDecomposerGaps:
    def test_decompose_response_not_json_array(self):
        """Cover line 106: valid JSON but not a list raises DecompositionError."""
        provider = MagicMock()
        provider.create.return_value = MagicMock(content='{"result": "not-a-list"}')
        decomposer = LLMDecomposer(provider=provider)
        context = SwarmContext()
        with patch(
            "mullet.core.orchestration.llm_decomposer._extract_json_array",
            return_value='{"result": "not-a-list"}',
        ):
            with pytest.raises(DecompositionError, match="Response is not a JSON array"):
                decomposer.decompose("test goal", context)

    def test_decompose_context_files(self):
        """Cover line 155: context_section when context.files is set."""
        provider = MagicMock()
        provider.create.return_value = MagicMock(content="[]")
        decomposer = LLMDecomposer(provider=provider)
        context = SwarmContext(files=["a.py"])
        decomposer.decompose("test goal", context)
        prompt = provider.create.call_args[1]["messages"][0]["content"]
        assert "## Files available" in prompt
        assert "- a.py" in prompt

    def test_decompose_context_items(self):
        """Cover line 157: context_section when context.items is set."""
        provider = MagicMock()
        provider.create.return_value = MagicMock(content="[]")
        decomposer = LLMDecomposer(provider=provider)
        context = SwarmContext(items=["item1"])
        decomposer.decompose("test goal", context)
        prompt = provider.create.call_args[1]["messages"][0]["content"]
        assert "## Items available" in prompt
        assert "- item1" in prompt
