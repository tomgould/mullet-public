"""Tests for batch runner and learned skills."""
from unittest.mock import MagicMock

from mullet.core.orchestration.batch import run_batch, BatchResult
from mullet.core.agent.types import AgentResult, TokenUsage
from mullet.core.skills.learner import extract_skill, save_skill, find_similar_skill, LearnedSkill


# === Batch Runner Tests ===

def test_batch_result_defaults():
    r = BatchResult()
    assert r.completed == 0
    assert r.failed == 0
    assert r.total == 0


def test_run_batch_completes():
    agent = MagicMock()
    agent.run.return_value = AgentResult(
        completed=True, final_response="done",
        tokens_used=TokenUsage(total_tokens=10),
    )
    result = run_batch(["task1", "task2"], agent, max_parallel=2)
    assert result.completed == 2
    assert result.total == 2
    assert result.failed == 0


def test_run_batch_handles_errors():
    agent = MagicMock()
    agent.run.side_effect = [
        AgentResult(completed=True, final_response="ok", tokens_used=TokenUsage(total_tokens=10)),
        Exception("boom"),
    ]
    result = run_batch(["task1", "task2"], agent, max_parallel=1)
    assert result.completed == 1
    assert result.failed == 1
    assert "boom" in result.errors[0]


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
