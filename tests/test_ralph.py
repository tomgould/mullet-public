"""Tests for ralph mode — PRD generation, story tracking, persistence loop."""

import json
import pytest
from mullet.core.orchestration.ralph import PRD, UserStory, StoryTask, RalphMode
from mullet.core.agent.types import TokenUsage
from mullet.core.providers.base import LLMResponse, Provider
from mullet.core.agent.runner import AgentRunner
from mullet.core.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def clean_state(tmp_path, monkeypatch):
    """Run each test in a temp directory to avoid state leakage."""
    monkeypatch.chdir(tmp_path)


# --- PRD data structure tests ---

def test_prd_all_pass():
    prd = PRD(task="test", stories=[
        UserStory(id="US-001", title="A", passes=True),
        UserStory(id="US-002", title="B", passes=True),
    ])
    assert prd.all_pass is True


def test_prd_not_all_pass():
    prd = PRD(task="test", stories=[
        UserStory(id="US-001", title="A", passes=True),
        UserStory(id="US-002", title="B", passes=False),
    ])
    assert prd.all_pass is False


def test_prd_next_story():
    prd = PRD(task="test", stories=[
        UserStory(id="US-001", title="A", passes=True, priority=1),
        UserStory(id="US-002", title="B", passes=False, priority=2),
        UserStory(id="US-003", title="C", passes=False, priority=3),
    ])
    assert prd.next_story.id == "US-002"


def test_prd_next_story_all_done():
    prd = PRD(task="test", stories=[
        UserStory(id="US-001", title="A", passes=True),
    ])
    assert prd.next_story is None


def test_prd_progress():
    prd = PRD(task="test", stories=[
        UserStory(id="US-001", title="A", passes=True),
        UserStory(id="US-002", title="B", passes=False),
        UserStory(id="US-003", title="C", passes=True),
    ])
    assert prd.progress == (2, 3)


def test_prd_serialization():
    prd = PRD(task="test", stories=[
        UserStory(id="US-001", title="A", acceptance_criteria=["works"], passes=True),
    ])
    data = prd.to_dict()
    restored = PRD.from_dict(data)
    assert restored.task == "test"
    assert restored.stories[0].id == "US-001"
    assert restored.stories[0].passes is True


# --- Ralph execution tests ---

class SequentialProvider(Provider):
    """Provider that returns different responses in sequence."""
    def __init__(self, responses: list[str]):
        self.name = "test"
        self._responses = responses
        self._idx = 0

    def create(self, messages, tools=None, **kwargs) -> LLMResponse:
        resp = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return LLMResponse(content=resp, usage=TokenUsage(total_tokens=50))


def _make_agent(responses: list[str]) -> AgentRunner:
    registry = ToolRegistry()
    return AgentRunner(
        provider=SequentialProvider(responses),
        tool_registry=registry,
        max_iterations=20,
    )


def test_ralph_generates_prd():
    prd_json = json.dumps({
        "stories": [
            {"id": "US-001", "title": "Add function", "description": "Add a hello function",
             "acceptance_criteria": ["function exists", "returns greeting"], "priority": 1},
        ]
    })

    agent = _make_agent([
        prd_json,                    # PRD generation
        "Function implemented.",     # Implementation
        "ALL CRITERIA PASS",         # Verification
        "APPROVAL — all stories verified.",  # Architect review
    ])

    ralph = RalphMode(max_iterations=5)
    result = ralph.execute("add a hello function", agent)

    assert result.completed is True
    assert result.stories_completed == 1
    assert result.stories_total == 1


def test_ralph_retries_on_failure():
    prd_json = json.dumps({
        "stories": [
            {"id": "US-001", "title": "Fix bug", "acceptance_criteria": ["test passes"], "priority": 1},
        ]
    })

    # verify_criteria (max_retries=2) calls agent.run() up to 2 times per story attempt
    agent = _make_agent([
        prd_json,                    # PRD generation
        "Attempted fix.",            # Implementation (iteration 1)
        "FAIL: test still broken",   # Verification attempt 1/2 (fail)
        "FAIL: still broken",        # Verification attempt 2/2 (fail) — verify_criteria retries
        "Fixed properly.",           # Implementation (iteration 2)
        "ALL CRITERIA PASS",         # Verification attempt 1/2 (pass)
        "APPROVAL — all verified.",  # Architect review
    ])

    ralph = RalphMode(max_iterations=5)
    result = ralph.execute("fix the bug", agent)

    assert result.completed is True
    assert result.iterations_used == 2


def test_ralph_hits_max_iterations():
    prd_json = json.dumps({
        "stories": [
            {"id": "US-001", "title": "Impossible", "acceptance_criteria": ["never passes"], "priority": 1},
        ]
    })

    # Always fails verification
    responses = [prd_json] + ["Trying..."] * 20 + ["FAIL"] * 20
    agent = _make_agent(responses)

    ralph = RalphMode(max_iterations=3)
    result = ralph.execute("impossible task", agent)

    assert result.completed is False
    assert result.iterations_used == 3


def test_ralph_fallback_on_bad_prd():
    """If PRD generation fails, ralph falls back to single-agent execution."""
    agent = _make_agent([
        "I can't generate a PRD for this.",  # Bad PRD response
    ])

    ralph = RalphMode(max_iterations=3)
    result = ralph.execute("do something simple", agent)

    # Should fall back to single-agent mode
    assert result.summary == "I can't generate a PRD for this."


# --- PRD JSON extraction tests ---

def test_generate_prd_with_markdown_fences_and_prose():
    """Markdown fences and surrounding prose should not break JSON extraction."""
    prd_data = {
        "stories": [
            {"id": "US-001", "title": "Fix bug", "acceptance_criteria": ["test passes"], "priority": 1},
        ]
    }
    response = (
        "Here is the PRD you requested:\n\n"
        "```json\n"
        f"{json.dumps(prd_data)}\n"
        "```\n\n"
        "Let me know if you need anything else!"
    )

    agent = _make_agent([response])
    ralph = RalphMode(max_iterations=3)
    prd = ralph._generate_prd("fix the bug", agent)

    assert len(prd.stories) == 1
    assert prd.stories[0].id == "US-001"


def test_generate_prd_with_trailing_prose():
    """Trailing prose after the JSON object should not break extraction."""
    prd_data = {
        "stories": [
            {"id": "US-002", "title": "Add feature", "acceptance_criteria": ["works"], "priority": 1},
        ]
    }
    response = (
        f"{json.dumps(prd_data)}\n\n"
        "Note: make sure to also update the docs."
    )

    agent = _make_agent([response])
    ralph = RalphMode(max_iterations=3)
    prd = ralph._generate_prd("add feature", agent)

    assert len(prd.stories) == 1
    assert prd.stories[0].id == "US-002"


def test_generate_prd_retry_then_fallback():
    """A genuinely invalid response triggers retry, then falls back cleanly."""
    agent = _make_agent([
        "This is not JSON.",            # First attempt — invalid
        "Still not JSON.",              # Retry attempt — also invalid
        "Fallback result.",             # Fallback single-agent execution
    ])

    ralph = RalphMode(max_iterations=3)
    result = ralph.execute("do something", agent)

    assert result.completed is True
    assert result.summary == "Fallback result."


# --- StoryTask tracking tests ---

def test_story_task_dataclass():
    task = StoryTask(id="T-001", title="Write function", description="Create hello()")
    assert task.completed is False
    assert task.result == ""


def test_story_with_tasks():
    story = UserStory(
        id="US-001", title="Auth module",
        tasks=[
            StoryTask(id="T-001", title="Create auth.py"),
            StoryTask(id="T-002", title="Add login endpoint"),
        ],
    )
    assert len(story.tasks) == 2
    assert not story.tasks[0].completed


def test_prd_serialization_with_tasks():
    """PRD should serialize and deserialize nested tasks."""
    prd = PRD(task="build auth", stories=[
        UserStory(
            id="US-001", title="Auth",
            acceptance_criteria=["tests pass"],
            tasks=[
                StoryTask(id="T-001", title="Create module", description="auth.py"),
                StoryTask(id="T-002", title="Add tests", completed=True, result="3 tests"),
            ],
        ),
    ])
    data = prd.to_dict()
    assert len(data["stories"][0]["tasks"]) == 2
    assert data["stories"][0]["tasks"][1]["completed"] is True

    restored = PRD.from_dict(data)
    assert len(restored.stories[0].tasks) == 2
    assert restored.stories[0].tasks[0].title == "Create module"
    assert restored.stories[0].tasks[1].completed is True
    assert restored.stories[0].tasks[1].result == "3 tests"


def test_ralph_with_nested_tasks():
    """PRD with tasks should execute and mark tasks completed."""
    prd_json = json.dumps({
        "stories": [
            {
                "id": "US-001", "title": "Build feature",
                "description": "Build the feature",
                "acceptance_criteria": ["feature works"],
                "priority": 1,
                "tasks": [
                    {"id": "T-001", "title": "Create file", "description": "Create main.py"},
                    {"id": "T-002", "title": "Add tests", "description": "Write test_main.py"},
                ],
            },
        ]
    })

    agent = _make_agent([
        prd_json,                    # PRD generation (with tasks)
        "Feature implemented.",      # Implementation
        "ALL CRITERIA PASS",         # Verification
        "APPROVAL — looks good.",    # Architect review
    ])

    ralph = RalphMode(max_iterations=5)
    result = ralph.execute("build a feature", agent)

    assert result.completed is True
    assert result.stories_completed == 1
