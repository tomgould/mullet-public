"""
Tests for PRD.from_dict() validation.
"""
import pytest
from mullet.core.orchestration.ralph import PRD


def test_not_a_dict_raises_typeerror():
    """PRD.from_dict("not a dict") raises TypeError with "expects a dict"."""
    with pytest.raises(TypeError, match="expects a dict"):
        PRD.from_dict("not a dict")


def test_not_a_dict_list_raises():
    """PRD.from_dict([]) raises TypeError."""
    with pytest.raises(TypeError, match="expects a dict"):
        PRD.from_dict([])


def test_empty_dict_ok():
    """PRD.from_dict({}) does not raise (returns PRD with no stories)."""
    try:
        prd = PRD.from_dict({})
        assert prd.stories == []
    except Exception as e:
        pytest.fail(f"PRD.from_dict({{}}) raised {type(e).__name__}: {e}")


def test_story_missing_id():
    """PRD.from_dict({"stories":[{"title":"t"}]}) raises KeyError with "UserStory requires"."""
    with pytest.raises(KeyError, match="UserStory requires"):
        PRD.from_dict({"stories": [{"title": "t"}]})


def test_story_missing_title():
    """PRD.from_dict({"stories":[{"id":"s1"}]}) raises KeyError."""
    with pytest.raises(KeyError, match="UserStory requires"):
        PRD.from_dict({"stories": [{"id": "s1"}]})


def test_task_missing_id():
    """PRD.from_dict with task missing id raises KeyError with "StoryTask requires"."""
    with pytest.raises(KeyError, match="StoryTask requires"):
        PRD.from_dict({
            "stories": [{
                "id": "s1",
                "title": "t",
                "tasks": [{"title": "x"}]
            }]
        })


def test_task_missing_title():
    """PRD.from_dict with task missing title raises KeyError."""
    with pytest.raises(KeyError, match="StoryTask requires"):
        PRD.from_dict({
            "stories": [{
                "id": "s1",
                "title": "t",
                "tasks": [{"id": "t1"}]
            }]
        })


def test_valid_prd_roundtrips():
    """A complete dict with stories and tasks produces PRD with correct stories and tasks."""
    data = {
        "task": "Implement feature X",
        "stories": [
            {
                "id": "story-1",
                "title": "User authentication",
                "tasks": [
                    {"id": "task-1", "title": "Create login form"},
                    {"id": "task-2", "title": "Implement backend auth"}
                ]
            }
        ]
    }
    
    try:
        prd = PRD.from_dict(data)
        assert len(prd.stories) == 1
        assert prd.stories[0].id == "story-1"
        assert prd.stories[0].title == "User authentication"
        assert len(prd.stories[0].tasks) == 2
        assert prd.stories[0].tasks[0].id == "task-1"
        assert prd.stories[0].tasks[0].title == "Create login form"
        assert prd.stories[0].tasks[1].id == "task-2"
        assert prd.stories[0].tasks[1].title == "Implement backend auth"
    except Exception as e:
        pytest.fail(f"PRD.from_dict(valid_data) raised {type(e).__name__}: {e}")