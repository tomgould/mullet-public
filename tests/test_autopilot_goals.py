import pytest
from mullet.core.orchestration.autopilot_goals import GoalDetector

@pytest.fixture
def goal_detector():
    return GoalDetector()

def test_goal_met_exact_match(goal_detector):
    task = "Create a file named example.txt"
    result = "Successfully created the file example.txt"
    assert goal_detector.is_goal_met(task, result) is True

def test_goal_not_met_no_keywords(goal_detector):
    task = "Create a file named example.txt"
    result = "Operation failed"
    assert goal_detector.is_goal_met(task, result) is False

def test_goal_not_met_insufficient_keywords(goal_detector):
    task = "Implement user authentication and authorization"
    result = "User authentication implemented"
    assert goal_detector.is_goal_met(task, result) is False

def test_goal_met_case_insensitive(goal_detector):
    task = "create a REPORT"
    result = "Report created successfully"
    assert goal_detector.is_goal_met(task, result) is True

def test_goal_met_different_phrasing(goal_detector):
    task = "Set up database connection"
    result = "Established connection to the database"
    assert goal_detector.is_goal_met(task, result) is True

def test_empty_task_keywords(goal_detector):
    task = "a b c"
    result = "a b c d e f g"
    assert goal_detector.is_goal_met(task, result) is False

def test_empty_result_keywords(goal_detector):
    task = "Create file"
    result = ""
    assert goal_detector.is_goal_met(task, result) is False
