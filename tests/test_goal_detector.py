import pytest
from mullet.core.orchestration.autopilot_goals import GoalDetector

@pytest.fixture
def goal_detector():
    return GoalDetector()

def test_goal_met_true_exact_match(goal_detector):
    task = "Implement the login feature"
    result = "Login feature implemented successfully."
    assert goal_detector.is_goal_met(task, result) is True

def test_goal_met_true_partial_match(goal_detector):
    task = "Create user authentication module"
    result = "User authentication module developed and integrated."
    assert goal_detector.is_goal_met(task, result) is True

def test_goal_met_false_no_match(goal_detector):
    task = "Fix database connection bug"
    result = "Implemented new UI design."
    assert goal_detector.is_goal_met(task, result) is False

def test_goal_met_false_insufficient_keywords(goal_detector):
    task = "Develop a robust API for data retrieval and storage"
    result = "API created."
    assert goal_detector.is_goal_met(task, result) is False

def test_goal_met_handles_casing_and_punctuation(goal_detector):
    task = "  Implement the User Interface!  "
    result = "user interface implemented."
    assert goal_detector.is_goal_met(task, result) is True

def test_goal_met_empty_task(goal_detector):
    task = ""
    result = "Task completed."
    assert goal_detector.is_goal_met(task, result) is False

def test_goal_met_empty_result(goal_detector):
    task = "Complete task"
    result = ""
    assert goal_detector.is_goal_met(task, result) is False

def test_goal_met_irrelevant_words(goal_detector):
    task = "Develop feature X"
    result = "I developed feature X, which was a lot of work."
    assert goal_detector.is_goal_met(task, result) is True

