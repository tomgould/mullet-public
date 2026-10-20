from unittest.mock import Mock
from mullet.core.knowledge.workflow import Workflow, WorkflowStep, WorkflowRunner

def test_workflow_runs_successfully():
    mock_task1 = Mock(return_value=True)
    mock_task2 = Mock(return_value=True)
    workflow = Workflow(
        name="Test Workflow",
        steps=[
            WorkflowStep(name="Step 1", task=mock_task1),
            WorkflowStep(name="Step 2", task=mock_task2),
        ],
    )
    runner = WorkflowRunner(workflow)
    assert runner.run() is True
    mock_task1.assert_called_once()
    mock_task2.assert_called_once()

def test_workflow_retries_on_failure():
    mock_task = Mock(side_effect=[Exception("Failed"), True])
    workflow = Workflow(name="Retry Workflow", steps=[WorkflowStep(name="Step 1", task=mock_task)])
    runner = WorkflowRunner(workflow, max_retries=1, retry_delay_seconds=0)
    assert runner.run() is True
    assert mock_task.call_count == 2

def test_workflow_fails_after_max_retries():
    mock_task = Mock(side_effect=Exception("Failed"))
    workflow = Workflow(name="Fail Workflow", steps=[WorkflowStep(name="Step 1", task=mock_task)])
    runner = WorkflowRunner(workflow, max_retries=2, retry_delay_seconds=0)
    assert runner.run() is False
    assert mock_task.call_count == 3

def test_workflow_with_args_kwargs():
    mock_task = Mock()
    workflow = Workflow(
        name="Args Kwargs Workflow",
        steps=[
            WorkflowStep(name="Step 1", task=mock_task, args=("arg1",), kwargs={"kwarg1": "value1"}),
        ],
    )
    runner = WorkflowRunner(workflow)
    runner.run()
    mock_task.assert_called_once_with("arg1", kwarg1="value1")

def test_empty_workflow():
    workflow = Workflow(name="Empty Workflow", steps=[])
    runner = WorkflowRunner(workflow)
    assert runner.run() is True

def test_workflow_partial_success_then_failure():
    mock_task1 = Mock(return_value=True)
    mock_task2 = Mock(side_effect=Exception("Fail"))
    workflow = Workflow(
        name="Partial Fail Workflow",
        steps=[
            WorkflowStep(name="Step 1", task=mock_task1),
            WorkflowStep(name="Step 2", task=mock_task2),
        ],
    )
    runner = WorkflowRunner(workflow, max_retries=0)
    assert runner.run() is False
    mock_task1.assert_called_once()
    mock_task2.assert_called_once()

def test_workflow_step_name_logging(capsys):
    mock_task = Mock(return_value=True)
    workflow = Workflow(name="Logging Workflow", steps=[WorkflowStep(name="Log Step", task=mock_task)])
    runner = WorkflowRunner(workflow)
    runner.run()
    captured = capsys.readouterr()
    assert "Step 1/1: Log Step" in captured.out

