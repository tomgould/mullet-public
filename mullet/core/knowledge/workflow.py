"""Manages workflow execution and task coordination for agents."""

from dataclasses import dataclass, field
from typing import List, Callable, Any
import time

@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    name: str
    task: Callable[..., Any]
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)

@dataclass
class Workflow:
    """A named sequence of workflow steps."""
    name: str
    steps: List[WorkflowStep]

class WorkflowRunner:
    """Execute a workflow with retry logic."""

    def __init__(self, workflow: Workflow, max_retries: int = 3, retry_delay_seconds: int = 1):
        self.workflow = workflow
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

    def run(self) -> bool:
        """Run all steps in the workflow."""
        print(f"Starting workflow '{self.workflow.name}'")
        for i, step in enumerate(self.workflow.steps):
            print(f"  Step {i+1}/{len(self.workflow.steps)}: {step.name}")
            retries = 0
            while retries <= self.max_retries:
                try:
                    step.task(*step.args, **step.kwargs)
                    print(f"    Step '{step.name}' completed successfully.")
                    break
                except Exception as e:
                    retries += 1
                    print(f"    Step '{step.name}' failed (attempt {retries}/{self.max_retries + 1}): {e}")
                    if retries <= self.max_retries:
                        time.sleep(self.retry_delay_seconds)
                    else:
                        print(f"Workflow '{self.workflow.name}' failed at step '{step.name}' after {self.max_retries + 1} attempts.")
                        return False
        print(f"Workflow '{self.workflow.name}' completed.")
        return True

