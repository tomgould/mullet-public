"""Task planner — decompose complex tasks into structured plans.

Used by autopilot for the planning phase and by team mode
for creating task graphs with dependencies.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mullet.core.agent.runner import AgentRunner

logger = logging.getLogger(__name__)


@dataclass
class PlanStep:
    """A single step in a plan."""
    id: str
    title: str
    description: str = ""
    depends_on: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    assigned_to: str | None = None  # For team mode


@dataclass
class Plan:
    """A structured plan with ordered steps."""
    task: str
    overview: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert the plan to a dictionary representation."""
        return {
            "task": self.task,
            "overview": self.overview,
            "steps": [
                {
                    "id": s.id,
                    "title": s.title,
                    "description": s.description,
                    "depends_on": s.depends_on,
                    "files": s.files,
                    "acceptance_criteria": s.acceptance_criteria,
                }
                for s in self.steps
            ],
            "risks": self.risks,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Plan":
        """Create a Plan instance from a dictionary."""
        steps = [
            PlanStep(
                id=s["id"],
                title=s["title"],
                description=s.get("description", ""),
                depends_on=s.get("depends_on", []),
                files=s.get("files", []),
                acceptance_criteria=s.get("acceptance_criteria", []),
            )
            for s in data.get("steps", [])
        ]
        return cls(
            task=data.get("task", ""),
            overview=data.get("overview", ""),
            steps=steps,
            risks=data.get("risks", []),
        )

    @property
    def execution_order(self) -> list[PlanStep]:
        """Return steps in dependency-respecting order (topological sort)."""
        completed: set[str] = set()
        ordered: list[PlanStep] = []
        remaining = list(self.steps)

        max_passes = len(remaining) + 1
        for _ in range(max_passes):
            if not remaining:
                break
            next_round = []
            for step in remaining:
                if all(dep in completed for dep in step.depends_on):
                    ordered.append(step)
                    completed.add(step.id)
                else:
                    next_round.append(step)
            remaining = next_round

        # Add any remaining (circular deps) at the end
        ordered.extend(remaining)
        return ordered


def create_plan(task: str, agent: "AgentRunner") -> Plan:
    """Use an agent to create a structured plan for a task."""
    prompt = (
        "Create a detailed implementation plan for this task. "
        "Return ONLY valid JSON with this structure:\n"
        '{\n'
        '  "overview": "Brief architecture description",\n'
        '  "steps": [\n'
        '    {"id": "S-001", "title": "...", "description": "...", '
        '"depends_on": [], "files": ["path/to/file.py"], '
        '"acceptance_criteria": ["criterion 1"]}\n'
        '  ],\n'
        '  "risks": ["risk 1", "risk 2"]\n'
        '}\n\n'
        f"Task: {task}"
    )

    result = agent.run(prompt)
    if not result.completed:
        return Plan(task=task)

    try:
        text = result.final_response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            plan = Plan.from_dict(data)
            plan.task = task
            return plan
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to parse plan: %s", e)

    return Plan(task=task, overview=result.final_response)


def consensus_plan(task: str, agent: "AgentRunner", max_rounds: int = 3) -> Plan:
    """RALPLAN-DR: planner proposes, architect reviews, critic challenges."""
    plan_text = ""
    for _ in range(max_rounds):
        plan = agent.run(f"As a planner, create a step-by-step plan for: {task}")
        plan_text = plan.final_response
        review = agent.run(f"As an architect, review this plan and say APPROVE or list issues:\n{plan_text}")
        critique = agent.run(f"As a critic, find weaknesses. Say NO ISSUES if none:\n{plan_text}")
        if "approve" in review.final_response.lower() and "no issues" in critique.final_response.lower():
            return Plan(task=task, overview=plan_text)
    return Plan(task=task, overview=plan_text)
