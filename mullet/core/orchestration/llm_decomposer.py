"""LLM-based task decomposition for the Mullet orchestration system."""

from __future__ import annotations

import json
import logging
from typing import List, TYPE_CHECKING

from .swarm_types import SwarmTask, SwarmContext

if TYPE_CHECKING:
    from mullet.core.providers.kimi import KimiProvider

logger = logging.getLogger(__name__)


class DecompositionError(Exception):
    """Raised when LLM decomposition returns unusable output."""


DECOMPOSE_PROMPT = """You are a task decomposition assistant. Your job is to break down a high-level goal into independent, self-contained sub-tasks that can be executed in parallel.

Given the goal: "{goal}"

Please return a JSON array of task objects. Each task object must have these exact keys:
- "id": a short unique string identifier for the task
- "instruction": a clear, self-contained instruction that can be executed independently (do not reference other sub-tasks)
- "expected_output_shape": what format the output should take (optional, defaults to "markdown report")
- "context_refs": array of file paths from the available files that this task needs to access (optional, defaults to [])

{context_section}

Important rules:
1. If the goal cannot be split into independent sub-tasks, return an empty array: []
2. Each instruction must be complete and not depend on other sub-tasks
3. Maximum {max} sub-tasks
4. Return ONLY the JSON array with no additional text or formatting

Example response:
[{{"id": "analyze", "instruction": "Analyze the provided data for patterns", "expected_output_shape": "markdown report", "context_refs": ["data.json"]}}]"""


def _extract_json_array(raw: str) -> str:
    """Extract the outermost JSON array using bracket counting.

    Regex fails when Kimi appends notes containing brackets after the array.
    """
    start = raw.find('[')
    if start == -1:
        raise DecompositionError(f"No JSON array found in response: {raw[:200]}")

    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(raw[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                return raw[start:i + 1]

    raise DecompositionError(f"Unclosed JSON array in response: {raw[:200]}")


def _parse_task_list(raw: str, output_shape: str) -> List[SwarmTask]:
    """Parse LLM response into list of SwarmTask objects.

    Args:
        raw: Raw string response from LLM
        output_shape: Default output shape to use if not specified in task

    Returns:
        List of SwarmTask objects

    Raises:
        DecompositionError: If response cannot be parsed or validated
    """
    # Strip markdown code fences Kimi sometimes wraps responses in
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    json_str = _extract_json_array(cleaned)

    try:
        task_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise DecompositionError(
            f"json.loads failed ({e}) on extracted ({len(json_str)} chars): {json_str[:300]!r}"
        ) from e
    
    if not isinstance(task_data, list):
        raise DecompositionError(f"Response is not a JSON array: {raw}")
    
    tasks = []
    for item in task_data:
        if not isinstance(item, dict):
            raise DecompositionError(f"Task item is not an object: {item}")
        
        if "id" not in item or "instruction" not in item:
            raise DecompositionError(f"Task missing required keys: {item}")
        
        task = SwarmTask(
            id=item["id"],
            instruction=item["instruction"],
            expected_output_shape=item.get("expected_output_shape", output_shape),
            context_refs=item.get("context_refs", [])
        )
        tasks.append(task)
    
    return tasks


class LLMDecomposer:
    """Decomposes high-level goals into sub-tasks using LLM."""
    
    def __init__(self, provider: KimiProvider, max_subtasks: int = 20):
        """Initialize the decomposer.
        
        Args:
            provider: KimiProvider instance for API calls
            max_subtasks: Maximum number of sub-tasks to generate
        """
        self._provider = provider
        self.max_subtasks = max_subtasks
    
    def decompose(self, goal: str, context: SwarmContext) -> List[SwarmTask]:
        """Decompose a goal into sub-tasks using LLM.
        
        Args:
            goal: High-level goal to decompose
            context: SwarmContext containing configuration
            
        Returns:
            List of SwarmTask objects, empty list if not swarmable
            
        Raises:
            DecompositionError: If decomposition fails and caller should handle fallback
        """
        context_section = ""
        if context.files:
            context_section = "## Files available\nThe following files are available as context. Assign relevant file paths to each task's context_refs field.\n" + "\n".join(f"- {f}" for f in context.files)
        elif context.items:
            context_section = "## Items available\nThe following items are available as context. Assign relevant item identifiers to each task's context_refs field.\n" + "\n".join(f"- {item}" for item in context.items)
        
        prompt = DECOMPOSE_PROMPT.format(
            goal=goal, 
            max=self.max_subtasks,
            context_section=context_section
        )
        
        try:
            response = self._provider.create(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048
            )
            
            tasks = _parse_task_list(response.content, context.output_shape)
            
            if not tasks:
                logger.warning("LLMDecomposer: Kimi returned empty decomposition — not swarmable")
                return []
            
            if len(tasks) > self.max_subtasks:
                logger.warning(f"LLMDecomposer: Kimi returned {len(tasks)} tasks, capping at {self.max_subtasks}")
                tasks = tasks[:self.max_subtasks]

            for task in tasks:
                task.apply = context.apply
                task.target_dir = context.target_dir

            return tasks
            
        except DecompositionError:
            # Re-raise for caller to handle fallback
            raise