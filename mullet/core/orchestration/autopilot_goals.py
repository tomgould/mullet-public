"""Detects if task goals are met using keyword analysis."""

import re

class GoalDetector:
    """Detect whether a task goal has been met."""
    def is_goal_met(self, task: str, result: str) -> bool:
        """
        Determines if a task's goal has been met by a given result using keyword analysis.
        """
        task_keywords = self._extract_keywords(task)
        result_keywords = self._extract_keywords(result)

        if not task_keywords:
            return False

        # Check if a significant portion of task keywords are present in the result
        matched_keywords = [kw for kw in task_keywords if kw in result_keywords]
        
        # Consider the goal met if at least 50% of the task keywords are found in the result
        # and there's at least one matched keyword to avoid division by zero if task_keywords is empty
        return len(matched_keywords) >= len(task_keywords) * 0.5 and len(matched_keywords) > 0

    def _extract_keywords(self, text: str) -> set[str]:
        """
        Extracts keywords from a given text, converting them to lowercase for case-insensitive matching.
        """
        # Simple tokenization: split by non-alphanumeric characters and filter out short words
        words = re.findall(r'\b\w+\b', text.lower())
        return {word for word in words if len(word) > 2}
