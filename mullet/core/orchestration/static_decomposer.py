"""StaticDecomposer — produce SwarmTasks from explicit file/item lists.

No LLM call. Used when the swarm task has an explicit enumeration:
  - files list from --files flag
  - items list from --spec YAML
  - glob patterns (already expanded by the caller before passing in context.files)
"""
import logging
from typing import List

from mullet.core.orchestration.swarm_types import SwarmContext, SwarmTask

logger = logging.getLogger(__name__)


class StaticDecomposer:
    """Static decomposer that creates tasks from explicit file/item lists.
    
    This decomposer does not make any LLM calls. It simply converts
    the provided files or items into individual SwarmTask objects.
    """
    
    def decompose(self, goal: str, context: SwarmContext) -> list[SwarmTask]:
        """Decompose the context into individual SwarmTasks.
        
        Args:
            goal: The overall goal/prompt for the swarm run
            context: The SwarmContext containing files/items to process
            
        Returns:
            List of SwarmTask objects to be executed
        """
        # Determine targets - files take priority over items
        targets: List[str] = []
        if context.files:
            targets = context.files
        elif context.items:
            targets = context.items
        else:
            return []
            
        # Deduplicate while preserving order
        seen = set()
        deduplicated_targets = []
        for target in targets:
            if target not in seen:
                seen.add(target)
                deduplicated_targets.append(target)
        
        # Log deduplication info
        original_count = len(targets)
        deduplicated_count = len(deduplicated_targets)
        if deduplicated_count < original_count:
            logger.warning(
                f"StaticDecomposer: deduplicated {original_count} targets to "
                f"{deduplicated_count} unique targets"
            )
        
        # Create tasks
        tasks = []
        for i, target in enumerate(deduplicated_targets):
            task = SwarmTask(
                id=f"task-{i:03d}",
                instruction=f"{goal}\n\nTarget: {target}",
                context_refs=[target] if context.files else [],
                expected_output_shape=context.output_shape,
                apply=context.apply,
                target_dir=context.target_dir,
            )
            tasks.append(task)
        
        return tasks