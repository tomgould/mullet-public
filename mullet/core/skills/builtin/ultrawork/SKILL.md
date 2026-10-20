---
name: ultrawork
description: Parallel agent execution for independent subtasks
triggers:
  - ultrawork
  - ulw
  - parallel
mode: ultrawork
tags:
  - orchestration
---

# Ultrawork Mode

Decompose → spawn parallel agents → merge results.

<Steps>
1. Decompose the task into independent subtasks
2. Route each subtask via the Router (children often get LOCAL tier)
3. Spawn via delegate_batch — up to 3 concurrent
4. Collect and merge results
</Steps>
