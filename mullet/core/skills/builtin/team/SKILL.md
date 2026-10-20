---
name: team
description: N coordinated agents with staged pipeline
triggers:
  - team
  - coordinated
mode: team
tags:
  - orchestration
---

# Team Mode

Staged pipeline with N parallel workers.

<Steps>
1. **PLAN**: Explore codebase, create task graph with dependencies
2. **PRD**: Refine acceptance criteria for each task
3. **EXEC**: Spawn N workers, each claims tasks from the pool
4. **VERIFY**: Check all outputs against acceptance criteria
5. **FIX**: Fix defects, loop back to verify (max 3 cycles)
</Steps>
