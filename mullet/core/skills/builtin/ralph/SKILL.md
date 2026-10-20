---
name: ralph
description: Persistence loop with PRD verification until done
triggers:
  - ralph
  - don't stop
  - until done
  - keep going
mode: ralph
tags:
  - orchestration
---

# Ralph Mode

Persistence loop: keep working until ALL acceptance criteria pass.

<Steps>
1. Generate or read PRD with user stories and acceptance criteria
2. Pick the highest-priority incomplete story
3. Implement the story
4. Verify EACH acceptance criterion with fresh evidence
5. Mark story as complete if all criteria pass
6. If incomplete stories remain, loop to step 2
7. Final reviewer verification when all stories pass
</Steps>
