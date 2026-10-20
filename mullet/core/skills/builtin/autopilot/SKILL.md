---
name: autopilot
description: Full autonomous execution from idea to working code
triggers:
  - autopilot
  - full auto
mode: autopilot
tier_override: HIGH
tags:
  - orchestration
---

# Autopilot Mode

Full lifecycle autonomous execution: plan, execute, verify.

<Steps>
1. **PLAN**: Analyze the task, produce an architecture plan with acceptance criteria
2. **EXECUTE**: Implement via ralph mode (persistence + verification per story)
3. **QA**: Run tests, verify all acceptance criteria with fresh evidence
4. **COMMIT**: If all passes, produce final summary
</Steps>

<Verification>
Every acceptance criterion must be verified with concrete evidence (test output, file reads, build results) — not assumptions.
</Verification>
