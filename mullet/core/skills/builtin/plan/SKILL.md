---
name: plan
description: Strategic planning and task decomposition with TDD enforcement
triggers:
  - plan
  - design
tier_override: HIGH
tags:
  - planning
---

# Plan Mode

Analyze a task and produce a structured implementation plan before coding.

## Pre-Flight (mandatory before generating any plan)

List ALL assumptions explicitly. For each unknown, state the assumption and mark it as UNVERIFIED.
If ambiguity exists, surface 2+ approaches and state why one is chosen.
Single-option plans without stated reasoning are REJECTED.

## TDD Step Structure (mandatory for all implementation tasks)

Each implementation step MUST follow RED → GREEN → REFACTOR:
1. Write the failing test first (describe exact assertion, expected input/output)
2. Run the test — confirm it fails (RED)
3. Implement the minimum code to pass
4. Run the test — confirm it passes (GREEN)
5. Refactor if needed, run again (REFACTOR)
6. Commit with descriptive message

## Plan Format

<Steps>
1. **Assumptions** — list all with VERIFIED/UNVERIFIED status
2. **Ambiguities** — list open questions, chosen resolution, alternatives considered
3. **Architecture overview** — components and their dependencies
4. **Implementation steps** (each must include):
   - Exact file paths (no "the utils file" — full path required)
   - TDD sequence: test file + test name → impl target
   - Acceptance criteria (specific, measurable, no placeholders)
   - Risk: what could go wrong and how to detect it
5. **Self-review checklist** before declaring done:
   - [ ] All tests pass (`pytest -x`)
   - [ ] No placeholder TODOs remain in production code
   - [ ] No "pass" stubs, no commented-out code
   - [ ] Syntax check passes (`python3 -m py_compile`)
   - [ ] Behaviour tested end-to-end (not just unit tests)
</Steps>

## Rules

- No placeholders. If a file path is unknown, find it before writing the plan.
- No speculative features — implement exactly what's asked.
- Surgical changes — touch only files the task requires.
- Define success criteria before the first line of code is written.
