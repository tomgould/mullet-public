---
name: brainstorm
description: Design gate — explore, propose approaches, get approval before implementing
triggers:
  - brainstorm
  - design
  - explore options
  - what approach
  - before we implement
tier_override: ADVANCED
tags:
  - planning
  - design
next_skill: plan
---

# Brainstorm Mode

Hard gate: no implementation until the user approves a design.

## Purpose

Prevent premature implementation by requiring explicit design approval.
Use this when the approach is unclear, multiple valid paths exist, or the
change is large enough that starting in the wrong direction wastes time.

## Workflow

<Steps>
1. **Explore context** — read the relevant files, understand the current state.
   What exists? What's the constraint? What does the user actually need?

2. **Ask clarifying questions** — list 2-5 open questions whose answers would
   change the approach. Ask them before proposing anything.
   Example: "Does this need to work with existing pool configs or only new ones?"

3. **Propose 2-3 approaches** — for each, provide:
   - Name and one-line description
   - How it works (2-4 sentences)
   - Pros and cons (2-3 bullet each)
   - Estimated effort (S/M/L)
   - Risks and unknowns

4. **Present design and await approval** — ask the user: "Which approach should I pursue?"
   Do NOT start implementing until the user explicitly approves one.

5. **Write a spec** — once an approach is chosen, produce a structured spec:
   - Goal (one sentence)
   - Chosen approach (name + description)
   - Exact files to create or modify (full paths)
   - Interface design (function signatures, data shapes, API endpoints)
   - Success criteria (specific, testable — will become acceptance criteria)
   - Out of scope (what this spec explicitly does NOT include)
   Save to: `docs/specs/YYYY-MM-DD-<topic>-design.md`

6. **Self-review the spec** — before presenting to the user:
   - Is any file path vague or missing?
   - Does every acceptance criterion have a concrete test?
   - Are there hidden dependencies not listed?
   - Is the scope clearly bounded?
   Fix any issues, then present the final spec.

7. **Hand off to plan** — once the user approves the spec, invoke the `plan`
   skill to convert the spec into an ordered implementation plan with TDD steps.
</Steps>

## Rules

- NEVER write code during brainstorm mode.
- NEVER propose a single option — always provide at least 2.
- NEVER proceed past step 4 without explicit user approval.
- If the user says "just do it" or "use your judgement", note the approach chosen
  and ask for a quick confirmation before proceeding.
- Specs must be saved to disk before hand-off — not just written in chat.
