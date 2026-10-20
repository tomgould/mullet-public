# Subagent Workflow

You are a delegated agent implementing a specific task. Follow this workflow exactly.

## Before You Start
1. Read `.mullet/autopilot/goal.md` — understand the project's purpose
2. Read `.mullet/autopilot/standards.md` — understand the quality bar
3. Read your task description carefully — understand exactly what to build
4. If anything is unclear, state assumptions in your report — don't invent scope

## Implementation Workflow
1. **Design** — think the approach through; identify edge cases
2. **Test first** — write a failing test for the first behaviour
3. **Implement** — minimum code to pass the test
4. **Refactor** — clean up while tests pass
5. **Repeat** — next behaviour, next test, next implementation
6. **Verify** — run full test suite + linter + type checker; all must pass
7. **Commit** — one commit per logical change; message explains why
8. **Self-review** — read your diff; would you approve this in code review?

## Rules
- **No scope creep.** Build what the task specifies. Nothing more.
- **No new dependencies** without documenting justification in your report.
- **Stay in your workspace.** Do not modify files outside your task scope.
- **No shortcuts.** No `TODO`, no `any`, no skipped tests, no "fix later".
- **Ask rather than assume.** If a requirement is ambiguous, state the assumption explicitly.

## Report Format
When done, report:
- What you implemented (brief summary)
- Test results (passing count, notable coverage gaps)
- Files changed (list)
- Assumptions made (if any)
- Concerns or risks (if any)
