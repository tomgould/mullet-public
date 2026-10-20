# Project Standards

Quality bar for this autopilot run. Every line of code must meet it.
Delegated agents: read this file before writing any code.

## Code Quality
- Names reveal intent. If you need a comment to explain what something does, rename it.
- Functions do one thing. If you describe behaviour using "and", split it.
- Error handling is explicit. No swallowed errors, no empty catch blocks.
- No escape hatches: no `any`, `# type: ignore`, `@ts-ignore`.

## Testing
- TDD first. Test describes behaviour, not implementation.
- Every public API has tests. Every error path has tests.
- Tests are independent. No shared mutable state. No order dependence.
- Assertions are specific. `assert x == 5`, not `assert x`.

## Architecture
- Dependencies flow inward. Core logic never imports infrastructure.
- Interfaces at boundaries. Concrete implementations behind abstractions.
- No premature abstraction. Three concrete uses before extracting a pattern.
- Configuration is explicit. No magic strings. No implicit defaults.

## Git
- Conventional Commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`.
- Each commit is one logical change that compiles and passes tests.
- Commit messages explain WHY. The diff shows WHAT.
- No `Co-Authored-By` trailers on this project.
