# Project Goal — Refactor

## Problem Statement
[Which module/subsystem is being refactored? What pain is it causing today?
Coupling? Performance? Testability? Readability?]

## Desired Outcome
Same observable behaviour, better structure:
- Clear separation of concerns
- No regressions — every existing test still passes
- Public API surface preserved (or deprecation path documented)
- Reduced [LOC / cyclomatic complexity / coupling / test-setup length]

## Acceptance Criteria
- [ ] Full existing test suite green on every commit
- [ ] No change to public API signatures (or deprecation shims in place)
- [ ] Target metric improvement: [e.g., "auth module LOC down 30%"]
- [ ] New architecture documented in `docs/architecture/`

## Non-Goals
- New features (strictly internal restructure)
- Test rewrites unless a test was testing implementation, not behaviour
- Dependency upgrades (separate ticket)
- Performance optimisation (unless it's the stated pain)

## Constraints
- Backwards compatibility: [strict / soft / breaking OK with migration guide]
- Existing test suite is the safety net — do not delete tests without replacement
- Refactor commits must be individually revertable

## Tech Stack (existing)
- [Read the existing code and fill this in verbatim from what's there]
