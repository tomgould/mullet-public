# Project Goal — Feature Sprint

## Problem Statement
[Which user need does this feature serve? What's the current gap? Evidence?]

## Desired Outcome
A new feature integrated into the existing codebase:
- Follows the codebase's existing patterns
- Gated behind a feature flag if risky
- Observable (logs + metrics) from day 1
- Documented for downstream consumers

## Acceptance Criteria
- [ ] Feature works end-to-end for [primary user journey]
- [ ] Unit + integration tests green
- [ ] Feature flag in place (default off until rollout)
- [ ] User-facing docs updated
- [ ] No regressions in adjacent features

## Non-Goals
- Refactoring adjacent code (separate ticket if needed)
- Rewriting existing abstractions to fit the feature "better"
- Performance optimisation beyond existing baselines
- Backfill / data migration (explicit separate milestone if needed)

## Constraints
- Must follow existing architecture — no new frameworks or patterns
- Must coexist with flag OFF (no broken state when disabled)
- PR size ≤ reviewer's comfort zone — split into milestones

## Tech Stack (existing, do not change)
- [Read the existing code and fill in verbatim]
