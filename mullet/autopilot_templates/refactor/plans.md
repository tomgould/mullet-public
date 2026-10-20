# Project Plan — Refactor

## Architecture Overview
Before: [describe current structure, its pain points]
After: [describe target structure, what it enables]
Migration strategy: [parallel / strangle-fig / in-place]

## Milestones

### Milestone 1: Safety net
**Goal:** We can refactor without fear
**Depends on:** None

#### Tasks
##### Task 1.1: Characterisation tests
- **Parallel:** no
- **Approach:** pin current observable behaviour in high-level tests
  (end-to-end if needed) BEFORE changing anything
- **Acceptance criteria:** any accidental behavioural change fails a test
- **Status:** pending

##### Task 1.2: Test suite baseline
- Run existing suite, record green state, capture any pre-existing flakes
- Anything flakey gets quarantined or fixed BEFORE refactor starts
- **Status:** pending

### Milestone 2: Extract seams
**Goal:** Isolate the thing being refactored behind an interface
**Depends on:** Milestone 1

#### Tasks
##### Task 2.1: Identify the seam
##### Task 2.2: Introduce interface / adapter
##### Task 2.3: Route existing callers through the seam
##### Task 2.4: Tests still green

### Milestone 3: Replace
**Goal:** Swap the old implementation for the new one
**Depends on:** Milestone 2

#### Tasks
##### Task 3.1: New implementation (on the seam interface)
##### Task 3.2: Flip callers to new impl behind a feature flag (if risk is high)
##### Task 3.3: Remove old implementation
##### Task 3.4: Remove seam shims if no longer needed

### Milestone 4: Cleanup
**Goal:** Tidy up, document, confirm metrics hit
**Depends on:** Milestone 3

#### Tasks
##### Task 4.1: Delete dead code
##### Task 4.2: Update docs and architecture diagrams
##### Task 4.3: Confirm target metric from goal.md
##### Task 4.4: Update CHANGELOG
