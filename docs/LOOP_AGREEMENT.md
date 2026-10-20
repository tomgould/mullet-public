# Development Loop Agreement

Canonical procedure for autonomous Claude loops on this project. Generic enough to port to other projects — swap the "Mullet project defaults" section for their equivalents.

Version 2026-04-24 (rev. 2).

---

## 0. Core Philosophy

**Quality over speed. Zero-defect tolerance.** Speed is irrelevant. Every phase requires explicit validation before proceeding. The system must recursively evaluate its own code, tests, and decision-making to continuously improve.

**BUT:** ceremony must match scope. A typo fix and a multi-file refactor don't deserve the same 8-step process. The loop exposes three modes — **Full**, **Mini**, **Micro** — and the user can override via speed/effort knobs (§5).

---

## 1. Constraints (Hard)

- **Local LLM is free and fast for small tasks.** Reads, greps, summaries, extraction. Not for reasoning or multi-file refactors. Use in economy mode only.
- **OpenRouter is the second tier** (pay-per-token — DeepSeek V4 Flash default, 1M context). Default cloud for implementation, review, refactor, test gen.
- **Anthropic (Haiku/Sonnet) substitutes OpenRouter** when `check_budget()` pressure <60% — it's prepaid (Max plan) so using it saves spend on OpenRouter/Kimi. Check pressure before dispatching.
- **Kimi K2.7-Code is the fourth tier** (pay-per-token, strong coder, large context, thinking always ON). Use when OpenRouter is unavailable or the user explicitly directs it.
- **Opus is orchestration only, never delegate.** Tie-breaker and committer.
- **Gemini is last resort.** Use only when all other cloud tiers are unavailable or insufficient.
- **Anthropic substitution rule:** check `check_budget()` before each delegation. If Anthropic pressure <60%, substitute Haiku/Sonnet for the equivalent OpenRouter task — same cost to you, saves pay-per-token spend. If pressure >60%, stick to OpenRouter/Kimi.

---

## 2. Delegation Ladder

Cheapest first. Escalate explicitly, never default upward.

| Rung | Provider | Role | Invocation |
|------|----------|------|------------|
| 1 | **Local LLM** (LM Studio / Ollama, free) | Context gathering: reads, greps, summaries, "what does this file do". Economy mode only. | `mullet --provider local` |
| 2 | **OpenRouter** (DeepSeek V4 Flash default) | Default cloud: code implementation, review, refactor, test gen. Cheap pay-per-token, 1M context. | `mullet --provider openrouter` |
| 3 | **Haiku / Sonnet** (Anthropic — prepaid) | Substitute for OpenRouter when pressure <60% — saves pay-per-token spend since Max plan is already paid. | `mullet --provider haiku` / `mullet --provider sonnet` |
| 4 | **Kimi K2.7-Code** (via Mullet) | Pay-per-token, large context, thinking ON. Use when OpenRouter unavailable or user explicitly directs. | `mullet --provider kimi` |
| 5 | **Opus** (Anthropic — prepaid) | Orchestration only: tie-breaker, committer, one-shot arbitration. Never delegate implementation. | Direct in-context |
| 6 | **Gemini** (via Mullet) | Last resort only. Use when all other cloud tiers fail or are insufficient. | `mullet --provider gemini` |

**Concrete thresholds:**
- Reading / analysing any file >300 lines → start at Rung 1.
- Code implementation, test generation, refactor drafting → Rung 2 (OpenRouter) by default; substitute Rung 3 (Haiku/Sonnet) when Anthropic pressure <60%.
- `check_budget()` returns THROTTLE or pressure >0.60 → escalate to OpenRouter or Kimi (Rungs 3–4).
- Escalate to Gemini (Rung 6) ONLY if all other cloud tiers failed or the task exceeds their context.
- Opus (Rung 5) is the tie-breaker and the committer.
- **Reviewer-is-one-rung-up:** if Haiku implements, Sonnet reviews; if Sonnet implements, Opus reviews; if OpenRouter implements, Kimi reviews. Self-review (same rung) is low-value — skip it.

**Sizing before dispatch** (universal rule):
Before ANY cloud delegation, measure the composed prompt (system + context files + task) with `estimate_tokens()`. Pick a provider+model whose `max_context ≥ prompt_tokens × 1.1` (10% safety margin). Never send prompts to models that will blow their stack.

---

## 3. Loop Modes

Select the mode before starting the cycle. Stay in one mode per cycle; don't cross-mix.

| Mode | When to use | Steps | Validations |
|------|-------------|-------|-------------|
| **Full Loop** | New features, multi-file refactors, architectural changes, anything touching ≥2 systems | 8 | Plan-critique, TDD-Red, TDD-Green, Peer review, Refactor re-test, Retro |
| **Mini Loop** | Single-file fix <100 lines, adding tests for existing code, straightforward bug fix with clear repro | 5 | TDD-Red, TDD-Green |
| **Micro Loop** | Typo, docstring, one-line comment, simple config tweak, doc correction | 3 | Test suite stays green |

If unsure, err toward the more ceremonious mode. Escalating mid-cycle is fine; demoting mid-cycle is not.

---

## 4. The Full Loop (8 steps)

### 4.1 REQUIREMENTS & PLAN

- Read the next item from `TODO.md`, `ROADMAP.md`, or `GAP_ANALYSIS.md`. If ambiguous or blocked, STOP and ask — do not invent scope.
- Draft a step-by-step implementation plan (files to touch, test cases, success criteria).
- **VALIDATION CHECKPOINT — Plan critique**
     - Delegate to **OpenRouter (Rung 2)**: *Review this plan. Flag security issues, missed edge cases, architectural conflicts, hidden assumptions. Suggest test cases not in the plan.*
     - Apply findings. Re-dispatch if the plan was substantially rewritten.
     - **Circuit-breaker:** if plan-critique has looped >3 times, escalate to Opus (Rung 5) for a one-shot arbitration. No infinite review loops.

### 4.2 TEST DESIGN (TDD — Red)

- Before writing any application code, write the full test suite: unit + integration + edge cases.
- **VALIDATION CHECKPOINT — Tests must fail Red, and must fail for the right reason**
  - Run the suite. It MUST fail.
  - If it passes, the tests are wrong — they're not actually testing the missing behaviour. Fix tests first.
  - **Name the expected failure mode in the test's docstring** (`"fails because method X does not yet exist"` / `"fails because handler returns None for missing key"`). If the failure message doesn't match, tests are testing the wrong thing.

### 4.3 IMPLEMENTATION (Green)

   - Write the minimum code necessary to make tests pass. Delegate to **OpenRouter (Rung 2)**.
- **VALIDATION CHECKPOINT — Tests must pass Green**
  - Run the full suite. Must be green.
  - If red, loop back to 4.3 with the failing output fed to the same provider. Two fix attempts max — then STOP (see §6).

### 4.4 PEER REVIEW

   - Feed the unified diff to the reviewer **one rung above the implementer** (OpenRouter impl → Kimi review; Kimi impl → Sonnet review).
- Reviewer prompt must cover: time/space complexity, DRY, SOLID, security (injection, path traversal, unchecked input), error handling, log-level appropriateness.
- **VALIDATION CHECKPOINT — Address all findings**
  - Categorise findings as MUST-FIX / SHOULD-FIX / NICE-TO-HAVE.
  - MUST-FIX applies before the commit. SHOULD-FIX applies or becomes a `chore:` TODO item. NICE-TO-HAVE → `chore:` TODO only if clearly worthwhile.

### 4.5 RECURSIVE REFACTOR

- Now that code works and is reviewed, refactor for elegance, readability, performance.
- **VALIDATION CHECKPOINT — Full suite stays green**
  - Re-run full test suite. Any red → revert the refactor, surface the regression.

### 4.6 DOCUMENTATION & METRICS

- Update `TODO.md`: tick the item, update any `%` indicators, add new items surfaced during the cycle.
- Update `CHANGELOG.md`: one entry under `## [Unreleased]` in Conventional Commits style.
- Append to `progress.txt`: `YYYY-MM-DD HH:MM <what shipped>`.
- **Non-negotiable:** documentation happens in the same cycle as the code. Never batch.

### 4.7 COMMIT, PUSH & REINDEX

- `git pull --rebase origin <branch>` before every commit.
- Commit using Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `perf:`, `ci:`, `build:`.
- `git push origin <branch>`.
- If code or tests changed (not docs-only), reindex the project's knowledge pool.

### 4.8 RETROSPECTIVE

- Briefly analyse the just-completed cycle:
  - Did the ladder work efficiently? Did we escalate?
  - Did plan-critique catch anything real?
  - Did tests actually catch bugs or just pass coincidentally?
  - Did peer review add value or just re-state the tests?
  - Token cost: how many delegate calls, roughly how many tokens, any runaway?
- **Output:** if there's an actionable process improvement, add a line to `docs/PROCESS_IMPROVEMENTS.md`. That file aggregates learnings across cycles — read it at the start of every new overnight loop.
- **Session-end:** run the `/session-review` skill (or `python3 scripts/session_review.py --apply`). It measures adherence against `docs/DESIGN_PRINCIPLES.md`, applies improvements via `mullet_opencode` (local LLM first), and evolves the principles document. This is how design principles improve over time — retros feed directly back into the principles doc.
- Pick next task. Back to 4.1.

---

## 5. User Overrides — Speed / Effort / Budget

The user can override the loop's ceremony level at any point. Recognise these as triggers and apply for the rest of the session (or until countermanded).

### 5.1 Speed mode

| User says | Mode | Behaviour |
|-----------|------|-----------|
| "go slow", "economy", "save tokens", "spend the night" | **economy** | Full Loop default; all validation checkpoints; prefer local LLM first |
| "balanced" | **balanced** | Full Loop for ≥2-file changes; Mini Loop for single-file; skip local LLM for code work |
| "go fast", "I'm in a rush", "quick" | **fast** | Mini Loop default; skip plan-critique + peer review unless touching security-sensitive code; OpenRouter/Kimi only |
| "max speed", "don't worry about cost" | **max** | Full Loop using Sonnet end-to-end; parallel agent dispatch where safe; skip OpenRouter/Kimi unless specifically requested |

### 5.2 Effort override for a single cycle

The user can pin the cycle's loop mode directly:
- "use Mini Loop", "Mini" → current cycle uses Mini Loop.
- "use Micro", "just do it" → current cycle uses Micro Loop. Requires a corresponding user-acknowledged risk: tests stay green, no new scope.
- "Full Loop please", "be thorough" → force Full Loop even for small changes.

### 5.3 Budget override

- "budget $X", "stay under $X" → use `check_budget()` between cycles; if projected spend exceeds the cap, stop and report.
- "no Gemini", "no API calls" → strip Rung 6 from the ladder for this session.

### 5.4 Scope override

- "skip tests" → ACK with a loud warning; commit message gets a `[SKIP-TESTS]` trailer; add a `chore: backfill tests` TODO. Never silently skip tests.
- "ignore the lint" / "just ship" → same pattern: acknowledge, warn, log a backfill TODO.
- "force push" → refuse without explicit per-command confirmation.

### 5.5 Skipping specific checkpoints

The user may selectively skip checkpoints:
- "skip plan critique" — 4.1 validation only.
- "skip peer review" — 4.4 only.
- "skip refactor" — 4.5 only.

Honour these for the current cycle. Do not carry over to the next cycle unless phrased as session-wide.

---

## 6. Error Handling & Root-Cause Analysis

- **Model failure (any Rung):** retry once. If it fails again, fall back one rung down (for simpler abstraction) OR escalate one rung up (if the task is too complex). Log in `progress.txt` with the failure message.
- **Reindex failure:** log, continue the loop, flag at end of session.
- **Test failure after 2 fix attempts:** **STOP.** Write a Root Cause Analysis to `progress.txt`: failing stack, hypothesised cause, what was tried. Surface to user. Do not guess a third fix.
- **Ambiguous requirement:** **STOP.** Do not invent scope. Ask.
   - **Checkpoint circuit-breaker:** any validation checkpoint (plan-critique, peer review, etc.) looping >3 times without convergence → escalate to Opus Rung 5 for one-shot arbitration, then proceed with the arbitrated decision.
- **Provider hallucinates tool calls:** record in `progress.txt` as a delegation-safety incident; strip tool mentions from delegate prompts where the provider has no tool access.

---

## 7. Graceful Stop — Usage Limits & State Preservation

Approaching the token/context budget, do this in order:

1. Finish the current loop **sub-step only**. Do not start a new step.
2. Commit WIP: `wip: <summary> [resume-here]`. No push. No reindex.
3. Append to `progress.txt`:
   - **Completed:** what's done.
   - **Pending:** exact step (4.1–4.8) to resume at.
   - **State:** files mid-edit, failing tests, pending reviews, unmerged decisions.
4. **STOP.** Do not begin another iteration.

**Resume protocol (mandatory first action in the next session):**
1. Read `progress.txt` — last 100 lines.
2. Check `git status` + `git log --oneline -10`.
3. Identify the `[resume-here]` marker.
4. Resume at the step noted, NOT at the next cycle.

---

## 8. Empty Backlog Protocol

When `TODO.md` and `ROADMAP.md` are both empty:

1. **Archive:** move to `docs/archive/YYYY-MM-DD-<name>.md`.
2. **Deep audit:** review the codebase against reference repos / comparable projects.
3. **Delegate the diff:** OpenRouter generates the gap analysis.
4. **Validate the audit:** escalate to Kimi (one rung up) to catch subtle technical debt OpenRouter missed. Kimi's findings are additive, not a replacement.
5. **Generate:** new `GAP_ANALYSIS.md`, fresh prioritised `TODO.md`, updated `ROADMAP.md`.
6. **Resume:** step 4.1.

---

## 9. Non-Negotiables

- No batching doc updates across cycles.
- No skipping reindex when code changed (unless in graceful-stop mode).
- No committing red tests.
- No defaulting to Gemini.
- No starting a new cycle when near the token limit.
- No self-review on the same rung.
- No prompts dispatched to models that will blow their context window.
- Conventional Commits only.
- No Co-Authored-By lines (project convention).

---

## 10. Reviewer Matrix (quick reference)

| Implementer | Reviewer (default) |
|-------------|---------------------|
| Local LLM | Haiku |
| Haiku | Sonnet |
| Sonnet | Opus |
| OpenRouter | Kimi |
| Kimi | Sonnet |
| Gemini | Opus |
| Opus | — (peer review skipped; require user to tag second opinion) |

---

## 11. Mullet project defaults (customisation point — swap for other projects)

For THIS project:
- Knowledge reindex command: `mullet reindex --pool mullet-dev` after code/test changes.
- Environment loader: `source scripts/_env_from_mcp.sh` before any `mullet --provider kimi|gemini` invocation.
- Budget check: `python3 -c "from mullet.hud.guard import check_budget; r = check_budget(); print(r.status.value, r.message)"`.
- After N cycles: `mullet review-session` for a loop retrospective; tie observations, decisions, and routing outcomes into a single report.
- Bridge PreToolUse rewriter is registered; `mcp__mullet__*` calls inherit session override + pressure-aware tier autofill without further orchestration.
  - The 5h budget check respects an active Opus override: the loop auto-downgrades opus→sonnet/haiku rather than disabling the override.
- Self-update between phases: `pip install -e .`.
- Default branch: `master`.
- Session state: `.mullet/progress.txt`.
- Retrospective log: `docs/PROCESS_IMPROVEMENTS.md`.

Porting to another project: replace the above five lines with the equivalents. Everything else in this doc is project-agnostic.
