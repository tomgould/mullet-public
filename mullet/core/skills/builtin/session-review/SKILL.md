---
name: session-review
description: End-of-session design principles review — assess adherence, evolve principles, apply improvements
triggers:
  - session review
  - wrap up
  - end session
  - done for today
  - review principles
  - end of session
tags:
  - process
  - review
  - principles
---

# Session Review

End-of-session ritual: review what happened, evolve DESIGN_PRINCIPLES.md, and apply improvements.

## Purpose

Capture lessons learned during a coding session before context is lost.
Use this when the user signals they are wrapping up, done for the day, or explicitly asks for a session review.

## Workflow

<Steps>
1. **Run the review script** — execute via Bash:
   `python3 scripts/session_review.py --apply`
   Read the full output. It will contain zero or more of these sections:
   - IMPROVEMENT items
   - PRINCIPLE_UPDATE items
   - NEW_PRINCIPLE items

2. **Apply IMPROVEMENT items** — for each one, delegate to `mullet_opencode` with a focused, single-item task.
   Process them one at a time, not batched. Wait for each to complete before starting the next.

3. **Apply PRINCIPLE_UPDATE items** — for each one, edit `docs/DESIGN_PRINCIPLES.md` directly.
   These are small edits; do them in-context rather than delegating.

4. **Apply NEW_PRINCIPLE items** — for each one, append the new principle to `docs/DESIGN_PRINCIPLES.md`.

5. **Commit changes** — once all items are processed:
   `git add docs/DESIGN_PRINCIPLES.md && git commit -m "docs: session review YYYY-MM-DD"`

6. **Reindex** — run `mullet reindex --pool <your-project-pool>` so the updated principles are available for future sessions.
</Steps>

## Rules

- NEVER use Anthropic (Opus / Sonnet / Haiku) for the improvements themselves.
- For improvements, prefer local LLM first; fall back to Kimi only if local is unavailable.
- One `mullet_opencode` call per improvement item — never batch multiple improvements into a single delegation.
- PRINCIPLE_UPDATE and NEW_PRINCIPLE changes are small; apply them directly without delegation.
- If the review script output contains no items, skip steps 2–5 and only run the reindex.
