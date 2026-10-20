---
name: mullet-observations
description: When the user references prior work, progress, or "what we were doing", invoke this to pull a summary of recent tool activity from Mullet's observation log before answering.
---

# Recent Activity Retrieval

When to use:
- User says "what were we working on", "where did we leave off", "remind me what I was doing"
- User references a prior session without naming it
- Before starting a new multi-step task, if observations exist in .mullet/observations/

How:
1. Run `mullet observations summarise --window 120` (last 2h) for a quick recap
2. Or `mullet observations summarise --session <sid>` if a specific session ID is known
3. Parse the returned summary + bullet_points and quote the relevant 2-3 bullets to the user
4. If no observations exist: say "no recent activity captured — the observation hook may not have fired yet"

Output format: always cite the summary's `started_at` → `ended_at` window so the user knows the time range.
