---
name: Prefer concurrent processing
description: Batch independent tasks in parallel; don't serialise what can fan out
type: feedback
---

When multiple independent tasks exist, dispatch them in parallel (multiple tool calls in one message).

**Why:** Reduces wall-clock time significantly for exploratory or multi-file work.
**How to apply:** Before starting sequential work, check if tasks are truly dependent; fan out if not.
