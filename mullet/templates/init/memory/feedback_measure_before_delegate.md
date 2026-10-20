---
name: Measure prompt tokens before delegating
description: Never send a prompt to a model that will blow its context window
type: feedback
---

Estimate prompt token count before delegating. Don't paste multi-thousand-line files into a small-context model.

**Why:** Context exceeded errors waste a full LLM round-trip.
**How to apply:** Use estimate_tokens() or rough word-count * 1.3 before composing delegation prompt.
