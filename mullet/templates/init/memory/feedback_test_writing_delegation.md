---
name: Test writing must use mullet_opencode
description: ALL test file creation/rewrites go to mullet_opencode, never written directly
type: feedback
---

When writing new test files or rewriting existing ones, delegate to mullet_opencode rather than writing inline.

**Why:** Test files are often large; inline generation burns expensive Anthropic tokens and is error-prone.
**How to apply:** Any test file creation → mullet_opencode task.
