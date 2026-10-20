---
name: code-review
description: Review code for bugs, quality, and security
triggers:
  - code review
  - review code
  - review this
tier_override: HIGH
tags:
  - development
---

# Code Review

Review code with severity-rated feedback.

<Steps>
1. Read the target file(s) using read_file
2. Analyze for:
   - **Bugs**: logic errors, off-by-one, null handling, race conditions
   - **Security**: injection, XSS, hardcoded secrets, insecure defaults
   - **Quality**: naming, complexity, duplication, SOLID violations
   - **Performance**: unnecessary allocations, N+1 queries, blocking calls
3. Rate each issue: CRITICAL / HIGH / MEDIUM / LOW
4. Suggest specific fixes with code examples
5. Note what's done well (not just problems)
</Steps>

<Output_Format>
## Code Review: {filename}

### Issues Found

**CRITICAL** - {description}
```suggestion
{fix}
```

**HIGH** - {description}
...

### Positive Notes
- {what's good about this code}
</Output_Format>
