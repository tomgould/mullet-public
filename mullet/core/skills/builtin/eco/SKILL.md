---
name: eco
description: Token-saving modifier — downtier to save costs
triggers:
  - eco
  - save tokens
  - cheap mode
tags:
  - orchestration
---

# Eco Mode

Downtier all routing decisions to save tokens and money.

HIGH → MEDIUM, MEDIUM → LOW, LOW → stays LOW.

Quality floor: verification stays at least MEDIUM tier.

Use `mullet eco ralph "task"` to run ralph with reduced cost.
