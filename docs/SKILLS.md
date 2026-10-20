# Mullet Skill Authoring Guide

## What is a Skill?

A skill is a SKILL.md file that instructs the Mullet agent how to perform a specific task. Skills are markdown with YAML frontmatter — they extend the agent's capabilities without writing Python code.

## Creating a Skill

### 1. Create the directory

```bash
mkdir -p ~/.mullet/skills/my-skill
```

### 2. Write the SKILL.md

```markdown
---
name: my-skill
description: What this skill does (shown in skill index)
triggers:
  - my trigger phrase
  - another trigger
tags:
  - category
---

# My Skill

Instructions for the agent...
```

### 3. Use it

```bash
mullet /my-skill              # Slash command
mullet "my trigger phrase"     # Auto-detected from triggers
```

## Frontmatter Fields

| Field | Required | Description |
|---|---|---|
| `name` | Yes | Skill identifier (lowercase, hyphens) |
| `description` | Yes | One-line description for skill index |
| `triggers` | No | Keywords that auto-activate the skill |
| `tags` | No | Categories for grouping |
| `mode` | No | Orchestration mode to activate (autopilot, ralph, ultrawork, team) |
| `version` | No | Semantic version string (defaults to "1.0.0") |

## Body Structure

Use XML-like tags for LLM-friendly parsing:

```markdown
<Purpose>
What this skill does and why.
</Purpose>

<Steps>
1. First step
2. Second step
</Steps>

<Verification>
How to verify the work is correct.
</Verification>
```

## Skill Pipelines

Skills can be chained together using orchestration modes rather than pipeline declarations in frontmatter. The mode field activates specific orchestration behaviors:

- **autopilot**: Full autonomous execution with planning and verification
- **ralph**: Deterministic, don't-stop-until-done execution
- **ultrawork**: Parallel task execution
- **team**: Coordinated multi-agent execution

The orchestration system handles skill chaining based on the selected mode.

## Scope Priority

1. **Project skills** (`.mullet/skills/`) — highest priority
2. **User skills** (`~/.mullet/skills/`)
3. **External dirs** (from `mullet.yaml` skills.external_dirs)
4. **Built-in skills** (bundled with mullet)

Project skills override user skills with the same name.

## Built-in Skills

| Skill | Triggers | Mode |
|---|---|---|
| autopilot | "autopilot", "full auto" | autopilot |
| brainstorm | "brainstorm", "design", "explore options" | — |
| code-review | "code review" | — |
| eco | "eco", "save tokens" | — |
| plan | "plan", "design" | — |
| pool-improve | "improve pool", "fix pool config" | — |
| ralph | "ralph", "don't stop" | ralph |
| team | "team", "coordinated" | team |
| ultrawork | "ultrawork", "parallel" | ultrawork |

## Learned Skills

Mullet can learn skills from your usage patterns. Learned skills are automatically created from successful agent executions and stored with confidence scoring for future matching.

The learned skill system:
- Automatically captures successful task patterns
- Uses confidence scoring (exact, pattern, fuzzy matching)
- Integrates with the skill matching system
- Stored separately from user-created skills

Learned skills complement user-created skills and built-ins, expanding the agent's capabilities over time.
