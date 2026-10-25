# Mullet Skill Authoring Guide

## What is a Skill?

A skill is a SKILL.md file that instructs the Mullet agent how to perform a specific task. Skills are markdown with YAML frontmatter — they extend the agent's capabilities without writing Python code.

## Creating a Skill

### 1. Create the directory

```bash
mkdir -p "$(mullet paths data)/skills/my-skill"   # ~/.local/share/mullet/skills on a fresh install
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
/my-skill                      # Slash command in the mullet REPL
mullet --skills                # List every skill Mullet can see
```

## Frontmatter Fields

| Field | Required | Description |
|---|---|---|
| `name` | Yes | Skill identifier (lowercase, hyphens) |
| `description` | Yes | One-line description for skill index |
| `triggers` | No | Keywords that auto-activate the skill |
| `tags` | No | Categories for grouping |
| `mode` | No | Read but ignored. The orchestration modes it used to name (autopilot, ralph, ultrawork, team) were removed. |
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

Skills are not chained by frontmatter. The orchestration modes that used to chain them (autopilot, ralph, ultrawork, team) were removed. For multi-step work on a repository, use `mullet job`; the `mullet-job` skill walks Claude Code through it.

## Scope Priority

1. **Project skills** (`.mullet/skills/`) — highest priority
2. **User skills** (`skills/` in the data directory, `mullet paths data`; `~/.mullet/skills/` on an install that has not been migrated)
3. **Built-in skills** (bundled with mullet)

`mullet.yaml` accepts `skills.external_dirs`, but the CLI and MCP server don't load
skills from those folders yet.

Project skills override user skills with the same name.

## Built-in Skills

| Skill | What it does |
|---|---|
| brainstorm | Explore options and get approval before implementing |
| code-review | Review code for bugs, quality and security |
| mullet-job | Run a big coding job unattended in a sandboxed clone and hand back a branch |
| plan | Strategic planning and task decomposition |
| pool-improve | Audit a Trawl pool config and apply improvements |
| session-review | End-of-session review against the design principles |

`mullet setup-claude` also installs `mullet-job` and `pool-improve` into `~/.claude/skills/` for Claude Code.

## Learned Skills

Mullet can learn skills from your usage patterns. Learned skills are automatically created from successful agent executions and stored with confidence scoring for future matching.

The learned skill system:
- Automatically captures successful task patterns
- Uses confidence scoring (exact, pattern, fuzzy matching)
- Integrates with the skill matching system
- Stored separately from user-created skills

Learned skills complement user-created skills and built-ins, expanding the agent's capabilities over time.
