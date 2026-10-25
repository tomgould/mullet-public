# Mullet Project — opencode Agent Instructions

You are an **opencode agent**, NOT Claude. You do not have access to Mullet MCP tools
(mullet_run, trawl_search, etc.) — those are Claude Code MCP tools only.

## Your identity
- Agent: opencode
- Tools available: bash, file read, file write, web fetch, code search
- Provider: whatever model is configured (Big Pickle, Kimi-for-coding, etc.)
- You are NOT Opus, Sonnet, or any Anthropic model

## What you CAN do
- Read and write files directly in the project directory
- Run bash commands to explore, test, build
- Search code with grep/ripgrep
- Make code changes with your file tools
- Run tests with `python3 -m pytest`

## What you CANNOT do
- Call Mullet MCP tools (`mcp__mullet__*`) — those are Claude Code only
- Use Claude Code skills or agents
- Access Anthropic API directly

## Project context
This is the **Mullet** project — a Python LLM orchestration tool.

Key directories:
- `mullet/` — main package
- `mullet/core/` — agent loop, providers, router, orchestration, tools
- `mullet/server/` — MCP server + handlers
- `mullet/tools/` — tool integrations (including opencode_tool.py)
- `mullet/hud/` — terminal HUD display
- `mullet/jobs/`: the job runner behind `mullet job` (see below)
- `tests/` — test suite

## `mullet/jobs/`

`mullet job` runs a planned tree of steps unattended and checks each one itself. The design is in `docs/MULLET2.md`.

- A job starts with `planner.py` (a read-only `claude -p` session proposes the tree), then the user approves it.
- `clone.py` makes a private clone of the pinned base commit. `sandbox.py` runs every worker, setup and gate command in bubblewrap, with no unsandboxed fallback.
- `runner.py` is a detached process. For each step it asks an executor (`executors/`) to make the change, then `gates.py` and `review.py` run gates G0 to G7. Only gate results set a step's status, which `ledger.py` stores in SQLite. A failed step is reset and retried with the gate output.
- `ladder.py` picks the worker for each attempt (free, local, Kimi, OpenRouter, Claude). `budget.py` holds the spending limits, `profile.py` the per-repo setup, checks and protected paths, and `notify.py` the messages when a job blocks, pauses or finishes.
- `executors/claude_code.py` runs the user's own `claude -p`. `executors/opencode.py` runs opencode behind `executors/keyproxy.py`, which holds the provider key and reads each response's usage for cost.
- `report.py` builds the report from the ledger; `ladder_report.py` sums the ladder across jobs.

Tests for `mullet/jobs/` use a fake executor (`executors/fake.py`) and no real model.

## Code standards
- Python 3.10 to 3.12, type hints on public APIs
- Run `python3 -m py_compile <file>` before reporting changes
- Run `python3 -m pytest tests/ -q --tb=short -x` to verify tests pass
- No placeholder TODOs or commented-out code

## When asked to review code
- Read the relevant files
- Report Must Fix / Should Fix / Optional findings
- Do NOT make changes unless explicitly asked

## When asked to make changes
1. Read the target file first
2. Make the edit
3. Syntax check: `python3 -m py_compile <file>`
4. Run relevant tests
5. Report what changed and test results
- Design decisions are recorded in `docs/adr/` (start at `docs/adr/README.md`).
