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
- `tests/` — test suite

## Code standards
- Python 3.10+, type hints on public APIs
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
