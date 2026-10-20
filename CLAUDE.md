# Mullet — Context for Claude

Mullet is an AI coding orchestrator. It routes tasks to cheap/free models (Kimi K2,
local LLM, Gemini) and integrates with Claude Code via MCP. The goal is to keep
Anthropic token spend near zero by delegating actual work elsewhere.

## Architecture

Mullet is an MCP-server-first orchestrator. The two primary entry points are:

- **`mullet mcp`** (or `mullet-mcp`) — stdio MCP server used by Claude Code and other MCP clients. Exposes `mullet_run`, `mullet_delegate`, `mullet_opencode`, `mullet_swarm`, and 42+ Trawl knowledge tools.
- **`mullet serve`** — OpenAI-compatible REST + WebSocket API server for external integrations.

The core runtime (`mullet/core/`) handles provider routing, tool dispatch, skills, knowledge pools, and orchestration modes. It is consumed through the MCP layer rather than a standalone interactive shell.

The legacy harness layer (`mullet/harness/`) and its interactive TUI (`mullet shell`, `mullet cli`) were removed in 2026-07. They are preserved in git history but no longer shipped.

## Critical: do not suggest these things to users

1. **Do not suggest adding `ANTHROPIC_API_KEY`** — Mullet integrates with Claude Code
   via the user's existing Max/Pro subscription. It does not use the Anthropic API directly.
   Adding an API key wastes money and confuses routing.

2. **Do not suggest `~/.zshrc` or `~/.bashrc` for API keys** — Mullet loads a `.env`
   file from the repo root via `python-dotenv`. Keys go in `.env`, not shell exports.

3. **Do not suggest 7B+ models on machines with ≤10GB RAM** — they won't fit.
   For 8GB Mac: recommend `qwen3.5-2b-kimi-and-opus-distillation` (1.6GB, reasoning tasks)
   and `qwen2.5-coder-3b-instruct` (1.9GB, code tasks). The router picks between them
   automatically based on task type.

4. **Do not suggest `pip install -e ".[gpu]"`on Mac** — the `[gpu]` extra pulls `pynvml`
   (NVIDIA-only). On Apple Silicon, `sentence-transformers` brings in PyTorch with MPS
   support automatically.

## Providers

Token economy order: **local LLM first** (when appropriate) → **OpenRouter** (default cheap cloud) → **Anthropic** (only with a real `ANTHROPIC_API_KEY`, pay-per-token) → **Kimi/Moonshot** (under budget pressure or when explicitly directed)

- **Local LLM** — LM Studio or Ollama on localhost:1234 / 11434. Free. First choice for simple tasks.
- **Anthropic** — Haiku/Sonnet/Opus via a real `ANTHROPIC_API_KEY` (metered). Mullet does **not** use your Claude Code Max/Pro/Free subscription's OAuth token to call Anthropic directly -- that's disabled as of 2026-07 because it violates Anthropic's Consumer ToS (see `mullet/core/providers/claude_max.py`). When Mullet is called from Claude Code and no API key is configured, it hands Anthropic-only requests back so Claude Code can do the work natively instead (that's always been a fully legitimate use of the subscription).
- **Kimi K2** — Moonshot cloud fallback when Anthropic budget pressure is high, or user-directed. Key at platform.kimi.ai (~£20).
- **Gemini Flash** — last-resort cloud fallback when Kimi unavailable. Optional.

## Local model routing

Mullet's router classifies local models by name pattern:
- Models matching `coder`, `code`, `starcoder` → selected for code tasks
- Models matching `distillation`, `thinking`, `reasoning`, `r1` → selected for reasoning/analysis tasks

Having one of each type gives full routing coverage.

## First-time setup

Run `python3 scripts/onboard.py` from the repo root — it detects the system and
prints a personalised checklist. Do this before `pip install -e .`.

## setup-claude is safe to run

`mullet setup-claude` is idempotent and fully reversible. It writes to `~/.claude/settings.json`,
`~/.claude/mcp.json`, and `~/.claude/CLAUDE.md`. To undo: `pip uninstall mullet` and remove
the two `mullet` blocks from those files. No vendor lock-in, no data migration. Run it.
