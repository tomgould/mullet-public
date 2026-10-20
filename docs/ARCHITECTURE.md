# Mullet Architecture

> Current state of the system — not aspirational, what's actually built.

## Overview

Mullet is a Python monolith (~97K lines, ~337 modules, 17,700+ tests) that routes AI tasks to the cheapest capable model. Local GPU for grunt work, cloud for architecture. One `pip install`, one command.

```
User → CLI/API/MCP → Router → Provider → Agent Loop → Tools → Response
                        ↕           ↕           ↕         ↕
                     Budget     Health      Knowledge  Analytics
                        ↕
                    Learning (improves routing from past outcomes)
```

## Core Components

### Agent Engine (`mullet/core/agent/`)
- **runner.py** — Main conversation loop. Send messages → LLM → tool calls → dispatch → repeat. Handles interrupts, budget warnings, context compression, fallback.
- **delegation.py** — Spawn isolated child agents. Parallel execution via ThreadPoolExecutor. Depth-limited (max 2).
- **compression.py** — Auto-summarize when approaching context limits. Model-aware thresholds. Pre-pass prunes old tool outputs before expensive summarization.
- **prompts.py** — System prompt assembly with model-specific guidance (Gemini, GPT, Mistral, Llama).
- **memory.py** — Persistent agent memory across sessions. Trawl-backed or local JSON fallback.
- **session.py** — Session tracking and cross-session search.
- **cost.py** — Cost estimation with savings display vs premium baseline.
- **context.py** — Multi-source context injection with priority ordering.

### Providers (`mullet/core/providers/`) — 10 backends
- **openai_compat.py** — Covers LM Studio, Ollama, vLLM, OpenRouter, Kimi, OpenAI. Retry with exponential backoff on 429/5xx.
- **anthropic.py** — Native Anthropic with prompt caching (~75% savings), tool_result batching, streaming.
- **gemini.py** — Native Google Gemini via google-genai SDK. Tool calling via FunctionDeclaration.
- **gemini_cli.py** — Free-tier Gemini via the `gemini` CLI (OAuth, no API key).
- **bedrock.py** — AWS Bedrock Converse API.
- **ollama.py** — Native Ollama integration.
- **claude_cli.py** — Routes via a local Claude Code CLI/Max subscription.
- **openrouter.py** — OpenRouter meta-provider.
- **kimi.py** — Moonshot AI Kimi K2 provider. Used by swarm orchestration and default STANDARD tier.
- **kimi_async.py** — Async facade over KimiProvider via `asyncio.to_thread`. Used by SwarmDispatcher.
- **model_info.py** — Model database: context windows, pricing, capabilities.
- **model_registry.py** — Dynamic model registry with 24h cache; discovers available models from Anthropic/Kimi/Gemini APIs. `get_best_model()` used by mcp_handlers for dynamic model selection.
- **health.py** — Exponential backoff per error type (rate_limit=30s, connection=120s, auth=3600s).

### Router (`mullet/core/router/`)
- **complexity.py** — Signal extraction + scoring. Lexical (keywords) + structural (subtasks, deps, domain) + context (failures, chain depth, plan complexity) → score → tier.
- **decision.py** — Routes to optimal provider. Health-filtered, budget-gated, tier-mapped. Auto-downgrades when budget blocks premium.
- **budget.py** — Token spend tracking with dollar limits. Per-task, per-session, per-day.
- **fallback.py** — FallbackManager with tried-provider tracking and turn-scoped restoration.
- **resolver.py** — Bridges routing decisions to actual Provider instances.

### Presets (`mullet/core/presets.py`)

Named routing policies (allowed provider sets, speed mode, opencode model, native-only flag) persisted to `~/.config/mullet/presets.yaml`. Built-ins (`balanced-mix`, `local-only`, `claude-only`, `cheap-cloud`) are always merged from code on load. Enforced in the MCP handlers before model selection.

### Pre-flight routing pipeline (added 2026-04-28)

Routing now uses a 3-stage pipeline before any paid model call:

1. **`mullet/core/router/preflight.py`** — calls a near-free LLM (local LM Studio first, then `gemini-2.5-flash`, then heuristic fallback) to estimate input tokens, expected output tokens, and complexity tier. Results are cached to `~/.mullet/cache/preflight.jsonl` (7-day TTL, atomic writes).
2. **`mullet/core/router/routing_matrix.py`** — `select_kimi_model()`, `select_gemini_model()`, and `select_local_model()` implement a 3-axis cheapest-viable-model selector. `select_local_model()` is adaptive: it classifies whatever models LM Studio exposes by parsing model ID strings (param count, coder/vision/reasoning flags), so it works on any machine.
3. **Fallback chain** — `_best_local()` → `_best_kimi()` → `_best_gemini()` in `ModelRegistry`. Local is attempted first for LOW/STANDARD tasks on economy/balanced speed mode.

`KNOWN_PRICING`, `KNOWN_CACHE_HIT_PRICING`, and `KNOWN_DEPRECATIONS` in `model_registry.py` hold corrected official pricing. All `moonshot-v1-*` models removed.

### Tools (`mullet/core/tools/`) — 25 tools across 15 toolsets

See `README.md` "Agent Tools" for the full list. Highlights:

| Toolset | Tools |
|---|---|
| terminal | terminal (cwd tracking, output limits) |
| file | file_ops (read/write/patch/list/search), patch (unified diffs, fuzzy matching) |
| web | web_search (DuckDuckGo), fetch_page |
| browser | browse (read, links, forms, extract, CSS selectors) |
| code | code_exec (sandboxed subprocess), python_repl (persistent state) |
| delegation | delegate (child agent spawning, depth-limited) |
| git | git (status, diff, log, blame — read-only) |
| testing | run_tests (pytest + npm with structured results) |
| vision | analyze_image (multimodal base64) |
| creative | image_gen (DALL-E, FAL.ai FLUX, local) |
| docker | docker (ps, build, run, exec, logs, stop, rm) |
| lsp | lsp (diagnostics, go-to-def, references, hover, symbols) |
| memory | memory (KV + TTL), notepad (agent scratchpad), shared_mem (multi-agent KV) |
| session | session_search, session_list |
| skills | skill_invoke, skill_mgr |
| planning | todo (in-session tracking) |
| interaction | clarify (structured user questions) |
| mcp | mcp_client (call external MCP servers) |

### Knowledge — Trawl (`mullet/knowledge/` + `mullet/server/mcp_knowledge.py`)

Trawl knowledge engine. 41 MCP tools covering pools, search, graphs, ingest, sources, memory, workflows, agents, prompts. Pool directory layout: `config.yaml` + `index/` (ChromaDB + graph) + `documents/` + `.pool_metadata.json` (usage tracking).

- **engine/** — vendored knowledge engine: client/searcher/indexer/splitter/reranker/query_transform/extract.
- **engine/pool_metadata.py** — per-pool `.pool_metadata.json`: `created_at`, `last_indexed_at`, `last_searched_at`, `search_count`. Atomic writes, best-effort failure mode.
- **graph/extractor.py** — LLM-based entity/relationship extraction. Circuit breaker + garbage-response detection (SPECIAL_* spam, low token diversity) to bail early on broken models.
- **graph/store.py** — SQLite + NetworkX backed graph. Tracks `processed_chunk_ids` for incremental extraction.
- **predigest.py** — Local LLM pre-processes heavy content → stores summary → cloud reasons over it.

### Orchestration (`mullet/core/orchestration/`)
- **ultrawork** — Decompose → parallel children → merge results.
- **ralph** — PRD-driven persistence loop. Generate stories → implement → verify → loop.
- **autopilot** — Plan → ralph → QA → fix cycle → summary.
- **team** — 5-stage pipeline: plan → prd → exec → verify → fix.
- **planner** — Structured plan generation with dependency ordering.
- **verifier** — Per-criterion verification with evidence collection.
- **eco** — Downtier modifier (CRITICAL→ADVANCED→STANDARD→LOCAL). Tier names: LOCAL (<30) / STANDARD (30-75) / ADVANCED (76-89) / CRITICAL (90+).
- **swarm** — Kimi Agent Swarm: decompose goal → dispatch N tasks in parallel → synthesize. Components: `swarm.py` (orchestrator), `swarm_dispatcher.py` (async fan-out with semaphore concurrency), `llm_decomposer.py` (LLM-based task split), `static_decomposer.py` (file/item enumeration), `synthesizer.py` (final merge call), `swarm_types.py` (shared dataclasses).

### Skills (`mullet/core/skills/`) — 7 built-in
- **loader.py** — Discovers SKILL.md files across builtin/user/project scopes.
- **registry.py** — Trigger matching (slash commands + keywords + learned).
- **invoker.py** — Pipeline context, requirements, handoff artifacts.
- **learned.py** — User-created skills with confidence scoring.

### Server (`mullet/server/`)
- **app.py** — aiohttp with CORS, auth, rate limiting, error handling, graceful shutdown.
- **api.py** — REST: /health, /v1/chat/completions (SSE), /api/v1/projects, search, memory.
- **websocket.py** — Real-time task execution with full tool parity.
- **mcp.py** + **mcp_handlers.py** — 10 mullet_* MCP tools + 41 trawl_* knowledge tools for Claude Code integration.
- **admin.py** — User/project/access/usage CRUD.
- **auth.py** — PAT generation, verification, 0600 permissions.
- **db.py** — SQLite schema (6 tables), solo bootstrap.

### CLI (`mullet/cli/`)
- 40+ commands including serve, mcp, doctor, init, setup-claude, history, config, stats, completion, explain, autopilot, ralph, ultrawork, team, eco, login, hud, pool, ingest, search, reindex, evaluate, live-test, review-session, and others
- --verbose shows routing decisions, cost savings
- --resume resumes interrupted orchestration modes
- --export saves conversation to JSON or Markdown
- --tools, --skills for discovery
- Pipe support: `cat file.py | mullet "review this code"`

## Harness Layer (`mullet/harness/`)

> **Removed 2026-07.** The harness layer and its interactive TUI (`mullet shell`, `mullet cli`, `mullet web`, `mullet acp`, etc.) have been deleted from the codebase.
>
> Mullet is now **MCP-server-first**. The core runtime is consumed through the MCP layer rather than a standalone interactive shell. The harness code is preserved in git history for reference.

## CLI Surface

Live subcommands:

- **mullet serve** — OpenAI-compatible REST + WebSocket API server.
- **mullet mcp** — stdio MCP server for Claude Code and other MCP clients.
- **mullet doctor** — diagnostic checks for install / models / pools / version drift.
- **mullet init** — initialize a new project profile (writes `.mullet/`).
- **mullet setup-claude** — idempotent Claude Code integration setup.
- **mullet config** — read/write user + project config keys (get/set/list/edit/reset/budget).
- **mullet preset** — manage routing presets.
- **mullet budget** — token spend tracking and limits.
- **mullet routing** — routing matrix and provider selection info.
- **mullet search** — top-level knowledge search across pools.
- **mullet reindex** — reindex knowledge pools.
- **mullet pool** — knowledge pool management.
- **mullet ingest** — ingest documents into knowledge pools.
- **mullet graph** — knowledge graph operations.
- **mullet logs** — view system logs.
- **mullet history** — session history.
- **mullet stats** — usage statistics.
- **mullet models** — available models and info.
- **mullet cost** — cost estimation and tracking.
- **mullet explain** — explain routing decisions.
- **mullet completion** — shell completion setup.
- **mullet override** — temporarily override routing decisions.
- **mullet --install-help** — install-specific help text.
- **mullet --version** — print the installed Mullet version.

## Server Surface

`mullet/server/` exposes two protocol surfaces:

- **mcp.py** + **mcp_handlers.py** + **mcp_knowledge.py** — stdio MCP server. 10 `mullet_*` orchestration tools (run, delegate, opencode, swarm, etc.) and 41 `trawl_*` knowledge tools (pools, search, graphs, ingest, etc.) for Claude Code and other MCP clients.
- **app.py** + **api.py** + **websocket.py** — OpenAI-compatible REST + WebSocket API (`mullet serve`). SSE streaming, CORS, auth, rate limiting.

## Data Flow

```
1. User input → CLI parser
2. Router analyzes complexity → selects tier + provider
3. System prompt assembled: base + model guidance + knowledge + skills + memory
4. AgentRunner loop:
   a. Send messages to LLM
   b. If tool_calls: dispatch each, append results, check budget, loop
   c. If no tool_calls: return final response
5. Results: print, export, track cost
```

### Agent Enhancements
- **prompt_cache.py** — Anthropic prompt caching (~75% cost savings).
- **reasoning.py** — Thinking block extraction (DeepSeek-R1, Claude, QwQ).
- **redact.py** — Secret redaction (20+ API key patterns, DB connstrings).
- **verification.py** — Standardized BUILD/TEST/LINT checks for orchestration.

## Dependencies

Required: openai, anthropic, pyyaml, aiohttp, chromadb, sentence-transformers, networkx
Optional: boto3 (Bedrock)

## Testing

17,700+ tests (mostly unit with mocked providers; integration tests require LM Studio on localhost:1234).

### Agent Intelligence
- **Project detection** — auto-detects language, framework, test command from directory contents.
- **Context references** — `@file:path`, `@diff`, `@staged`, `@folder`, `@git:N`, `@url:url` syntax.
- **Injection guard** — scans external content for prompt injection patterns before injection.
- **Reasoning extraction** — strips thinking blocks from DeepSeek-R1, Claude, QwQ outputs.
