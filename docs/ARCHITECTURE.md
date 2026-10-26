# Mullet Architecture

> Current state of the system — not aspirational, what's actually built.

## Overview

Mullet is a Python monolith (about 84K lines in about 370 modules, about 18,000 tests). Its main job is `mullet job`: large coding jobs that run unattended in a sandboxed clone, with every step checked by gates Mullet runs itself (see Jobs below). It also routes single tasks to the cheapest capable model, serves the Trawl knowledge engine over MCP, and keeps an OpenAI-compatible API. It installs from GitHub with `scripts/install.sh` (never `pip install mullet`: that PyPI package is someone else's).

The diagram below is the legacy single-task path (`mullet "<task>"`, the API and the delegation tools). It runs unless `MULLET_DELEGATION=off` is set (the default is on), and jobs do not use it. Jobs have their own path, described under Jobs.

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

### Providers (`mullet/core/providers/`) — 9 backends
- **openai_compat.py** — Covers LM Studio, Ollama, vLLM, OpenRouter, Kimi, OpenAI. Retry with exponential backoff on 429/5xx.
- **anthropic.py** — Native Anthropic with prompt caching (~75% savings), tool_result batching, streaming.
- **gemini.py** — Native Google Gemini via google-genai SDK. Tool calling via FunctionDeclaration.
- **bedrock.py** — AWS Bedrock Converse API.
- **ollama.py** — Native Ollama integration.
- **claude_cli.py**: Runs the user's own `claude` command as a provider. Mullet never reads Claude's credentials; Anthropic models through Mullet's own providers need `ANTHROPIC_API_KEY`.
- **openrouter.py** — OpenRouter meta-provider.
- **kimi.py** — Moonshot AI Kimi K2 provider. Used by swarm orchestration and default STANDARD tier.
- **kimi_async.py** — Async facade over KimiProvider via `asyncio.to_thread`. Used by SwarmDispatcher.
- **model_info.py** — Model database: context windows, pricing, capabilities.
- **model_registry.py** — Dynamic model registry with 24h cache; discovers available models from Anthropic/Kimi/Gemini APIs. `get_best_model()` used by mcp_handlers for dynamic model selection.
- **health.py** — Exponential backoff per error type (rate_limit=30s, connection=120s, auth=3600s).

### Router (`mullet/core/router/`)
- **complexity.py** — Signal extraction + scoring. Lexical (keywords) + structural (subtasks, deps, domain) + context (failures, chain depth, plan complexity) → score → tier.
- **decision.py** — Routes to optimal provider. Health-filtered, budget-gated, tier-mapped. Auto-downgrades when budget blocks premium.
- **budget.py** — Token spend tracking with dollar limits. Per-task, per-session, per-day. Gated jobs do not use it; their limits are in `mullet/jobs/budget.py`.
- **fallback.py** — FallbackManager with tried-provider tracking and turn-scoped restoration.
- **resolver.py** — Bridges routing decisions to actual Provider instances.

### Presets (`mullet/core/presets.py`)

Named routing policies (allowed provider sets, speed mode, opencode model, native-only flag) persisted to `presets.yaml` in the config directory (`~/.config/mullet` by default). Built-ins (`balanced-mix`, `local-only`, `claude-only`, `cheap-cloud`) are always merged from code on load. Enforced in the MCP handlers before model selection.

### Pre-flight routing pipeline (added 2026-04-28)

Routing now uses a 3-stage pipeline before any paid model call:

1. **`mullet/core/router/preflight.py`**: calls a near-free LLM (local LM Studio first, then `gemini-2.5-flash`, then heuristic fallback) to estimate input tokens, expected output tokens, and complexity tier. Results are cached to `preflight.jsonl` in the cache directory (`mullet paths cache`; 7-day TTL, atomic writes).
2. **`mullet/core/router/routing_matrix.py`** — `select_kimi_model()`, `select_gemini_model()`, and `select_local_model()` implement a 3-axis cheapest-viable-model selector. `select_local_model()` is adaptive: it classifies whatever models LM Studio exposes by parsing model ID strings (param count, coder/vision/reasoning flags), so it works on any machine.
3. **Fallback chain** — `_best_local()` → `_best_kimi()` → `_best_gemini()` in `ModelRegistry`. Local is attempted first for LOW/STANDARD tasks on economy/balanced speed mode.

`KNOWN_PRICING`, `KNOWN_CACHE_HIT_PRICING`, and `KNOWN_DEPRECATIONS` in `model_registry.py` hold corrected official pricing. All `moonshot-v1-*` models removed.

### Tools (`mullet/core/tools/`) — 24 tools

`mullet --tools` prints the full list. Highlights:

| Toolset | Tools |
|---|---|
| terminal | terminal (cwd tracking, output limits) |
| file | file_ops (read/write/patch/list/search), patch (unified diffs, fuzzy matching) |
| web | web_search (DuckDuckGo), fetch_page |
| browser | browse (read, links, forms, extract, CSS selectors) |
| code | code_exec (subprocess in a temp directory with a timeout; no OS sandbox), python_repl (persistent state) |
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

Which toolsets a provider gets is set by `delegation.toolsets`: Claude's own providers get all of them, any other provider gets a safe subset unless you list more. See [CONFIGURATION.md](CONFIGURATION.md#delegationtoolsets).

### Knowledge — Trawl (`mullet/knowledge/` + `mullet/server/mcp_knowledge.py`)

Trawl knowledge engine. Over 40 MCP tools covering pools, search, graphs, ingest, sources, memory, workflows, agents, prompts. Pool directory layout: `config.yaml` + `index/` (ChromaDB + graph) + `documents/` + `.pool_metadata.json` (usage tracking).

- **engine/** — vendored knowledge engine: client/searcher/indexer/splitter/reranker/query_transform/extract.
- **engine/pool_metadata.py** — per-pool `.pool_metadata.json`: `created_at`, `last_indexed_at`, `last_searched_at`, `search_count`. Atomic writes, best-effort failure mode.
- **graph/extractor.py** — LLM-based entity/relationship extraction. Circuit breaker + garbage-response detection (SPECIAL_* spam, low token diversity) to bail early on broken models.
- **graph/store.py** — SQLite + NetworkX backed graph. Tracks `processed_chunk_ids` for incremental extraction.
- **`mullet/core/knowledge/predigest.py`** — Local LLM pre-processes heavy content → stores summary → cloud reasons over it.

### Orchestration (`mullet/core/orchestration/`)
- **swarm** (legacy single-task path; `mullet_swarm` refuses to run when `MULLET_DELEGATION=off`) — Kimi Agent Swarm: decompose goal → dispatch N tasks in parallel → synthesize. It returns text, can save a JSON report, and never edits project files. Components: `swarm.py` (orchestrator), `swarm_dispatcher.py` (async fan-out with semaphore concurrency), `llm_decomposer.py` (LLM-based task split), `static_decomposer.py` (file/item enumeration), `synthesizer.py` (final merge call), `swarm_types.py` (shared dataclasses).
- **shared_memory.py**: file-backed key/value store behind the `shared_mem` tool.
- **scheduler.py**: recurring tasks for `mullet schedule`.
- **autoresearch.py**: research pipeline object behind `mullet research`.

### Jobs (`mullet/jobs/`)

Long, unattended work runs as a `mullet job`: a planned set of nodes, each run by the user's own `claude -p` in a sandboxed private clone and checked by gates Mullet runs itself (G0 baseline, then real change, scope, static checks, build, tests against the baseline, test weakening, and a fresh review). A node's status comes from gate results only, never from words in a model's reply. The job hands back a branch and an evidence report. See [MULLET2.md](MULLET2.md).

- **planner.py**: task tree (at most three levels and 40 nodes).
- **clone.py**: private clone; the worker never sees its git directory.
- **sandbox.py**: picks a backend from `sandbox_backends/` (bubblewrap by default on Linux, or a hardened Docker or Podman container per command), no unsandboxed fallback. Jobs refuse to start on macOS for now.
- **gates.py**, **review.py**: gates G0 to G7.
- **ledger.py**: SQLite ledger; **report.py** builds the report from it, and **ladder_report.py** sums the ladder across jobs.
- **profile.py**: per-repo setup, checks and protected paths. The stored profile lives in the `projects` folder of the config directory.
- **runner.py**: detached runner with resume; **notify.py**: desktop and Slack messages when a job blocks, pauses or finishes.
- **ladder.py**: which worker attempts each step (free, local, Kimi, OpenRouter, Claude); **night.py**: night mode; **budget.py**: spending limits across jobs; **reconcile.py**: the provider check at job end.
- **machine.py**: the GPU lease and headroom waits; **reaper.py**: `mullet reap`.
- **executors/claude_code.py**: the user's own `claude -p`. **executors/opencode.py** runs opencode behind **executors/keyproxy.py**, which holds the provider key and meters each response's usage (**executors/usage_meter.py**).

Job data (clones, ledger, logs) lives in `jobs/` under the data directory, and profiles in the `projects` folder of the config directory. `MULLET_JOBS_DIR` and `MULLET_PROFILES_DIR` override them, and `mullet job` and `mullet ladder report` need the same values when jobs live elsewhere.

Not built: `mullet job bench` and a macOS sandbox.

### Skills (`mullet/core/skills/`) — 6 built-in (brainstorm, code-review, mullet-job, plan, pool-improve, session-review)
- **loader.py** — Discovers SKILL.md files across builtin/user/project scopes.
- **registry.py** — Trigger matching (slash commands + keywords + learned).
- **invoker.py** — Pipeline context, requirements, handoff artifacts.
- **learned.py** — User-created skills with confidence scoring.

### Server (`mullet/server/`)
- **app.py** — aiohttp with CORS, auth, rate limiting, error handling, graceful shutdown.
- **guard.py**: checks `Host` and `Origin` on every request and, in solo mode, the install's token (`serve-token`). See [CONFIGURATION.md](CONFIGURATION.md#mullet-serve).
- **api.py** — REST: /health, /v1/chat/completions (SSE), /api/v1/projects, search, memory.
- **websocket.py** — Real-time task execution with full tool parity.
- **mcp.py** + **mcp_handlers.py** — 10 mullet_* MCP tools + over 40 trawl_* knowledge tools for Claude Code integration.
- **admin.py** — User/project/access/usage CRUD.
- **auth.py** — PAT generation, verification, 0600 permissions.
- **db.py** — SQLite schema (6 tables), solo bootstrap.

### CLI (`mullet/cli/`)
- 40+ commands including serve, mcp, doctor, init, setup-claude, job, history, config, stats, completion, explain, login, hud, pool, ingest, search, reindex, evaluate, live-test, review-session, and others
- --verbose shows routing decisions, cost savings
- --export saves conversation to JSON or Markdown
- --tools, --skills for discovery
- Pipe support: `cat file.py | mullet "review this code"`

## Harness Layer (`mullet/harness/`)

> **Removed 2026-07.** The harness layer and its interactive TUI (`mullet shell`, `mullet cli`, `mullet web`, `mullet acp`, etc.) have been deleted from the codebase.
>
> Mullet is now **MCP-server-first**. The core runtime is mostly consumed through the MCP layer; bare `mullet` still starts a small REPL (`mullet/cli/repl.py`). The harness code is preserved in git history for reference.

## CLI Surface

Live subcommands:

- **mullet serve** — OpenAI-compatible REST + WebSocket API server.
- **mullet mcp** — stdio MCP server for Claude Code and other MCP clients.
- **mullet doctor**: read-only checks of Python, the `claude` CLI, git, the job sandbox (bubblewrap and `socat` on Linux), the Claude Code setup, providers and pools.
- **mullet init** — initialize a new project profile (writes `.mullet/`).
- **mullet setup-claude**: registers the MCP server and skills with Claude Code, adds hooks only with `--hooks`, and records every change so `--uninstall` removes exactly those.
- **mullet hook**: runs one of the Claude Code hooks in `mullet/hooks/` (`mullet hook list`).
- **mullet job**: plan, approve, run and merge gated jobs in a sandboxed clone.
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

- **mcp.py** + **mcp_handlers.py** + **mcp_knowledge.py** — stdio MCP server. 10 `mullet_*` orchestration tools (run, delegate, opencode, swarm, etc.) and over 40 `trawl_*` knowledge tools (pools, search, graphs, ingest, etc.) for Claude Code and other MCP clients.
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
- **prompt_cache.py** (`mullet/core/providers/`) — Anthropic prompt caching (~75% cost savings).
- **reasoning.py** — Thinking block extraction (DeepSeek-R1, Claude, QwQ).
- **redact.py** — Secret redaction (20+ API key patterns, DB connstrings).

## Dependencies

Required: openai, anthropic, pyyaml, aiohttp, chromadb, sentence-transformers, networkx
Optional: boto3 (Bedrock)

## Testing

About 16,900 tests, about 5 minutes in parallel (`python3 -m pytest tests -q -n 8 --timeout 300`). Tests run with no network, no provider keys and a throwaway home directory (mostly unit with mocked providers; live tests, which need a real model, are excluded by default).

### Agent Intelligence
- **Project detection** — auto-detects language, framework, test command from directory contents.
- **Context references** — `@file:path`, `@diff`, `@staged`, `@folder`, `@git:N`, `@url:url` syntax.
- **Injection guard** — scans external content for prompt injection patterns before injection.
- **Reasoning extraction** — strips thinking blocks from DeepSeek-R1, Claude, QwQ outputs.
