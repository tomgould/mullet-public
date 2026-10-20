# Mullet — AI coding orchestrator

> **Cloud in the front, local GPU in the back. Multi-LLM. Self-hosted. Beautiful.**

[![Python](https://img.shields.io/badge/python-%3E%3D%203.10-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-17%2C700%2B-brightgreen)](tests/)
[![License](https://img.shields.io/badge/license-BSL%201.1-blue)](LICENSE)


Mullet is an MCP-server-first AI coding orchestrator. It routes tasks to cheap/free
models (Kimi K2, local LLM, Gemini) and integrates with Claude Code via MCP. The
goal is to route the bulk of work to free or cheap models (local, Kimi, OpenRouter) so you're not burning a metered API budget or your Claude Code session's rate-limited quota on routine tasks.
It also exposes an OpenAI-compatible REST + WebSocket API via `mullet serve`.

### The 30-second pitch

- 🎯 **Multi-provider native** — Kimi K2.7-Code / K2.6, Claude (Fable 5 / Opus 4.8 / Sonnet 5 / Sonnet 4.6 / Haiku), GPT-5, Gemini 2.5, DeepSeek V4 (Flash/Pro), Qwen3-Coder-Next, GLM 5.1, MiniMax M3, Ollama, LM Studio, OpenRouter, Bedrock, Claude/Gemini CLI subscriptions. Smart routing picks the cheapest model that can handle the task.
- 🔌 **MCP-server-first** — `mullet mcp` exposes tools to Claude Code and any MCP client. `mullet serve` provides an OpenAI-compatible REST + WebSocket API.
- ⭐ **30+ subsystems** — skills, plugins, hooks, knowledge engine (semantic search + graph), swarm, voice transcription, multimodal images, PDF reading, browser automation, Prometheus metrics, secret redaction, auto-compaction.

---

## Quick start

```bash
# Install from source (needs Python 3.10+ and git)
git clone https://github.com/tomgould/mullet-public.git && cd mullet-public && pip install -e .

# Set your Kimi key (or skip and use a local LLM via Ollama / LM Studio)
echo "KIMI_API_KEY=sk-..." > .env

# Jump straight in
mullet doctor                    # verify your install
mullet preset list               # see the 4 built-in presets
mullet preset use balanced-mix   # activate cost-aware routing
```

Or, with an optional extra such as Gemini:

```bash
git clone https://github.com/tomgould/mullet-public.git && cd mullet-public
pip install -e ".[gemini]"       # optional extras: gemini, bedrock, postgres, all
mullet doctor                    # verify your install
mullet preset list               # see the 4 built-in presets
```

> Mullet reads `.env` from the repo root via `python-dotenv`. There is no need
> to add API keys to your shell rc file. Run `mullet --install-help` for the full
> menu of optional extras (`gemini`, `bedrock`, `postgres`, `all`).

---

## What's Inside

17,700+ tests. 30+ subsystems. Pure Python, no JS runtime. Here's the full taxonomy:

### Core Runtime

| Capability | Detail |
|:-----------|:-------|
| **AgentRunner** | Async agent loop with tool execution, streaming, and fallback retry |
| **Streaming** | Token-by-token via `assistant_chunk` events; tool-call deltas; `thinking_delta` for Claude reasoning |
| **Auto-compaction** | Summarises older messages when context window exceeds 70% |
| **Auto-context** | Trawl-backed `AutoContextResolver` injects relevant code chunks |
| **IntentGate** | Keyword classifier biases routing |
| **Named subagents** | Sisyphus, Hephaestus, Oracle, Librarian — primary + fallback provider chains |
| **Memory auto-discovery** | Walks up to git root collecting `CLAUDE.md` / `AGENTS.md` / `MEMORY.md` |
| **Multimodal** | `ImageAttachment` + `MultimodalMessageBuilder` via `--image PATH` / `--image-url URL` |

### Coordinator / Workflows

| Capability | Detail |
|:-----------|:-------|
| **Swarm** | `mullet_swarm` MCP tool — fan out a goal across parallel Kimi K2 calls, synthesize results, configurable concurrency |
| **Workflows** | Durable SQLite 3-table state (`WorkflowRun` + `WorkflowEvent` + `WorkflowMessage`); `replay_events()` for corruption recovery |
| **Auto-swarm** | Detects parallelisable tasks and routes to swarm automatically |

### CLI commands

| Command | What it does |
|:--------|:-------------|
| `mullet mcp` | MCP server for Claude Code integration (stdio transport) |
| `mullet serve` | OpenAI-compatible REST + WebSocket API server |
| `mullet preset` | Manage provider presets (list / use / show / add / remove) |
| `mullet budget` | Show per-provider balance + spend cap status |
| `mullet routing` | Show provider routing outcomes, quality history, insights |
| `mullet doctor` | System health check (providers, pools, config, hooks) |
| `mullet doctor --fix` | Auto-remediate fixable issues |
| `mullet init` | Generate `mullet.yaml` config for current project |
| `mullet setup-claude` | Install recommended Claude Code global config for Mullet |
| `mullet search` | Semantic search across indexed knowledge pools |
| `mullet pool` | Manage knowledge pools (list, audit, prune, tune, validate) |
| `mullet reindex` | Rebuild pool index |
| `mullet ingest` | Ingest files/URLs into a knowledge pool |
| `mullet graph` | Knowledge graph operations (build, stats, export) |
| `mullet logs` | View LLM interaction history (tail/stats/suggest) |
| `mullet history` | Show recent sessions or search past conversations |
| `mullet stats` | Show usage statistics from session history |
| `mullet models` | List available models and pricing by provider |
| `mullet cost` | Show model pricing across tiers |
| `mullet explain` | Show complexity analysis for a task (no LLM call) |
| `mullet completion` | Generate shell completion scripts (bash/zsh) |
| `mullet override` | Toggle budget override (mullet override on/off) |

### Tools (25+ built-in)

| Category | Tools |
|:---------|:------|
| **Filesystem** | `Read`, `Write`, `Edit`, `MultiEdit`, `NotebookEdit`, `Glob`, `Grep` |
| **Shell** | `Bash`, `Terminal`, `RunBash`, `RunTests`, `RunPython`, `code_exec` |
| **Code intelligence** | `View`, `ViewCode`, `SearchCode`, `trawl_search_code`, `trawl_find_symbol` |
| **Web / browser** | `WebSearch`, `WebFetch`, `WebScrape`, `Browse` (Playwright multi-session: goto/extract/click/fill/screenshot/links/text) |
| **Agent / orchestration** | `TodoWrite`, `Task`, `SwarmDispatch`, `MCPCall` |
| **Multimodal** | `ImageView`, `Image` (attach + process), `ReadPDF` (pypdf-backed), `VoiceTranscribe`, `Clipboard`, `MetricsQuery` |

### REPL commands

The bare `mullet` REPL supports:

`/skill-name` · `clear` · `history` · `help` · `exit`

Type `/` followed by a skill name to invoke it (e.g. `/code-review`).

### Hooks

| Hook | Detail |
|:-----|:-------|
| **5 tool guards** | `writeExistingFileGuard`, `hashlineReadEnhancer`, `directoryAgentsInjector`, `bashFileReadGuard`, `toolOutputTruncator` |
| **Secret redaction** | `PreToolUse` guard with 12 patterns: AWS, GitHub, OpenAI, Anthropic, JWT, Slack, private keys, generic password |
| **IntentGate** | Message hook keyword classifier biases routing |
| **Memory injection** | `AGENTS.md` / `CLAUDE.md` auto-injected into prompts |
| **Auto-compaction** | Transform hook summarises when context > 70% |
| **Per-session fallback** | Event hook with cooldown map, error classifier, retry signal extraction |

### Sessions

| Capability | Detail |
|:-----------|:-------|
| **Conversation search** | Full-text grep across `messages.jsonl` with role filter + recency scoring |
| **Tags + Pin** | `/tag pin/unpin/add/remove`, `list_pinned`, `list_by_tag` |
| **Undo** | Rewind to any prior point, restoring previous state |
| **Branch** | `fork_session` at any index with `parent_session_id` genealogy |
| **Storage** | Append-only event log with `project_hash` isolation at `~/.config/mullet/sessions/{hash}-{cwd}/` |
| **Resume** | Interactive picker via `/resume` |

### Servers

| Server | Detail |
|:-------|:-------|
| **MCP server** | Registers as Claude Code MCP server — 10 Mullet tools + 42 Trawl knowledge tools |
| **API server** | OpenAI-compatible `/v1/chat/completions`, `/v1/models`, `/health`, `/metrics` via `mullet serve` |

### Polish

- 🧠 **Thinking blocks** — Anthropic reasoning stream
- 🔄 **Auto-update checker** — PyPI version check on `mullet doctor`; `maybe_print_update_notice()` warns when a newer version is available
- 💬 **FriendlyError** — actionable messages with install hints and docs links
- 🎨 **ASCII banner** — startup banner with version info via `mullet --banner`

### Security

- 🔒 **Secret scanner** — 12 regex patterns covering AWS keys, GitHub tokens, OpenAI/Anthropic keys, JWT, Slack webhooks, private keys, generic passwords
- 🛡️ **Redactor** — automatic redaction in tool inputs, outputs, and session exports
- 🚫 **PreToolUse guard** — blocks or masks secrets before tool execution
- 🔐 **Permissions** — `default` / `auto` / `plan` modes; `path_rules`; `denied_commands`; `denied_tools`; `allowed_tools` whitelist
- 🔍 **Diff-risk classifier** — deterministic, zero-LLM-cost heuristic scans every `mullet_opencode` diff for risky changes (literal edits, quote flattening, new control flow, tampered assertions) delegates won't self-report

### Observability

- 📊 **MetricsRegistry** — counters, gauges, histograms with dimensional tags (thread-safe)
- 📝 **LogAggregator** — structured log collection, filtering by level/source/since, FIFO eviction
- 📈 **Prometheus export** — `/metrics` endpoint for external monitoring stacks
- 📡 **Opt-in telemetry** — privacy-first session analytics, zero PII / paths / code, jcode-style disclosure banner

---

## Why Not X?

| Capability | **Mullet** | aider | opencode | Claude Code |
|:-----------|:----------:|:-----:|:--------:|:-----------:|
| Self-hosted, no cloud account required | ✅ | ✅ | ✅ | ❌ |
| Multi-provider native (Kimi/Claude/OpenAI/Gemini/Local) | ✅ | ✅ | ✅ | partial |
| Hooks (PreToolUse / PostToolUse / Message / Event / Transform / Params) | ✅ | ❌ | ❌ | ✅ |
| Swarm / parallel delegation | ✅ | ❌ | ❌ | partial |
| Voice input (whisper.cpp / OpenAI / Groq) | ✅ | ❌ | ❌ | ❌ |
| Knowledge engine (semantic search + graph) | ✅ | ❌ | ❌ | ❌ |
| Multimodal (image + PDF) | ✅ | partial | partial | ✅ |
| Browser automation (Playwright) | ✅ | ❌ | ❌ | ❌ |
| One-line install | ✅ | ✅ | ✅ | ✅ |
| Skills marketplace | ✅ | ❌ | ❌ | ✅ |
| Plugin manifest (`plugin.json`) | ✅ | ❌ | ❌ | ✅ |
| MCP client (stdio + HTTP/SSE) | ✅ | ❌ | partial | ✅ |
| Pure Python (no JS runtime) | ✅ | ✅ | ❌ | ❌ |
| Token-cost router with budget guard | ✅ | partial | ❌ | ❌ |
| Secret redaction in tool args + exports | ✅ | ❌ | ❌ | partial |

---

## Architecture

### Routing Tiers

Every task is scored by `mullet/core/router/complexity.py` using signals — file count,
dependency depth, task type, context size — and routed to the cheapest model that can
handle it. Mullet uses a **3-stage routing pipeline**:

1. **Pre-flight analysis** (`mullet/core/router/preflight.py`) — a near-free LLM call (local LM Studio → Gemini Flash → heuristic fallback) estimates input tokens, expected output tokens, and complexity *before* committing to a paid model.
2. **3-D routing matrix** (`mullet/core/router/routing_matrix.py`) — selects the cheapest model satisfying all three constraints across Kimi, Gemini, and local providers.
3. **Fallback chain** — if the selected model is unavailable, escalates automatically.

| Tier | Default model | Cost (input/output per 1M) | Use |
|:-----|:-------------|:--------------------------|:----|
| **LOW** | Local LLM — smallest available (e.g. 2B) | Free | Simple lookups, analysis, short edits |
| **STANDARD** | Local LLM 7B → OpenRouter `deepseek-v4-flash` ($0.098/$0.196) → Kimi K2.7-Code | Free or ~$0.10/$0.20 | Refactoring, unit tests, isolated changes |
| **ADVANCED** | OpenRouter `deepseek-v4-flash` ($0.098/$0.196) → Kimi K2.7-Code | $0.098/$0.196 | Cross-file reasoning, integration tasks |
| **CRITICAL** | Kimi K2.7-Code ($0.95/$4.00) | $0.95/$4.00 | Architecture, complex debugging |
| **PREMIUM** | Opus 4.8 (primary frontier model; Fable 5 deactivated by Anthropic) | $5.00/$25.00 | Orchestrator only |

Reasoning tasks route to `deepseek/deepseek-v4-pro` (OpenRouter) or `kimi-k2.7-code` (built-in
thinking enabled by default) — the `kimi-k2-thinking` series was discontinued 2026-05-25. Local models are
classified adaptively from LM Studio's live model list — no hardcoded names required.

A **1.1× context safety margin** is applied before model selection. If the padded
estimate exceeds the target model's context window, the router escalates automatically.

### Fallback Chain

`kimi → openrouter → gemini → haiku → sonnet`

Opus is excluded from automatic fallback. It only runs when explicitly authorised
(`mullet override on` or `PREMIUM` tier).

### Orchestration Modes

| Mode | Behaviour | Trigger |
|:-----|:----------|:--------|
| `standard` | Single agent call | Default |
| `ultrawork` | Decomposes task, spawns up to 3 parallel workers via `delegate_batch`, merges results | `#ultrawork` or `--mode ultrawork` |
| `ralph` | Wraps ultrawork in a persistence loop — re-runs until all PRD user stories pass, up to 10 iterations | `#ralph` |
| `autopilot` | Full lifecycle: plan → ralph → verify. Runs unsupervised. | `#autopilot` |
| `team` | N coordinated agents with non-overlapping file ownership boundaries | `#team` |
| `eco` | Modifier: forces all calls to local LLM for the session | `#eco` |

Magic keywords (`#ralph`, `#ultrawork`, etc.) in any prompt trigger the mode inline
without changing global config.

---

## Providers & Cost Reference

| Provider | Model | Input $/M | Output $/M | Context | Notes |
|:---------|:------|----------:|----------:|:--------|:------|
| Local LLM | Any LM Studio / Ollama model | Free | Free | Hardware-limited | Auto-classified by param count |
| Moonshot | Kimi K2.7-Code | $0.95 | $4.00 | 256K | **Default** — code-first, ~30% fewer reasoning tokens (2026-06-12) |
| Moonshot | Kimi K2.6 | $0.95 | $4.00 | 262K | Premium quality, general-purpose |
| Moonshot | Kimi K2.5 | $0.60 | $3.00 | 262K | Standard |
| Moonshot | Kimi K2-0905-preview | $0.60 | $2.50 | 262K | Deprecated 2026-05-25 |
| Moonshot | Kimi K2-0711-preview | $0.60 | $2.50 | 131K | Deprecated 2026-05-25 |
| Moonshot | Kimi K2-turbo-preview | $1.15 | $8.00 | 262K | Speed premium |
| Moonshot | Kimi K2-thinking | $0.60 | $2.50 | 262K | Reasoning, cost-optimised |
| Moonshot | Kimi K2-thinking-turbo | $1.15 | $8.00 | 262K | Reasoning + speed |
| Google | Gemini 2.5 Flash Lite | $0.10 | $0.40 | 1M | Ultra-cheap fallback |
| Google | Gemini 2.5 Flash | $0.30 | $2.50 | 1M | Standard cloud |
| Google | Gemini 2.5 Pro | $2.00 | $12.00 | 1M | Premium cloud |
| OpenAI | GPT-4o | $2.50 | $10.00 | 128K | Profile available |
| Anthropic | Claude Haiku 4.5 | $0.80 | $4.00 | 200K | Tool use only |
| Anthropic | Claude Sonnet 4.6 | $3.00 | $15.00 | 1M | Orchestration |
| Anthropic | Claude Sonnet 5 | $2.00 | $10.00 | 1M | ADVANCED/CRITICAL default (2026-07-10) |
| Anthropic | Claude Opus 4.8 | $5.00 | $25.00 | 1M | Premium / orchestrator (2026-05-28) |
| Anthropic | Claude Fable 5 | $10.00 | $50.00 | 1M | Mythos-class — deactivated by Anthropic (catalog only, not selectable) |
| OpenRouter | `deepseek/deepseek-v4-flash` | $0.098 | $0.196 | 1M | Default workhorse — cheapest, newest |
| OpenRouter | `deepseek/deepseek-v4-pro` | $0.435 | $0.87 | 1M | Advanced tier — large MoE |
| OpenRouter | `z-ai/glm-5.1` | $0.98 | $3.08 | 203K | Strong coder |
| OpenRouter | `minimax/minimax-m3` | $0.30 | $1.20 | 1M | Multimodal, 1M ctx (launch promo) |
| OpenRouter | `qwen/qwen3-coder-next` | $1.00 | $3.20 | 203K | 80B/3B MoE coder |
| OpenRouter | `deepseek/deepseek-v3.2-exp` | $0.27 | $0.41 | 164K | Prior default (superseded by V4 Flash) |
| AWS Bedrock | `anthropic.claude-sonnet-4-6` | $3.00 | $15.00 | 1M | Enterprise |
| Claude CLI | Sonnet | included | included | — | Max subscription |
| Gemini CLI | Gemini 2.5 Flash | included | included | — | Subscription |

### Real Cost: Typical 10K-token Coding Task

| Model | Cost | vs Kimi |
|:------|-----:|:--------|
| Local LLM | $0.00 | — |
| **Kimi K2** | **$0.031** | **1×** |
| Sonnet 4.6 | $0.180 | 6× more |
| Opus 4.8 | $0.300 | 10× more |

Mullet routes >90% of coding tasks to Kimi K2, OpenRouter, or local LLMs. Anthropic tokens are
reserved for orchestration and final commit review.

> **Note:** Alibaba Cloud / DashScope is configured but not yet available on international accounts — pending future activation.

### Budget Guard

```python
from mullet.hud.guard import check_budget
r = check_budget()
# → {'status': 'GO', '5h_pressure': 0.10, '7d_pressure': 0.11}
```

Three states: **GO** (green, <50% pressure) → **THROTTLE** (yellow, >50%, downgrade
tier) → **PAUSE** (red, >90%, block calls).

### Provider Presets

A **preset** is a named bundle of `{allowed_providers, speed_mode, opencode_model, native_only}` that controls how Claude Code MCP delegations (`mullet_run`, `mullet_opencode`) route work. Switching presets changes which providers are eligible and how aggressively Mullet spends budget — without editing config files by hand.

| Preset | Providers | Speed | opencode model | Behaviour |
|:---------|:----------|:------|:---------------|:----------|
| `balanced-mix` (default) | all | `balanced` | `moonshot/kimi-k2.6` | Cost-aware routing: local for simple tasks, cloud for hard ones. |
| `local-only` | `local` | `economy` | `lmstudio/qwen2.5-coder-14b-instruct` | Everything on local LM Studio coders. Zero cloud cost. |
| `claude-only` | `sonnet`, `haiku`, `opus-4-8` | `balanced` | — | Claude Code native only — never delegates to opencode or proxies Anthropic. |
| `cheap-cloud` | `kimi`, `gemini` | `balanced` | `moonshot/kimi-k2.6` | OpenRouter/Kimi/Gemini cloud. No local, no Anthropic quota. |

Manage presets from the CLI:

```bash
mullet preset list                # show all; * marks the active one
mullet preset use <name>          # switch active preset
mullet preset show <name>         # print preset as YAML
mullet preset add --name my-preset --label "My preset" \
  --providers local,kimi --speed balanced \
  --opencode-model moonshot/kimi-k2.6
mullet preset remove <name>       # built-ins require --force
```

From the CLI:

```bash
mullet preset use cheap-cloud     # switch active preset
```

Each new Claude Code session reports the active preset via the `SessionStart` hook, and Claude will ask which preset to use when the default doesn't fit the task.

---

## MCP Tools & Claude Code Integration

Mullet registers as a Claude Code MCP server. Tools appear automatically in Claude
Code once `mullet setup-claude` has run.

### Mullet MCP Tools (10 total)

| Tool | What it does | Key params |
|:-----|:-------------|:-----------|
| `mullet_run` | Route task to cheapest capable model | `task`, `tier`, `files`, `apply`, `providers`, `mode` |
| `mullet_swarm` | Fan-out across parallel Kimi K2 workers | `goal`, `items`, `concurrency`, `output_shape` |
| `mullet_delegate` | Child agent with explicit provider control | `task`, `provider`, `tier` |
| `mullet_opencode` | opencode — native filesystem access, returns git diff | `task`, `apply`, `commit_message`, `timeout` |
| `mullet_search` | Semantic search via Trawl knowledge engine | `query` |
| `mullet_remember` | Store memory in project knowledge base | `memory` |
| `mullet_analyze` | Complexity score + routing decision for a task | `task` |
| `mullet_status` | Provider health, resource info | — |
| `mullet_providers` | Live health check per provider | — |
| `mullet_budget` | Token spend and budget limits | — |

Plus **42 Trawl knowledge tools** covering pool management, graph search, and code indexing.

### Typical Claude Code Workflow

```python
# Delegate to Kimi — paste files in, get EDIT blocks back, apply to disk
mullet_run(
    task="refactor auth module to use OAuth2",
    files=["src/auth.py", "src/tokens.py"],
    apply=True,   # writes EDIT blocks directly to disk
    tier="STANDARD",
)

# Or give Kimi full filesystem access via opencode
mullet_opencode(task="add type hints to all public APIs", apply=False)
# → returns {"response": "...", "diff": "--- a/src/auth.py\n+++ ..."}
# Review the diff, then commit:
mullet_opencode(task="add type hints to all public APIs", apply=True,
                commit_message="feat: add type hints to public APIs")
```

---

## opencode Integration

[opencode](https://github.com/sst/opencode) (MIT) is an open-source terminal coding
agent. Mullet drives it via the `opencode run` CLI (one-shot, non-interactive — not
the ACP/JSON-RPC session protocol) for cloud dispatch; local LM Studio-resolved
models route through a lightweight local agent instead, falling back to the opencode
CLI only if that can't run. Zero Anthropic tokens for the delegated work.

### mullet_run vs mullet_opencode

| | `mullet_run` / Kimi API | `mullet_opencode` / opencode |
|:--|:------------------------|:-----------------------------|
| Filesystem | None — files must be pasted into the prompt | Full native read/write/exec |
| Context feeding | Manual: paste relevant file sections | Autonomous: agent reads what it needs |
| Multi-file edits | Manual coordination | Autonomous navigation |
| Test runs | No | Yes — runs `pytest`, `npm test`, etc. |
| Anthropic tokens | 0 | 0 |

When opencode is not installed, `mullet_opencode` falls back to `mullet_run/Kimi`
with a warning.

### Setup

```bash
npm i -g opencode-ai@latest
export KIMI_API_KEY=sk-...        # Moonshot API key (moonshot.cn)
mullet setup-claude               # writes ~/.config/opencode/opencode.json
```

Generated `~/.config/opencode/opencode.json`:

```json
{
  "model": "moonshot/kimi-k2.7-code",
  "provider": {
    "moonshot": {
      "api": "https://api.moonshot.ai/v1",
      "env": ["KIMI_API_KEY"],
      "models": {
        "kimi-k2.7-code": {
          "tool_call": true,
          "limit": { "context": 262144, "output": 32768 },
          "cost": { "input": 0.95, "output": 4.0 }
        }
      }
    }
  }
}
```

`AGENTS.md` in the project root scopes the agent's identity, available tools, and
code standards for each opencode session.

See [docs/opencode_integration.md](docs/opencode_integration.md) for the full guide
and troubleshooting.

---

## Terminal HUD

Mullet maintains a live status bar showing budget pressure, active agents, and tool
attribution:

```
[Sonnet 4.6 | mullet-dev | master]
Ctx 28% ████░░░░░░  •  5h: 10%  •  7d: 11%  •  Resets: 3h 52m
Agent: moonshot/kimi-k2.7-code  Refactoring auth module...
MCP:1 | Hooks:4 | CLAUDE.md:2
```

| Field | Description |
|:------|:------------|
| `Ctx` | Context window usage % for the current leader model |
| `5h` / `7d` | Spend pressure against rolling 5-hour and 7-day budget caps |
| `Resets` | Time until the nearest budget window resets |
| `Agent:` | One live row per active agent, labelled `provider/model`. Appears on `add_agent`, drops on `remove_agent`. |
| Tool pills | In-flight MCP call counters, each carrying a `provider/model` badge |

---

## Getting Started

> **New to Mullet?** See [docs/INSTALL.md](docs/INSTALL.md) for the full
> onboarding guide.

### Prerequisites

- Python 3.10+
- Node.js 18+ (only if you want opencode)
- LM Studio or Ollama — optional, for free local LLM fallback
- [Moonshot API key](https://platform.moonshot.cn) — Kimi K2, $0.60/M tokens
- Anthropic API key — only required for direct Anthropic routing (pay-per-token; orchestrator tier). Not needed when using Mullet via Claude Code, which handles Anthropic access natively through your subscription.

### Install

```bash
# Clone and install
git clone https://github.com/tomgould/mullet-public.git && cd mullet-public && pip install -e .
```

Step by step, with the optional extras:

```bash
git clone https://github.com/tomgould/mullet-public.git
cd mullet-public
pip install -e .

# Optional extras
pip install -e ".[gemini]"    # Google Gemini
pip install -e ".[bedrock]"   # AWS Bedrock
pip install -e ".[postgres]"  # PostgreSQL backend
pip install -e ".[all]"       # Everything

# Bootstrap Claude Code integration (MCP server, hooks, opencode)
mullet setup-claude
```

`setup-claude` is idempotent and handles:
1. Writes `~/.mcp.json` registering Mullet + Trawl MCP servers
2. Installs global Claude Code hooks into `~/.claude/settings.json` — fires in **any project directory**, not just this repo: delegation reminder (UserPromptSubmit), provider/tier rewriter (PreToolUse on mullet_run), observation capture (PostToolUse)
3. Appends Mullet delegation rules to `~/.claude/CLAUDE.md`
4. Installs skills (`/delegate`, `/commit`, `/review`) and agents (`patch-writer`, `test-runner`, `verifier`) into `~/.claude/`
5. Installs opencode via npm and writes `~/.config/opencode/opencode.json`

> **Note:** The hook scripts live in `scripts/hooks/` inside the source tree.
> `.claude/settings.json` references them via Claude Code's `${CLAUDE_PROJECT_DIR}`
> variable, so the path itself is portable — but a regular `pip install mullet`
> still won't ship the `scripts/` directory, so an editable install
> (`pip install -e .` after cloning) is required either way. Run `mullet doctor`
> to verify hooks are active.

```bash
export KIMI_API_KEY="sk-..."        # Moonshot (primary delegation target)
export OPENROUTER_API_KEY="sk-..."  # OpenRouter — 8 curated models, OpenAI-compatible
export ANTHROPIC_API_KEY="sk-..."   # Direct Anthropic routing only (pay-per-token); skip if using Claude Code MCP
```

### Quick Test

```bash
# Check budget status
python3 -c "from mullet.hud.guard import check_budget; r = check_budget(); print(r.status.value, r.message)"

# Run a task (auto-routes to Kimi K2 by default)
mullet "explain this codebase in one paragraph"

# Force Kimi explicitly — watch the HUD agent row appear
mullet --provider kimi "add docstrings to auth.py"

# List available models and pricing
mullet models
```

---

## Orchestration Modes In Practice

```bash
# Standard: single Kimi call
mullet "add a retry decorator to the API client"

# Ultrawork: parallel decomposition
mullet "#ultrawork add error handling to every endpoint in api/"

# Ralph: persistence loop — keeps going until tests pass
mullet "#ralph implement OAuth2 login flow with full test coverage"

# Autopilot: plan, execute, verify unsupervised
mullet "#autopilot migrate database from SQLite to PostgreSQL"

# Team: coordinated agents with file ownership
mullet "#team refactor the entire auth subsystem"

# Eco: local LLM only, zero cloud cost
mullet "#eco add docstrings to all files in utils/"

# Swarm: parallel fan-out (via MCP)
mullet_swarm(goal="summarise each file in src/", items=[...])
```

---

## Agent Profiles

Pre-configured YAML templates in `mullet/agents/`:

- `architecture-analyst` — system design review
- `code-reviewer` — PR review with Must Fix / Should Fix / Optional findings
- `documentation-writer` — docstrings, READMEs, API docs
- `performance-analyzer` — bottleneck identification
- `security-reviewer` — OWASP-aligned vulnerability scan
- `test-generator` — unit and integration test generation
- `verifier` — correctness checks and test validation

---

## API Server

```bash
mullet serve
# → http://localhost:3000

# OpenAI-compatible chat completions:
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}]}'

# Health, models, Prometheus metrics:
curl http://localhost:3000/health
curl http://localhost:3000/v1/models
curl http://localhost:3000/metrics
```

---

## Configuration

`mullet.yaml` in project root or `~/.mullet/config.yaml`:

```yaml
providers:
  local:
    type: openai_compatible
    base_url: http://localhost:1234/v1
    models:
      default: mistral-nemo-instruct-2407

  cheap_cloud:
    type: openai_compatible
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
    models:
      default: google/gemini-2.5-flash

routing:
  default_tier: STANDARD
  budget:
    daily_max_dollars: 10.00
```

### Knowledge Pools Directory

Mullet stores indexed code/document pools (vector + graph DBs) under a single
pools directory. Resolution order:

1. `TRAWL_POOLS_DIR` env var, if set
2. `~/.mullet/pools/` (default — created automatically by the installer)

To keep your pools on a separate drive or in an existing location, you have
two options:

```bash
# Option A — point Mullet at an existing pools dir via env var
export TRAWL_POOLS_DIR=/data/mullet-pools

# Option B — symlink the default path at your existing pools dir
ln -sfn /data/mullet-pools ~/.mullet/pools
```

Pool data is never tracked in the Mullet repo — `mullet/knowledge/pools/` is
gitignored, and the runtime never reads from inside the source tree.

---

## Development

```bash
pip install -e ".[dev]"
pytest                            # 17,700+ tests
pytest tests/integration/ -v      # integration tests (needs LM Studio)
mullet doctor                     # system health check
```

---

## Documentation

- **[Quick Start](docs/QUICKSTART.md)** — zero to productive in 5 minutes
- **[opencode Integration](docs/opencode_integration.md)** — full dispatch guide with troubleshooting
- **[API Reference](docs/API.md)** — REST, WebSocket, MCP, admin endpoints
- **[Architecture](docs/ARCHITECTURE.md)** — system design and data flow
- **[Skill Authoring](docs/SKILLS.md)** — custom skills and pipelines
- **[Reindexing](docs/REINDEX.md)**: rebuilding the indexes behind knowledge search
- **[Changelog](CHANGELOG.md)** — release history

---

## Companion Tools

- **[rtk](https://github.com/rtk-ai/rtk)** — Rust CLI that compresses shell output
  by 60–90% before it hits the context window. Pairs well with Mullet when piping
  large `grep`/`find`/`test` output into prompts. Install: `cargo install rtk`

---

## License

[BSL 1.1](LICENSE) — solo/internal use unrestricted. Production multi-tenant
deployment requires a commercial license. Converts to Apache 2.0 on 2030-04-03.

For commercial licensing, contact the Licensor named in [LICENSE](LICENSE).

---

⭐ **Star us on GitHub if you like Mullet.** PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
