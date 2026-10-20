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
- ⭐ **Batteries included** — skills, plugins, hooks, knowledge engine (semantic search + graph, PDF and document ingestion), swarm, image analysis, web browsing, Prometheus metrics, secret redaction, context compression.

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

17,700+ tests. Pure Python, no JS runtime. Here's what's in the box:

### Core Runtime

| Capability | Detail |
|:-----------|:-------|
| **AgentRunner** | Async agent loop with tool execution, an iteration budget, streaming and fallback providers |
| **Streaming** | Tool-call arguments that arrive in fragments are held until they parse |
| **Context compression** | Summarises older messages once the conversation passes half the model's context window |
| **Reasoning blocks** | Extracts thinking output from Claude, DeepSeek-R1, Qwen and OpenRouter responses |
| **Context references** | `@file:path`, `@folder:path`, `@git:N`, `@diff`, `@staged` and `@url:url` in a prompt pull that content in |
| **Agent memory** | Durable facts (preferences, environment details, conventions) injected into every turn |
| **Agent profiles** | YAML profiles in `mullet/agents/` (see [Agent Profiles](#agent-profiles)) |

### Coordinator / Workflows

| Capability | Detail |
|:-----------|:-------|
| **Swarm** | `mullet_swarm` MCP tool — fan out a goal across parallel Kimi K2 calls, synthesize results, configurable concurrency |
| **Orchestration modes** | `mullet ultrawork`, `ralph`, `autopilot`, `team` and `eco` (see [Orchestration Modes](#orchestration-modes)) |
| **Auto-swarm** | Detects parallelisable tasks and routes to swarm automatically (`mullet_run` with `force_single` turns it off) |

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
| `mullet ingest` | Preview the chunks a directory would produce (writes nothing; use `mullet reindex` to index a pool) |
| `mullet graph` | Knowledge graph operations (build, search, stats) |
| `mullet logs` | View LLM interaction history (tail/stats/suggest) |
| `mullet history` | Show recent sessions or search past conversations |
| `mullet stats` | Show usage statistics from session history |
| `mullet models` | List available models and pricing by provider |
| `mullet cost` | Show model pricing across tiers |
| `mullet explain` | Show complexity analysis for a task (no LLM call) |
| `mullet completion` | Generate shell completion scripts (bash/zsh) |
| `mullet override` | Toggle budget override (mullet override on/off) |

### Tools (24 built-in)

| Category | Tools |
|:---------|:------|
| **Files and code** | `file_ops`, `patch`, `git`, `lsp` |
| **Shell and execution** | `terminal`, `code_exec`, `python_repl`, `run_tests`, `docker` |
| **Web** | `web_search`, `fetch_page`, `browse` (fetch pages, extract content, follow links, fill forms) |
| **Agent / orchestration** | `delegate`, `todo`, `clarify`, `skill_invoke`, `skill_mgr` |
| **Memory and sessions** | `memory`, `notepad`, `shared_mem`, `session_search`, `session_list` |
| **Images** | `analyze_image` (needs a vision-capable provider), `image_gen` |

Tools from external MCP servers (stdio) can be added through the MCP client in `mullet/core/tools/mcp_client.py`.

### REPL commands

The bare `mullet` REPL supports:

`/skill-name` · `clear` · `history` · `help` · `exit`

Type `/` followed by a skill name to invoke it (e.g. `/code-review`).

### Hooks

`mullet setup-claude` installs Claude Code hooks from `scripts/hooks/`, such as the
delegation reminder (UserPromptSubmit), the provider/tier rewriter (PreToolUse on
`mullet_run`) and observation capture (PostToolUse).

### Sessions

| Capability | Detail |
|:-----------|:-------|
| **Storage** | Each session is saved as JSON in `~/.mullet/sessions/` |
| **Search** | A SQLite FTS5 index (`~/.mullet/sessions.db`) backs `mullet history` and the `session_search` tool |

### Servers

| Server | Detail |
|:-------|:-------|
| **MCP server** | Registers as Claude Code MCP server — 10 Mullet tools + 42 Trawl knowledge tools |
| **API server** | OpenAI-compatible `/v1/chat/completions`, `/v1/models`, `/health`, `/metrics` via `mullet serve` |

### Polish

- 🧠 **Reasoning blocks** — thinking output from reasoning models is kept apart from the answer
- 🔄 **Auto-update checker** — `mullet doctor` compares your version with the latest on the public GitHub repository and says when a newer one is available
- 💬 **Helpful errors** — common exceptions come with a suggested fix, such as the package to install
- 🎨 **ASCII banner** — version banner via `mullet --version --banner`

### Security

- 🛡️ **Redactor** — masks API keys, tokens and credentials in logs and tool-result previews
- 🚫 **Injection guard** — the terminal tool blocks commands that match known prompt-injection patterns, and content pulled in by `@` context references is sanitised
- 🌐 **URL safety** — URL ingestion into knowledge pools refuses non-HTTP schemes, localhost and private-network addresses
- 🔍 **Diff-risk classifier** — deterministic, zero-LLM-cost heuristic scans every `mullet_opencode` diff for risky changes (literal edits, quote flattening, new control flow, tampered assertions) delegates won't self-report

### Observability

- 📈 **Prometheus metrics** — `mullet serve` exposes `/metrics` with request, token, latency and tool-call counters
- 📝 **LLM interaction log** — `mullet logs tail|stats|suggest` inspects the history the provider routing layer writes
- 🧭 **Routing history** — `mullet routing` shows provider outcomes, quality history and insights

---

## Why Not X?

| Capability | **Mullet** | aider | opencode | Claude Code |
|:-----------|:----------:|:-----:|:--------:|:-----------:|
| Self-hosted, no cloud account required | ✅ | ✅ | ✅ | ❌ |
| Multi-provider native (Kimi/Claude/OpenAI/Gemini/Local) | ✅ | ✅ | ✅ | partial |
| Swarm / parallel delegation | ✅ | ❌ | ❌ | partial |
| Knowledge engine (semantic search + graph) | ✅ | ❌ | ❌ | ❌ |
| Multimodal (image + PDF) | ✅ | partial | partial | ✅ |
| One-line install | ✅ | ✅ | ✅ | ✅ |
| MCP client (stdio) | ✅ | ❌ | partial | ✅ |
| Pure Python (no JS runtime) | ✅ | ✅ | ❌ | ❌ |
| Token-cost router with budget guard | ✅ | partial | ❌ | ❌ |
| Secret redaction in logs and tool output | ✅ | ❌ | ❌ | partial |

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

| Mode | Behaviour | Command |
|:-----|:----------|:--------|
| `standard` | Single agent call | `mullet "<task>"` |
| `ultrawork` | Decomposes task, spawns up to 3 parallel workers via `delegate_batch`, merges results | `mullet ultrawork "<task>"` |
| `ralph` | Wraps ultrawork in a persistence loop — re-runs until all PRD user stories pass, up to 10 iterations | `mullet ralph "<task>"` |
| `autopilot` | Full lifecycle: plan → ralph → verify. Runs unsupervised. | `mullet autopilot "<task>"` |
| `team` | N coordinated agents with non-overlapping file ownership boundaries | `mullet team N "<task>"` |
| `eco` | Modifier: runs ralph, ultrawork or autopilot on cheaper tiers to cut token cost | `mullet eco <mode> "<task>"` |

Over MCP, `mullet_run` takes the mode as its `mode` parameter (`standard`, `ultrawork`
or `ralph`).

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

> **Note:** Alibaba Cloud / DashScope is configured but not yet available on international accounts — pending future activation.

### Budget Guard

```python
from mullet.hud.guard import check_budget
r = check_budget()   # a BudgetCheck
print(r.status.value, r.five_hour_pct, r.weekly_pct, r.recommended_provider)
```

Three states: **go**; **throttle** once 5-hour usage reaches 70% (55% when weekly
usage is 85% or more) or the daily share of the weekly allowance drops below 8%,
which steers work to a cheaper provider; and **pause** when 5-hour usage is at least
95% and weekly usage at least 90%.

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
| `mullet_run` | Route task to cheapest capable model (no file access) | `task`, `tier`, `providers`, `mode`, `force_single` |
| `mullet_swarm` | Fan-out across parallel Kimi K2 workers | `goal`, `items`, `concurrency`, `output_shape` |
| `mullet_delegate` | Child agent with explicit provider control | `goal`, `context`, `providers` |
| `mullet_opencode` | opencode — native filesystem access, edits files in place and lists what changed | `task`, `cwd`, `model` |
| `mullet_search` | Semantic search via Trawl knowledge engine | `query`, `client` |
| `mullet_remember` | Store memory in project knowledge base | `key`, `content`, `client` |
| `mullet_analyze` | Complexity score + routing decision for a task | `task` |
| `mullet_status` | Provider health, resource info | — |
| `mullet_providers` | Live health check per provider | — |
| `mullet_budget` | Token spend and budget limits | — |

Plus **42 Trawl knowledge tools** covering pool management, graph search, and code indexing.

### Typical Claude Code Workflow

```python
# Delegate a self-contained task. mullet_run has no file access, so put
# everything the model needs into the task text.
mullet_run(
    task="write a regex that matches UK postcodes, with five test cases",
    tier="STANDARD",
)

# For code changes, give the delegate filesystem access via opencode
mullet_opencode(task="add type hints to all public APIs")
# → returns {"response": "...", "files_changed": ["src/auth.py", ...], ...}
# opencode edits files in place and never commits: review with git diff,
# then commit yourself.
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
1. Writes `~/.claude/mcp.json` registering Mullet + Trawl MCP servers
2. Installs global Claude Code hooks into `~/.claude/settings.json` — fires in **any project directory**, not just this repo: delegation reminder (UserPromptSubmit), provider/tier rewriter (PreToolUse on mullet_run), observation capture (PostToolUse)
3. Appends Mullet delegation rules to `~/.claude/CLAUDE.md`
4. Installs skills (`/delegate`, `/commit`, `/review`) and agents (`patch-writer`, `test-runner`, `verifier`) into `~/.claude/`
5. Installs opencode via npm and writes `~/.config/opencode/opencode.json`

> **Note:** The hook scripts live in `scripts/hooks/` inside the source tree.
> `.claude/settings.json` references them via Claude Code's `${CLAUDE_PROJECT_DIR}`
> variable, so the path itself is portable — but a regular (non-editable)
> `pip install .` still won't ship the `scripts/` directory, so an editable install
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
# Standard: a single routed call
mullet "add a retry decorator to the API client"

# Ultrawork: parallel decomposition
mullet ultrawork "add error handling to every endpoint in api/"

# Ralph: persistence loop — keeps going until tests pass
mullet ralph "implement OAuth2 login flow with full test coverage"

# Autopilot: plan, execute, verify unsupervised
mullet autopilot "migrate database from SQLite to PostgreSQL"

# Team: 3 coordinated agents with file ownership
mullet team 3 "refactor the entire auth subsystem"

# Eco: ralph on cheaper tiers
mullet eco ralph "add docstrings to all files in utils/"
```

Swarm runs over MCP: `mullet_swarm(goal="summarise each file in src/", items=[...])`.

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

[Business Source License 1.1](LICENSE). You may copy, modify and redistribute
Mullet and use it outside production. The licence's Additional Use Grant also
allows production use, free of charge, by one individual developer using it
themselves on one running instance. It does not cover providing Mullet, or a
service based on it, to anyone else. Any other production use needs a
commercial licence.

Each version of Mullet becomes available under the Apache License, Version 2.0
on 2030-04-03, or on the fourth anniversary of that version's first public
release if that comes sooner.

For other licences, contact the Licensor as described in [LICENSE](LICENSE).

---

⭐ **Star us on GitHub if you like Mullet.** PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
