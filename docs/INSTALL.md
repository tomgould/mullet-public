# Mullet — Install Guide

> **Goal:** Get Mullet installed, wired into Claude Code, and running your first delegated task in ~10 minutes.
>
> **What Mullet is:** An AI coding orchestrator that routes tasks to cheap or free models (Kimi K2, local LLM, Gemini) and integrates with Claude Code via MCP. It keeps your Anthropic token spend low by delegating the actual code work elsewhere.
>
> **📚 Full documentation** (architecture, API, troubleshooting, skills, agents): see `docs/` directory or the project README.

---

## 1. Prerequisites

Before you start, make sure you have:

- **Python 3.10+**
  ```bash
  python3 --version
  ```
- **Git**
  ```bash
  git --version
  ```
- **Claude Code CLI**
  ```bash
  npm i -g @anthropic-ai/claude-code
  ```
  Or install via the [Anthropic desktop app](https://www.anthropic.com/claude-code).

- **At least one LLM provider.** You only need one to get started:

  | Provider | Cost | Setup | Best for |
  |:---------|:-----|:------|:---------|
  | **Local LLM (LM Studio)** | Free | Download [LM Studio](https://lmstudio.ai), load any 4–14B model, toggle the server on | Default for quick tasks, zero cost |
  | **Kimi K2 (Moonshot)** | $0.60 / $2.50 per 1M tokens | API key from [platform.moonshot.cn](https://platform.moonshot.cn) | **Recommended primary** — fast, cheap, great at code |
  | **OpenRouter** | $0.00–$0.80 / $0.30–$2.50 per 1M tokens | API key from [openrouter.ai](https://openrouter.ai) | Aggregates 8 curated models (DeepSeek, Llama 4, Qwen, Grok) |
  | **Google Gemini** | $0.15 / $0.35 per 1M tokens | API key from [aistudio.google.com](https://aistudio.google.com) | Cheap fallback, huge context |
  | **Anthropic** | $3.00 / $15.00 per 1M tokens (Sonnet) | API key from [console.anthropic.com](https://console.anthropic.com) | **Orchestrator only** — Mullet uses this sparingly |

  *Minimum viable config: just Kimi, or just a local LLM. You don't need all of them.*

---

## 2. Install Mullet

Mullet is installed from source. We use an editable install so that Claude Code hooks can reference scripts inside the source tree by absolute path.

```bash
git clone https://github.com/tomgould/mullet-public.git
cd mullet-public
pip install -e .
mullet --version
```

> **Why `-e` (editable) is required:**
> `mullet setup-claude` installs hook scripts that live in `scripts/hooks/` inside this repo. A regular `pip install .` copies the Python package into your site-packages but skips the `scripts/` directory. Without `-e`, the hooks won't resolve and `mullet doctor` will warn you.

Optional extras if you need them later:

```bash
pip install -e ".[gemini]"    # Google Gemini support
pip install -e ".[bedrock]"   # AWS Bedrock
pip install -e ".[postgres]"  # PostgreSQL backend
pip install -e ".[all]"       # Everything
```

---

## 3. Configure API Keys

Mullet loads keys from a `.env` file in the repo root via `python-dotenv`.

```bash
cp .env.example .env
# then open .env in your editor and fill in your keys
```

The keys you need:

```
KIMI_API_KEY=sk-...           # Primary provider — platform.kimi.ai
OPENROUTER_API_KEY=sk-...     # Optional — 8 curated models, OpenAI-compatible
GOOGLE_API_KEY=AIza...        # Optional — cheap fallback
```

> **Do not add `ANTHROPIC_API_KEY`.** Mullet integrates with Claude Code via your
> existing Max or Pro subscription — it does not call the Anthropic API directly.
> Adding a key won't improve anything and will confuse the routing.

> **Minimum viable setup:** `KIMI_API_KEY` alone is enough to get started.
> Alternatively, a local LLM running in LM Studio requires no key at all.

---

## 4. Verify Setup

Run the health check:

```bash
mullet doctor
```

**What the output means:**

- **Green ✓** — component is ready.
- **Yellow !** — component has a warning (e.g. local LLM running but no model loaded).
- **Cyan ·** — informational only (e.g. a provider key is not set — that's fine if you don't plan to use it).

The check covers:
- Python version
- Which providers are live (local, Kimi, Gemini, Anthropic)
- Hook installation status in `~/.claude/settings.json`
- `~/.claude/CLAUDE.md` delegation rules
- Knowledge engine, tools, skills, git, and database

If you see `X/Y checks passed` and at least one provider is green, you're good to go.

---

## 5. Wire into Claude Code (MCP)

Mullet registers as an MCP server inside Claude Code. Once wired, you can ask Claude to delegate work via Mullet tools like `mullet_run`, `mullet_opencode`, and `trawl_search`.

### Auto (recommended)

```bash
mullet setup-claude
```

This is idempotent — safe to re-run. It installs:

| What | Where | Effect |
|:-----|:------|:-------|
| **MCP server config** | `~/.claude/mcp.json` | Registers Mullet + Trawl knowledge tools globally |
| **Hooks** | `~/.claude/settings.json` | Fires in **any** project directory you open with Claude Code |
| **Delegation rules** | `~/.claude/CLAUDE.md` | Provider priority, AI-Bridge validation rules, filesystem guardrails |
| **Skills** | `~/.claude/skills/` | `/delegate`, `/commit`, `/review`, `pool-config` |
| **Agents** | `~/.claude/agents/` | `patch-writer`, `test-runner`, `verifier` |
| **opencode** | `~/.config/opencode/opencode.json` | Zero-Anthropic coding agent using Kimi K2 |
| **Status HUD** | `~/.claude/settings.json` → `statusLine` | Live budget pressure + active agent row in Claude Code's status bar |

**What the hooks do:**
- `delegation_reminder.py` — treats delegation as a last resort. Suggests it when a prompt
  looks like delegable work,
  but only once the Anthropic 5-hour or weekly allowance is genuinely short. Below those
  thresholds it stays silent, because in-context work is already paid for and delegating
  spends real third-party tokens. When it does fire it tells Claude to ask you first.
- `mullet_pretooluse_rewriter.py` — auto-fills the cheapest viable `provider`/`tier` on every `mullet_run` call.
- `mullet_observation_capture.py` — logs tool calls so you can replay them with `mullet observations`.

After running `mullet setup-claude`, **restart Claude Code** so the MCP server and hooks load.

### Manual (if you prefer)

Add to `~/.claude/settings.json` under `mcpServers`:

```json
{
  "mcpServers": {
    "mullet": {
      "command": "mullet",
      "args": ["mcp"]
    }
  }
}
```

Then restart Claude Code.

---

## 6. Optional: Local LLM with LM Studio

If you want a free provider for quick tasks:

1. **Download** [LM Studio](https://lmstudio.ai) and install it.
2. **Load a model.** Recommended for code tasks: **Qwen2.5-Coder-7B** (fast, good at Python/JS).
   - Any 4–14B instruction-tuned model works.
   - On 8GB RAM Macs, the 7B model is too large — use **Qwen2.5-Coder-3B** instead.
   - **Apple Silicon (M1/M2/M3):** download **MLX format** models, not GGUF. Search `mlx-community`
     in the Discover tab — MLX runs natively on Apple Silicon and uses less memory.
3. **Start the local server** — toggle the server switch in the LM Studio UI (default port `1234`).
4. **Verify:**
   ```bash
   mullet doctor
   ```
   You should see a green check next to **Local LLM**.

If you prefer **Ollama** instead of LM Studio:

```bash
ollama pull qwen2.5-coder:7b
ollama serve
```

Mullet auto-detects Ollama on port `11434`.

If your local server is on a non-standard port, set:

```bash
export LOCAL_BASE_URL=http://localhost:8080/v1
```

> **Mac warning:** Do not run `pip install -e ".[gpu]"` on Apple Silicon — the `[gpu]` extra pulls `pynvml` (NVIDIA GPU monitoring) which does nothing on Mac and may error on install.

> **Context size warning:** `mullet doctor` will flag if your loaded model has a very small context window (`< 4K`). For code tasks, aim for at least 8K context.

---

## 7. Encoding and Graph Extraction

**Encoding** is handled by `sentence-transformers` (already in core dependencies — no extras needed).
The embedding model (`all-MiniLM-L6-v2`, ~90MB) downloads automatically from HuggingFace on first use.
On Apple Silicon it uses MPS (Metal) automatically. No manual download or LM Studio involvement required.

**Graph extraction** uses your configured LLM provider — whichever is active (local LM Studio or Kimi).
No additional install is needed. The local LLM you load in LM Studio handles graph extraction prompts directly.

**Mac / Apple Silicon note:** Do not run `pip install -e ".[gpu]"` — the `[gpu]` extra pulls `pynvml`
(NVIDIA GPU monitoring) which does nothing on Mac and may error on install. The `sentence-transformers`
package brings in a compatible PyTorch (CPU/MPS) automatically.

---

## 8. First Test

### CLI test

```bash
mullet "what Python version am I running?"
```

You should get an answer routed to your cheapest available provider (local LLM if running, otherwise Kimi or Gemini).

### See where a task would route

```bash
mullet explain "refactor auth module for OAuth2"
```

This prints the complexity score, chosen tier/provider, and estimated cost — without actually calling the LLM. Useful for checking your wallet impact before big tasks.

### MCP test (inside Claude Code)

Restart Claude Code, then ask:

> "Use `mullet_run` to summarise what this project does."

Claude should invoke the MCP tool and return a summary generated by Kimi or your local LLM — zero Anthropic tokens for the actual generation.

### List models and pricing

```bash
mullet models
```

Shows every model Mullet knows about, grouped by provider, with per-1M-token pricing.

---

## 9. Knowledge Pools (Optional but Powerful)

Mullet includes a **Trawl** knowledge engine that indexes codebases and documents for semantic search. Once indexed, you can query them from Claude Code without pasting entire files into context.

### Where pools live

Pool data (vector indexes, graph DBs, embeddings) lives outside the source tree.
The installer creates `~/.mullet/pools/` as the default. Resolution order at
runtime:

1. `TRAWL_POOLS_DIR` env var, if set (used verbatim)
2. `~/.mullet/pools/`

If you already have a pools directory elsewhere — on a separate disk, or shared
across machines — use either of:

```bash
# Option A — env var (greppable, explicit)
export TRAWL_POOLS_DIR=/data/mullet-pools

# Option B — symlink (invisible, survives every shell + cron job)
ln -sfn /data/mullet-pools ~/.mullet/pools
```

The Mullet repo itself never reads pool data from inside the source tree —
`mullet/knowledge/pools/` is gitignored and unused.

### Managing pools

```bash
# See what pools exist
mullet pool list

# Create a pool for the current repo (one-off)
mullet init

# Index or re-index a pool
mullet reindex --pool <name>
```

See [docs/REINDEX.md](REINDEX.md) for full flag reference, source sync behavior, and the overnight multi-pool workflow.

**Concrete example:**

```bash
# In the mullet repo itself
mullet pool list
mullet reindex --pool mullet-dev
```

Then inside Claude Code:

> "Use `trawl_search` to find where authentication is handled."

Or for code-specific queries:

> "Use `trawl_search_code` to find the `resolve_provider` function."

*Tip: If you're working in a large repo, create a pool for it and reindex after major refactors. Search quality depends on up-to-date indexes.*

---

## 10. Troubleshooting

| Problem | Fix |
|:--------|:----|
| `mullet: command not found` | Run `pip install -e .` from the repo root. If it's still missing, add `~/.local/bin` to your `PATH` (common on Linux). |
| `mullet doctor` shows no providers | Add `KIMI_API_KEY` or `GOOGLE_API_KEY` to your `.env` file, or start LM Studio and load a model. Do not add `ANTHROPIC_API_KEY` — it is not used. |
| MCP tools not appearing in Claude Code | Run `mullet setup-claude`, then **fully restart** Claude Code. Check Settings → MCP → `mullet` is listed. |
| `mullet mcp: server ready` not seen | In Claude Code, go to Settings → MCP → `mullet` → **View logs**. Look for Python tracebacks or missing dependencies. |
| Kimi API key rejected | Ensure the key starts with `sk-` and has active credits at [platform.moonshot.cn](https://platform.moonshot.cn). If you have a coding-specific key (`sk-kimi-...`), Mullet auto-routes to the coding endpoint. |
| `mullet doctor` warns about missing hooks | You probably did a non-editable install. Run `pip install -e .` from the repo root, then re-run `mullet setup-claude`. |
| Local LLM detected but model shows "very small context" | Load a larger model in LM Studio (7B+) or increase the context slider in the model settings. |
| `mullet_run` returns empty or garbled output | The delegated model may not support tool calling. Try forcing a different provider: `mullet_run(task="...", tier="STANDARD")`. |

---

## Next Steps

- **Quick reference:** See [`docs/QUICKSTART.md`](QUICKSTART.md) for common commands and orchestration modes (`#ralph`, `#ultrawork`, `#autopilot`).
- **opencode deep-dive:** See [`docs/opencode_integration.md`](opencode_integration.md) for filesystem-agent setup and ACP protocol details.
- **Architecture:** See [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) for routing tiers, fallback chains, and system design.
- **Preset management:** `mullet preset use <name>` switches routing policy. Built-ins: `balanced-mix` (default), `local-only`, `claude-only`, `cheap-cloud`. See README.md § "Provider Presets" for details.

If something still doesn't work, run `mullet doctor --full` for extra diagnostics and paste the output into a GitHub issue.
