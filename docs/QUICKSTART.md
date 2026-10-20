# Mullet Quick Start

Get from zero to productive in 5 minutes.

## 1. Install

From source (development):

```bash
git clone https://github.com/tomgould/mullet-public.git
cd mullet-public
pip install -e .
```

> **Editable install required:** Hook scripts referenced by `mullet setup-claude` use absolute paths inside the source tree. A plain `pip install` without `-e` will skip the `scripts/` directory and hooks will be skipped (`mullet doctor` will warn).

## 2. Set Up Claude Code Integration

```bash
mullet setup-claude
```

Installs Mullet's hooks and rules globally so they fire in **any project directory**:

| What | Where | Effect |
|------|-------|--------|
| `delegation_reminder.py` | `~/.claude/settings.json` → UserPromptSubmit | Treats delegation as a last resort: silent unless the Anthropic 5h or weekly allowance is too tight, and always tells Claude to ask you first |
| `mullet_pretooluse_rewriter.py` | `~/.claude/settings.json` → PreToolUse | Auto-fills provider/tier on every `mullet_run` call |
| `mullet_observation_capture.py` | `~/.claude/settings.json` → PostToolUse | Logs all tool calls for `mullet observations` |
| Delegation rules | `~/.claude/CLAUDE.md` | Provider priority, AI-Bridge rules, filesystem rules |
| Skills + agents | `~/.claude/skills/` and `~/.claude/agents/` | `/delegate`, `/commit`, `/review`, patch-writer, test-runner, verifier |

## 3. Check Setup

```bash
mullet doctor
```

This shows which providers are available and verifies hooks are installed. You need at least one provider:
- **Local LLM**: Start [LM Studio](https://lmstudio.ai) or [Ollama](https://ollama.com) (8GB Mac: use a 2–3B model; 7B+ models won't fit)
- **Anthropic**: `export ANTHROPIC_API_KEY=sk-ant-...`
- **Google Gemini**: `export GOOGLE_API_KEY=AIza...`
- **Kimi**: `export KIMI_API_KEY=sk-...`
- **OpenRouter**: `export OPENROUTER_API_KEY=sk-or-...` — 8 curated models, OpenAI-compatible

> **Encoding:** `sentence-transformers` installs with the core package. The `all-MiniLM-L6-v2` model
> (~90MB) downloads from HuggingFace automatically on first use — no manual step needed.
> **Graph extraction** uses your active LLM provider (local or Kimi) — no separate install.

## 4. Your First Task

```bash
mullet "list the Python files in this directory"
```

Mullet routes to the cheapest capable model automatically. Simple tasks use your local GPU (free), complex tasks use the cloud.

## 5. See How Routing Works

```bash
mullet explain "refactor auth module for OAuth2"
```

Shows the complexity score, which tier/provider would be selected, and estimated cost.

## 6. Use Tools

The agent has 24 tools. Here are the most useful:

```bash
# Read and modify files
mullet "read main.py and add error handling"

# Run tests
mullet "run the tests and fix any failures"

# Git operations
mullet "check git status and show recent changes"

# Review code with context
mullet "review @file:src/auth.py and check @diff"
```

## 7. Orchestration Modes

For bigger tasks, use orchestration:

```bash
# Ralph: persistence loop until done
mullet ralph "implement user authentication with JWT"

# Autopilot: plan → implement → verify
mullet autopilot "build a REST API for inventory management"

# Ultrawork: parallel execution
mullet ultrawork "optimize all Python files for performance"

# Team: coordinated agents
mullet team 3 "build a microservice architecture"

# Eco: same modes but cheaper
mullet eco ralph "add input validation"
```

## 8. Watch Mode

Re-run tasks automatically when files change:

```bash
mullet watch "run tests and fix failures"
```

## 9. API Server

Start the API server for web UI or programmatic access:

```bash
mullet serve --port 3000
```

Then use the OpenAI-compatible API:

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "hello"}]}'
```

## 10. MCP Integration

Use Mullet from Claude Code:

```json
{
  "mcpServers": {
    "mullet": {"command": "mullet", "args": ["mcp"]}
  }
}
```

## 11. Swarm Mode

Process multiple files or items in parallel using Kimi K2 via the `mullet_swarm` MCP tool:

```json
// Fan out across files
{
  "goal": "add docstrings to all functions",
  "files": ["**/*.py"]
}

// Fan out across arbitrary items
{
  "goal": "review each item",
  "items": ["file1.py", "file2.py"],
  "concurrency": 5
}

// Dry run to see task plan
{
  "goal": "audit security",
  "files": ["src/**/*.py"],
  "dry_run": true
}
```

## 11a. Auto-Swarm Routing (MCP)

When using Mullet via Claude Code MCP, the `mullet_run` tool automatically fans out to a swarm when it detects any of these signals:

| Signal | Threshold | Example |
|---|---|---|
| Many files | ≥ 5 | `files=["a.py","b.py","c.py","d.py","e.py"]` |
| CRITICAL tier | forced | `tier="CRITICAL"` |
| Long prompt | ≥ 8,000 chars | Large task description |
| Mixed languages | 2+ extensions | `.py` + `.ts` + `.go` |
| Many doc files | 3+ `.md`/`.rst` | Documentation sweep |

```json
{
  "task": "add type hints to all modules",
  "files": ["src/a.py", "src/b.py", "src/c.py", "src/d.py", "src/e.py"],
  "apply": true
}
```
The above auto-swarms. To disable: `"force_single": true`.

## 12. Model Management

```bash
mullet models              # List all available models
mullet models refresh      # Refresh registry from provider APIs
mullet models --provider kimi
```

## 13. Configuration

Generate a project config:

```bash
mullet init
```

This creates `mullet.yaml` with detected providers and project settings.

## Provider presets

Switch routing policy on the fly:

```bash
mullet preset list                        # show built-ins and custom presets
mullet preset use local-only              # force local models only
```

From the CLI:

```bash
mullet preset use cheap-cloud
```

Built-in presets:
- `balanced-mix` — default, allows all providers with balanced speed
- `local-only` — local LLM only, economy speed, no cloud fallback
- `claude-only` — native Anthropic only (refuses delegation via opencode/Mullet)
- `cheap-cloud` — cheapest cloud providers, skips premium tiers

## Tips

- **Force cheap routing**: Prefix with `quick:` to use local GPU regardless of complexity:
  ```bash
  mullet "quick: refactor the auth module"
  ```
- **Context references**: Inject files with `@file:path`, diffs with `@diff`, URLs with `@url:...`
- **Verbose mode**: Add `--verbose` to see routing decisions and cost
- **JSON output**: Add `--json` for structured output (great for scripting)

## Key Concepts

- **Routing**: Tasks are analyzed for complexity and routed to the cheapest capable provider
- **Escape hatches**: Prefix with `quick:`, `simple:`, or `just:` to force local routing
- **Tools**: 25 tools including file ops, terminal, git, tests, vision, Docker, LSP, MCP client
- **Context**: Use `@file:path`, `@diff`, `@staged` to inject context into prompts
- **Learning**: The router learns from past outcomes to make better decisions over time
- **Sessions**: Conversations are saved and searchable across sessions
