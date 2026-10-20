# Mullet

Mullet takes work off an AI coding tool such as Claude Code. Over the Model Context Protocol (MCP), it sends each task to the cheapest model that can handle it, from a local model in LM Studio or Ollama to Kimi, OpenRouter, Gemini or Claude, and it searches your own documents and code.

## Status

Mullet is paused, not finished. I still run its knowledge search, Trawl, but I've turned delegation off: a larger Anthropic subscription turned out quicker than handing work to other models. If Anthropic's prices rise, I'll switch it back on and send most of the work to cheaper models through Kimi and OpenRouter. Set `MULLET_DELEGATION=off` in `.env` to turn delegation off; Trawl, status and budget keep working. The code, model list and prices date from October 2026.

## Install

Mullet isn't on PyPI (the name there belongs to another package), so install it from a clone. It needs Python 3.10 or later.

```bash
git clone https://github.com/tomgould/mullet-public.git
cd mullet-public
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Keep the install editable (`-e`): the Claude Code hooks run from `scripts/hooks/` in the clone. Optional extras are `gemini`, `bedrock`, `postgres`, `gpu`, `all` and `dev`, for example `pip install -e ".[gemini]"`. opencode needs Node.js 18 or later.

## Quick start

```bash
cp .env.example .env     # add the API keys you have
mullet doctor            # checks the install and which providers it can reach
mullet explain "add a retry with exponential backoff to the HTTP client"   # scores a task, calls no model
mullet "explain this codebase in one paragraph"                            # runs a task
```

Mullet reads `.env` from the root of the clone. `KIMI_API_KEY` alone is enough to start, and a local model in LM Studio or Ollama needs no key. `ANTHROPIC_API_KEY` is optional: without it, Mullet hands Anthropic work back to Claude Code or uses the `claude` command-line tool.

The MCP server is `mullet mcp`. To add it to Claude Code for every project:

```bash
claude mcp add --scope user mullet -- mullet mcp
```

`mullet setup-claude` goes further. Read what it does first, or run it with `--dry-run`: it installs hooks that run in every project, adds delegation rules to `~/.claude/CLAUDE.md`, installs skills, agents and a status line into `~/.claude/`, and installs opencode with npm.

## Documentation

The guides are in [`docs/`](docs/):

- [Quick start](docs/QUICKSTART.md)
- [Installation](docs/INSTALL.md), with guides for [Linux](docs/ONBOARDING_LINUX.md), [macOS](docs/ONBOARDING_MAC.md) and [ChromeOS](docs/ONBOARDING_CHROMEBOOK.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Model routing](docs/MODEL_ROUTING.md)
- [opencode integration](docs/opencode_integration.md)
- [API reference](docs/API.md)
- [Writing skills](docs/SKILLS.md)
- [Reindexing knowledge pools](docs/REINDEX.md)
- [Changelog](CHANGELOG.md)

The rest of this page is a reference.

## What it does

- Scores each task out of 100 before it picks a model, from keywords, the number of subtasks, how much of the system it touches and whether it needs tests. The score puts the task in a tier: LOW, STANDARD, ADVANCED or CRITICAL. `mullet explain` shows the working without calling a model.
- Watches the budget. It reads how much of a Claude subscription's five-hour and weekly allowance has gone and, as that runs low, moves work away from Claude to providers such as OpenRouter and Kimi.
- Edits files through [opencode](https://github.com/sst/opencode), an open-source coding agent, then checks what it did. The list of changed files comes from `git status`, not from the model's report, and a classifier with no model behind it flags edits that often mean trouble.
- Splits a goal across parallel Kimi calls and merges the answers, and has longer-running modes that break a task up, work through it and check the result.
- Searches your own documents and code with Trawl, its knowledge engine. Trawl indexes collections called pools, searches them by keyword and by meaning together, re-ranks the results and can build a knowledge graph.
- Limits which providers it may use through presets.
- Serves an OpenAI-compatible API with `mullet serve`.

## Using it from Claude Code

The MCP server has 10 Mullet tools and 43 Trawl tools.

| Tool | What it does |
| --- | --- |
| `mullet_run` | Sends a task to the cheapest suitable model. The model can't see your files, so the task must carry everything it needs. |
| `mullet_opencode` | Gives the task to opencode, which edits files in place and returns the list of changed files. It never commits. |
| `mullet_swarm` | Fans a goal out across parallel Kimi calls and merges the results. |
| `mullet_delegate` | Runs a child agent on providers you choose. |
| `mullet_search` | Searches a Trawl pool. |
| `mullet_remember` | Stores a note in a project's knowledge base. |
| `mullet_analyze` | Scores a task and shows the routing decision. |
| `mullet_status`, `mullet_providers`, `mullet_budget` | Show provider health and budget. |

The Trawl tools manage pools, search code, documents and the knowledge graph, and list a pool's newest documents. With `MULLET_DELEGATION=off`, `mullet_run`, `mullet_opencode`, `mullet_swarm` and `mullet_delegate` refuse, and the rest keep working.

## Routing

Routing has three stages.

1. A preflight step estimates how much the task will read and write and how hard it is. It asks a local model first, then Gemini Flash, and falls back to heuristics.
2. A routing matrix picks the cheapest model that can take that input and output at that complexity. Size estimates get a 10% margin before they're compared with a model's context window.
3. A fallback chain takes over if that model isn't available.

Local models are classified from their names, by size and by whether they're built for code, vision or reasoning, so routing works with whatever is loaded in LM Studio. A fifth tier, PREMIUM, is for the orchestrating model only: `mullet override on` sends work to Opus until the current five-hour window resets.

`mullet models` and `mullet cost` list the models and prices Mullet knows about. [docs/MODEL_ROUTING.md](docs/MODEL_ROUTING.md) has the detail.

### Budget guard

`mullet.hud.guard.check_budget()` returns one of three states. It's "go" normally. It's "throttle", which steers work to a cheaper provider, once five-hour usage reaches 70% (55% when weekly usage is 85% or more) or the day's share of the weekly allowance drops below 8%. It's "pause" when five-hour usage is at least 95% and weekly usage at least 90%.

### Provider presets

A preset limits which providers Mullet may use.

| Preset | Providers | opencode model |
| --- | --- | --- |
| `balanced-mix` (default) | all | `moonshot/kimi-k2.6` |
| `local-only` | `local` | `lmstudio/qwen2.5-coder-14b-instruct` |
| `claude-only` | `sonnet`, `haiku`, `opus-4-8`, used by Claude Code itself, never delegated | none |
| `cheap-cloud` | `kimi`, `gemini` | `moonshot/kimi-k2.6` |

```bash
mullet preset list          # * marks the active one
mullet preset use cheap-cloud
mullet preset show local-only
mullet preset add --name my-preset --label "My preset" --providers local,kimi --speed balanced
mullet preset remove my-preset
```

### Orchestration modes

| Command | What it does |
| --- | --- |
| `mullet "<task>"` | One routed call |
| `mullet ultrawork "<task>"` | Splits the task and runs the parts in parallel |
| `mullet ralph "<task>"` | Repeats ultrawork until every user story passes, up to ten times |
| `mullet autopilot "<task>"` | Plans, runs ralph, then checks the result |
| `mullet team N "<task>"` | Runs N coordinated agents |
| `mullet eco <mode> "<task>"` | Runs another mode on cheaper tiers |

Over MCP, `mullet_run` takes `standard`, `ultrawork` or `ralph` as its `mode`.

### Built-in tools

A task run by Mullet's own agent can use 24 tools, in `mullet/core/tools/`:

| Kind | Tools |
| --- | --- |
| Files and code | `file_ops`, `patch`, `git`, `lsp` |
| Shell | `terminal`, `code_exec`, `python_repl`, `run_tests`, `docker` |
| Web | `web_search`, `fetch_page`, `browse` |
| Agents | `delegate`, `todo`, `clarify`, `skill_invoke`, `skill_mgr` |
| Memory and sessions | `memory`, `notepad`, `shared_mem`, `session_search`, `session_list` |
| Images | `analyze_image` (needs a provider that takes images), `image_gen` |

Tools from other MCP servers can be added through the client in `mullet/core/tools/mcp_client.py`.

## Commands

| Command | What it does |
| --- | --- |
| `mullet mcp` | Starts the MCP server (stdio) |
| `mullet serve` | Starts the API server on `127.0.0.1:3000` (`--host`, `--port`) |
| `mullet doctor` | Checks the install, providers, pools, config and hooks |
| `mullet setup-claude` | Sets up Claude Code, as above |
| `mullet init` | Writes a `mullet.yaml` for the current project |
| `mullet preset` | Lists, switches, shows, adds or removes presets |
| `mullet explain` | Scores a task without calling a model |
| `mullet models`, `mullet cost` | List models and prices |
| `mullet budget`, `mullet routing` | Show spend and routing history |
| `mullet search`, `mullet pool`, `mullet reindex`, `mullet graph` | Search and manage knowledge pools |
| `mullet history`, `mullet logs`, `mullet stats` | Show past sessions and model calls |
| `mullet override` | Turns the Opus override on or off |

`mullet --help` lists the rest.

## API server

`mullet serve` offers `/v1/chat/completions`, `/v1/models`, `/health`, `/metrics` (Prometheus) and a WebSocket at `/ws`.

```bash
mullet serve
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```

[docs/API.md](docs/API.md) covers the REST, WebSocket and admin endpoints.

## Configuration

- API keys and switches such as `MULLET_DELEGATION` go in `.env`; `.env.example` lists them.
- Project settings go in `mullet.yaml` in the project, or `~/.mullet/config.yaml`; `mullet.yaml.example` shows the shape. For example, to add an OpenAI-compatible server on a port Mullet doesn't probe:

  ```yaml
  providers:
    my_server:
      type: openai_compatible
      base_url: http://localhost:8080/v1
      models:
        default: my-model
  ```

- Knowledge pools live in `~/.mullet/pools/`, or wherever `TRAWL_POOLS_DIR` points.
- A pool can also crawl a website on every `mullet reindex`, from its sitemap, with linked PDFs and Office files. [docs/REINDEX.md](docs/REINDEX.md#web-sources-crawled-websites) has the options.

## Development

```bash
pip install -e ".[dev]"
pytest
```

There are more than 17,800 tests. GitHub Actions runs them on Python 3.10, 3.11 and 3.12, and ruff checks for errors. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

Mullet is source-available under the [Business Source License 1.1](LICENSE). You may copy, change and redistribute it, and use it outside production. One developer may also use it in production, free, for their own work on one running instance, but not offer it, or a service based on it, to anyone else. Any other production use needs a commercial licence; [LICENSE](LICENSE) says how to ask.

Each version becomes available under the Apache License 2.0 on 3 April 2030, or four years after its first public release if that comes sooner.
