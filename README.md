# Mullet

Mullet runs a large coding job in a repo while you are away, and checks every step itself before it counts. You approve a plan, leave it running, and come back to a branch and an evidence report. Your own checkout is not touched until you merge.

Each step is done by a worker in a sandboxed private clone. A step passes only when Mullet's own gates pass: real changes in the diff, files within scope, no secrets, a clean build, tests no worse than before, and a fresh review. Nothing is marked done because a model said so.

Mullet also serves Trawl, a knowledge search for your own documents and code, over MCP.

## Status

Mullet is maintained. Jobs run on Linux. On macOS they refuse to start for now, because there is no sandbox there yet; the rest of Mullet works. The code, model list and prices date from October 2026.

## Install

Mullet isn't on PyPI (the name there belongs to another package), so install it from GitHub. It needs Python 3.10 to 3.12, git, and uv or pipx. Jobs also need the `claude` command-line tool, logged in, and on Linux, bubblewrap and socat.

```bash
curl -fsSL https://raw.githubusercontent.com/tomgould/mullet-public/main/scripts/install.sh | bash
```

That runs `uv tool install git+https://github.com/tomgould/mullet-public.git` (or the pipx equivalent). Run it again to update. Add `--version X.Y.Z` or `--ref <branch|tag|sha>` to pin a release or ref, and `--dry-run` to print the command without running it (with curl: `... | bash -s -- --version X.Y.Z`). Optional extras are `gemini`, `bedrock`, `postgres`, `gpu` and `all`. To work on Mullet itself, clone it and run `pip install -e ".[dev]"` in a virtualenv.

## Quick start

```bash
mullet doctor                                  # checks Claude Code, the sandbox and your setup
cd your-repo
mullet job plan "add retry with backoff to the HTTP client"   # a read-only session proposes a task tree
mullet job approve <id>                        # you approve the plan
mullet job run <id>                            # runs in the background; status, tail, stop, resume
mullet job report <id>                         # evidence for each step
mullet job merge <id>                          # copies the branch into your repo; never pushes
```

`mullet job show <id>` prints the plan. Add `--yaml` to get an editable plan file, and pass it back with `--plan-file`. `--single` skips the planner and makes one step. `mullet job list` shows recent jobs, and `mullet job discard <id>` removes one. A job with steps that could only be partly checked (shown as unverified) is refused by `merge` until you review them and add `--allow-unverified`.

## How a job works

1. The planner breaks the goal into a tree of steps, up to three levels deep and 40 steps, until each leaf fits one pass. Only Mullet creates steps. Workers never start workers.
2. You read the plan and approve it. This is the first of two approvals.
3. Each step runs in a private clone of the commit the job started from. A worker edits the files. Mullet then snapshots the tree, runs the gates, and commits exactly the tree it judged.
4. A failed step is reset and tried again with the gate output, up to three attempts. After that it is blocked. A parent step passes only when its children pass and the full checks pass on their combined change.
5. When the job finishes, you read the report and run `mullet job merge`. This is the second approval.

One Claude worker runs at a time across all jobs, because they share one subscription's allowance; cheaper workers have slots of their own (three by default, `MULLET_API_WORKERS`). When Claude Code reports a usage limit, the job pauses until the reset and carries on. A pause does not count as an attempt. Stop and `kill -9` are safe: `mullet job resume` returns the clone to the last commit Mullet recorded.

## The gates

A gate that fails fails the step. A gate that could not run shows as unverified, never as passed.

| Gate | What it checks |
| --- | --- |
| G0 | The repo's checks on the starting commit, once per job, so later results have a baseline |
| G1 | The diff from git shows a real change |
| G2 | Files stay in scope; protected paths such as `.git*`, `CLAUDE.md`, CI workflows and lockfiles are untouched |
| G3 | Added lines hold no secrets and no risky patterns |
| G4 | Lint, type check and build |
| G5 | Tests, compared with G0: new failures, and tests that passed before and are skipped now, count against the step |
| G6 | No removed assertions or newly skipped tests, unless the step is allowed to |
| G7 | A fresh read-only Claude session reviews the diff, without seeing the worker's transcript. It can block a step, never pass one |

Job state is kept in a SQLite ledger, and a step's status comes from gate results only. The report is built from the ledger. Anything the worker says about itself is labelled as a claim.

## Safety

- Workers run inside bubblewrap on Linux: read-only root, your home directory hidden, no network, and an allowlisted environment so none of Mullet's loaded keys get through. Gate commands run in the same sandbox. If the sandbox can't start, the job refuses to run. There is no unsandboxed fallback.
- The private clone keeps its git directory out of the worker's reach, so a worker can't move refs or rewrite history. Mullet runs its own git commands with hooks off.
- Credentials: the Claude Code worker is your own installed `claude -p`, using your own login. Mullet never reads, stores or passes Claude's credentials. `ANTHROPIC_API_KEY` is left out of the worker's environment, because in `claude -p` it silently replaces your login and bills per token. Plan a job with `--billing api` to opt in. `mullet doctor` warns when the key is set. Check Anthropic's current terms for your own use.
- `mullet job merge` fetches the job branch into your repo as a new branch and stops. It refuses an existing branch name. It never pushes.

## Workers

A step's worker is Claude Code, your own `claude -p`, or opencode on a model you name: Kimi on Moonshot's API, an OpenRouter model, an OpenRouter `:free` model, or a model in LM Studio on your own machine. opencode is off unless a repo's stored profile lists it in `allowed_executors`. Every worker runs in the same sandbox and faces the same gates, and reaches its provider through a key proxy on a Unix socket, so the sandbox never sees an API key and each attempt may only use its own model.

### The delegation ladder

With a `ladder:` section in `~/.config/mullet/config.yaml`, each step goes up a ladder of workers, one attempt per rung, cheapest first unless Claude's week has room:

```yaml
ladder:
  kimi: moonshotai/kimi-k2.7-code          # Moonshot's own API (MOONSHOT_API_KEY)
  openrouter: [openrouter/z-ai/glm-5.3]    # paid OpenRouter models, in order
  free: openrouter/<model>:free            # optional; verified-public GitHub repos only
  local: lmstudio/qwen3.6-35b-a3b          # optional; LM Studio on 127.0.0.1:1234
  local_by_day: false                      # local only in night mode
  max_spend: 4.0                           # dollars per job on the paid rungs
  routing:                                 # optional OpenRouter host choice per model
    openrouter/z-ai/glm-5.3: {only: [z-ai]}
```

- The planner marks each step mechanical, design, analysis or integration; mechanical and checkable design steps start low, the rest start with Claude or Kimi. Planning and review always stay on Claude.
- The order follows Claude's paced weekly usage, read from what Claude Code reports: with room, Claude comes before the paid rungs; when the week is tight, Claude is the last resort.
- Free models only see repos GitHub confirms are public. Paid OpenRouter requests always ask for hosts that don't keep prompts, and by default skip hosts serving a 4-bit copy of the model.
- Each job has a spending cap on the paid rungs. A provider error moves a step up without using an attempt, and a model that failed a step's gates isn't tried again elsewhere.
- `mullet ladder report` adds up every job's attempts by rung, step kind and model from the ledger, with a short list of what to change.
- `mullet job ladder` shows which rungs work now; `mullet job retry <job> <step> --rung claude` gives a blocked step another go. The report shows each attempt's rung, model and cost.

### Local models and night mode

- `mullet night on [--until HH:MM]` lets overnight jobs lean on the local GPU: local attempts get longer and wait for room on the machine. It ends at its time or when you come back.
- `mullet local status|restart|check --fix` keeps LM Studio usable, and `mullet local watchdog --install` checks it every two minutes and restarts it when it is stuck.
- `mullet strategy set --budget USD --delegation ladder|claude-only|off` records this session's budget and delegation choice; `mullet job plan` follows it while it is fresh.
- `mullet reap` stops idle or orphaned Mullet work (test workers, stray opencode runs, idle local models, stale temp directories); `--dry-run` shows what it would do.
- A machine guard (one GPU lease across Mullet's processes, plus VRAM, RAM and temperature limits) keeps jobs, local models and reindexing from fighting over the card.

Not built: a macOS sandbox, so jobs cannot run there yet.

## Spending limits and other settings

Spending is unlimited until you set limits in `~/.config/mullet/config.yaml`. `ladder.max_spend` caps one job; `budget.envelope` and `budget.per_rung` cap all jobs together per UTC day or week:

```yaml
budget:
  envelope: {daily: 25.0, weekly: 100.0}
  per_rung:
    kimi: {daily: 15.0}
    openrouter: {weekly: 40.0}
  warn_below_usd: 5.0                      # doctor and notifications flag provider credit under this
```

Cost comes from what the provider's responses say. Each attempt gets an allowance, and the key proxy refuses requests with HTTP 402 once it is spent. `mullet job report` ends with a provider check that sets the ledger's spend beside each provider's own figure. [docs/CONFIGURATION.md](docs/CONFIGURATION.md) covers these, which toolsets and files a delegated model may touch (`delegation.toolsets`, `delegation.file_deny`), and how `mullet serve` guards itself (Host and Origin checks, a token in `serve-token`). `mullet config show` lists every active setting and where it came from.

## Notifications

Mullet tells you when a job blocks, pauses or finishes. Desktop notifications are on by default. A Slack incoming webhook is sent only in working hours. Each message ends with your paid providers' credit and Claude's weekly use, when Mullet can read them.

| Variable | Meaning |
| --- | --- |
| `MULLET_NOTIFY` | Channels: `desktop`, `slack`, both separated by a comma, or `off` |
| `MULLET_SLACK_WEBHOOK` | The Slack incoming webhook URL |
| `MULLET_NOTIFY_HOURS` | Working hours for Slack, default `Mon-Fri 09:00-17:30` |
| `MULLET_NOTIFY_TZ` | Time zone for those hours, default the machine's local time zone |

## Where files live

`mullet paths` prints every directory. Settings and keys go in `~/.config/mullet/.env` (`$MULLET_CONFIG_DIR` if set). State, data and caches follow the XDG directories. An older `~/.mullet` still works, and `mullet migrate-state --dry-run` shows what moving it would do. Your stored job profile for each repo is in the `projects` folder of the config directory. A repo can add `.mullet/project.yaml`, which can only tighten it.

`mullet setup-claude` registers the MCP server and Mullet's skills with Claude Code for every project. Run it with `--dry-run` first, and `--uninstall` removes exactly what it added. It never edits `CLAUDE.md`. Hooks are opt-in: `--hooks list` describes them, and `mullet hook <name>` runs one. `--hooks session-strategy` opens each session with one question about budget and delegation, which `mullet strategy set` records and `mullet job plan` follows.

## Trawl and MCP

Trawl indexes collections of documents and code, called pools. It searches them by keyword and by meaning together, re-ranks the results and can build a knowledge graph. `mullet mcp` serves it to Claude Code and other MCP clients, with `mullet search`, `mullet pool`, `mullet reindex` and `mullet graph` on the command line. [docs/REINDEX.md](docs/REINDEX.md) covers pools and crawling websites.

The MCP server still has the older delegation tools, which send a task to another model provider. `MULLET_DELEGATION=off` in `~/.config/mullet/.env` switches them off. Trawl, status and budget keep working. Presets such as `local-only` limit which providers they may use:

| Preset | Providers | opencode model |
| --- | --- | --- |
| `balanced-mix` (default) | all | `moonshot/kimi-k2.6` |
| `local-only` | `local` | `lmstudio/ornith-1.5-35b-a3b` |
| `claude-only` | `sonnet`, `haiku`, `opus-4-8`, used by Claude Code itself, never delegated | none |
| `cheap-cloud` | `kimi`, `gemini` | `moonshot/kimi-k2.6` |

The old orchestration modes (ultrawork, ralph, autopilot, team, eco) and the overnight loop are gone.

## Documentation

The guides are in [`docs/`](docs/):

- [Quick start](docs/QUICKSTART.md)
- [Installation](docs/INSTALL.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Settings: limits, toolsets, the API server](docs/CONFIGURATION.md)
- [Model routing](docs/MODEL_ROUTING.md)
- [API reference](docs/API.md)
- [Reindexing knowledge pools](docs/REINDEX.md)
- [Changelog](CHANGELOG.md)

`mullet --help` lists every command. GitHub Actions runs the tests on Linux and macOS; see [CONTRIBUTING.md](CONTRIBUTING.md) to work on Mullet.

## Licence

Mullet is source-available under the [Business Source License 1.1](LICENSE). You may copy, change and redistribute it, and use it outside production. You may also use it in production, free, for any purpose: on your own, inside a company or team, on any number of machines, to build software you sell or ship. The one thing the licence does not cover is offering Mullet, or a service whose value comes substantially from it, to other people as a hosted, managed or embedded service (an API or SaaS that runs Mullet's job runner or search for them). That needs a commercial licence; [LICENSE](LICENSE) says how to ask.

Each version becomes available under the Apache License 2.0 on 3 April 2030, or four years after its first public release if that comes sooner.
