# Settings

Mullet reads one YAML file, `config.yaml` in the config directory (`~/.config/mullet` by default; `mullet paths` prints it). `mullet config show [--json] [section]` lists every active setting with its default and where its value came from. The sections are `ladder`, `budget`, `delegation`, `session`, `routing`, `providers`, `skills`, `other` and `unrecognised`. A setting whose name contains `key`, `token`, `secret` or `password` shows as `***`, and a provider's API key shows only `set` or `unset`. Keys in the file that Mullet does not recognise are listed under `unrecognised`.

Job commands (`mullet job ...`, `mullet ladder report`) read job data from the jobs directory and profiles from the profiles directory. If your jobs live elsewhere, set `MULLET_JOBS_DIR` and `MULLET_PROFILES_DIR` for those commands too.

## Spending limits

Nothing is limited until you set a value. All amounts are dollars.

```yaml
ladder:
  max_spend: 2.0              # per job, on the paid rungs (default 2.0)
  prices:                     # $ per million tokens, for providers that don't state cost
    moonshotai/kimi-k2.7-code: {input: 0.6, output: 2.5, cache_read: 0.15}

budget:
  envelope:                   # every paid rung, every job together
    daily: 25.0               # per UTC day
    weekly: 100.0             # per week, from Monday 00:00 UTC
  per_rung:                   # the same windows for one paid rung (kimi or openrouter)
    kimi: {daily: 15.0}
    openrouter: {weekly: 40.0}
  warn_below_usd: 5.0         # `mullet doctor --full` and notifications flag credit under this (default 5.0)
```

- `ladder.max_spend` caps one job. `budget.envelope` and `budget.per_rung` cap all jobs together, over the day or the week. The window spend is every paid attempt in the ledger that started inside the window, running attempts included.
- Each attempt on a paid rung gets an allowance: the least of what is left of the job's cap (lowered by the envelope) and what is left under that rung's `per_rung` limits. The key proxy refuses further requests with HTTP 402 once the attempt has spent it, and the attempt moves up the ladder.
- A paid rung with no room is skipped for the next attempt, and the reason is recorded. A per-rung limit pauses the job only when no paid rung is left. `mullet job resume <job> --max-spend <dollars>` raises a job's cap.
- `budget.per_task`, `budget.per_session` and `budget.daily` belong to the older single-task path. Gated jobs do not read them; `mullet config show budget` marks them as such.
- `mullet doctor` warns when no envelope or per-rung limit is set. With `--full` it also probes each paid provider's credit and warns under `budget.warn_below_usd`. `mullet job ladder` prints the same credit line and warnings unless you pass `--offline`.

### Where a cost comes from

The key proxy reads the usage in every provider response. OpenRouter states the real cost of each request (Mullet asks for it with `usage.include`). Other providers are priced from `ladder.prices` (`input`, `output` and `cache_read`, in dollars per million tokens). A model with no configured price falls back to `ladder.fallback_price`, which defaults to $2 in and $8 out, set high on purpose so an unpriced model reaches its cap early. An attempt's recorded cost never falls below what the provider's responses say.

### Provider check

At the start and end of a job Mullet records each paid provider's own usage or remaining credit. `mullet job report` then prints a "Provider check" block that sets the ledger's spend for that provider beside the provider's figure and says whether they agree. A difference of up to 3% of the recorded spend, or $0.25 if that is more, counts as agreeing. The provider's figure also counts anything else using the same key, so a difference is a prompt to look, not proof of a fault.

## Session start: budget and delegation

The `session-strategy` hook (`mullet setup-claude --hooks session-strategy`) opens each Claude Code session with one short question: this session's budget in dollars, and the delegation strategy. It gives Claude local facts only (envelope left today and this week, Claude's weekly use, the active preset, night mode) and a recommendation, then Claude records the answer:

```bash
mullet strategy set --budget 3 --delegation ladder   # also --note TEXT
mullet strategy                                      # show it, when it was set and when it expires
mullet strategy clear
```

| Choice | Meaning |
|:-------|:--------|
| `ladder` | Cheap and free models first, Claude last (the job ladder) |
| `claude-only` | Jobs run on Claude Code alone, no cheaper rungs |
| `off` | Nothing is delegated |

While the strategy is fresh, `mullet job plan` uses its budget as the job's cap when `--max-spend` is not given, plans a `claude-only` job with one Claude Code worker and no ladder, and refuses under `off`. Explicit flags always win, and `budget.envelope` still applies when the job runs. `off` also turns delegation off exactly as `MULLET_DELEGATION=off` does: the MCP delegation tools refuse and the delegation hooks stay quiet.

The recommendation follows simple rules (`mullet.hooks.session_strategy.recommend`): no room left under the envelope means `claude-only`, or `off` when Claude's paced weekly pressure is also above `ladder.claude_high`; otherwise `ladder`. The suggested budget is `ladder.max_spend`, capped at the envelope room left. With a fresh strategy the hook tells Claude not to ask again.

```yaml
session:
  ask_at_start: true          # false silences the hook (default true)
  strategy_ttl_hours: 12      # how long a strategy stays in force (default 12)
```

## Notifications

Every job notification (blocked, paused, spend cap reached, finished) ends with a line such as `Credit: openrouter $12.40 left; Claude week 62% used`, when Mullet can read those figures. It holds only provider names and numbers, so it is also sent to Slack.

## What delegated models may touch

These settings apply to the single-task path (`mullet "<task>"`), the API server and the MCP delegation tools. Job workers have their own sandbox.

### `delegation.toolsets`

A model that reads a hostile web page or file could be talked into misusing tools that act on your machine, so non-Claude providers get a safe set by default.

```yaml
delegation:
  toolsets:
    default: [web, browser, session, vision, trove, lsp]   # every provider not listed below
    kimi: all                                              # `all` is every toolset
    local: [terminal, file, web, code]
```

- Without a `delegation.toolsets` block, Claude's own providers (`anthropic`, `claude-cli`, `claude_max`) get every toolset and any other provider gets `web`, `browser`, `session`, `vision`, `trove` and `lsp`.
- A value is `all` or a list of toolset names. To give a provider command or file access, list `terminal` or `file` for it, as `local` does above. `mullet --tools` lists the toolsets.
- `default` replaces the safe default for providers you do not name.

### `delegation.file_deny`

The file and docker tools refuse locations that hold credentials. The default list covers SSH, cloud and token files, Mullet's own config and keys, and `.env` files: `~/.ssh`, `~/.gnupg`, `~/.aws`, `~/.azure`, `~/.kube`, `~/.config/gcloud`, `~/.config/gh`, `~/.docker/config.json`, `~/.netrc`, `~/.git-credentials`, `~/.npmrc`, `~/.pypirc`, `~/.mullet`, `~/.config/mullet`, `~/.local/share/mullet`, `~/.local/share/opencode/auth.json`, `~/.claude/.credentials.json`, `~/.claude.json`, `.env`, `.env.*`, `*.pem`, `id_rsa`, `id_ed25519`, `id_ecdsa`, `/var/run/docker.sock`, `/run/docker.sock` and `/etc/shadow`.

```yaml
delegation:
  file_deny: ["~/.ssh", "~/.aws", "*.pem", ".env", ".env.*"]
```

- Setting `file_deny` replaces the whole default list.
- A pattern with no `/` matches a file or directory name at any depth. A pattern with a `/` is a path (`~` is your home directory) that covers everything under it, and may use `*`.
- Symlinks are resolved before the check, so a link cannot hide its target. The docker tool also refuses a mount of a denied location or of a directory that contains one.

## `mullet serve`

`mullet serve [--host HOST] [--port PORT] [--show-token]` serves the REST and WebSocket API on `127.0.0.1:3000`. A web page in your browser can send requests to `localhost` as easily as to its own site, so the server checks every request before it does anything:

- The `Host` header must be `localhost`, `127.0.0.1`, `::1`, the address the server is bound to (unless that is `0.0.0.0` or `::`), or an entry in `server.allowed_hosts`. Anything else gets 403.
- An `Origin` header, when present, must match `server.allowed_origins` (default `http://localhost:*` and `http://127.0.0.1:*`). Anything else gets 403.
- In solo mode, every path except `/health` needs this install's token, or the request gets 401. The token is made on first use and kept in the file `serve-token` in the config directory (mode 0600). Send it as `Authorization: Bearer <token>` or `X-Mullet-Token: <token>`. A WebSocket on `/ws` may also send `?token=<token>`, because browsers cannot set headers there. `mullet serve --show-token` prints the token file's path and the token.

```yaml
server:
  require_token: true                          # default; false turns the token check off
  allowed_origins: ["http://localhost:*", "https://tools.example.com"]
  allowed_hosts: ["mullet.internal"]           # add names when binding to a non-local address
```

In enterprise mode (`MULLET_MODE=enterprise`) the shared token is not used. REST requests carry a personal access token as `Authorization: Bearer <token>`, and a WebSocket client sends it in an `authenticate` message.

## Job protections

- **Protected paths.** A step may not change `.git`, `.gitmodules`, `.gitattributes`, `.lfsconfig`, CI workflows (`.github/workflows`, `.github/actions`, `.gitlab-ci.yml`), `.husky`, `.pre-commit-config.yaml`, files your own tools run when you open the repo (`.envrc`, `.vscode`, `.devcontainer`, `.npmrc`, `.yarnrc`, `.yarnrc.yml`, `.pypirc`), `AGENTS.md`, `CLAUDE.md`, `.claude`, `.mcp.json`, `opencode.json`, `.mullet`, `.env` files, lockfiles, or test runner configuration, unless the approved plan allows it for that step. A repo's stored profile can set its own `protected_paths`.
- **Crashed gates pause, they do not judge.** If a gate crashes, or a required review gives no verdict, the checks run again on the same work up to three times. Then the job pauses, nothing is committed, and `mullet job resume` runs the checks again without asking the worker to redo the work.
- **Flaky tests.** A failing test is rerun up to twice. It is excused only if it passed in one of the runs.
- **Attempts below Claude.** `ladder.max_below_claude` (default 3) caps the counted attempts per step on the rungs below Claude, before or after Claude's own. It never goes over the cap.
- **OpenRouter privacy.** Every request that goes to OpenRouter asks for hosts that do not keep or train on prompts and, by default, only for hosts serving fp8, fp16, bf16 or fp32 copies. Hosts that do not say which copy they serve (`unknown`) are left out unless you list `unknown` under `ladder.routing.<model>.quantizations`.
- **Single-worker opencode jobs and `:free` models.** A job planned with `--executor opencode --no-ladder` has the same spending cap and shared limits when its model is paid. A `:free` model runs only on a repo that passes the free rung's public check.

## Doctor checks

`mullet doctor` is read-only and offline. It checks Python, the `claude` command, git, the job sandbox, the GPU, the Claude Code setup that `mullet setup-claude` recorded, and whether `ANTHROPIC_API_KEY` is set. It warns when no spending limit is set, and when the toolset policy lets a non-Claude provider run commands or touch files (`delegation.toolsets`). `--full` adds the network checks, including each paid provider's credit, which it flags under `budget.warn_below_usd`.
