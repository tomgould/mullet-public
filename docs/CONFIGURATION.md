# Settings

Mullet reads one YAML file, `config.yaml` in the config directory (`~/.config/mullet` by default; `mullet paths` prints it). `mullet config show [--json] [section]` lists every active setting with its default and where its value came from. The sections are `ladder`, `budget`, `delegation`, `session`, `sandbox`, `routing`, `providers`, `skills`, `other` and `unrecognised`. A setting whose name contains `key`, `token`, `secret` or `password` shows as `***`, and a provider's API key shows only `set` or `unset`. Keys in the file that Mullet does not recognise are listed under `unrecognised`.

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

Scripts and CI can drive jobs through the Python API, which applies the same strategy and budget envelope: see [API_PYTHON.md](API_PYTHON.md).

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

## Sandbox

Every command a job runs on your machine (setup, checks, gates and the opencode worker) runs in a sandbox: the filesystem read-only except the clone and the paths the job names, no network except for setup commands, an environment built from an allowlist with no provider keys, and a timeout that kills the whole process tree. If the sandbox can't start, the job is refused; nothing ever runs unsandboxed instead. `mullet doctor` shows which backend is active and why. The guarantees every backend keeps are in `docs/adr/0007-sandbox-backends.md`.

```yaml
sandbox:
  backend: bubblewrap         # bubblewrap, docker, podman or auto (default bubblewrap)
  image: mullet-sandbox:2.3.1 # docker and podman: the image (default mullet-sandbox:<Mullet version>)
  memory: 8g                  # docker and podman: each container's memory limit, swap included (default 8g, at least 64m)
  pids_limit: 4096            # docker and podman: the most processes and threads per container (default 4096, 64 to 1000000)
```

- `bubblewrap` uses Linux user namespaces through `bwrap`, with no daemon and no root.
- `docker` and `podman` run each command in a throwaway container from a local image, as your own uid, with every capability dropped, no-new-privileges, a read-only root, no network (setup commands get the engine's default network), only the clone and the job's paths mounted, and the environment passed on the container's stdin, never on a command line. Each container's PID 1 is Mullet's own init, which kills the container when Mullet goes away, and every container is labelled with the Mullet process and user that started it, so runner start and resume, `mullet job stop` and `mullet doctor` remove your containers whose owner has died (each gives up after 15 s if the daemon is wedged). Without `/etc/machine-id`, a random id kept in Mullet's state directory (`sandbox-host-id`) tells this install's containers from those of another sharing the daemon, such as a second WSL2 distro. If the engine or its daemon is missing, the daemon runs without seccomp, or the image is missing or lacks the `mullet-sandbox` labels, jobs are refused with the reason. Build the image once with `mullet sandbox build-image` (`--engine podman`, `--tag NAME`, `--print` to show the command); for toolchains it lacks, build your own image `FROM` it (it keeps the labels; keep `/usr/bin/python3` too) and name it in `sandbox.image`. An image built by an older Mullet needs rebuilding once, as it has no labels. The opencode worker can't run under these backends, since it needs a sandbox nested inside the container; use bubblewrap for opencode jobs. Linux only for now.
- `auto` takes the first backend that works on this machine, and refuses all work when none does: bubblewrap, then docker, then podman.
- `seatbelt` is reserved for a backend still to come. Choosing it refuses all work with a message that says so.
- A value Mullet doesn't recognise is reported by `mullet config show` and `mullet doctor`, and its default is used.

The container backend's design is in `docs/adr/0011-container-sandbox-backend.md`.

## Job protections

- **Protected paths.** A step may not change `.git`, `.gitmodules`, `.gitattributes`, `.lfsconfig`, CI workflows (`.github/workflows`, `.github/actions`, `.gitlab-ci.yml`), `.husky`, `.pre-commit-config.yaml`, files your own tools run when you open the repo (`.envrc`, `.vscode`, `.devcontainer`, `.npmrc`, `.yarnrc`, `.yarnrc.yml`, `.pypirc`), `AGENTS.md`, `CLAUDE.md`, `.claude`, `.mcp.json`, `opencode.json`, `.mullet`, `.env` files, lockfiles, or test runner configuration, unless the approved plan allows it for that step. A repo's stored profile can set its own `protected_paths`.
- **Review standards.** `review_standards` in a profile (stored, or the repo's `.mullet/project.yaml`) names a repo-relative Markdown file of your coding standards; default none. Only the G7 reviewer sees it, fenced as data and capped at 20,000 characters, and it is read from the job's base commit, so a step can't edit the standards its own review applies. A missing file is ignored with a note in the gate details. A review that finished but gave no verdict is asked again in the same session, only for the verdict, `jobs.reask_attempts` times (default 1; 0 pauses at once); one cut off by its turn limit or clock pauses the job instead. A one-way step is always reviewed, even with `--no-review`, and an approval it only got when asked again pauses the job for you to check.
- **Crashed gates pause, they do not judge.** If a gate crashes, or a required review gives no verdict, the checks run again on the same work up to three times. Then the job pauses, nothing is committed, and `mullet job resume` runs the checks again without asking the worker to redo the work.
- **Flaky tests.** A failing test is rerun up to twice. It is excused only if it passed in one of the runs.
- **Attempts below Claude.** `ladder.max_below_claude` (default 3) caps the counted attempts per step on the rungs below Claude, before or after Claude's own. It never goes over the cap.
- **OpenRouter privacy.** Every request that goes to OpenRouter asks for hosts that do not keep or train on prompts and, by default, only for hosts serving fp8, fp16, bf16 or fp32 copies. Hosts that do not say which copy they serve (`unknown`) are left out unless you list `unknown` under `ladder.routing.<model>.quantizations`.
- **Single-worker opencode jobs and `:free` models.** A job planned with `--executor opencode --no-ladder` has the same spending cap and shared limits when its model is paid. A `:free` model runs only on a repo that passes the free rung's public check.
- **Looking at a failed attempt.** `mullet job show <job> --attempt N` prints the attempt's outcome, reason and gate results; add `--diff` for the work it left. `mullet job export <job> --attempt N --branch NAME` copies that work into your repo as a new branch, takes only a plain branch name (`rescue`, `rescue/attempt-1`; not `refs/heads/...`, `heads/...` or `@{-1}`), refuses an existing branch unless you pass `--force`, and never touches the branch you have checked out. If the work was not kept, both say so.

## Jobs

How a job run treats slow or silent steps, under `jobs:` in the config file. A bad value, including `.inf` and `.nan`, is reported by `mullet config show` and its default kept; it never stops a job.

```yaml
jobs:
  reask_attempts: 1          # times to ask again, in the same session, for a reply without its structured part (0 turns it off)
  idle_minutes: null         # minutes of silence before a worker is stopped; null derives it
  grace_seconds: 60          # seconds a worker may take to exit after its final reply (5 to 3600)
  git_timeout_seconds: 600   # longest a single git command on a job clone may run
  commit_body: full          # full or minimal: what goes under each step commit's subject
```

- `idle_minutes`: any line on the worker's output or stderr resets the clock. A stopped worker's attempt does not count against it. Unset, it is the larger of 15 minutes and 1.5 times the profile's longest check or test timeout.
- `grace_seconds`: after that, the worker is killed and its work is judged by the gates as a finished run. It is at least 5 (at most 3600): Claude Code saves its session file after its final reply, and killing it sooner could leave a failed attempt with no session to resume.
- `git_timeout_seconds`: a git command that runs longer gets SIGTERM, then 5 seconds to remove its lock files and exit, then SIGKILL.
- `commit_body`: `full` (the default) puts a "Mullet step n1 (kind) passed gates ..." line and the worker's own summary, quoted and marked unverified, under each step commit's subject. `minimal` commits the subject alone. If your repo's rules forbid tool or AI lines in commit messages, set `minimal`; the gate results stay in the job's evidence report either way.

The design is in `docs/adr/0010-worker-idle-timeout-and-grace.md`.

## Doctor checks

`mullet doctor` is read-only and offline. It checks Python, the `claude` command, git, the job sandbox and its backend (`sandbox.backend`), the GPU, the Claude Code setup that `mullet setup-claude` recorded, and whether `ANTHROPIC_API_KEY` is set. It warns when no spending limit is set, and when the toolset policy lets a non-Claude provider run commands or touch files (`delegation.toolsets`). `--full` adds the network checks, including each paid provider's credit, which it flags under `budget.warn_below_usd`.

Design decisions behind these settings are recorded in [adr/](adr/README.md).
