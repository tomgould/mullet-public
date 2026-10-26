# Mullet install guide

> **Goal:** install Mullet, connect it to Claude Code, and plan your first job in about ten minutes.
>
> **What Mullet is:** a runner for large coding jobs. You approve a task tree; Mullet runs each step with your own logged-in Claude Code in a sandboxed private clone, checks every step with tests and checks it runs itself, and hands back a branch with an evidence report. It also gives Claude Code an MCP server with knowledge-search tools. See [MULLET2.md](MULLET2.md) for the design.

---

## 1. Prerequisites

- **Linux or macOS.** On Windows, use WSL. Jobs run in a bubblewrap sandbox, so they need Linux for now; the macOS sandbox is planned.
- **Python 3.10 to 3.12**, and **git**.
- **uv** ([install](https://docs.astral.sh/uv/)) or **pipx** ([install](https://pipx.pypa.io/)). The installer uses whichever it finds.
- **Claude Code**, installed and logged in. Mullet's jobs run your own `claude` CLI under your subscription:
  ```bash
  npm i -g @anthropic-ai/claude-code
  claude          # then /login
  ```
- **On Linux, for jobs:** `bubblewrap` (Mullet's sandbox) and `socat` (Claude Code's own sandbox):
  ```bash
  sudo apt install bubblewrap socat     # Debian and Ubuntu
  ```
  Where bubblewrap can't run, Docker or Podman can confine checks and gates instead: `mullet sandbox build-image`, then `sandbox.backend: docker` (see [CONFIGURATION.md](CONFIGURATION.md#sandbox)). opencode jobs still need bubblewrap.

Provider API keys are optional. Jobs need none. They matter only for Mullet's delegation tools and model-backed knowledge features (section 6).

---

## 2. Install Mullet

```bash
curl -fsSL https://raw.githubusercontent.com/tomgould/mullet-public/main/scripts/install.sh | bash
```

The installer runs `uv tool install` (or `pipx install`) from the public GitHub repository. The install is not editable, so Mullet never runs from a tree it may be editing. Run it again to update. To pin a version, add `--version X.Y.Z` (the release tag `vX.Y.Z`) or `--ref <branch|tag|sha>`; `--dry-run` prints the install command and stops. With curl: `curl -fsSL <url> | bash -s -- --version X.Y.Z`. You can also run the same command yourself:

```bash
uv tool install git+https://github.com/tomgould/mullet-public.git
# or: pipx install git+https://github.com/tomgould/mullet-public.git
```

With an optional extra (`gemini`, `bedrock`, `postgres` or `all`):

```bash
curl -fsSL https://raw.githubusercontent.com/tomgould/mullet-public/main/scripts/install.sh | MULLET_EXTRAS=gemini bash
uv tool install "mullet[gemini] @ git+https://github.com/tomgould/mullet-public.git"
```

> **Never `pip install mullet`.** The `mullet` package on PyPI is an unrelated project.

**Working on Mullet itself:** clone the repository and run `pip install -e ".[dev]"` in a virtualenv. Keep that separate from the copy you use for jobs, so a job never runs the code it is changing.

---

## 3. Check your machine

```bash
mullet doctor
```

It is read-only and makes no network calls (add `--full` to probe the provider APIs and check GitHub for a newer Mullet). It checks:

- Python, the `claude` CLI (and its version) and git;
- on Linux, bubblewrap, Mullet's sandbox self-test and `socat`;
- whether `ANTHROPIC_API_KEY` is set. In `claude -p` that key silently replaces your subscription login and bills the API, so the doctor warns about it. Mullet's jobs leave it out unless the job was planned with `--billing api`;
- what `mullet setup-claude` has set up, and any hooks an older version left pointing into a source checkout;
- the optional providers, knowledge engine, skills, tools and config.

Green ✓ is ready, yellow ! needs a look, cyan · is information only.

---

## 4. Connect Mullet to Claude Code

```bash
mullet setup-claude --dry-run    # shows what it would change
mullet setup-claude
```

By default it:

| What | How |
|:-----|:----|
| Registers the MCP server | `claude mcp add --scope user mullet -- <mullet> mcp`, available in every project |
| Installs skills | `~/.claude/skills/mullet-job/SKILL.md` and `~/.claude/skills/pool-improve/SKILL.md` |

It does not touch `CLAUDE.md`, adds no hooks and never writes into the current repository. `<mullet>` is the `mullet` on your PATH. Restart Claude Code afterwards.

**Hooks are opt-in.** `mullet setup-claude --hooks list` shows what each one does; add the ones you want with `--hooks skill-activator,observation-capture` (or `--hooks all`). They run as `mullet hook <name>` from the installed Mullet, so they keep working after an update.

**Status line:** `--statusline` sets Claude Code's status line to `mullet-hud`, if you don't have one already.

**Undo:** `mullet setup-claude --uninstall` removes exactly what setup-claude added and nothing else. Every change is recorded in `~/.config/mullet/setup-manifest.json` (`$MULLET_CONFIG_DIR` if that is set). Running setup-claude again changes nothing unless something is missing.

**One project only:** instead of `setup-claude`, run `claude mcp add --scope project mullet -- mullet mcp` in that project.

---

## 5. Your first job

In any git repository, with the changes you want the job to start from committed:

```bash
mullet job plan "add type hints to the utils package"
mullet job show <id>       # the plan and the checks that will run
mullet job approve <id>
mullet job run <id>        # runs in the background; mullet job status <id> to follow it
mullet job report <id>     # what each step did, with its gate results
mullet job merge <id>      # copies the job's branch into your repo; you merge it
```

The first `plan` saves a project profile with the setup and test commands it detected and prints its path. Without a test command every step ends up "unverified", never "passed". Inside Claude Code, the `/mullet-job` skill walks through the same steps.

---

## 6. Optional: provider keys and a local model

Mullet's delegation tools (`mullet_run`, `mullet_opencode`, `mullet_swarm`) and CLI tasks send work to other models. Put their keys in `~/.config/mullet/.env` (`$MULLET_CONFIG_DIR/.env` if that is set), not in your shell profile. A source checkout's `.env` is still read, for development. Variables already in the environment win, and a project's own `.env` is never read.

```
KIMI_API_KEY=sk-...           # Kimi K2 (platform.kimi.ai)
OPENROUTER_API_KEY=sk-or-...  # OpenRouter
GOOGLE_API_KEY=AIza...        # Google Gemini
```

`.env.example` lists the rest. `ANTHROPIC_API_KEY` is optional and billed per token; see the warning in section 3.

**Turning delegation on:** delegation is off by default, so Mullet sends no work to any model provider until you set `MULLET_DELEGATION=on` in `~/.config/mullet/.env`. While it is off, the delegation tools, CLI task runs, the REPL and `mullet serve`'s chat endpoints say that delegation is off. Trawl search, knowledge pools and the read-only tools work as normal.

**A local model:** start [LM Studio](https://lmstudio.ai) (port `1234`) or [Ollama](https://ollama.com) (port `11434`) and load a model; Mullet finds it, and `mullet doctor` shows it. On a machine with 10GB of RAM or less, use a 2B to 3B model. On a Mac, don't install the `[gpu]` extra: it pulls NVIDIA-only `pynvml`.

---

## 7. Local LLM watchdog

`mullet local` keeps the local LLM server usable and restarts it when it gets stuck.
It probes LM Studio on `localhost:1234` (or whatever `LOCAL_LLM_URL` points at) without
sending a real completion.

```bash
mullet local status              # URL, reachability, latency, loaded models, GPU memory
mullet local restart [--hard]    # restart the LM Studio service or lms server
mullet local check [--fix]       # decide whether a restart is needed; --fix applies it
```

`mullet local watchdog --install` writes a systemd user timer that runs
`mullet local check --fix` every two minutes. If your system has no systemd user
units, it prints the equivalent crontab line instead.

Restart rules:
- unreachable, or a probe slower than 10 s, on two checks in a row;
- a model whose state is "loading" for longer than 5 minutes across checks;
- when a model is loaded, a single-token generation request times out on two checks in a row.

A 5-minute cooldown after each restart prevents a broken install from looping. Every
restart is logged as JSON at `<state dir>/local-watchdog.log` and sent through the
usual notification channels.

---

## 8. Optional: knowledge pools

Mullet's Trawl engine indexes codebases and documents for semantic search, which Claude Code can query through the `trawl_*` MCP tools.

```bash
mullet pool list
mullet reindex --pool <name>
```

Set `TRAWL_POOLS_DIR` to keep pools somewhere else, such as a larger disk. See [REINDEX.md](REINDEX.md) for the full reference.

---

## 9. Troubleshooting

| Problem | Fix |
|:--------|:----|
| `mullet: command not found` | Run `uv tool update-shell` (or `pipx ensurepath`) and open a new terminal. |
| `uv` refuses because a `mullet` command already exists | Another install is in the way. Remove it, or run the installer with `MULLET_FORCE=1`. |
| MCP tools don't appear in Claude Code | Run `mullet setup-claude`, then fully restart Claude Code. `claude mcp list` should show `mullet`. |
| `mullet doctor` warns about hooks that point into a checkout | An older setup-claude added them by path. Remove those entries from `~/.claude/settings.json`, then add hooks with `mullet setup-claude --hooks ...`. |
| `mullet job run` refuses | It says why: no sandbox, no `socat`, or no `claude`. `mullet doctor` shows the same checks. Don't work around the sandbox. |
| Jobs bill the API instead of your subscription | The job was planned with `--billing api`. Plan it again without that flag. |

If something still doesn't work, run `mullet doctor --full` and include its output when you ask for help.
