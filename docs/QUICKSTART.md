# Mullet Quick Start

From nothing to a planned job in a few minutes. [INSTALL.md](INSTALL.md) has the details.

## 1. Install

On Linux or macOS, with Python 3.10 to 3.12, git and either uv or pipx:

```bash
curl -fsSL https://raw.githubusercontent.com/tomgould/mullet-public/main/scripts/install.sh | bash
```

That runs `uv tool install git+https://github.com/tomgould/mullet-public.git` (or the pipx equivalent). Never `pip install mullet`: that PyPI package is someone else's.

You also need Claude Code installed and logged in, and on Linux `bubblewrap` and `socat` for jobs.

## 2. Check your machine

```bash
mullet doctor
```

It checks Python, the `claude` CLI, git, the job sandbox (bubblewrap and `socat` on Linux) and your Claude Code setup, and warns if `ANTHROPIC_API_KEY` is set, because `claude -p` would then bill the API instead of your subscription. It is read-only and offline; `--full` adds the network checks.

## 3. Connect Mullet to Claude Code

```bash
mullet setup-claude --dry-run    # see the changes first
mullet setup-claude
```

| What | Where |
|------|-------|
| MCP server `mullet` | `claude mcp add --scope user mullet -- <mullet> mcp` |
| Skills `mullet-job` and `pool-improve` | `~/.claude/skills/<name>/SKILL.md` |
| The `session-strategy` hook, plus any others named with `--hooks <names>` | `~/.claude/settings.json`, as `<mullet> hook <name>` |

`mullet setup-claude --hooks list` describes the hooks; only `session-strategy` is added by default (`--hooks none` skips it). `mullet setup-claude --uninstall` removes exactly what setup-claude added. Restart Claude Code afterwards.

## 4. Run a job

In a git repository, with the starting point committed:

```bash
mullet job plan "split the payments module into smaller files"
mullet job show <id>
mullet job approve <id>
mullet job run <id>
mullet job status <id>       # or: mullet job tail <id> -f
mullet job report <id>
mullet job merge <id>        # copies the job's branch into your repo
```

Each step runs with your own `claude -p` in a sandboxed private clone and passes only when Mullet's gates pass: a real change, within scope, no secrets or weakened tests, and the build and tests no worse than on the base commit. `stop` pauses a job and `resume` carries on. Your checkout is untouched until `merge`, and `merge` never touches a checked-out branch.

## 5. Optional: knowledge search

```bash
mullet pool list
mullet reindex --pool <name>
```

Then, in Claude Code: "Use `trawl_search_code` to find where authentication is handled."

## 6. Optional: delegation to other models

Mullet can also send work to cheaper models, but only after you opt in with `MULLET_DELEGATION=on` in `~/.config/mullet/.env`: `mullet "<task>"` on the command line, and `mullet_run`, `mullet_opencode` and `mullet_swarm` over MCP. Put the provider keys in `~/.config/mullet/.env` (`$MULLET_CONFIG_DIR/.env` if that is set):

```
KIMI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...
GOOGLE_API_KEY=AIza...
```

or start LM Studio or Ollama for a free local model. Delegation stays off until `MULLET_DELEGATION=on` is set.

```bash
mullet explain "refactor auth module for OAuth2"   # where a task would route, without running it
mullet models                                      # models and pricing by provider
mullet preset use local-only                       # routing policy: balanced-mix, local-only, claude-only, cheap-cloud
```
